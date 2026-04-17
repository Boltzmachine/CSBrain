"""
DINOv2 self-distillation components for EEG foundation model.

Provides:
    - DINOHead: projection head with weight-normalised prototype layer
    - FrequencySubBandView: EEG augmentation via random frequency-band selection
    - cosine_schedule: schedule builder for EMA momentum / teacher temperature
    - DINOEEGModel: teacher–student wrapper with EMA, DINO + iBOT losses,
      and retained image-alignment objective

References:
    - DINOv2: https://github.com/facebookresearch/dinov2
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dino_loss import DINOLoss, iBOTPatchLoss, KoLeoLoss


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class DINOHead(nn.Module):
    """
    DINOv2 projection head.

    Architecture: multi-layer MLP → L2-normalise → weight-normalised
    linear projection to *n_prototypes* dimensions.
    """

    def __init__(self, in_dim, hidden_dim=256, bottleneck_dim=256,
                 n_prototypes=4096, n_layers=3, use_bn=True):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dim_in = in_dim if i == 0 else hidden_dim
            dim_out = bottleneck_dim if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(dim_in, dim_out))
            if i < n_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dim_out))
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.Linear(bottleneck_dim, n_prototypes, bias=False)
        nn.utils.parametrizations.weight_norm(self.last_layer, name='weight')

    def freeze_last_layer(self, freeze: bool):
        """Freeze/unfreeze the weight-norm magnitude of the prototype layer.

        DINOv2 keeps the magnitude frozen for the first ~1 epoch to stabilise
        training and prevent uniform collapse. With
        torch.nn.utils.parametrizations.weight_norm, the magnitude is stored
        at ``.parametrizations.weight.original0``.
        """
        for name, p in self.last_layer.named_parameters():
            if 'original0' in name:
                p.requires_grad = not freeze

    def forward(self, x):
        """
        Args:
            x: (..., in_dim)  — works for both (B, d) and (B, C, N, d)
        Returns:
            (..., n_prototypes)
        """
        shape = x.shape[:-1]
        x = x.reshape(-1, x.size(-1))
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x.reshape(*shape, -1)


# ---------------------------------------------------------------------------
# EEG view augmentations
# ---------------------------------------------------------------------------

class FrequencySubBandView(nn.Module):
    """
    Create an augmented EEG view by randomly retaining a subset of
    frequency sub-bands.

    The full time-series (patches concatenated) is FFT-transformed, divided
    into ``n_bands`` equal-width bins, a random subset of bins is kept, and
    the signal is reconstructed via inverse FFT.

    This forces the student to learn representations robust to partial
    spectral information (e.g. only delta + beta, or only alpha + gamma).
    """

    def __init__(self, n_bands=5, min_bands=1, max_bands=None):
        super().__init__()
        self.n_bands = n_bands
        self.min_bands = min_bands
        self.max_bands = max_bands or n_bands

    def forward(self, x):
        """
        Args:
            x: (B, C, N, d) — EEG patches
        Returns:
            (B, C, N, d) — frequency-filtered version
        """
        if not self.training:
            return x

        B, C, N, d = x.shape
        T = N * d

        # Concatenate patches → full time-series
        ts = x.reshape(B, C, T)

        # Forward FFT
        spectral = torch.fft.rfft(ts, dim=-1)  # (B, C, n_freq)
        n_freq = spectral.size(-1)

        # Divide frequency axis into equal-width bands
        band_edges = torch.linspace(0, n_freq, self.n_bands + 1).long()

        # Randomly select a subset of bands (same for all samples in batch)
        n_selected = torch.randint(
            self.min_bands, self.max_bands + 1, (1,)).item()
        selected = torch.randperm(self.n_bands)[:n_selected].sort().values

        # Build binary frequency mask
        freq_mask = torch.zeros(n_freq, device=x.device, dtype=torch.float32)
        for band_idx in selected:
            start = band_edges[band_idx].item()
            end = band_edges[band_idx + 1].item()
            freq_mask[start:end] = 1.0

        # Apply mask and reconstruct
        filtered = spectral * freq_mask.unsqueeze(0).unsqueeze(0)
        ts_filtered = torch.fft.irfft(filtered, n=T, dim=-1)

        return ts_filtered.reshape(B, C, N, d)


# ---------------------------------------------------------------------------
# EEG multi-crop (analogue of DINOv2 global/local crops)
# ---------------------------------------------------------------------------

class EEGLocalCrop(nn.Module):
    """
    Create a *local* EEG crop by selecting a random temporal sub-window
    and a random channel subset.

    This is the EEG analogue of the local spatial crops in DINOv2.  The
    teacher sees the full input (global view); the student additionally
    processes several local crops through the DINO CLS-token loss,
    forcing it to infer global structure from partial context.

    The crop is implemented by zeroing non-selected regions and setting
    ``valid_channel_mask`` / ``valid_length_mask`` so the transformer
    only attends to the selected subset.
    """

    def __init__(self, time_scale=(0.3, 0.7), channel_scale=(0.5, 1.0)):
        super().__init__()
        self.time_scale = time_scale
        self.channel_scale = channel_scale

    def forward(self, batch):
        """
        Args:
            batch: dict with at least 'timeseries' (B, C, N, d) and
                   'ch_coords'.  May contain 'valid_channel_mask' and
                   'valid_length_mask'.
        Returns:
            crop_batch: new dict with cropped timeseries and updated masks.
        """
        x = batch['timeseries']  # (B, C, N, d)
        B, C, N, d = x.shape
        device = x.device

        # ---- random temporal window (same for all samples in batch) ----
        time_frac = torch.empty(1).uniform_(*self.time_scale).item()
        n_time = max(1, int(N * time_frac))
        t_start = torch.randint(0, max(1, N - n_time + 1), (1,)).item()

        # ---- random channel subset ----
        ch_frac = torch.empty(1).uniform_(*self.channel_scale).item()
        n_ch = max(1, int(C * ch_frac))
        ch_idx = torch.randperm(C, device=device)[:n_ch].sort().values

        # ---- build cropped timeseries ----
        x_crop = torch.zeros_like(x)
        # Copy only selected channels in selected time window
        x_crop[:, ch_idx, t_start:t_start + n_time, :] = \
            x[:, ch_idx, t_start:t_start + n_time, :]

        # ---- build masks ----
        vcm = torch.zeros(B, C, dtype=torch.bool, device=device)
        vcm[:, ch_idx] = True
        orig_vcm = batch.get('valid_channel_mask')
        if orig_vcm is not None:
            vcm = vcm & orig_vcm

        vlm = torch.zeros(B, N, dtype=torch.bool, device=device)
        vlm[:, t_start:t_start + n_time] = True
        orig_vlm = batch.get('valid_length_mask')
        if orig_vlm is not None:
            vlm = vlm & orig_vlm

        # ---- assemble crop batch (shallow copy + overrides) ----
        crop_batch = {k: v for k, v in batch.items()}
        crop_batch['timeseries'] = x_crop
        crop_batch['valid_channel_mask'] = vcm
        crop_batch['valid_length_mask'] = vlm

        return crop_batch


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def cosine_schedule(base_value, final_value, total_iters,
                    warmup_iters=0, start_warmup_value=0.0):
    """Build a cosine-annealing schedule array (numpy).

    Optionally includes a linear warmup phase.
    """
    warmup = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(total_iters - warmup_iters)
    cos_values = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / max(len(iters), 1))
    )
    return np.concatenate([warmup, cos_values])


# ---------------------------------------------------------------------------
# Teacher–student wrapper
# ---------------------------------------------------------------------------

class DINOEEGModel(nn.Module):
    """
    DINOv2-style self-distillation wrapper for the EEG foundation model.

    Wraps an existing ``CSBrainAlign`` backbone as the **student**.  A deep
    copy serves as the **teacher**, updated via exponential moving average
    (EMA).  Separate DINO projection heads are attached for CLS-token
    (global) and patch-token (iBOT) distillation.  The student's existing
    image-alignment objective is retained as a secondary loss.

    During ``training_step`` (multi-crop):

    1. The **teacher** receives full, un-augmented EEG input (global view).
    2. The **student** processes one *global* view (freq-augmented +
       patch-masked) for both DINO CLS and iBOT patch losses, plus
       ``n_local_crops`` *local* views (temporal + channel crops +
       freq augmentation) for DINO CLS loss only.
    3. DINO loss (CLS, averaged over all views), iBOT loss (masked
       patches, global view only), KoLeo loss (all CLS tokens), plus
       any backbone losses are combined via weighted sum.
    """

    def __init__(
        self,
        student_backbone: nn.Module,
        d_model: int = 40,
        dino_head_hidden_dim: int = 256,
        dino_head_bottleneck_dim: int = 256,
        n_prototypes: int = 4096,
        dino_head_n_layers: int = 3,
        student_temp: float = 0.1,
        teacher_temp_base: float = 0.04,
        teacher_temp_final: float = 0.07,
        ema_momentum_base: float = 0.992,
        ema_momentum_final: float = 0.9995,
        dino_loss_weight: float = 1.0,
        ibot_loss_weight: float = 1.0,
        koleo_loss_weight: float = 0.1,
        use_freq_subband: bool = True,
        freq_n_bands: int = 5,
        freq_min_bands: int = 1,
        freq_max_bands: int = None,
        # Multi-crop
        n_local_crops: int = 4,
        local_crop_time_scale: tuple = (0.3, 0.7),
        local_crop_channel_scale: tuple = (0.5, 1.0),
        # Last-layer freeze
        last_layer_freeze_iters: int = 1250,
    ):
        super().__init__()
        self.d_model = d_model
        self.dino_loss_weight = dino_loss_weight
        self.ibot_loss_weight = ibot_loss_weight
        self.koleo_loss_weight = koleo_loss_weight
        self.use_freq_subband = use_freq_subband
        self.ema_momentum_base = ema_momentum_base
        self.ema_momentum_final = ema_momentum_final
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_final = teacher_temp_final
        self.last_layer_freeze_iters = last_layer_freeze_iters

        # ---- student backbone (trainable) ----
        self.student = student_backbone

        # ---- teacher backbone (EMA copy, frozen) ----
        self.teacher = copy.deepcopy(student_backbone)
        # Remove components the teacher never uses (saves GPU memory)
        for attr in ('pretrained_image_encoder', 'semantic_readout',
                     'contrastive_proj', 'equiv_projector'):
            if hasattr(self.teacher, attr):
                delattr(self.teacher, attr)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # ---- DINO heads (CLS token) ----
        self.student_dino_head = DINOHead(
            d_model, dino_head_hidden_dim, dino_head_bottleneck_dim,
            n_prototypes, dino_head_n_layers,
        )
        self.teacher_dino_head = copy.deepcopy(self.student_dino_head)
        for p in self.teacher_dino_head.parameters():
            p.requires_grad = False

        # ---- iBOT heads (patch tokens) ----
        self.student_ibot_head = DINOHead(
            d_model, dino_head_hidden_dim, dino_head_bottleneck_dim,
            n_prototypes, dino_head_n_layers,
        )
        self.teacher_ibot_head = copy.deepcopy(self.student_ibot_head)
        for p in self.teacher_ibot_head.parameters():
            p.requires_grad = False

        # Freeze the weight-norm magnitude of the prototype layers for the
        # first ``last_layer_freeze_iters`` iterations. Unfreezed later by
        # ``maybe_unfreeze_last_layer``.
        if last_layer_freeze_iters > 0:
            self.student_dino_head.freeze_last_layer(True)
            self.student_ibot_head.freeze_last_layer(True)
        self._last_layer_unfrozen = last_layer_freeze_iters <= 0

        # ---- loss functions ----
        self.dino_criterion = DINOLoss(
            n_prototypes, student_temp, teacher_temp_base,
        )
        self.ibot_criterion = iBOTPatchLoss(
            n_prototypes, student_temp, teacher_temp_base,
        )
        self.koleo_criterion = KoLeoLoss()

        # ---- augmentation ----
        if use_freq_subband:
            self.freq_subband_view = FrequencySubBandView(
                n_bands=freq_n_bands,
                min_bands=freq_min_bands,
                max_bands=freq_max_bands,
            )

        # ---- multi-crop ----
        self.n_local_crops = n_local_crops
        if n_local_crops > 0:
            self.local_crop = EEGLocalCrop(
                time_scale=local_crop_time_scale,
                channel_scale=local_crop_channel_scale,
            )

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """EMA-update teacher backbone and heads from student weights.

        Only parameters present in *both* student and teacher are updated
        (the teacher has fewer modules — e.g. no pretrained_image_encoder).
        """
        # Backbone — match by name since teacher has fewer modules
        teacher_dict = dict(self.teacher.named_parameters())
        for name, ps in self.student.named_parameters():
            if name in teacher_dict:
                teacher_dict[name].data.mul_(momentum).add_(
                    ps.data, alpha=1 - momentum)

        # DINO head
        for ps, pt in zip(self.student_dino_head.parameters(),
                          self.teacher_dino_head.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)

        # iBOT head
        for ps, pt in zip(self.student_ibot_head.parameters(),
                          self.teacher_ibot_head.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------

    def get_ema_momentum(self, iteration: int, total_iters: int) -> float:
        """Cosine schedule for EMA momentum (increases from base to final)."""
        return (self.ema_momentum_final
                - (self.ema_momentum_final - self.ema_momentum_base)
                * (1 + math.cos(math.pi * iteration / max(total_iters, 1)))
                / 2)

    def get_teacher_temp(self, iteration: int, total_iters: int,
                         warmup_iters: int = 0) -> float:
        """Linear warmup then constant for teacher temperature."""
        if warmup_iters > 0 and iteration < warmup_iters:
            return (self.teacher_temp_base
                    + (self.teacher_temp_final - self.teacher_temp_base)
                    * iteration / warmup_iters)
        return self.teacher_temp_final

    def update_schedules(self, iteration: int, total_iters: int,
                         warmup_iters: int = 0):
        """Update teacher temperature in both loss modules."""
        t_temp = self.get_teacher_temp(iteration, total_iters, warmup_iters)
        self.dino_criterion.teacher_temp = t_temp
        self.ibot_criterion.teacher_temp = t_temp

    def maybe_unfreeze_last_layer(self, iteration: int):
        """Unfreeze the prototype-layer magnitude once warmup has elapsed."""
        if self._last_layer_unfrozen:
            return
        if iteration >= self.last_layer_freeze_iters:
            self.student_dino_head.freeze_last_layer(False)
            self.student_ibot_head.freeze_last_layer(False)
            self._last_layer_unfrozen = True

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, mask=None):
        """
        Full DINOv2 training step with multi-crop.

        **Teacher** receives the full (global) EEG input.
        **Student** processes:
          - 1 *global* view (freq-augmented + patch-masked) → DINO + iBOT
          - N *local* views (temporal + channel crops + freq-augmented) → DINO only

        Returns ``(student_out, info_dict)`` in the same format as
        ``CSBrainAlign.forward`` so the existing ``Trainer`` can process
        it without changes.
        """
        x = batch['timeseries']  # (B, C, N, d)

        # ---- teacher: full input, no augmentation, no gradient ----
        # eval() disables dropout for clean, stable targets
        self.teacher.eval()
        self.teacher_dino_head.eval()
        self.teacher_ibot_head.eval()
        with torch.no_grad():
            teacher_cls, teacher_patches = self.teacher.encode(
                batch, mask=None)
            teacher_cls_out = self.teacher_dino_head(teacher_cls)
            teacher_patch_out = self.teacher_ibot_head(teacher_patches)
        self.teacher.train()
        self.teacher_dino_head.train()
        self.teacher_ibot_head.train()

        # Pre-compute teacher targets (centered + sharpened), update
        # center once so it is not inflated by multi-crop repetitions.
        dino_targets = self.dino_criterion.get_teacher_targets(
            teacher_cls_out)

        # ==============================================================
        # Student global view: freq augmentation + patch mask
        # ==============================================================
        global_batch = {k: v for k, v in batch.items()}
        if self.use_freq_subband and self.training:
            global_batch['timeseries'] = self.freq_subband_view(x)

        student_out, student_info = self.student.forward(
            global_batch, mask=mask)
        global_cls = student_info['global_rep']
        global_patches = student_info['patch_tokens']

        global_cls_out = self.student_dino_head(global_cls)
        global_patch_out = self.student_ibot_head(global_patches)

        # DINO loss on global view
        dino_loss = self.dino_criterion.loss_from_targets(
            global_cls_out, dino_targets)

        # ==============================================================
        # Student local views: temporal + channel crops (DINO CLS only)
        # ==============================================================
        local_cls_list = []
        for _ in range(self.n_local_crops):
            local_batch = self.local_crop(batch)
            if self.use_freq_subband and self.training:
                local_batch['timeseries'] = self.freq_subband_view(
                    local_batch['timeseries'])
            local_cls, _ = self.student.encode(local_batch, mask=None)
            local_cls_list.append(local_cls)
            local_cls_out = self.student_dino_head(local_cls)
            dino_loss = dino_loss + self.dino_criterion.loss_from_targets(
                local_cls_out, dino_targets)

        # Average across all views (1 global + N local)
        n_views = 1 + self.n_local_crops
        dino_loss = dino_loss / n_views
        student_info['dino_loss'] = (self.dino_loss_weight, dino_loss)

        # ==============================================================
        # iBOT loss: global view only (patches must align with teacher)
        # ==============================================================
        if (mask is not None
                and not getattr(self.student, 'project_to_source', False)):
            ibot_targets = self.ibot_criterion.get_teacher_targets(
                teacher_patch_out)
            ibot_loss = self.ibot_criterion.loss_from_targets(
                global_patch_out, ibot_targets, mask)
            student_info['ibot_loss'] = (self.ibot_loss_weight, ibot_loss)

        # ==============================================================
        # KoLeo diversity loss on all student CLS tokens
        # ==============================================================
        if self.koleo_loss_weight > 0:
            all_cls = torch.cat([global_cls] + local_cls_list, dim=0)
            cls_normed = F.normalize(all_cls, dim=-1, p=2)
            koleo_loss = self.koleo_criterion(cls_normed)
            student_info['koleo_loss'] = (self.koleo_loss_weight, koleo_loss)

        # Signal that the primary reconstruction / causal loss should be
        # skipped — DINO + iBOT replace the masked-patch reconstruction.
        student_info['dino_mode'] = True

        return student_out, student_info

    # ------------------------------------------------------------------
    # Inference delegation
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        """Delegate to student backbone for inference."""
        return self.student.forward(*args, **kwargs)

    def encode(self, *args, **kwargs):
        """Delegate to student backbone."""
        return self.student.encode(*args, **kwargs)

    def _load_weights(self, *args, **kwargs):
        """Delegate weight loading to student."""
        return self.student._load_weights(*args, **kwargs)
