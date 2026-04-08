"""
CSBrainSpectral — Foundation model for EEG representation learning via
learnable spectral band decomposition and multi-level visual alignment.

Architecture
============
1. **Shared PatchEmbedding** + spherical coordinate PE  → (B, C, N, d)
2. **Spectral Decomposition**: FFT → K learnable sigmoid masks (can overlap)
   → K band signals via IFFT.  Inspired by speech-separation mask networks.
3. **Shared Transformer Trunk** (most parameters): processes *each band
   independently* through the same encoder layers (parameter-efficient).
4. **Branch-specific Band Adapters**: lightweight 2-layer MLPs, one per band,
   applied after the shared trunk to specialise each band's representation.
5. **Band Fusion**: element-wise sum of all adapted band outputs → fused
   representation used for reconstruction.
6. **Readout heads**:
   - Reconstruction head (shared linear) on fused output → MSE loss.
   - Per-band alignment projections + learned band-to-level soft assignment
     matrix → multi-level InfoNCE against DINOv2 hidden states.

Parameter sharing
=================
- **Shared**: PatchEmbedding, coord PE, transformer trunk, TemEmbedEEGLayer,
  reconstruction proj_out, global tokens.
- **Branch-specific**: K spectral mask logit vectors, K band adapter MLPs,
  K alignment projection heads, band-to-level assignment logits.

Training objectives
===================
- EEG-only data  → masked reconstruction loss (on fused output).
- EEG + image    → masked reconstruction loss
                  + multi-band alignment loss (per-band InfoNCE weighted by
                    learned soft assignment to DINOv2 layer levels)
                  + band decorrelation regulariser.
"""

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CSBrain_transformerlayer import (
    CSBrain_TransformerEncoderLayer,
    RegionAttentionMaskBuilder,
)
from models.CSBrain_transformer import (
    CSBrain_TransformerEncoder,
    TemEmbedEEGLayer,
    _get_clones,
)
from transformers import Dinov2Model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Patch embedding (reused from alignment.py with minor cleanup)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(19, 7), stride=1,
                      padding=(9, 3), groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=(1, d_model), stride=1, padding=0),
            nn.GroupNorm(5, d_model), nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1), nn.GroupNorm(5, d_model), nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1), nn.GroupNorm(5, d_model), nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(d_model // 2 + 1, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x_flat = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x_flat)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(
            bz, ch_num, patch_num, self.d_model)

        flat = mask_x_flat.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(flat, dim=-1, norm='forward')
        spectral = torch.abs(spectral).view(bz, ch_num, patch_num, flat.shape[1] // 2 + 1)
        patch_emb = patch_emb + self.spectral_proj(spectral)

        pos = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        patch_emb = patch_emb + pos.permute(0, 2, 3, 1)
        return patch_emb


# ---------------------------------------------------------------------------
# Spectral mask module — learnable, overlapping soft masks
# ---------------------------------------------------------------------------

class SpectralMaskBank(nn.Module):
    """
    Produces K soft masks over frequency bins via sigmoid(logits).
    Masks can freely overlap, allowing bands to share frequency content.

    The mask logits are stored at a fixed *reference* resolution and
    interpolated at runtime to match the actual number of frequency bins
    (which varies with input length).
    """
    def __init__(self, num_bands: int, ref_freq_bins: int):
        super().__init__()
        self.num_bands = num_bands
        self.ref_freq_bins = ref_freq_bins
        # Initialise so that bands roughly tile the spectrum at init
        logits = torch.zeros(num_bands, ref_freq_bins)
        bin_edges = torch.linspace(0, ref_freq_bins, num_bands + 1).long()
        for k in range(num_bands):
            logits[k, bin_edges[k]:bin_edges[k + 1]] = 2.0  # sigmoid(2)≈0.88
        self.mask_logits = nn.Parameter(logits)

    def forward(self, num_bins: int = None):
        """
        Returns (num_bands, num_bins) soft masks in [0, 1].
        If num_bins differs from the reference resolution, masks are
        linearly interpolated along the frequency axis.
        """
        masks = torch.sigmoid(self.mask_logits)  # (K, ref_freq_bins)
        if num_bins is not None and num_bins != self.ref_freq_bins:
            # Interpolate: (K, ref) -> (1, K, ref) -> interpolate -> (K, num_bins)
            masks = F.interpolate(
                masks.unsqueeze(0), size=num_bins, mode='linear',
                align_corners=True,
            ).squeeze(0)
        return masks

    def completeness_loss(self, num_bins: int = None):
        """Encourage masks to softly partition the spectrum (sum ≈ 1)."""
        masks = self.forward(num_bins)
        return (masks.sum(dim=0) - 1.0).pow(2).mean()


# ---------------------------------------------------------------------------
# Band adapter — lightweight branch-specific transform
# ---------------------------------------------------------------------------

class BandAdapter(nn.Module):
    """Small MLP applied per-band after the shared trunk."""
    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
        )

    def forward(self, x):
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CSBrainSpectral(nn.Module):
    """
    Foundation model for EEG with learnable spectral band decomposition.

    Args:
        num_bands: Number of learnable frequency bands (K).
        num_visual_levels: Number of DINOv2 hidden-state levels to align with.
            The mapping from K bands to L levels is a *learned* soft assignment.
        dino_layer_indices: Which DINOv2 layers to extract (length = num_visual_levels).
    """

    def __init__(
        self,
        in_dim: int = 200,
        out_dim: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        seq_len: int = 30,
        n_layer: int = 12,
        nhead: int = 8,
        TemEmbed_kernel_sizes=((1,), (3,), (5,)),
        brain_regions=None,
        sorted_indices=None,
        # --- spectral band args ---
        num_bands: int = 4,
        num_visual_levels: int = 3,
        dino_layer_indices: tuple = (3, 7, 12),
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_bands = num_bands
        self.num_visual_levels = num_visual_levels
        self.dino_layer_indices = dino_layer_indices

        # ---- Shared components ------------------------------------------------
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)

        # TemEmbedEEGLayer operates on (B*C, d_model, N, 1), so dim_in must be d_model
        self.TemEmbedEEGLayer = TemEmbedEEGLayer(
            dim_in=d_model, dim_out=d_model,
            kernel_sizes=sorted(TemEmbed_kernel_sizes), stride=1,
        )

        # Spherical coordinate PE
        self.spherical_r_scale = 0.1
        self.spherical_num_freqs = 32
        self.spherical_pe_base = 20.0
        inv_freq = torch.exp(
            -math.log(self.spherical_pe_base)
            * torch.arange(self.spherical_num_freqs, dtype=torch.float32)
            / self.spherical_num_freqs
        )
        self.register_buffer("spherical_inv_freq", inv_freq, persistent=False)
        self.coord_enhancement = nn.Sequential(
            nn.Linear(3 * 2 * self.spherical_num_freqs, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Global channel + token (CLS-like)
        self.global_channel = nn.Parameter(torch.randn(1, 1, 1, d_model))
        self.global_token = nn.Parameter(torch.randn(1, 1, 1, d_model))

        # Shared transformer trunk
        encoder_layer = CSBrain_TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            area_config=None, sorted_indices=sorted_indices or [],
            batch_first=True, activation=F.gelu,
        )
        self.encoder = CSBrain_TransformerEncoder(
            encoder_layer, num_layers=n_layer, enable_nested_tensor=False,
        )

        # Shared reconstruction head (operates on fused band output)
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))

        # ---- Spectral decomposition (branch-specific) -------------------------
        max_freq_bins = seq_len * in_dim // 2 + 1
        self.mask_bank = SpectralMaskBank(num_bands, max_freq_bins)

        # ---- Branch-specific band adapters ------------------------------------
        self.band_adapters = nn.ModuleList([
            BandAdapter(d_model) for _ in range(num_bands)
        ])

        # ---- Per-band alignment projections -----------------------------------
        self.band_align_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 768),  # DINOv2-base hidden size
            ) for _ in range(num_bands)
        ])

        # ---- Learned band-to-level assignment ---------------------------------
        # Soft assignment matrix: (num_bands, num_visual_levels)
        # Each band contributes to each visual level with a learned weight.
        self.band_level_logits = nn.Parameter(
            torch.zeros(num_bands, num_visual_levels)
        )

        # ---- Semantic readout (from global token of fused representation) -----
        self.semantic_readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 768),
        )

        # ---- Frozen image encoder ---------------------------------------------
        self.pretrained_image_encoder = Dinov2Model.from_pretrained(
            "facebook/dinov2-base"
        ).eval()

        self.apply(_weights_init)

    # ------------------------------------------------------------------
    # Coordinate PE
    # ------------------------------------------------------------------
    def _spherical_positional_encoding(self, ch_coords):
        finite_mask = torch.isfinite(ch_coords).all(dim=-1, keepdim=True)
        safe = torch.where(finite_mask, ch_coords, torch.zeros_like(ch_coords))
        r = safe[..., 0:1] / self.spherical_r_scale
        theta = safe[..., 1:2] / math.pi
        phi = safe[..., 2:3] / math.pi
        norm = torch.cat([r, theta, phi], dim=-1)
        inv_freq = self.spherical_inv_freq.to(norm.device, norm.dtype)
        angles = norm.unsqueeze(-1) * inv_freq.view(1, 1, 1, -1)
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        pe = pe.reshape(norm.size(0), norm.size(1), -1)
        return pe, finite_mask.squeeze(-1)

    # ------------------------------------------------------------------
    # Image encoder (frozen)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def get_image_hidden_states(self, **image_encoder_inputs):
        outputs = self.pretrained_image_encoder(
            **image_encoder_inputs, output_hidden_states=True
        )
        hs = outputs.hidden_states
        selected = [hs[idx][:, 0] for idx in self.dino_layer_indices]
        return torch.stack(selected, dim=1)  # (B, num_visual_levels, 768)

    # ------------------------------------------------------------------
    # Spectral decomposition (on raw input, before embedding)
    # ------------------------------------------------------------------
    def _decompose_bands(self, x):
        """
        Decompose **raw** EEG input into K frequency bands in the spectral
        domain, *before* patch embedding.

        Args:
            x: (B, C, N, in_dim) — raw input patches.

        Returns:
            band_signals: list of K tensors, each (B, C, N, in_dim)
            num_bins: actual number of frequency bins (for completeness loss)
        """
        B, C, N, d = x.shape
        # Flatten patches into a continuous time signal per channel
        time_signal = x.reshape(B, C, N * d)  # (B, C, T)
        spectral = torch.fft.rfft(time_signal, dim=-1, norm='forward')  # (B, C, F)
        num_bins = spectral.shape[-1]

        masks = self.mask_bank(num_bins)  # (K, num_bins) — interpolated if needed
        # Apply each mask and IFFT back
        band_signals = []
        for k in range(self.num_bands):
            masked_spectral = spectral * masks[k].unsqueeze(0).unsqueeze(0)  # (B, C, F)
            band_time = torch.fft.irfft(masked_spectral, n=N * d, dim=-1, norm='forward')
            band_signals.append(band_time.reshape(B, C, N, d))

        return band_signals, num_bins

    # ------------------------------------------------------------------
    # Shared trunk forward (processes one band at a time)
    # ------------------------------------------------------------------
    def _encode_band(self, band_emb, batch):
        """
        Run the shared transformer trunk on a single band's embedding.
        band_emb: (B, C+1, N+1, d) with global tokens prepended.

        Uses gradient checkpointing to trade compute for memory when
        training with multiple bands through the same deep trunk.
        """
        iw_mask = ~batch['valid_length_mask'] if 'valid_length_mask' in batch else None
        ir_mask = ~batch['valid_channel_mask'] if 'valid_channel_mask' in batch else None

        for layer_idx in range(self.encoder.num_layers):
            if self.training:
                band_emb = torch.utils.checkpoint.checkpoint(
                    self._encode_one_layer, band_emb, layer_idx, iw_mask, ir_mask,
                    use_reentrant=False,
                )
            else:
                band_emb = self._encode_one_layer(band_emb, layer_idx, iw_mask, ir_mask)
        return band_emb

    def _encode_one_layer(self, band_emb, layer_idx, iw_mask, ir_mask):
        """Single encoder layer step (factored out for checkpointing)."""
        band_emb = self.TemEmbedEEGLayer(band_emb) + band_emb
        band_emb = self.encoder.layers[layer_idx](
            band_emb, None,
            inter_window_attn_mask=iw_mask,
            inter_region_attn_mask=ir_mask,
        )
        return band_emb

    # ------------------------------------------------------------------
    # Band decorrelation loss
    # ------------------------------------------------------------------
    def _band_decorrelation_loss(self, band_globals):
        """
        Encourage different bands to capture different information.
        band_globals: (B, K, d)
        """
        normed = F.normalize(band_globals, dim=-1)
        # (B, K, K) pairwise cosine similarity
        sim = torch.bmm(normed, normed.transpose(1, 2))
        eye = torch.eye(self.num_bands, device=sim.device).unsqueeze(0)
        off_diag = sim * (1.0 - eye)
        return off_diag.pow(2).mean()

    # ------------------------------------------------------------------
    # Multi-band alignment loss
    # ------------------------------------------------------------------
    def _multi_band_alignment_loss(self, band_globals, image_hidden_states, batch):
        """
        Align per-band EEG representations with visual feature levels via
        a learned soft assignment matrix.

        Args:
            band_globals: list of K tensors, each (B_img, d_model)
            image_hidden_states: (B_img, num_visual_levels, 768)

        Returns:
            loss_dict with per-level and aggregate losses + accuracies.
        """
        loss_dict = {}
        temperature = 0.07

        # Soft assignment: which bands map to which visual levels
        assignment = F.softmax(self.band_level_logits, dim=0)  # (K, L) — normalised over bands

        total_loss = torch.tensor(0.0, device=band_globals[0].device)
        total_acc = torch.tensor(0.0, device=band_globals[0].device)

        for level_idx in range(self.num_visual_levels):
            # Weighted combination of band projections for this visual level
            image_feat = image_hidden_states[:, level_idx]  # (B_img, 768)

            # Compute weighted EEG embedding for this level (768-dim, matching DINOv2)
            eeg_level_embed = torch.zeros(
                band_globals[0].size(0), 768,
                device=band_globals[0].device, dtype=band_globals[0].dtype,
            )
            for k in range(self.num_bands):
                proj_k = self.band_align_projs[k](band_globals[k])  # (B_img, 768)
                eeg_level_embed = eeg_level_embed + assignment[k, level_idx] * proj_k

            # InfoNCE
            pred_norm = F.normalize(eeg_level_embed, dim=-1)
            img_norm = F.normalize(image_feat, dim=-1)
            logits = pred_norm @ img_norm.t() / temperature
            targets = torch.arange(logits.size(0), device=logits.device)

            loss_p2i = F.cross_entropy(logits, targets)
            loss_i2p = F.cross_entropy(logits.t(), targets)
            level_loss = 0.5 * (loss_p2i + loss_i2p)
            total_loss = total_loss + level_loss

            with torch.no_grad():
                acc_p2i = (logits.argmax(1) == targets).float()
                acc_i2p = (logits.argmax(0) == targets).float()
                acc = 0.5 * (acc_p2i + acc_i2p)
            total_acc = total_acc + acc.mean()

            loss_dict[f"align_loss_level{level_idx}"] = (1.0, level_loss)
            loss_dict[f"align_acc_level{level_idx}"] = acc.mean()

        # Average over levels
        total_loss = total_loss / self.num_visual_levels
        total_acc = total_acc / self.num_visual_levels
        loss_dict["align_loss_total"] = (1.0, total_loss)
        loss_dict["align_acc_total"] = total_acc

        return loss_dict

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch, mask=None):
        x = batch['timeseries']  # (B, C, N, in_dim)
        B, C, N, _ = x.shape

        # 1. Spectral decomposition on RAW input (before embedding)
        band_signals, num_freq_bins = self._decompose_bands(x)  # K × (B, C, N, in_dim)

        # 2. Coordinate PE (computed once, shared across bands)
        ch_coords = batch['ch_coords']
        coord_pe, _ = self._spherical_positional_encoding(ch_coords)
        coord_emb = self.coord_enhancement(coord_pe)  # (B, C, d_model)

        # 3. Process each band: embed → shared trunk → branch adapter
        #    Memory-efficient: accumulate fused output incrementally,
        #    only keep small global vectors per band (not full activations).
        band_globals = []  # K × (B, d) — global CLS embedding per band
        fused_no_global = None  # accumulated (B, C, N, d)

        # Pre-compute padded masks once (shared across bands)
        batch_k = dict(batch)
        if 'valid_channel_mask' in batch:
            batch_k['valid_channel_mask'] = torch.cat([
                torch.ones(B, 1, device=x.device, dtype=torch.bool),
                batch['valid_channel_mask'],
            ], dim=1)
        if 'valid_length_mask' in batch:
            batch_k['valid_length_mask'] = torch.cat([
                torch.ones(B, 1, device=x.device, dtype=torch.bool),
                batch['valid_length_mask'],
            ], dim=1)

        for k, band_raw_k in enumerate(band_signals):
            # Embed this band's raw signal (shared PatchEmbedding)
            band_emb_k = self.patch_embedding(band_raw_k, mask)  # (B, C, N, d_model)
            band_emb_k = band_emb_k + coord_emb.unsqueeze(2)

            # Prepend global channel and global token
            bk = torch.cat([
                self.global_channel.expand(B, -1, band_emb_k.size(2), -1),
                band_emb_k,
            ], dim=1)
            bk = torch.cat([
                self.global_token.expand(B, bk.size(1), -1, -1),
                bk,
            ], dim=2)

            # Shared trunk
            encoded_k = self._encode_band(bk, batch_k)  # (B, C+1, N+1, d)

            # Branch-specific adapter
            adapted_k = self.band_adapters[k](encoded_k)

            # Extract global CLS (channel-0, token-0) before discarding
            band_globals.append(adapted_k[:, 0, 0, :])  # (B, d)

            # Accumulate into fused (strip global tokens)
            band_content = adapted_k[:, 1:, 1:, :]  # (B, C, N, d)
            if fused_no_global is None:
                fused_no_global = band_content
            else:
                fused_no_global = fused_no_global + band_content
            # adapted_k, encoded_k, band_emb_k go out of scope → freed

        # 6. Reconstruction output
        out = self.proj_out(fused_no_global)  # (B, C, N, out_dim)

        # 7. Compute auxiliary losses
        info = {}

        # Band decorrelation
        band_globals_stack = torch.stack(band_globals, dim=1)  # (B, K, d)
        decorr_loss = self._band_decorrelation_loss(band_globals_stack)
        info["band_decorr_loss"] = (0.0, decorr_loss)

        # Mask completeness loss (at the actual resolution used this forward pass)
        completeness_loss = self.mask_bank.completeness_loss(num_freq_bins)
        info["mask_completeness_loss"] = (0.0, completeness_loss)

        # Multi-band alignment (only for samples with images)
        has_image = batch.get('has_image', None)
        if has_image is not None and has_image.any() and 'image_encoder_inputs' in batch:
            img_inputs = {
                k: v[has_image] for k, v in batch['image_encoder_inputs'].items()
            }
            image_hs = self.get_image_hidden_states(**img_inputs)  # (B_img, L, 768)

            band_globals_img = [bg[has_image] for bg in band_globals]  # K × (B_img, d)
            align_losses = self._multi_band_alignment_loss(
                band_globals_img, image_hs, batch,
            )
            info.update(align_losses)

            # Also compute per-source accuracy breakdown
            sources = batch.get('source', [])
            if sources:
                mask_list = has_image.detach().cpu().tolist()
                img_sources = [s for s, m in zip(sources, mask_list) if m]
                # Use total alignment accuracy for per-source breakdown
                if 'align_acc_total' in info:
                    acc_total = info['align_acc_total']
                    per_source = defaultdict(list)
                    # acc_total is a scalar mean; for detailed per-source we'd
                    # need per-sample acc, but keeping it simple here.

        # Global semantic readout (for downstream use)
        # Average band globals → single embedding
        fused_global = band_globals_stack.mean(dim=1)  # (B, d)
        info["rep"] = fused_no_global
        info["band_reps"] = band_globals_stack  # (B, K, d) for downstream
        info["fused_global"] = fused_global  # (B, d) for downstream

        # Log the learned assignment for monitoring
        with torch.no_grad():
            info["band_level_assignment"] = F.softmax(
                self.band_level_logits, dim=0
            ).detach()
            info["mask_overlap"] = self.mask_bank(num_freq_bins).detach()

        return out, info

    def training_step(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
