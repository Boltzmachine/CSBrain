import torch
import torch.nn as nn
from functools import partial
from .CSBrain import *
from models import get_model
from utils.util import load_pretrain_checkpoint


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.param = param
        # Bilateralization-prior flip augmentation (training only): reuse the
        # pretrained learned x_bi/x_lat split to build x_flip = x_bi+flip(x_lat)
        # and remap labels (left<->right hand; bilateral classes unchanged).
        self.lateral_flip_aug = getattr(param, 'lateral_flip_aug', False)
        # Test-time flip augmentation: average logits(x) with the label-aligned
        # logits(x_flip). Uses the same learned split + label map.
        self.lateral_flip_tta = getattr(param, 'lateral_flip_tta', False)
        self.lateral_flip_prob = float(getattr(param, 'lateral_flip_prob', 0.5))
        # Either training aug or test-time aug needs the pretrained split loaded.
        self._needs_split = self.lateral_flip_aug or self.lateral_flip_tta
        self.flip_perm = None  # lazy homologous-channel perm (true data order)
        if self._needs_split and not getattr(param, 'use_pretrained_weights', False):
            raise ValueError(
                "lateral_flip_aug / lateral_flip_tta requires "
                "--use_pretrained_weights with a bilateralization-pretrained "
                "checkpoint (--lateralization_flip)")

        # 思路1 — symmetrization augmentation (training only): synthesize the
        # hard bilateral 'both fists' class from single-hand trials via the
        # symmetric projection x_sym = 0.5*(x + flip(x)). Uses the RAW full
        # homologous-channel swap (the learned split adds nothing for this axis,
        # per the |x_lat| extent probe), so it needs no pretrained split.
        self.symmetrize_aug = getattr(param, 'symmetrize_aug', False)
        self.symmetrize_aug_prob = float(getattr(param, 'symmetrize_aug_prob', 0.25))
        self.symmetrize_target_label = int(getattr(param, 'symmetrize_target_label', 2))
        _src = [int(t) for t in
                str(getattr(param, 'symmetrize_src_labels', '0,1')).split(',')
                if t.strip() != '']
        self.register_buffer('symmetrize_src_labels',
                             torch.tensor(_src, dtype=torch.long), persistent=False)
        selected_channels = [
            'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 
            'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
            'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
            'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FT8',
            'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
            'O1', 'OZ', 'O2', 'IZ'
        ]

        # Brain region encoding: Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
        brain_regions = [
            0, 0, 0, 0, 0, 0, 0,  # FC series
            4, 4, 4, 4, 4, 4, 4,  # C series
            4, 4, 4, 4, 4, 4, 4,  # CP series
            0, 0, 0, 0, 0, 0, 0, 0,  # FP and AF series
            0, 0, 0, 0, 0, 0, 0, 0, 0,  # F series
            2, 2,  # FT series
            2, 2, 2, 2, 2, 2,  # T and TP series
            1, 1, 1, 1, 1, 1, 1, 1, 1,  # P series
            3, 3, 3, 3, 3,  # PO series
            3, 3, 3, 3  # O and IZ series
        ]

        # Topological structure
        topology = {
            0: ['AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
                'FP1', 'FPZ', 'FP2',
                'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6'],
            4: ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'],
            2: ['T7', 'T8', 'T9', 'T10', 'TP7', 'TP8',
                'FT7', 'FT8',
                ],
            1: ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
            3: ['PO7', 'PO3', 'POZ', 'PO4', 'PO8',
                'O1', 'OZ', 'O2', 'IZ']
        }

        # Group electrode indices by brain region
        region_groups = {}
        for i, region in enumerate(brain_regions):
            if region not in region_groups:
                region_groups[region] = []
            region_groups[region].append((i, selected_channels[i]))

        # Sort based on topology
        sorted_indices = []
        for region in sorted(region_groups.keys()):
            region_electrodes = region_groups[region]
            sorted_electrodes = sorted(region_electrodes, key=lambda x: topology[region].index(x[1]))
            sorted_indices.extend([e[0] for e in sorted_electrodes])

        print("Sorted Indices:", sorted_indices)

        self.backbone = get_model(param, brain_regions, sorted_indices)

        if param.use_pretrained_weights:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict, _ = load_pretrain_checkpoint(
                param.foundation_dir, map_location=map_location
            )

            # --- Normalise key prefixes ---
            # DataParallel wrapping
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # --- Handle world-model wrapper checkpoints ---
            # Both wrappers store the CSBrainAlign EEG backbone under a prefix,
            # alongside extra heads we discard when finetuning the encoder:
            #   WorldModel       -> "encoder."  (+ predictor.)
            #   ActionWorldModel -> "eeg."      (+ predictor., action_head.)
            # CSBrainAlign itself has a top-level ``encoder`` submodule (the
            # transformer stack), so we can't key off ``encoder.`` alone --
            # use the ``TemEmbedEEGLayer`` sentinel, which only exists directly
            # under the wrapper's backbone prefix. (ActionWorldModel also has
            # ``predictor.`` keys, so detecting on ``predictor.`` would
            # mis-route it to the ``encoder.`` strip and wipe the state dict.)
            if any(k.startswith("eeg.TemEmbedEEGLayer.") for k in state_dict):
                use_prefix = "eeg."
            elif any(k.startswith("encoder.TemEmbedEEGLayer.") for k in state_dict):
                use_prefix = "encoder."
            else:
                use_prefix = None
            if use_prefix is not None:
                state_dict = {
                    k[len(use_prefix):]: v
                    for k, v in state_dict.items()
                    if k.startswith(use_prefix)
                }

                # del self.backbone.pretrained_image_encoder

            # --- Handle DINO-pretrained checkpoints ---
            # The teacher (EMA-smoothed) produces better representations than
            # the student, so we load teacher weights for finetuning.
            is_dino_ckpt = any(k.startswith("teacher.") for k in state_dict)
            if is_dino_ckpt:
                # Extract teacher backbone weights and strip the "teacher." prefix.
                # Sibling top-level modules (student.*, *_dino_head.*, *_ibot_head.*,
                # *_criterion.*, freq_subband_view.*) are dropped implicitly by
                # the filter below.
                teacher_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("student."):
                        teacher_dict[k[len("student."):]] = v
                state_dict = teacher_dict

            if is_dino_ckpt:
                # The teacher had these modules deleted to save GPU memory
                # during DINO training, so they won't be in the checkpoint.
                # Remove them from the backbone before loading.
                for attr in ('pretrained_image_encoder', 'semantic_readout',
                             'contrastive_proj', 'equiv_projector'):
                    if hasattr(self.backbone, attr):
                        delattr(self.backbone, attr)
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            # Pretraining-only heads that hang off the encoder but aren't part of
            # the finetune backbone — tolerate (drop) them, fail loudly on
            # anything else. ``equiv_projector``: equivariance head;
            # ``lateralization_split`` / ``flip_align_proj``: bilateralization-
            # prior heads (learned x_bi/x_lat split + horizontal-flip alignment);
            # ``frame_split``: feature-space split gate of the equivariant
            # frame-averaging frontend (equi 2.0); ``frame_flip_align_proj``:
            # its flip-alignment head — both pretrain-only like the above.
            _pretrain_only = ('equiv_projector', 'lateralization_split',
                              'flip_align_proj', 'frame_split',
                              'frame_flip_align_proj')
            unexpected_keys = [k for k in unexpected_keys
                               if not any(t in k for t in _pretrain_only)]
            if unexpected_keys:
                # for unexpected_key in unexpected_keys:
                #     if not 'source_projector' in unexpected_key: #FIXME
                raise ValueError(f"UNEXPECTED KEYS: {unexpected_keys}")
            if missing_keys:
                raise ValueError(f"MISSING KEYS: {missing_keys}")

            # --- Bilateralization-prior flip augmentation (optional) ---
            # Load the pretrained learned x_bi/x_lat split as a FROZEN module so
            # finetune can augment with x_flip = x_bi + flip(x_lat) — the same
            # construction as pretraining — paired with a left<->right label
            # remap. Weights live under 'lateralization_split.' in the ckpt.
            if self._needs_split:
                from models.alignment import LateralizationSplit
                split_sd = {k[len('lateralization_split.'):]: v
                            for k, v in state_dict.items()
                            if k.startswith('lateralization_split.')}
                if not split_sd:
                    raise ValueError(
                        "lateral_flip_aug=True but the checkpoint has no "
                        "'lateralization_split.*' weights — pretrain with "
                        "--lateralization_flip first.")
                coord_pe_dim = 3 * 2 * self.backbone.spherical_num_freqs
                self.lateral_split = LateralizationSplit(
                    in_dim=param.in_dim, coord_pe_dim=coord_pe_dim,
                    hidden=getattr(param, 'flip_split_hidden', 64),
                )
                self.lateral_split.load_state_dict(split_sd, strict=True)
                for p in self.lateral_split.parameters():
                    p.requires_grad = False
                self.lateral_split.eval()
                lut = [int(t) for t in
                       str(getattr(param, 'flip_label_map', '1,0,2,3')).split(',')
                       if t.strip() != '']
                if len(lut) != param.num_of_classes:
                    raise ValueError(
                        f"flip_label_map has {len(lut)} entries but "
                        f"num_of_classes={param.num_of_classes}")
                self.register_buffer(
                    'flip_label_lut', torch.tensor(lut, dtype=torch.long),
                    persistent=False)

        self.backbone.proj_out = nn.Identity()

        if getattr(param, 'linear_probe', False):
            self.classifier = nn.LazyLinear(param.num_of_classes)
        else:
            self.classifier = nn.Sequential(
                nn.LazyLinear(4 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(4 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes)
            )

    def train(self, mode=True):
        super().train(mode)
        if getattr(self.param, 'linear_probe', False):
            # Keep backbone in eval mode so dropout / norm stats don't drift
            # while the linear head trains on frozen features.
            self.backbone.eval()
        if self._needs_split:
            # The split is a frozen pretrained module — always deterministic.
            self.lateral_split.eval()
        return self

    def _ensure_flip_perm(self, ch_names_list, device):
        """Lazily build (and cache) the homologous-channel swap permutation
        from the true data channel order (PhysioNet uses a fixed montage)."""
        if self.flip_perm is None:
            from models.alignment import build_flip_perm_batch
            self.flip_perm = build_flip_perm_batch(
                [ch_names_list[0]])[0].to(device)
        return self.flip_perm

    @torch.no_grad()
    def _lateral_flip_inputs(self, xr, ch_coords):
        """``xr`` (B, C, N, in_dim) -> x_bi + flip(x_lat) for every row, using
        the frozen pretrained split. ``flip_perm`` must already be built."""
        coord_pe = self.backbone._spherical_positional_encoding(ch_coords)[0]
        x_lat, _ = self.lateral_split(xr, coord_pe)
        return xr - x_lat + x_lat.index_select(1, self.flip_perm)

    @torch.no_grad()
    def flip_augment(self, batch):
        """Bilateralization-prior augmentation (training only).

        With prob ``lateral_flip_prob`` per sample, replace ``x`` with
        ``x_bi + flip(x_lat)`` (the pretrained learned split; only the lateral
        stream is hemisphere-swapped) and remap the label via ``flip_label_lut``
        (left<->right hand; bilateral classes map to themselves). Mutates
        ``batch['x']`` in place and returns the (possibly remapped) labels.
        No-op when disabled or in eval mode.
        """
        y = batch['y']
        if not (self.lateral_flip_aug and self.training):
            return y
        x = batch['x']
        orig_shape = x.shape
        B, C = x.shape[0], x.shape[1]
        xr = x.reshape(B, C, -1, self.param.in_dim)         # (B, C, N, in_dim)
        self._ensure_flip_perm(batch['ch_names'], x.device)

        sel = torch.rand(B, device=x.device) < self.lateral_flip_prob
        idx = sel.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return y

        xr_flip = self._lateral_flip_inputs(
            xr.index_select(0, idx),
            batch['ch_coords'].index_select(0, idx))
        xr = xr.index_copy(0, idx, xr_flip)
        batch['x'] = xr.reshape(orig_shape)
        y = y.clone()
        y[idx] = self.flip_label_lut.to(y.device)[y.index_select(0, idx)]
        return y

    @torch.no_grad()
    def symmetrize_augment(self, batch, y):
        """思路1: synthesize the hard bilateral class (both fists) from
        single-hand trials by the symmetric projection
        ``x_sym = 0.5 * (x + flip(x))`` over a RAW full homologous-channel swap
        (C3<->C4, ...), relabeled to ``symmetrize_target_label``. This teaches
        'bilateral = unilateral + its mirror' and rebalances toward the
        hardest class. Acts on the labels passed in (so it composes after
        flip_augment); mutates ``batch['x']`` in place. No learned split needed.
        No-op when disabled or in eval mode.
        """
        if not (self.symmetrize_aug and self.training):
            return y
        x = batch['x']
        B = x.shape[0]
        self._ensure_flip_perm(batch['ch_names'], x.device)

        # Eligible = single-hand source labels; sample by prob.
        src = self.symmetrize_src_labels.to(y.device)
        eligible = (y.unsqueeze(1) == src.unsqueeze(0)).any(dim=1)
        sel = eligible & (torch.rand(B, device=x.device) < self.symmetrize_aug_prob)
        idx = sel.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return y

        x_sel = x.index_select(0, idx)
        x_sym = 0.5 * (x_sel + x_sel.index_select(1, self.flip_perm))
        batch['x'] = x.index_copy(0, idx, x_sym)
        y = y.clone()
        y[idx] = self.symmetrize_target_label
        return y

    @torch.no_grad()
    def _forward_tta(self, batch):
        """Test-time flip augmentation: average logits(x) with the label-aligned
        logits(x_flip). x_flip = x_bi + flip(x_lat) (the learned split). The
        mirrored input's class lut[k] is evidence for the original class k, so
        we gather the flipped logits' columns by ``flip_label_lut`` before
        averaging (e.g. flipped 'right' score reinforces original 'left')."""
        x = batch['x']
        B, C = x.shape[0], x.shape[1]
        xr = x.reshape(B, C, -1, self.param.in_dim)
        self._ensure_flip_perm(batch['ch_names'], x.device)
        x_flip = self._lateral_flip_inputs(xr, batch['ch_coords'])

        # Shallow copies so each _forward_core pops its own 'x'.
        logits_o = self._forward_core({**batch, 'x': xr})
        logits_f = self._forward_core({**batch, 'x': x_flip})
        lut = self.flip_label_lut.to(logits_f.device)
        return 0.5 * (logits_o + logits_f.index_select(1, lut))

    def forward(self, batch):
        # Test-time flip augmentation (eval only): see _forward_tta.
        if self.lateral_flip_tta and self._needs_split and not self.training:
            return self._forward_tta(batch)
        return self._forward_core(batch)

    def _forward_core(self, batch):
        x = batch.pop('x')
        x = x.reshape(x.size(0), x.size(1), -1, self.param.in_dim)
        bz, ch_num, seq_len, patch_size = x.shape

        if getattr(self.param, 'use_initial_segment_only', False):
            seg_len = self.param.seq_len
            # n_starts = seq_len - seg_len + 1
            n_starts = 1

            # Which seg_len-patch segment to crop along the time axis. 0 keeps
            # the original first/native-window behaviour; >0 selects a later
            # second of the trial (e.g. for PhysioNet's 4s/20-patch input,
            # segment_index 0..3 -> seconds 1..4).
            seg_idx = getattr(self.param, 'segment_index', 0)
            seg_start = seg_idx * seg_len
            if seg_start + seg_len > seq_len:
                raise ValueError(
                    f"segment_index={seg_idx} (start patch {seg_start}, "
                    f"seg_len {seg_len}) exceeds input time dim {seq_len}")

            if n_starts <= 1:
                # The input already matches seg_len; nothing to crop or ensemble.
                x = x[:, :, seg_start:seg_start + seg_len, :].contiguous()
                batch['timeseries'] = x
                feats = self.backbone(batch)
                if not isinstance(feats, tuple):
                    raise ValueError("Expected backbone to return a tuple with a dictionary containing 'rep' key.")
                feats = feats[1]["rep"]
            elif self.training:
                # Per-sample sub-patch jitter on the raw time axis: each
                # example is shifted by 0..n_starts-1 *samples* before being
                # re-patchified. Sample-level (not patch-level) shifts keep
                # the patch grid and the encoder's positional slots stable --
                # each patch still holds the same content it would at shift=0
                # up to a few-sample boundary slide.
                flat_len = seq_len * patch_size
                seg_samples = seg_len * patch_size
                x_flat = x.view(bz, ch_num, flat_len)
                shift = torch.randint(0, n_starts, (bz,), device=x.device)
                sample_offsets = torch.arange(seg_samples, device=x.device)
                gather_idx = (shift.unsqueeze(1) + sample_offsets.unsqueeze(0)) \
                    .view(bz, 1, seg_samples) \
                    .expand(bz, ch_num, seg_samples)
                x_flat = torch.gather(x_flat, dim=2, index=gather_idx)
                x = x_flat.view(bz, ch_num, seg_len, patch_size).contiguous()
                batch['timeseries'] = x
                feats = self.backbone(batch)
                if not isinstance(feats, tuple):
                    raise ValueError("Expected backbone to return a tuple with a dictionary containing 'rep' key.")
                feats = feats[1]["rep"]
            else:
                # Eval-time sliding-window ensemble over the same sample-level
                # shifts used in training. Process one shift per forward pass
                # so eval memory stays at the input batch size regardless of
                # n_starts -- stacking all shifts into the batch dim OOMs once
                # n_starts gets large.
                flat_len = seq_len * patch_size
                seg_samples = seg_len * patch_size
                x_flat = x.view(bz, ch_num, flat_len)
                logits_sum = None
                for s in range(n_starts):
                    x_s = x_flat[:, :, s:s + seg_samples] \
                        .reshape(bz, ch_num, seg_len, patch_size).contiguous()
                    seg_batch = {'timeseries': x_s}
                    for k, v in batch.items():
                        seg_batch[k] = v
                    feats_s = self.backbone(seg_batch)
                    if not isinstance(feats_s, tuple):
                        raise ValueError("Expected backbone to return a tuple with a dictionary containing 'rep' key.")
                    feats_s = feats_s[1]["rep"]
                    out_s = feats_s.contiguous().view(bz, -1)
                    logits_s = self.classifier(out_s)
                    logits_sum = logits_s if logits_sum is None else logits_sum + logits_s
                return logits_sum / n_starts
        elif getattr(self.param, 'segment_forward', False) and seq_len > self.param.seq_len:
            # Pretrained encoder expects seq_len=self.param.seq_len; the finetune
            # input is longer, so split along time and encode each segment
            # independently. Segments are packed into the batch dim so the
            # encoder runs in a single forward call.
            seg_len = self.param.seq_len
            assert seq_len % seg_len == 0, (
                f"time dim {seq_len} not divisible by pretrained seq_len {seg_len}")
            n_seg = seq_len // seg_len

            x_seg = x.view(bz, ch_num, n_seg, seg_len, patch_size) \
                     .permute(0, 2, 1, 3, 4).contiguous() \
                     .view(bz * n_seg, ch_num, seg_len, patch_size)

            seg_batch = {'timeseries': x_seg}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    seg_batch[k] = v.repeat_interleave(n_seg, dim=0)

            feats = self.backbone(seg_batch)
            if not isinstance(feats, tuple):
                raise ValueError("Expected backbone to return a tuple with a dictionary containing 'rep' key.")
            feats = feats[1]["rep"]  # (bz*n_seg, C', N', D)
            _, Cp, Np, D = feats.shape
            feats = feats.view(bz, n_seg, Cp, Np, D) \
                         .permute(0, 2, 1, 3, 4).contiguous() \
                         .view(bz, Cp, n_seg * Np, D)
        else:
            batch['timeseries'] = x
            feats = self.backbone(batch)
            if not isinstance(feats, tuple):
                raise ValueError("Expected backbone to return a tuple with a dictionary containing 'rep' key.")
            feats = feats[1]["rep"]

        out = feats.contiguous().view(bz, -1)
        out = self.classifier(out)
        return out