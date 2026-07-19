import torch
import torch.nn as nn
from functools import partial
from .CSBrain import *
from models import get_model
from utils.util import load_pretrain_checkpoint


# Weibo2014 60-channel EEG montage, in the MOABB acquisition order (== the LMDB
# sample channel order). CB1/CB2 (misc), VEO/HEO (eog) and STIM014 are dropped
# upstream, so this is the pure 60-channel scalp EEG set.
SELECTED_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
    'O1', 'Oz', 'O2',
]


# Brain-region encoding (matches model_for_physio / model_for_finemi):
#   Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
# Case-insensitive: Weibo channel names are mixed-case (Fp1, Cz, FCz, POz).
def _region_of(ch):
    ch = ch.upper()
    if ch.startswith('FT'):
        return 2
    if ch.startswith(('FC', 'FP', 'AF', 'F')):
        return 0
    if ch.startswith('CB'):
        return 3
    if ch.startswith('CP') or ch.startswith('C'):
        return 4
    if ch.startswith('TP') or ch.startswith('T'):
        return 2
    if ch.startswith('PO'):
        return 3
    if ch.startswith('P'):
        return 1
    if ch.startswith('O') or ch.startswith('I'):
        return 3
    raise ValueError(f'unclassified channel {ch}')


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.param = param
        selected_channels = SELECTED_CHANNELS

        brain_regions = [_region_of(ch) for ch in selected_channels]

        # Group electrodes by region, then order channels region-by-region.
        # generate_area_config (in CSBrain) sorts brain_regions and slices
        # contiguous per-region blocks, so sorted_indices must reorder the raw
        # channels to match sorted(brain_regions). A stable sort by region id
        # keeps the (roughly topographic) acquisition order within each region.
        sorted_indices = sorted(range(len(brain_regions)),
                                key=lambda i: brain_regions[i])

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

            # --- Handle WorldModelWrapper checkpoints ---
            is_world_model_ckpt = any(
                k.startswith("encoder.patch_embed.")
                or k.startswith("encoder.TemEmbedEEGLayer.")
                or k.startswith("predictor.")
                for k in state_dict
            )
            if is_world_model_ckpt:
                use_prefix = "encoder."
                state_dict = {
                    k[len(use_prefix):]: v
                    for k, v in state_dict.items()
                    if k.startswith(use_prefix)
                }

            # --- Handle DINO-pretrained checkpoints ---
            is_dino_ckpt = any(k.startswith("teacher.") for k in state_dict)
            if is_dino_ckpt:
                teacher_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("student."):
                        teacher_dict[k[len("student."):]] = v
                state_dict = teacher_dict
                for attr in ('pretrained_image_encoder', 'semantic_readout',
                             'contrastive_proj', 'equiv_projector'):
                    if hasattr(self.backbone, attr):
                        delattr(self.backbone, attr)

            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            # Pretraining-only heads that hang off the encoder but aren't part of
            # the finetune backbone -- tolerate (drop) them, fail loudly on
            # anything else. Mirrors model_for_finemi's allowlist.
            _pretrain_only = ('equiv_projector', 'lateralization_split',
                              'flip_align_proj', 'frame_split',
                              'frame_flip_align_proj', 'hand_pred_head')
            unexpected_keys = [k for k in unexpected_keys
                               if not any(t in k for t in _pretrain_only)]
            if unexpected_keys:
                raise ValueError(f"UNEXPECTED KEYS: {unexpected_keys}")
            if missing_keys:
                raise ValueError(f"MISSING KEYS: {missing_keys}")

        self.backbone.proj_out = nn.Identity()

        # Equivariant frame-averaging backbone: pin the random per-step flip to 0
        # so the finetune forward is always the canonical frame-averaged pass.
        if getattr(self.backbone, 'frame_averaging', False):
            self.backbone.frame_avg_flip_prob = 0.0
            self.backbone.frame_rep_mode = getattr(param, 'frame_rep_mode', 'both')

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
            self.backbone.eval()
        return self

    def forward(self, batch):
        x = batch.pop('x')
        x = x.reshape(x.size(0), x.size(1), -1, self.param.in_dim)
        bz, ch_num, seq_len, patch_size = x.shape

        if getattr(self.param, 'use_initial_segment_only', False):
            seg_len = self.param.seq_len
            seg_idx = getattr(self.param, 'segment_index', 0)
            seg_start = seg_idx * seg_len
            if seg_start + seg_len > seq_len:
                raise ValueError(
                    f"segment_index={seg_idx} (start patch {seg_start}, "
                    f"seg_len {seg_len}) exceeds input time dim {seq_len}")
            x = x[:, :, seg_start:seg_start + seg_len, :].contiguous()

        batch['timeseries'] = x
        feats = self.backbone(batch)
        if not isinstance(feats, tuple):
            raise ValueError("Expected backbone to return a tuple with a dictionary containing 'rep' key.")
        feats = feats[1]["rep"]

        out = feats.contiguous().view(bz, -1)
        out = self.classifier(out)
        return out
