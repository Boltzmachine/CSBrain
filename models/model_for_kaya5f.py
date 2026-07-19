import torch
import torch.nn as nn
from functools import partial
from .CSBrain import *
from models import get_model
from utils.util import load_pretrain_checkpoint


# Kaya 5F 19-channel scalp EEG montage (10-20), in the stored LMDB order. A1/A2
# (earlobe refs) and X5 (sync) are dropped; old T3/T4/T5/T6 renamed T7/T8/P7/P8.
SELECTED_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
]


# Brain-region encoding (matches model_for_physio / model_for_finemi):
#   Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
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
        sorted_indices = sorted(range(len(brain_regions)),
                                key=lambda i: brain_regions[i])

        print("Sorted Indices:", sorted_indices)

        self.backbone = get_model(param, brain_regions, sorted_indices)

        if param.use_pretrained_weights:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict, _ = load_pretrain_checkpoint(
                param.foundation_dir, map_location=map_location
            )

            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

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
