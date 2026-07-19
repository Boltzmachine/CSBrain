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

        # Brain region encoding: Frontal lobe (0) | Parietal lobe (1) | Temporal lobe (2) | Occipital lobe (3) | Central region (4)
        bci42a_brain_regions = [
            0,  # 'Fz'
            4, 4, 4, 4, 4,  # FC3 FC1 FCZ FC2 FC4
            4, 4, 4, 4, 4, 4, 4,  # C5 C3 C1 CZ C2 C4 C6
            4, 4, 4, 4, 4,  # CP3 CP1 CPZ CP2 CP4
            1, 1, 1, 1,  # P1 PZ P2 POZ
        ]

        bci42a_electrode_labels = [
            "Fz",
            "FC3", "FC1", "FCZ", "FC2", "FC4",
            "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
            "CP3", "CP1", "CPZ", "CP2", "CP4",
            "P1", "PZ", "P2", "POZ"
        ]

        # Define local topological relationships within brain regions
        bci42a_topology = {
            0: ["Fz"],
            4: ["FC3", "FC1", "FCZ", "FC2", "FC4",
                "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
                "CP3", "CP1", "CPZ", "CP2", "CP4"],
            1: ["P1", "PZ", "P2", "POZ"]
        }

        # Group electrode indices by brain region
        bci42a_region_groups = {}
        for i, region in enumerate(bci42a_brain_regions):
            if region not in bci42a_region_groups:
                bci42a_region_groups[region] = []
            bci42a_region_groups[region].append((i, bci42a_electrode_labels[i]))

        # Sort based on topological relationships
        bci42a_sorted_indices = []
        for region in sorted(bci42a_region_groups.keys()):
            region_electrodes = bci42a_region_groups[region]
            sorted_electrodes = sorted(region_electrodes, key=lambda x: bci42a_topology[region].index(x[1]))
            bci42a_sorted_indices.extend([e[0] for e in sorted_electrodes])

        print("Sorted Indices:", bci42a_sorted_indices)

        self.backbone = get_model(param, bci42a_brain_regions, bci42a_sorted_indices)

        # --- Standard-protocol checkpoint loading (mirrors model_for_physio) ---
        if param.use_pretrained_weights:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict, _ = load_pretrain_checkpoint(
                param.foundation_dir, map_location=map_location
            )

            # DataParallel prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # World-model wrapper: the CSBrainAlign EEG backbone is stored under
            # "encoder." (WorldModel) or "eeg." (ActionWorldModel); the
            # TemEmbedEEGLayer sentinel only exists directly under the wrapper's
            # backbone prefix. Strip it and drop the predictor / action heads.
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

            # DINO teacher/student: load the student backbone weights.
            is_dino_ckpt = any(k.startswith("teacher.") for k in state_dict)
            if is_dino_ckpt:
                state_dict = {k[len("student."):]: v
                              for k, v in state_dict.items()
                              if k.startswith("student.")}
                for attr in ('pretrained_image_encoder', 'semantic_readout',
                             'contrastive_proj', 'equiv_projector'):
                    if hasattr(self.backbone, attr):
                        delattr(self.backbone, attr)

            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            # Pretraining-only heads not part of the finetune backbone — drop
            # them, fail loudly on anything else.
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
