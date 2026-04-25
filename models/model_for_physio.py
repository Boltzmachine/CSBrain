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

            # --- Handle WorldModelWrapper checkpoints ---
            # WorldModelWrapper wraps a CSBrainAlign under "encoder." and
            # (optionally) a LatentPredictor under "predictor.". Finetuning
            # only uses the encoder: strip the prefix and drop anything that
            # doesn't belong to it. CSBrainAlign itself has a top-level
            # ``encoder`` submodule (the transformer stack), so we can't key
            # off ``encoder.`` alone -- look for keys that only exist under
            # the wrapper, e.g. ``encoder.patch_embed.`` or ``predictor.``.
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
            # ``equiv_projector`` is an optional pretraining-only head; tolerate
            # it in the checkpoint but fail loudly on anything else.
            unexpected_keys = [k for k in unexpected_keys
                               if "equiv_projector" not in k]
            if unexpected_keys:
                raise ValueError(f"UNEXPECTED KEYS: {unexpected_keys}")
            if missing_keys:
                raise ValueError(f"MISSING KEYS: {missing_keys}")

        self.backbone.proj_out = nn.Identity()

        self.classifier = nn.Sequential(
            nn.LazyLinear(4 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(4 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, param.num_of_classes)
        )

    def forward(self, batch):
        x = batch.pop('x')
        x = x.reshape(x.size(0), x.size(1), -1, self.param.in_dim)
        bz, ch_num, seq_len, patch_size = x.shape

        self.param.segment_forward = True
        if getattr(self.param, 'segment_forward', False) and seq_len > self.param.seq_len:
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