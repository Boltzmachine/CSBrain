import torch
import torch.nn as nn
from functools import partial
from .CSBrain import *
from models import get_model

class Model(nn.Module):
    def __init__(self, param):
        self.param = param
        super(Model, self).__init__()

        # Brain region encoding: Frontal lobe (0) | Parietal lobe (1) | Temporal lobe (2) | Occipital lobe (3) | Central region (4)
        bci42a_brain_regions = [
            0, # 'Fz'
            4, # 'FC3'
            4, # 'FC1'
            4, # 'FCZ'
            4, # 'FC2'
            4, # 'FC4'
            4, # 'C5'
            4, # 'C3'
            4, # 'C1'
            4, # 'CZ'
            4, # 'C2'
            4, # 'C4'
            4, # 'C6'
            4, # 'CP3'
            4, # 'CP1'
            4, # 'CPZ'
            4, # 'CP2'
            4, # 'CP4'
            1, # 'P1'
            1, # 'PZ'
            1, # 'P2'
            1, # 'POZ'
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
            
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "")
                new_state_dict[new_key] = value

            model_state_dict = self.backbone.state_dict()

            # Filter matching weights by shape
            # matching_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

            model_state_dict.update(state_dict)
            self.backbone.load_state_dict(model_state_dict, strict=True)

        self.backbone.proj_out = nn.Identity()

        num_chs = 16
        self.feed_forward = nn.Sequential(
            nn.Linear(num_chs*200, 128),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(128, param.num_of_classes),
        )

    def forward(self, batch):
        x = batch.pop('x')
        bz, ch_num, seq_len, patch_size = x.shape

        batch['timeseries'] = x
        feats = self.backbone(batch)
        if isinstance(feats, tuple):
            feats = feats[0]
        feats = feats[:, :, 0].contiguous().view(bz, -1)
        out = self.feed_forward(feats)
        return out