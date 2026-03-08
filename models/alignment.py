import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CSBrain_transformerlayer import *
from models.CSBrain_transformer import *
from collections import Counter
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from transformers import Dinov2Model


class CNNSemanticReadout(nn.Module):
    def __init__(self, n_ch: int, out_dim: int, seq_len: int = 240, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # 输入投影：先把通道维映射到更高维
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_ch, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )

        # 多层 dilated temporal conv，扩大感受野
        self.temporal_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=8, dilation=8, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ),
        ])

        # 显式时间位置编码，保留绝对时序信息
        self.pos_embed = nn.Parameter(torch.zeros(1, hidden_dim, seq_len))

        # 不做全局池化，直接保留整段时间特征
        self.head = nn.Sequential(
            nn.Flatten(),  # (B, hidden_dim * seq_len)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * seq_len, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_ch, seq_len)
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (B, n_ch, seq_len), got {tuple(x.shape)}")

        if x.size(-1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.size(-1)}")

        h = self.input_proj(x)              # (B, hidden_dim, seq_len)
        h = h + self.pos_embed              # 加绝对位置编码

        for block in self.temporal_blocks:
            h = h + block(h)                # residual temporal modeling

        y = self.head(h)                    # (B, out_dim)
        return y

class MLPSemanticReadout(nn.Module):
    def __init__(self, n_ch: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(32 * 240, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, out_dim),
        )

        # self.mlp1 = nn.Sequential(
        #     nn.Linear(32, 256),
        #     nn.GELU(),
        #     nn.Linear(256, hidden_dim),
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(hidden_dim, 256),
        #     nn.GELU(),
        #     nn.Linear(256, out_dim),
        #     )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(x.view(*x.shape[:-2], -1))
        return y


class CSBrainAlign(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                 nhead=8, TemEmbed_kernel_sizes=[(1,), (3,), (5,)], brain_regions=[], sorted_indices=[]):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)

        self.TemEmbed_kernel_sizes = TemEmbed_kernel_sizes
        kernel_sizes = self.TemEmbed_kernel_sizes
        self.TemEmbedEEGLayer = TemEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim, kernel_sizes=kernel_sizes, stride=1)

        self.brain_regions = brain_regions
        self.area_config = None #generate_area_config(sorted(brain_regions))
        self.BrainEmbedEEGLayer = BrainEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim)
        self.sorted_indices = sorted_indices

        self.pretrained_image_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base").eval()

        n_branch_layers = 2

        encoder_layer = CSBrain_TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, area_config=self.area_config, sorted_indices=self.sorted_indices, batch_first=True,
            activation=F.gelu
        )
        self.encoder = CSBrain_TransformerEncoder(encoder_layer, num_layers=n_layer - n_branch_layers, enable_nested_tensor=False)

        self.recon = CSBrain_TransformerEncoder(encoder_layer, num_layers=n_branch_layers, enable_nested_tensor=False)

        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )

        encoder_layer_alinment = CSBrain_TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, area_config=self.area_config, sorted_indices=self.sorted_indices, batch_first=True,
            activation=F.gelu
        )
        self.encoder_alignment = CSBrain_TransformerEncoder(encoder_layer_alinment, num_layers=n_branch_layers, enable_nested_tensor=False)

        self.num_visual_levels = 10
        max_freq_bins = seq_len * in_dim // 2 + 1
        self.freq_mask_logits = nn.Parameter(torch.zeros(self.num_visual_levels, max_freq_bins))

        hidden_ch_dim = 64
        semantic_arch = 'cnn'

        if semantic_arch == 'mlp':
            self.semantic_readout = MLPSemanticReadout(n_ch=32, out_dim=d_model, hidden_dim=hidden_ch_dim)
        elif semantic_arch == 'cnn':
            self.semantic_readout = CNNSemanticReadout(n_ch=32, out_dim=d_model, hidden_dim=hidden_ch_dim)

        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 768),
        )

        self.apply(_weights_init)

        self.features_by_layer = []
        self.input_features = []

    def _spectral_branches(self, patch_emb, image_hidden_states):
        time_emb = patch_emb.view(patch_emb.size(0), patch_emb.size(1), -1)
        spectral = torch.fft.rfft(time_emb, dim=-1, norm='forward')

        masks = torch.sigmoid(self.freq_mask_logits[: image_hidden_states.size(-2), :]) # (num_visual_levels, n_freq_bins)
        branch_spectral = spectral.unsqueeze(0) * masks.unsqueeze(1).unsqueeze(1) # (n_branch, B, n_ch, n_freq_bins)
        branch_time = torch.fft.irfft(branch_spectral, norm='forward') # (n_branch, B, n_ch, seq_len)

        branch_embeds = []
        for i_branch in range(branch_time.size(0)):
            branch_embed = branch_time[i_branch].view(*patch_emb.size())
            for layer_idx in range(self.encoder_alignment.num_layers):
                branch_embed = self.TemEmbedEEGLayer(branch_embed) + branch_embed
                branch_embed = self.BrainEmbedEEGLayer(branch_embed, self.area_config) + branch_embed
                branch_embed = self.encoder_alignment.layers[layer_idx](branch_embed, self.area_config)
            branch_embeds.append(branch_embed)
        branch_embeds = torch.stack(branch_embeds, dim=1)

        return branch_embeds
    
    @torch.inference_mode()
    def get_image_hidden_states(self, **image_encoder_inputs):
        B, n_events = image_encoder_inputs['pixel_values'].shape[:2]
        for key in image_encoder_inputs:
            if key == 'pixel_values':
                image_encoder_inputs[key] = image_encoder_inputs[key].view(-1, *image_encoder_inputs[key].shape[2:]) # (B * n_events, C, H, W)
            else:
                raise ValueError(f"Unexpected key '{key}' in image_encoder_inputs")
        outputs = self.pretrained_image_encoder(**image_encoder_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        selected_hidden_states = []
        for layer_idx in [1, 6, 12]:
            selected_hidden_states.append(hidden_states[layer_idx][:, 0])
        selected_hidden_states = torch.stack(selected_hidden_states, dim=1) # (B * n_events, n_branch, d_model)
        selected_hidden_states = selected_hidden_states.view(B, n_events, selected_hidden_states.size(1), selected_hidden_states.size(2)) # (B, n_events, n_branch, d_model)
        return selected_hidden_states


    def forward(self, batch, mask=None):
        x = batch['timeseries'] # (B, n_ch, seq_len, in_dim)
        # x = x[:, self.sorted_indices, :, :]
        patch_emb = self.patch_embedding(x, mask)

        for layer_idx in range(self.encoder.num_layers):
            patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
            patch_emb = self.BrainEmbedEEGLayer(patch_emb, self.area_config) + patch_emb
            patch_emb = self.encoder.layers[layer_idx](patch_emb, self.area_config)


        if batch.get('events') is not None:
            events = batch['events'] # (B, n_events)
            # image_hidden_states = batch['image_hidden_states'] # (B, n_events, n_branch, d_model)
            image_hidden_states = self.get_image_hidden_states(**batch['image_encoder_inputs']) # (B, n_events, n_branch, d_model)

            T = patch_emb.size(-2) * patch_emb.size(-1)  # use x.shape[2] * x.shape[3] if events are in raw-sample indices

            start = events - 40
            branch_embs = self._spectral_branches(patch_emb, image_hidden_states) #(B, n_branch, n_ch, seq_len, d_model)
            branch_time = branch_embs.view(branch_embs.size(0), branch_embs.size(1), branch_embs.size(2), -1) # (B, n_branch, n_ch, seq_len * d_model)
            window_len = 240

            B, n_branch, n_ch, total_len = branch_time.shape
            n_events = events.size(1)

            starts = start.long()  # (B, n_events)
            offsets = torch.arange(window_len, device=branch_time.device).view(1, 1, -1)
            idx = starts.unsqueeze(-1) + offsets  # (B, n_events, window_len)

            valid = (idx >= 0) & (idx < total_len)
            idx = idx.clamp(0, total_len - 1)

            branch_time_exp = branch_time.unsqueeze(2).expand(B, n_branch, n_events, n_ch, total_len)
            gather_idx = idx[:, None, :, None, :].expand(B, n_branch, n_events, n_ch, window_len)

            event_windows = torch.gather(branch_time_exp, dim=-1, index=gather_idx) # LINE 1
            # event_windows = torch.gather(x.view(B, 32, -1).unsqueeze(1).unsqueeze(1).expand(-1, 3, 5, -1, -1), dim=-1, index=gather_idx) # LINE 2 FIXME
            event_windows = event_windows * valid[:, None, :, None, :].to(event_windows.dtype)
            # event_windows: (B, n_branch, n_events, n_ch, window_len)

            # event_windows = event_windows.view(B, 3, 5, -1) 
            event_windows = self.semantic_readout(event_windows.view(-1, event_windows.size(-2), event_windows.size(-1))) # (B * n_branch * n_events, d_model)
            event_windows = event_windows.view(B, n_branch, n_events, -1) # (B, n_branch, n_events, d_model)
            event_windows_for_contrastive = self.contrastive_proj(event_windows) # (B, n_branch, n_events, 768)

            contrastive_loss = dict()
            for i_branch in range(event_windows_for_contrastive.size(1)):
                event_windows_for_contrastive_branch = event_windows_for_contrastive[:, i_branch] # (B, n_events, d_model)
                image_hidden_states_branch = image_hidden_states[:, :, i_branch] # (B, n_events, d_model)

                pred_flatten = event_windows_for_contrastive_branch.reshape(-1, event_windows_for_contrastive_branch.size(-1))
                image_flatten = image_hidden_states_branch.reshape(-1, image_hidden_states_branch.size(-1))

                # contrastive learning
                temperature = 0.07
                pred_norm = F.normalize(pred_flatten, dim=-1)
                image_norm = F.normalize(image_flatten, dim=-1)

                # Pairwise similarity: positives are on the diagonal
                logits = torch.matmul(pred_norm, image_norm.t()) / temperature
                targets = torch.arange(logits.size(0), device=logits.device)

                # Symmetric InfoNCE
                loss_p2i = F.cross_entropy(logits, targets)
                loss_i2p = F.cross_entropy(logits.t(), targets)
                branch_loss = 0.5 * (loss_p2i + loss_i2p)

                contrastive_loss[f"contrastive_loss_{i_branch}"] = (1.0, branch_loss)
                with torch.no_grad():
                    pred_to_img = logits.argmax(dim=1)
                    img_to_pred = logits.argmax(dim=0)

                    acc_p2i = (pred_to_img == targets).float().mean()
                    acc_i2p = (img_to_pred == targets).float().mean()
                    acc = 0.5 * (acc_p2i + acc_i2p)

                contrastive_loss[f"contrastive_acc_{i_branch}"] = acc

        for layer_idx in range(self.recon.num_layers):
            patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
            patch_emb = self.BrainEmbedEEGLayer(patch_emb, self.area_config) + patch_emb

            patch_emb = self.recon.layers[layer_idx](patch_emb, self.area_config)

        out = self.proj_out(patch_emb)

        return out, {
            **contrastive_loss,
        }
    
    def training_step(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(d_model // 2 + 1, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, mask_x.shape[1] // 2 + 1)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def generate_area_config(brain_regions):
    region_to_channels = defaultdict(list)
    for channel_idx, region in enumerate(brain_regions):
        region_to_channels[region].append(channel_idx)

    area_config = {}
    for region, channels in region_to_channels.items():
        area_config[f'region_{region}'] = {
            'channels': len(channels),
            'slice': slice(channels[0], channels[-1] + 1)
        }
    return area_config

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CSBrain(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8).to(device)
    model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth',
                                     map_location=device))
    a = torch.randn((8, 16, 10, 200)).cuda()
    b = model(a)
    print(a.shape, b.shape)