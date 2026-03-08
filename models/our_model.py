import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CSBrain_transformerlayer import *
from models.CSBrain_transformer import *
from collections import Counter
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import uuid
import math

class _SetInteractionBlock(nn.Module):
    """
    Permutation-equivariant block over channel dimension:
    reordering channels reorders internal tokens the same way.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, C, H)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


# class SourceProjector(nn.Module):
#     """
#     Input:  (B, C, N, d)
#     Output: (B, K, N, d)

#     - Handles variable number of channels C.
#     - Channel order equivariant in interaction layers.
#     - Final source extraction is permutation-invariant to channel order
#       via learned source queries attending over the channel set.
#     """
#     def __init__(
#         self,
#         patch_dim: int,
#         num_sources: int,
#         hidden_dim: int,
#         num_heads: int = 8,
#         num_interaction_layers: int = 2,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.num_sources = num_sources
#         self.hidden_dim = hidden_dim

#         self.in_proj = nn.Sequential(
#             nn.Linear(patch_dim, patch_dim),
#             nn.GELU(),
#             nn.Linear(patch_dim, patch_dim),
#         )
#         # self.interaction = nn.ModuleList(
#         #     [_SetInteractionBlock(hidden_dim, num_heads, dropout) for _ in range(num_interaction_layers)]
#         # )

#         self.k_map = nn.Linear(patch_dim, hidden_dim)

#         self.source_queries = nn.Parameter(torch.randn(num_sources, hidden_dim))
#         nn.init.xavier_normal_(self.source_queries)
#         self.ch_attn = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=1,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.src_attn1 = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=1,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.src_attn2 = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True,
#         )
        
#         self.src_norm1 = nn.LayerNorm(hidden_dim)
#         self.k_norm = nn.LayerNorm(hidden_dim)
#         self.v_norm = nn.LayerNorm(hidden_dim)
#         self.src_ffn = nn.Sequential(
#             nn.Linear(hidden_dim, 4 * hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(4 * hidden_dim, hidden_dim),
#             nn.Dropout(dropout),
#         )
#         self.src_norm2 = nn.LayerNorm(hidden_dim)

#         self.out_proj = nn.Linear(patch_dim, patch_dim)

#         self.mixer = nn.Linear(patch_dim, num_sources, bias=False)

#     def forward(self, x: torch.Tensor, coord_emb: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, C, N, d)
#         coord_emb: (B, C, 1, d)
#         """
#         b, c, n, d = x.shape
#         x = x.permute(0, 2, 3, 1).contiguous().view(b, n * d, c)
#         mixer = self.mixer(coord_emb)  # (B, C, d) -> (B, C, K)
#         x = x @ mixer # (B, N*d, K)
#         s = x.view(b, n, d, self.num_sources).permute(0, 3, 1, 2).contiguous()  # (B, K, N, d)

#         # h = x                       # (B, C, N, H)
#         # h = h.reshape(b, c, n*d)  # (B*N, C, H)
#         # q = self.source_queries.unsqueeze(0).expand(b, -1, -1)    # (B, K, N*H)
#         # q0 = q
#         # h = self.ch_attn(h, h, h, need_weights=False)[0] + h
#         # s, _ = self.src_attn1(q, self.k_map(coord_emb), h, need_weights=False)
#         # q = q0 + s
#         # s = s + self.src_ffn(self.src_norm2(s))

#         # s = s.reshape(b, self.num_sources, n, d).contiguous()  # (B, K, N, d)
#         # s = self.out_proj(s)  # (B, K, N, d)
#         return s

class SourceProjector(nn.Module):
    """
    Input:  (B, C, N, d)
    Output: (B, K, N, d)

    - Handles variable number of channels C.
    - Channel order equivariant in interaction layers.
    - Final source extraction is permutation-invariant to channel order
      via learned source queries attending over the channel set.
    """
    def __init__(
        self,
        patch_dim: int,
        num_sources: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.hidden_dim = hidden_dim

        self.mixer = nn.Linear(patch_dim, num_sources, bias=False)

    def forward(self, x: torch.Tensor, coord_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, N, d)
        coord_emb: (B, C, 1, d)
        """
        b, c, n, d = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, n * d, c)
        mixer = self.mixer(coord_emb)  # (B, C, d) -> (B, C, K)
        x = x @ mixer # (B, N*d, K)
        s = x.view(b, n, d, self.num_sources).permute(0, 3, 1, 2).contiguous()  # (B, K, N, d)

        return s


class OurModel(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                 nhead=8, TemEmbed_kernel_sizes=[(1,), (3,), (5,)], brain_regions=[]):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)

        self.TemEmbed_kernel_sizes = TemEmbed_kernel_sizes
        kernel_sizes = self.TemEmbed_kernel_sizes
        self.TemEmbedEEGLayer = TemEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim, kernel_sizes=kernel_sizes, stride=1)

        self.brain_regions = brain_regions
        self.area_config = generate_area_config(sorted(brain_regions)) if brain_regions is not None else None
        self.BrainEmbedEEGLayer = BrainEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim)

        encoder_layer = CSBrain_TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, area_config=self.area_config, batch_first=True,
            activation=F.gelu
        )
        self.encoder = CSBrain_TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)

        self.source_projector = SourceProjector(
            patch_dim=in_dim,
            num_sources=16,
            hidden_dim=d_model * seq_len,
        )

        self.spherical_r_scale = 0.1
        self.spherical_num_freqs = 32
        self.spherical_pe_base = 1000.0
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
        self.coord_to_source_weights = nn.Sequential(
            nn.Linear(3 * 2 * self.spherical_num_freqs, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.source_projector.num_sources),
        )

        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

        self.features_by_layer = []
        self.input_features = []

    def compute_zero_lag_sync_loss(self, source_emb):
        # Zero-lag synchronization regularizer on projected sources.
        src_t = source_emb.view(source_emb.size(0), source_emb.size(1), -1)  # (B, K, N*d)

        # Per-source normalization over token/time axis
        src_t = src_t - src_t.mean(dim=-1, keepdim=True)
        src_t = src_t / (src_t.std(dim=-1, keepdim=True) + 1e-6)

        # Lag-0 correlation matrix across sources: (B, K, K)
        corr = torch.matmul(src_t, src_t.transpose(1, 2)) / src_t.size(-1)

        # Penalize cross-source instantaneous synchronization (off-diagonal terms)
        eye = torch.eye(corr.size(-1), device=corr.device, dtype=corr.dtype).unsqueeze(0)
        off_diag = corr * (1.0 - eye)
        zero_lag_sync_loss = (off_diag ** 2).mean()
        return zero_lag_sync_loss

    def _spherical_positional_encoding(self, ch_coords):
        # ch_coords: (B, C, 3) -> (r, theta, phi)
        finite_mask = torch.isfinite(ch_coords).all(dim=-1, keepdim=True)
        safe_coords = torch.where(finite_mask, ch_coords, torch.zeros_like(ch_coords))

        r = safe_coords[..., 0:1] / self.spherical_r_scale
        theta = safe_coords[..., 1:2] / math.pi
        phi = safe_coords[..., 2:3] / math.pi
        norm_coords = torch.cat([r, theta, phi], dim=-1)

        inv_freq = self.spherical_inv_freq.to(device=norm_coords.device, dtype=norm_coords.dtype)
        angles = norm_coords.unsqueeze(-1) * inv_freq.view(1, 1, 1, -1)
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        pe = pe.reshape(norm_coords.size(0), norm_coords.size(1), -1)
        return pe, finite_mask.squeeze(-1)

    def forward(self, batch, mask=None):
        x = batch['timeseries']
        # x = x[:, self.sorted_indices, :, :]

        patch_emb = self.patch_embedding(x, mask)

        ch_coords = batch['ch_coords']
        coord_pe, finite_coord_mask = self._spherical_positional_encoding(ch_coords)

        coord_emb = self.coord_enhancement(coord_pe)
        patch_emb = patch_emb + coord_emb.unsqueeze(-2)
        patch_emb = self.source_projector(patch_emb, coord_emb)

        zero_lag_sync_loss = self.compute_zero_lag_sync_loss(patch_emb)

        for layer_idx in range(self.encoder.num_layers):
            patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
            patch_emb = self.BrainEmbedEEGLayer(patch_emb, self.area_config) + patch_emb

            patch_emb = self.encoder.layers[layer_idx](patch_emb, self.area_config)

        # eeg_from_sources = patch_emb
        if 'valid_channel_mask' in batch:
            valid_channel_mask = batch['valid_channel_mask'].to(x.device).bool()
        else:
            valid_channel_mask = torch.ones_like(finite_coord_mask, dtype=torch.bool)

        coord_valid_mask = valid_channel_mask & finite_coord_mask

        source_weights = self.coord_to_source_weights(coord_pe)
        eeg_from_sources = torch.einsum('bck,bknd->bcnd', source_weights, patch_emb)
        eeg_from_sources = eeg_from_sources * coord_valid_mask.unsqueeze(-1).unsqueeze(-1).to(eeg_from_sources.dtype)

        out = self.proj_out(eeg_from_sources)

        return patch_emb, {
            # "rec": eeg_from_sources,
            "zero_lag_sync_loss": (0.1, zero_lag_sync_loss),
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
    model = OurModel(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8).to(device)
    model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth',
                                     map_location=device))
    a = torch.randn((8, 16, 10, 200)).cuda()
    b = model(a)
    print(a.shape, b.shape)