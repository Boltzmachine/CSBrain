import math

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


class SourceProjector(nn.Module):
    """
    Standalone sensor-to-source projector.

    Input:  x (B, C, N, d), ch_coords (B, C, 3)  — raw sensor signals + spherical coords
    Output: (B, K, N, d)  — source-space signals

    Mixing weights depend only on electrode coordinates (not data),
    analogous to a learned lead-field matrix.

    Pre-trainable with reconstruction + decorrelation (see training_step).
    """
    def __init__(
        self,
        in_dim: int,
        num_sources: int = 32,
        decorr_weight: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_sources = num_sources
        self.decorr_weight = decorr_weight

        # --- spherical positional encoding ---
        self.spherical_r_scale = 0.1
        self.spherical_num_freqs = 32
        self.spherical_pe_base = 20.0
        inv_freq = torch.exp(
            -math.log(self.spherical_pe_base)
            * torch.arange(self.spherical_num_freqs, dtype=torch.float32)
            / self.spherical_num_freqs
        )
        self.register_buffer("spherical_inv_freq", inv_freq, persistent=False)

        pe_dim = 3 * 2 * self.spherical_num_freqs  # 192

        # coord -> forward mixing weights  (sensors -> sources)
        self.forward_mix = nn.Sequential(
            nn.Linear(pe_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, num_sources),
        )

        # coord -> inverse mixing weights  (sources -> sensors, for reconstruction)
        self.inverse_mix = nn.Sequential(
            nn.Linear(pe_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, num_sources),
        )

    def _spherical_positional_encoding(self, ch_coords):
        """ch_coords: (B, C, 3) spherical (r, theta, phi)."""
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

    def _valid_channel_mask(self, ch_coords, valid_channel_mask=None):
        """Combine user-supplied validity mask with finite-coordinate check."""
        _, finite_mask = self._spherical_positional_encoding(ch_coords)
        if valid_channel_mask is not None:
            return valid_channel_mask.bool() & finite_mask
        return finite_mask

    def forward(self, x: torch.Tensor, ch_coords: torch.Tensor,
                valid_channel_mask: torch.Tensor = None):
        """
        x:         (B, C, N, d)  raw sensor timeseries
        ch_coords: (B, C, 3)    spherical coordinates
        valid_channel_mask: (B, C) bool — False for padded/invalid channels
        Returns:   (B, K, N, d) source signals
        """
        coord_pe, _ = self._spherical_positional_encoding(ch_coords)
        b, c, n, d = x.shape

        w_fwd = self.forward_mix(coord_pe)        # (B, C, K)

        # Zero out weights for invalid/padded channels
        valid = self._valid_channel_mask(ch_coords, valid_channel_mask)  # (B, C)
        w_fwd = w_fwd * valid.unsqueeze(-1).float()

        x_flat = x.permute(0, 2, 3, 1).contiguous().view(b, n * d, c)
        s = (x_flat @ w_fwd).view(b, n, d, self.num_sources)
        s = s.permute(0, 3, 1, 2).contiguous()    # (B, K, N, d)
        return s

    def inverse(self, s: torch.Tensor, ch_coords: torch.Tensor,
                valid_channel_mask: torch.Tensor = None):
        """Reconstruct sensors from sources: (B, K, N, d) -> (B, C, N, d)."""
        coord_pe, _ = self._spherical_positional_encoding(ch_coords)
        w_inv = self.inverse_mix(coord_pe)         # (B, C, K)

        valid = self._valid_channel_mask(ch_coords, valid_channel_mask)  # (B, C)
        w_inv = w_inv * valid.unsqueeze(-1).float()

        recon = torch.einsum('bck,bknd->bcnd', w_inv, s)
        return recon

    def compute_decorr_loss(self, sources):
        """Penalise cross-source instantaneous correlation."""
        src = sources.view(sources.size(0), sources.size(1), -1)  # (B, K, N*d)
        src = src - src.mean(dim=-1, keepdim=True)
        src = src / (src.std(dim=-1, keepdim=True) + 1e-6)
        corr = torch.matmul(src, src.transpose(1, 2)) / src.size(-1)
        eye = torch.eye(corr.size(-1), device=corr.device, dtype=corr.dtype).unsqueeze(0)
        return ((corr * (1.0 - eye)) ** 2).mean()

    def training_step(self, batch, mask=None):
        """
        Pre-training: sensor -> source -> sensor reconstruction.
        Returns (recon, loss_dict) compatible with the existing Trainer.
        """
        x = batch['timeseries']       # (B, C, N, d)
        ch_coords = batch['ch_coords']
        valid_channel_mask = batch.get('valid_channel_mask', None)

        valid = self._valid_channel_mask(ch_coords, valid_channel_mask)

        sources = self.forward(x, ch_coords, valid_channel_mask=valid)
        recon = self.inverse(sources, ch_coords, valid_channel_mask=valid)

        # Loss only on valid channels
        mask_exp = valid.unsqueeze(-1).unsqueeze(-1).float()  # (B, C, 1, 1)
        n_valid = mask_exp.sum().clamp(min=1)

        recon_loss = ((recon - x) ** 2 * mask_exp).sum() / (n_valid * x.size(2) * x.size(3))
        decorr_loss = self.compute_decorr_loss(sources)

        return recon, {
            "recon_loss": (1.0, recon_loss),
            "decorr_loss": (self.decorr_weight, decorr_loss),
        }



class MLPSemanticReadout(nn.Module):
    def __init__(self, n_ch: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_ch * 240, 1024),
            nn.GELU(),
            # nn.Linear(1024, 1024),
            # nn.GELU(),
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
    

class MLPSemanticReadoutGlobal(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(x)
        return y


class AttentionMLPSemanticReadout(nn.Module):
    def __init__(self, flatten_dim: int, out_dim: int, hidden_dim: int = 200, dropout: float = 0.1):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.LayerNorm(flatten_dim),
            nn.Linear(flatten_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(flatten_dim),
            nn.Dropout(dropout),
            nn.Linear(flatten_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (..., n_ch, flatten_dim)
        attn_logits = self.channel_attention(x).squeeze(-1)
        if channel_mask is not None:
            attn_logits = attn_logits.masked_fill(~channel_mask.bool(), float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1)
        aggregated = torch.sum(x * attn_weights.unsqueeze(-1), dim=-2)
        return self.mlp(aggregated)


class CSBrainAlign(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                 nhead=8, TemEmbed_kernel_sizes=[(1,), (3,), (5,)], brain_regions=[], sorted_indices=[],
                 causal=False, project_to_source=False, num_sources=32,
                 source_projector_ckpt=None, freeze_source_projector=True):
        super().__init__()
        self.causal = causal
        self.project_to_source = project_to_source

        # --- Source projector (operates on raw timeseries, before patch embedding) ---
        self.source_projector = SourceProjector(
            in_dim=in_dim,
            num_sources=num_sources,
        )
        if source_projector_ckpt is not None:
            ckpt = torch.load(source_projector_ckpt, map_location='cpu')
            self.source_projector.load_state_dict(ckpt, strict=True)
            print(f"Loaded pre-trained source projector from {source_projector_ckpt}")
        if self.project_to_source and freeze_source_projector:
            for p in self.source_projector.parameters():
                p.requires_grad = False
            print("Source projector weights frozen")

        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len, causal=causal)

        self.TemEmbed_kernel_sizes = TemEmbed_kernel_sizes
        kernel_sizes = self.TemEmbed_kernel_sizes
        self.TemEmbedEEGLayer = TemEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim, kernel_sizes=kernel_sizes, stride=1, causal=causal)

        self.brain_regions = brain_regions
        self.area_config = None #generate_area_config(sorted(brain_regions))
        self.BrainEmbedEEGLayer = BrainEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim)
        self.sorted_indices = sorted_indices

        self.pretrained_image_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base").eval()
        encoder_layer = CSBrain_TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, area_config=self.area_config, sorted_indices=self.sorted_indices, batch_first=True,
            activation=F.gelu, causal=causal
        )
        self.encoder = CSBrain_TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)

        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )

        self.add_global = True
        if self.add_global:
            self.global_channel = nn.Parameter(torch.randn(1, 1, 1, d_model))
            self.global_token = nn.Parameter(torch.randn(1, 1, 1, d_model))

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

        self.features_by_layer = []
        self.input_features = []

        self.num_visual_levels = 10
        max_freq_bins = seq_len * in_dim // 2 + 1
        self.freq_mask_logits = nn.Parameter(torch.zeros(self.num_visual_levels, max_freq_bins))

        hidden_ch_dim = 64
        semantic_arch = 'mlp' # 'mlp', 'attn_mlp', 'cnn'
        self.event_window_len = 240

        if semantic_arch == 'mlp':
            if self.add_global:
                self.semantic_readout = MLPSemanticReadoutGlobal(out_dim=d_model, hidden_dim=d_model)
            else:
                self.semantic_readout = MLPSemanticReadout(n_ch=num_sources, out_dim=d_model, hidden_dim=hidden_ch_dim)

        elif semantic_arch == 'attn_mlp':
            self.semantic_readout = AttentionMLPSemanticReadout(
                flatten_dim=12 * d_model,
                out_dim=d_model,
                hidden_dim=hidden_ch_dim,
            )
        elif semantic_arch == 'cnn':
            self.semantic_readout = CNNSemanticReadout(n_ch=num_sources, out_dim=d_model, hidden_dim=hidden_ch_dim, seq_len=self.event_window_len)

        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 768),
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
                # branch_embed = self.BrainEmbedEEGLayer(branch_embed, self.area_config) + branch_embed
                branch_embed = self.encoder_alignment.layers[layer_idx](branch_embed, self.area_config)
            branch_embeds.append(branch_embed)
        branch_embeds = torch.stack(branch_embeds, dim=1)

        return branch_embeds
    
    @torch.inference_mode()
    def get_image_hidden_states(self, **image_encoder_inputs):
        outputs = self.pretrained_image_encoder(**image_encoder_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        selected_hidden_states = []
        # for layer_idx in [1, 6, 12]:
        for layer_idx in [12]:
            selected_hidden_states.append(hidden_states[layer_idx][:, 0])
        selected_hidden_states = torch.stack(selected_hidden_states, dim=1) # (B, n_branch, d_model)
        return selected_hidden_states


    def forward(self, batch, mask=None):
        x = batch['timeseries'] # (B, n_ch, seq_len, in_dim)
        ch_coords = batch['ch_coords']

        # --- Apply mask in sensor space, then project to source space ---
        if self.project_to_source:
            valid_ch = batch.get('valid_channel_mask', None)
            if mask is not None:
                # Apply mask in original sensor space before source projection
                x_masked = x.clone()
                x_masked[mask == 1] = 0.0  # zero out masked patches in sensor space
                x = self.source_projector(x_masked, ch_coords, valid_channel_mask=valid_ch)
            else:
                x = self.source_projector(x, ch_coords, valid_channel_mask=valid_ch)
            # Don't pass mask to patch_embedding — already applied
            mask = None

        if not self.causal:
            # Spectral filtering mixes all time steps — skip in causal mode
            timeseries = x.view(x.size(0), x.size(1), -1)
            spectral = torch.fft.rfft(timeseries, dim=-1, norm='forward')
            masks = torch.sigmoid(self.freq_mask_logits[:1, :spectral.size(-1)]) # (num_visual_levels, n_freq_bins)
            branch_spectral = spectral * masks.unsqueeze(0) # (n_branch, B, n_ch, n_freq_bins)
            branch_time = torch.fft.irfft(branch_spectral, norm='forward') # (n_branch, B, n_ch, seq_len)
            x = branch_time.view_as(x)

        patch_emb = self.patch_embedding(x, mask)

        if self.project_to_source:
            # Sources are always valid (no padding in source dim)
            b = patch_emb.size(0)
            batch['valid_channel_mask'] = torch.ones(b, self.source_projector.num_sources,
                                                     dtype=torch.bool, device=patch_emb.device)
        else:
            coord_pe, finite_coord_mask = self._spherical_positional_encoding(ch_coords)
            coord_emb = self.coord_enhancement(coord_pe)
            patch_emb = patch_emb + coord_emb.unsqueeze(2)

        if self.add_global:
            patch_emb = torch.cat([self.global_channel.expand(patch_emb.size(0), -1, patch_emb.size(2), -1), patch_emb], dim=1)
            patch_emb = torch.cat([self.global_token.expand(patch_emb.size(0), patch_emb.size(1), -1, -1), patch_emb], dim=2)
            if 'valid_channel_mask' in batch:
                batch['valid_channel_mask'] = torch.cat([torch.ones_like(batch['valid_channel_mask'][:, :1], dtype=torch.bool), batch['valid_channel_mask']], dim=1)
            if 'valid_length_mask' in batch:
                batch['valid_length_mask'] = torch.cat([torch.ones_like(batch['valid_length_mask'][:, :1], dtype=torch.bool), batch['valid_length_mask']], dim=1)

        contrastive_loss = dict()

        for layer_idx in range(self.encoder.num_layers):
            patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
            # patch_emb = self.BrainEmbedEEGLayer(patch_emb, self.area_config) + patch_emb
            patch_emb = self.encoder.layers[layer_idx](patch_emb, self.area_config, inter_window_attn_mask=~batch['valid_length_mask'] if 'valid_length_mask' in batch else None, inter_region_attn_mask=~batch['valid_channel_mask'] if 'valid_channel_mask' in batch else None)

            if layer_idx == self.encoder.num_layers - 3:
                i_branch = 0
                has_image = batch.get('has_image', None)
                if has_image is not None:
                    branch_embs = patch_emb[has_image] # (B_with_image, K, N, d)
                else:
                    branch_embs = patch_emb
                if 'image_encoder_inputs' in batch and has_image.any():
                    image_encoder_inputs = {k: v[has_image] for k, v in batch['image_encoder_inputs'].items()}
                    if self.add_global:
                        semantic_emb = self.semantic_readout(branch_embs[:, 0, 0, :])
                    else:
                        semantic_emb = self.semantic_readout(branch_embs.view(branch_embs.size(0), branch_embs.size(1), -1))
                    pred_flatten = self.contrastive_proj(semantic_emb) # (B, n_events, d_model) <- does not work

                    image_hidden_states = self.get_image_hidden_states(**image_encoder_inputs) # (B, n_events, n_branch, d_model)
                    image_hidden_states_branch = image_hidden_states[:, i_branch] # (B, n_events, d_model)
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

                        acc_p2i = (pred_to_img == targets).float()
                        acc_i2p = (img_to_pred == targets).float()
                        acc = 0.5 * (acc_p2i + acc_i2p)

                    contrastive_loss[f"contrastive_acc_{i_branch}"] = acc.mean()
                    contrastive_loss_per_source = defaultdict(list)

                    sources = batch['source']
                    if has_image is not None:
                        mask_list = has_image.detach().cpu().tolist()
                        sources = [s for s, m in zip(sources, mask_list) if m]
                    for i, src in enumerate(sources):
                        contrastive_loss_per_source[src].append(acc[i])

                    for src, acc_list in contrastive_loss_per_source.items():
                        contrastive_loss[f"contrastive_acc_{src}"] = torch.stack(acc_list).mean()

        if self.project_to_source:
            source_emb = patch_emb[:, 1:, 1:, :] if self.add_global else patch_emb
            # Reconstruct back to sensor space for masked reconstruction loss
            out = self.source_projector.inverse(source_emb, ch_coords,
                                                valid_channel_mask=valid_ch)
            out = self.proj_out(out)
        else:
            out = self.proj_out(patch_emb)
            if self.add_global:
                out = out[:, 1:, 1:, :]

        rep = branch_embs # (B, K, N, d)
        #     rep = torch.cat([self.global_token.expand(out.size(0), -1, -1, -1), out], dim=2)

        return out, {
            "rep": rep,
            # "zero_lag_sync_loss": (0.1, zero_lag_sync_loss),
            **contrastive_loss,
        }
    
    def training_step(self, *args, **kwargs):
        return self.forward(*args, **kwargs)



class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len, causal=False):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        if causal:
            # Channel dim (kernel=19, pad=9) stays symmetric.
            # Time dim (kernel=7): no built-in padding; left-pad manually.
            self.positional_encoding = nn.Sequential(
                nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7),
                          stride=(1, 1), padding=(9, 0), groups=d_model),
            )
            self.pos_time_pad = 6  # kernel_time - 1
        else:
            self.positional_encoding = nn.Sequential(
                nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                          groups=d_model),
            )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=(1, d_model), stride=(1, 1), padding=(0, 0)),
            nn.GroupNorm(5, d_model),
            nn.GELU(),

            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GroupNorm(5, d_model),
            nn.GELU(),

            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GroupNorm(5, d_model),
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

        if self.causal:
            # Process each time step independently so GroupNorm doesn't mix across time
            proj_in_x = mask_x.permute(0, 2, 1, 3).contiguous().view(bz * patch_num, 1, ch_num, patch_size)
            patch_emb = self.proj_in(proj_in_x)  # (B*T, d_model, C, 1)
            patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, patch_num, ch_num, self.d_model)
            patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous()  # (B, C, T, d_model)
        else:
            mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
            patch_emb = self.proj_in(mask_x)
            patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, mask_x.shape[1] // 2 + 1)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        pe_input = patch_emb.permute(0, 3, 1, 2)  # (B, d_model, ch, time)
        if self.causal:
            pe_input = F.pad(pe_input, (self.pos_time_pad, 0))  # left-pad time dim
        positional_embedding = self.positional_encoding(pe_input)
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
