import copy
import json
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CSBrain_transformerlayer import *
from models.CSBrain_transformer import *
from models.adversarial import SessionDiscriminator
from models.llm_vq import LLMEmbeddingVQ
from collections import Counter
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel


# ---------------------------------------------------------------------------
# Hemispheric flip utilities
# ---------------------------------------------------------------------------

def _normalize_ch_name(name):
    """Normalize channel name: uppercase, strip common reference suffixes."""
    name = name.upper().strip()
    for suffix in ('-REF', '-LE', '-AR', '-AVG'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def _get_homologous_name(norm_name):
    """Return the contralateral homologue of a normalized channel name.

    Odd-numbered electrodes are left-hemisphere; even are right.
    Midline channels (ending in 'Z') and channels without a trailing
    digit map to themselves.
    """
    if norm_name in ('PAD', ''):
        return norm_name
    if norm_name.endswith('Z'):
        return norm_name
    # Split into alphabetic base and trailing number
    m = re.match(r'^([A-Z]+)(\d+)$', norm_name)
    if m is None:
        return norm_name  # no trailing digit → identity
    base, num_str = m.group(1), int(m.group(2))
    pair_num = num_str + 1 if num_str % 2 == 1 else num_str - 1
    return f'{base}{pair_num}'


def build_flip_perm_batch(ch_names_batch, valid_channel_mask=None):
    """Build a hemispheric-flip permutation for every sample in a batch.

    Parameters
    ----------
    ch_names_batch : list[list[str]]
        Per-sample channel name lists.
    valid_channel_mask : (B, C) bool tensor, optional
        ``False`` for padded / invalid channels.

    Returns
    -------
    perm : (B, C) long tensor
        ``perm[b, i]`` is the index of the channel whose *data* should
        appear at position *i* after the hemispheric flip.  Midline and
        unpaired channels map to themselves.
    """
    B = len(ch_names_batch)
    C = max(len(names) for names in ch_names_batch)
    perm = torch.arange(C, dtype=torch.long).unsqueeze(0).expand(B, -1).clone()

    for b, ch_names in enumerate(ch_names_batch):
        # normalised-name → original index  (valid channels only)
        norm_to_idx = {}
        for i, name in enumerate(ch_names):
            if valid_channel_mask is not None and not valid_channel_mask[b, i]:
                continue
            norm = _normalize_ch_name(name)
            if norm == 'PAD':
                continue
            norm_to_idx[norm] = i

        for i, name in enumerate(ch_names):
            if valid_channel_mask is not None and not valid_channel_mask[b, i]:
                continue
            norm = _normalize_ch_name(name)
            if norm == 'PAD':
                continue
            pair_norm = _get_homologous_name(norm)
            if pair_norm in norm_to_idx:
                perm[b, i] = norm_to_idx[pair_norm]
    return perm


class SourceProjector(nn.Module):
    """
    Standalone sensor-to-source projector.

    Input:  x (B, C, N, d), ch_coords (B, C, 3)  — raw sensor signals + spherical coords
    Output: (B, K, N, d)  — source-space signals

    Forward path is MLP + cross-attention + MLP, applied independently per
    patch index. Queries are K learnable source embeddings, keys come from
    a per-channel coordinate positional encoding, and values come from the
    per-channel, per-patch timeseries — so each source attends across
    sensors with weights driven by electrode geometry plus signal content.
    The inverse path mirrors this with channel-coord queries and learnable
    per-source keys.

    Pre-trainable with reconstruction + decorrelation (see training_step).
    """
    def __init__(
        self,
        in_dim: int,
        num_sources: int = 32,
        decorr_weight: float = 0.1,
        attn_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_sources = num_sources
        self.decorr_weight = decorr_weight
        self.attn_dim = attn_dim
        self.num_heads = num_heads

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

        # --- Forward path: MLP + cross-attention + MLP (sensors -> sources) ---
        self.forward_queries = nn.Parameter(torch.randn(num_sources, attn_dim) * 0.02)
        self.forward_key_mlp = nn.Sequential(
            nn.Linear(pe_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, attn_dim),
        )
        self.forward_value_mlp = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, attn_dim),
        )
        self.forward_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, batch_first=True,
        )
        self.forward_out_mlp = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, in_dim),
        )

        # --- Inverse path: MLP + cross-attention + MLP (sources -> sensors) ---
        self.inverse_keys = nn.Parameter(torch.randn(num_sources, attn_dim) * 0.02)
        self.inverse_query_mlp = nn.Sequential(
            nn.Linear(pe_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, attn_dim),
        )
        self.inverse_value_mlp = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, attn_dim),
        )
        self.inverse_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, batch_first=True,
        )
        self.inverse_out_mlp = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, in_dim),
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
                valid_channel_mask: torch.Tensor = None,
                sensor_mask: torch.Tensor = None):
        """
        x:         (B, C, N, d)  raw sensor timeseries
        ch_coords: (B, C, 3)    spherical coordinates
        valid_channel_mask: (B, C) bool — False for padded/invalid channels
        sensor_mask: (B, C, N) — 1/True at (channel, patch) positions that
            should be hidden from the cross-attention (masked patches).
            Folded per-timestep into the key_padding_mask so source tokens
            at time n genuinely cannot see masked sensors at time n.
        Returns:   (B, K, N, d) source signals
        """
        B, C, N, d = x.shape
        K = self.num_sources

        coord_pe, _ = self._spherical_positional_encoding(ch_coords)         # (B, C, pe_dim)
        keys = self.forward_key_mlp(coord_pe)                                # (B, C, attn_dim)

        x_perm = x.permute(0, 2, 1, 3).contiguous().view(B * N, C, d)        # (B*N, C, d)
        values = self.forward_value_mlp(x_perm)                              # (B*N, C, attn_dim)

        keys_exp = keys.unsqueeze(1).expand(B, N, C, self.attn_dim).reshape(B * N, C, self.attn_dim)
        queries = self.forward_queries.unsqueeze(0).expand(B * N, K, self.attn_dim)

        valid = self._valid_channel_mask(ch_coords, valid_channel_mask)      # (B, C)
        key_padding_mask = ~valid.unsqueeze(1).expand(B, N, C).reshape(B * N, C)

        if sensor_mask is not None:
            # sensor_mask: (B, C, N) -> (B*N, C). True = masked patch, hide.
            sm = sensor_mask.bool().permute(0, 2, 1).reshape(B * N, C)
            combined = key_padding_mask | sm
            # Guard against all-True rows (softmax over -inf -> NaN): fall back
            # to the valid-only mask for any (b, n) where every sensor is masked.
            all_padded = combined.all(dim=-1, keepdim=True)
            key_padding_mask = torch.where(all_padded, key_padding_mask, combined)

        attn_out, _ = self.forward_attn(
            queries, keys_exp, values, key_padding_mask=key_padding_mask,
            need_weights=False,
        )                                                                    # (B*N, K, attn_dim)
        out = self.forward_out_mlp(attn_out)                                 # (B*N, K, d)
        s = out.view(B, N, K, d).permute(0, 2, 1, 3).contiguous()            # (B, K, N, d)
        return s

    def inverse(self, s: torch.Tensor, ch_coords: torch.Tensor,
                valid_channel_mask: torch.Tensor = None):
        """Reconstruct sensors from sources: (B, K, N, d) -> (B, C, N, d)."""
        B, K_, N, d = s.shape

        coord_pe, _ = self._spherical_positional_encoding(ch_coords)         # (B, C, pe_dim)
        C = coord_pe.size(1)
        queries = self.inverse_query_mlp(coord_pe)                           # (B, C, attn_dim)

        s_perm = s.permute(0, 2, 1, 3).contiguous().view(B * N, K_, d)       # (B*N, K, d)
        values = self.inverse_value_mlp(s_perm)                              # (B*N, K, attn_dim)

        queries_exp = queries.unsqueeze(1).expand(B, N, C, self.attn_dim).reshape(B * N, C, self.attn_dim)
        keys = self.inverse_keys.unsqueeze(0).expand(B * N, K_, self.attn_dim)

        attn_out, _ = self.inverse_attn(queries_exp, keys, values, need_weights=False)  # (B*N, C, attn_dim)
        out = self.inverse_out_mlp(attn_out)                                 # (B*N, C, d)
        out = out.view(B, N, C, d).permute(0, 2, 1, 3).contiguous()          # (B, C, N, d)

        valid = self._valid_channel_mask(ch_coords, valid_channel_mask)      # (B, C)
        out = out * valid.unsqueeze(-1).unsqueeze(-1).to(out.dtype)
        return out

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


class TemporalConcatEncoder(nn.Module):
    """Temporal-only transformer over concatenated source features.

    After the source projector the K sources carry a fixed,
    montage-independent identity, so spatial (cross-channel) attention is
    unnecessary. Instead, all K source features at a timestep are
    concatenated into a single token of width ``num_sources * d_model``
    and only temporal self-attention is applied — ``num_layers`` of it,
    matching the criss-cross encoder's depth.

    A learnable global token carries the sequence-level (CLS)
    representation. In non-causal mode it is prepended; in causal mode it
    is appended as the *last* position so it can attend over the whole
    sequence while the per-timestep tokens still never see it (and so
    never leak future information).
    """

    def __init__(self, num_sources, d_model, nhead, dim_feedforward,
                 num_layers, dropout=0.1, causal=False):
        super().__init__()
        self.num_sources = num_sources
        self.d_model = d_model
        self.width = num_sources * d_model
        self.num_layers = num_layers
        self.causal = causal
        self.global_last = causal
        assert self.width % nhead == 0, (
            f"num_sources*d_model={self.width} must be divisible by "
            f"nhead={nhead}")

        self.global_token = nn.Parameter(torch.randn(1, 1, self.width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.width, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=F.gelu, batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)])

    def split_global(self, seq):
        """Split (B, N+1, width) into global (B, width) and tokens (B, N, width)."""
        if self.global_last:
            return seq[:, -1, :], seq[:, :-1, :]
        return seq[:, 0, :], seq[:, 1:, :]

    def forward(self, patch_emb, valid_length_mask=None, branch_layer_idx=None):
        """patch_emb: (B, K, N, d_model). Returns (enc_out, branch).

        ``enc_out`` is (B, N+1, width); the global token sits at index
        ``-1`` (causal) or ``0`` (non-causal) — use :meth:`split_global`.
        ``branch`` is the encoder state after layer ``branch_layer_idx``
        (or ``None`` if not requested).
        """
        B, K, N, d = patch_emb.shape
        x = patch_emb.permute(0, 2, 1, 3).reshape(B, N, K * d)        # (B, N, width)
        g = self.global_token.expand(B, -1, -1)

        if valid_length_mask is not None:
            g_valid = torch.ones_like(valid_length_mask[:, :1])

        if self.global_last:
            x = torch.cat([x, g], dim=1)                              # (B, N+1, width)
            if valid_length_mask is not None:
                vlm = torch.cat([valid_length_mask, g_valid], dim=1)
        else:
            x = torch.cat([g, x], dim=1)
            if valid_length_mask is not None:
                vlm = torch.cat([g_valid, valid_length_mask], dim=1)

        key_padding_mask = None
        if valid_length_mask is not None:
            key_padding_mask = ~vlm.bool()                            # (B, N+1)

        attn_mask = None
        if self.causal:
            # Pure causal mask over [t0..t_{N-1}, global]: timestep k sees
            # only timesteps 0..k; the global token (last) sees everything.
            L = x.size(1)
            attn_mask = torch.triu(
                torch.full((L, L), float('-inf'), device=x.device), diagonal=1)

        branch = None
        for i, layer in enumerate(self.layers):
            x = layer(x, src_mask=attn_mask, src_key_padding_mask=key_padding_mask)
            if branch_layer_idx is not None and i == branch_layer_idx:
                branch = x
        return x, branch


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
                 source_projector_ckpt=None, freeze_source_projector=False,
                 adversarial_weight=0.0, num_sessions=256,
                 equivariance_weight=0.0, info_max_weight=0.0,
                 alignment_weight=1.0,
                 patch_embed_type='cnn',
                 mamba_band_periods=None, n_mamba_layers=2,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 use_llm_vq=False, num_language_tokens=8,
                 max_llm_codebook_size=4096, llm_vq_aux_weight=0.1,
                 spectral_mode='static', stft_n_fft=64, stft_hop=1,
                 use_moe=False, num_experts=4, moe_top_k=2,
                 moe_gate_input_dim=32,
                 moe_balance_weight=0.01, moe_band_prior_weight=0.1,
                 moe_z_loss_weight=1e-3,
                 in_dim_for_gate=None,
                 contrastive_band=False, contrastive_n_bands=3,
                 contrastive_proj_dim=64,
                 vision_encoder='facebook/dinov2-base',
                 image_pool_heads=4):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        self.project_to_source = project_to_source # Does not work
        self.adversarial_weight = adversarial_weight
        self.equivariance_weight = equivariance_weight
        self.info_max_weight = info_max_weight
        self.alignment_weight = alignment_weight
        self.patch_embed_type = patch_embed_type
        self.use_llm_vq = use_llm_vq
        self.num_language_tokens = num_language_tokens
        self.llm_vq_aux_weight = llm_vq_aux_weight

        # --- MoE config ---
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        self.moe_gate_input_dim = moe_gate_input_dim
        self.moe_balance_weight = moe_balance_weight
        self.moe_band_prior_weight = moe_band_prior_weight
        self.moe_z_loss_weight = moe_z_loss_weight
        if use_moe and patch_embed_type != 'cnn':
            raise ValueError(
                "use_moe=True currently requires patch_embed_type='cnn' so the "
                "per-patch FFT magnitude is available as a gate input."
            )

        # --- Source projector (operates on raw timeseries, before patch embedding) ---
        if self.project_to_source:
            self.source_projector = SourceProjector(
                in_dim=in_dim,
                num_sources=num_sources,
            )
            # Learned sentinel that replaces masked sensor patches in raw
            # amplitude space (instead of hard zeros — see fix #2).
            self.sensor_mask_embed = nn.Parameter(torch.zeros(in_dim))
            # Learned mask-presence embedding added to source-space tokens at
            # time positions where sensors were masked, so the temporal
            # transformer can tell "predict me" from "preserve me" (fix #3).
            self.source_mask_embed = nn.Parameter(torch.zeros(d_model))
        if source_projector_ckpt is not None:
            raise 
            ckpt = torch.load(source_projector_ckpt, map_location='cpu')
            self.source_projector.load_state_dict(ckpt, strict=True)
            print(f"Loaded pre-trained source projector from {source_projector_ckpt}")

        if self.project_to_source and freeze_source_projector:
            raise
            for p in self.source_projector.parameters():
                p.requires_grad = False
            print("Source projector weights frozen")

        if patch_embed_type == 'mamba':
            self.patch_embedding = MambaPatchEmbedding(
                in_dim, out_dim, d_model, seq_len,
                band_periods=mamba_band_periods,
                n_mamba_layers=n_mamba_layers,
                d_state=mamba_d_state, d_conv=mamba_d_conv,
                expand=mamba_expand, causal=causal,
            )
        elif patch_embed_type == 'cnn':
            self.patch_embedding = PatchEmbedding(
                in_dim, out_dim, d_model, seq_len, causal=causal,
                spectral_mode=spectral_mode, stft_n_fft=stft_n_fft, stft_hop=stft_hop,
            )
        else:
            raise ValueError(f"Unknown patch_embed_type: {patch_embed_type}")

        # --- Band-contrastive pretraining: one specialised PatchEmbedding per
        # frequency band, plus a small projection head for the InfoNCE objective.
        # Later layers (coord PE, transformer, etc.) are shared.
        self.contrastive_band = contrastive_band
        if contrastive_band:
            if patch_embed_type != 'cnn':
                raise ValueError(
                    "contrastive_band currently requires patch_embed_type='cnn'."
                )
            self.band_patch_embeddings = nn.ModuleList([
                PatchEmbedding(
                    in_dim, out_dim, d_model, seq_len, causal=causal,
                    spectral_mode=spectral_mode,
                    stft_n_fft=stft_n_fft, stft_hop=stft_hop,
                )
                for _ in range(contrastive_n_bands)
            ])
            self.contrastive_band_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, contrastive_proj_dim),
            )

        self.TemEmbed_kernel_sizes = TemEmbed_kernel_sizes
        kernel_sizes = self.TemEmbed_kernel_sizes
        self.TemEmbedEEGLayer = TemEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim, kernel_sizes=kernel_sizes, stride=1, causal=causal)

        self.brain_regions = brain_regions
        self.area_config = None #generate_area_config(sorted(brain_regions))
        self.BrainEmbedEEGLayer = BrainEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim)
        self.sorted_indices = sorted_indices

        # Frozen pretrained vision encoder. Behavior dispatches on
        # config.model_type: V-JEPA 2 has no CLS and takes pixel_values_videos,
        # while DINOv2-style ViTs emit a CLS token from pixel_values. Only
        # the attention-pool head (V-JEPA 2 path) is trained.
        self.vision_encoder_name = vision_encoder
        self.pretrained_image_encoder = AutoModel.from_pretrained(vision_encoder).eval()
        for p in self.pretrained_image_encoder.parameters():
            p.requires_grad = False
        self.image_feature_dim = int(self.pretrained_image_encoder.config.hidden_size)
        self.encoder_kind = str(self.pretrained_image_encoder.config.model_type).lower()
        if self.encoder_kind == 'vjepa2':
            # Learned-query attention pool over V-JEPA 2 patch tokens.
            self.image_pool_query = nn.Parameter(
                torch.randn(1, 1, self.image_feature_dim) * 0.02)
            self.image_pool_attn = nn.MultiheadAttention(
                embed_dim=self.image_feature_dim, num_heads=image_pool_heads,
                batch_first=True,
            )
            self.image_pool_norm = nn.LayerNorm(self.image_feature_dim)
        encoder_layer = CSBrain_TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, area_config=self.area_config, sorted_indices=self.sorted_indices, batch_first=True,
            activation=F.gelu, causal=causal,
            use_moe=use_moe, num_experts=num_experts, moe_top_k=moe_top_k,
            moe_gate_input_dim=moe_gate_input_dim,
        )
        self.encoder = CSBrain_TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)

        # --- MoE gate input: project per-patch FFT magnitude to a small
        # descriptor consumed by every encoder layer's MoE gate. Building it
        # once here keeps each gate cheap and lets us anchor it on a fixed
        # F_bins = in_dim // 2 + 1 (the static rFFT magnitude length).
        if self.use_moe:
            f_bins = (in_dim_for_gate or in_dim) // 2 + 1
            self.moe_n_freq_bins = f_bins
            self.moe_gate_proj = nn.Sequential(
                nn.LayerNorm(f_bins),
                nn.Linear(f_bins, moe_gate_input_dim),
            )
            # Pre-compute contiguous frequency-bin ranges per expert so the
            # band-prior loss is parameter-free.
            edges = torch.linspace(0, f_bins, num_experts + 1).round().long()
            band_assign = torch.zeros(num_experts, f_bins)
            for k in range(num_experts):
                lo, hi = int(edges[k]), max(int(edges[k + 1]), int(edges[k]) + 1)
                band_assign[k, lo:hi] = 1.0
            self.register_buffer('moe_band_assign', band_assign, persistent=False)
            # Learned default for the gate input at global-token positions
            # (there is no underlying patch FFT there). The band-prior target
            # at those positions is pinned to uniform: leaving it learnable
            # lets it drift off the simplex and break the KL once it goes
            # negative, which manifests as NaN loss after several epochs.
            self.moe_global_gate_input = nn.Parameter(torch.zeros(moe_gate_input_dim))
            self.register_buffer(
                'moe_global_band_target',
                torch.full((num_experts,), 1.0 / num_experts),
                persistent=False,
            )

        # Source path: temporal-only attention over concatenated sources.
        if self.project_to_source:
            self.source_encoder = TemporalConcatEncoder(
                num_sources=num_sources, d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, num_layers=n_layer,
                causal=causal,
            )
            self.source_global_proj = nn.Linear(num_sources * d_model, d_model)

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
            nn.Linear(d_model, self.image_feature_dim),
        )

        if self.use_llm_vq:
            self._build_llm_vq_path(
                d_model=d_model,
                max_llm_codebook_size=max_llm_codebook_size,
            )

        # Equivariance projection head — discarded at fine-tuning so the
        # backbone representation stays unconstrained.
        if equivariance_weight > 0:
            self.equiv_projector = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        self.apply(_weights_init)

        self.features_by_layer = []
        self.input_features = []

    def _build_llm_vq_path(self, d_model, max_llm_codebook_size):
        """Project global token -> K language tokens via LLM-embedding VQ,
        then concat+MLP to produce the contrastive prediction."""
        from transformers import AutoModel

        llm_id = "meta-llama/Llama-2-7b-hf"
        llm_model = AutoModel.from_pretrained(llm_id, device_map='cpu')
        llm_embedding_bank = llm_model.get_input_embeddings().weight.data.clone()
        llama_dim = llm_embedding_bank.size(-1)
        del llm_model
        torch.cuda.empty_cache()

        self.global_to_k_tokens = nn.Linear(
            d_model, self.num_language_tokens * d_model,
        )
        self.llm_vq = LLMEmbeddingVQ(
            input_dim=d_model,
            llm_embedding_dim=llama_dim,
            max_codebook_size=max_llm_codebook_size,
        )

        token_freq_entries = json.load(open("data/llama2_token_frequencies.json", "r"))
        llm_token_frequencies = torch.zeros(
            llm_embedding_bank.size(0), dtype=torch.long,
        )
        for i, item in enumerate(token_freq_entries):
            token_id = int(item["token_id"])
            assert i == token_id, (
                "Token IDs in the frequency file should match their index."
            )
            llm_token_frequencies[token_id] = int(item["count"])

        self.llm_vq.set_llm_embedding_bank(
            llm_embedding_bank=llm_embedding_bank,
            token_frequencies=llm_token_frequencies,
        )

        self.llm_contrastive_proj = nn.Sequential(
            nn.Linear(self.num_language_tokens * llama_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.image_feature_dim),
        )

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
    
    @torch.no_grad()
    def _vjepa2_patch_tokens(self, pixel_values):
        """Run frozen V-JEPA 2 on a single still frame per sample.

        ``pixel_values``: (B, 3, H, W). V-JEPA 2 expects a video tensor
        ``pixel_values_videos`` of shape (B, T, C, H, W) with T divisible
        by tubelet_size (default 2). We replicate the frame T=2 times so
        the temporal patch count is 1.
        Returns the last hidden state — (B, N_patches, image_feature_dim).
        """
        pv = pixel_values.unsqueeze(1).expand(-1, 2, -1, -1, -1).contiguous()
        outputs = self.pretrained_image_encoder(pixel_values_videos=pv)
        return outputs.last_hidden_state  # (B, N, d_img)

    @torch.inference_mode()
    def _dinov2_cls_token(self, image_encoder_inputs):
        """Run frozen DINOv2-style ViT and return the CLS token at layer 12.

        Mirrors the pre-VJEPA behavior: ``hidden_states[12][:, 0]``.
        """
        outputs = self.pretrained_image_encoder(**image_encoder_inputs, output_hidden_states=True)
        return outputs.hidden_states[12][:, 0]  # (B, d_img)

    def get_image_hidden_states(self, **image_encoder_inputs):
        """Return (B, n_branch=1, image_feature_dim) for either encoder.

        Dispatches on ``self.encoder_kind``: V-JEPA 2 (no CLS) -> attention
        pool over patch tokens with a learnable query; DINOv2-style ViTs ->
        CLS token at layer 12 (the original pre-VJEPA path).
        """
        if self.encoder_kind == 'vjepa2':
            pixel_values = image_encoder_inputs.get('pixel_values')
            assert pixel_values is not None, (
                "V-JEPA 2 image encoder expects `pixel_values` in image_encoder_inputs"
            )
            patch_tokens = self._vjepa2_patch_tokens(pixel_values)  # (B, N, d_img)
            B = patch_tokens.size(0)
            q = self.image_pool_query.expand(B, -1, -1)
            pooled, _ = self.image_pool_attn(q, patch_tokens, patch_tokens, need_weights=False)
            pooled = self.image_pool_norm(pooled.squeeze(1))  # (B, d_img)
            return pooled.unsqueeze(1)  # (B, 1, d_img)
        # DINOv2-style fallback (also covers any future ViT with a CLS token).
        cls = self._dinov2_cls_token(image_encoder_inputs)  # (B, d_img)
        return cls.unsqueeze(1)


    # ------------------------------------------------------------------
    # Hemispheric equivariance helpers
    # ------------------------------------------------------------------

    def _encode_to_global_rep(self, x, ch_coords,
                              valid_channel_mask=None,
                              valid_length_mask=None,
                              mask=None):
        """Return only the global-token representation (backward compat).

        Routes through ``forward(encoder_only=True)`` so this helper and
        the main encoder share a single code path. ``encoder_only`` also
        prevents the recursive call into the equivariance branch.
        """
        _batch = {'timeseries': x, 'ch_coords': ch_coords}
        if valid_channel_mask is not None:
            _batch['valid_channel_mask'] = valid_channel_mask
        if valid_length_mask is not None:
            _batch['valid_length_mask'] = valid_length_mask
        _, info = self.forward(_batch, mask=mask, encoder_only=True)
        return info['global_rep']

    @staticmethod
    def _info_max_loss(z_inv, z_eq):
        """VICReg-style regulariser preventing collapse of either subspace.

        Returns a scalar combining:
        * **variance**: per-dimension std across the batch ≥ 1
        * **covariance**: off-diagonal covariance within each subspace → 0
        * **cross-covariance**: covariance between the two subspaces → 0
        """
        def _var(z):
            std = torch.sqrt(z.var(dim=0) + 1e-4)
            return torch.clamp(1.0 - std, min=0).mean()

        def _cov(z):
            z_c = z - z.mean(dim=0)
            N = max(z.size(0) - 1, 1)
            cov = (z_c.T @ z_c) / N
            off = cov - torch.diag(cov.diag())
            return (off ** 2).sum() / z.size(-1)

        var_loss = _var(z_inv) + _var(z_eq)
        cov_loss = _cov(z_inv) + _cov(z_eq)

        # cross-subspace decorrelation
        inv_c = z_inv - z_inv.mean(dim=0)
        eq_c = z_eq - z_eq.mean(dim=0)
        N = max(z_inv.size(0) - 1, 1)
        cross = (inv_c.T @ eq_c) / N
        cross_loss = (cross ** 2).sum() / z_inv.size(-1)

        return var_loss + cov_loss + cross_loss

    # ------------------------------------------------------------------

    def _encode_source(self, patch_emb, batch):
        """Temporal-only encode of concatenated source features.

        ``patch_emb`` is (B, K, N, d_model) in source space. Returns
        ``(global_rep, patch_tokens, branch_embs, branch_global)`` where
        ``global_rep``/``branch_global`` are (B, d_model) and
        ``patch_tokens``/``branch_embs`` are (B, K, N, d_model). The
        branch tensors are taken from layer ``num_layers - 3`` to feed
        the contrastive readout, mirroring the criss-cross path.
        """
        B, K, N, d = patch_emb.shape
        vlm = batch.get('valid_length_mask', None)
        branch_idx = self.source_encoder.num_layers - 3
        enc_out, branch_x = self.source_encoder(
            patch_emb, valid_length_mask=vlm, branch_layer_idx=branch_idx)

        def _split(seq):
            # seq: (B, N+1, width) -> global (B, d_model), tokens (B, K, N, d_model)
            g_raw, tok_seq = self.source_encoder.split_global(seq)
            g = self.source_global_proj(g_raw)
            toks = tok_seq.reshape(B, N, K, d).permute(0, 2, 1, 3).contiguous()
            return g, toks

        global_rep, patch_tokens = _split(enc_out)
        branch_global, branch_embs = _split(branch_x)
        return global_rep, patch_tokens, branch_embs, branch_global

    def _image_contrastive(self, pred_flatten, batch, has_image, i_branch=0):
        """Symmetric InfoNCE between predicted features and DINOv2 image
        embeddings. Returns a dict of loss/accuracy entries."""
        image_encoder_inputs = {k: v[has_image]
                                for k, v in batch['image_encoder_inputs'].items()}
        image_hidden_states = self.get_image_hidden_states(**image_encoder_inputs)
        image_hidden_states_branch = image_hidden_states[:, i_branch]
        image_flatten = image_hidden_states_branch.reshape(
            -1, image_hidden_states_branch.size(-1))

        temperature = 0.07
        pred_norm = F.normalize(pred_flatten, dim=-1)
        image_norm = F.normalize(image_flatten, dim=-1)

        logits = torch.matmul(pred_norm, image_norm.t()) / temperature
        targets = torch.arange(logits.size(0), device=logits.device)

        loss_p2i = F.cross_entropy(logits, targets)
        loss_i2p = F.cross_entropy(logits.t(), targets)
        branch_loss = 0.5 * (loss_p2i + loss_i2p)

        out = {f"contrastive_loss_{i_branch}": (self.alignment_weight, branch_loss)}
        with torch.no_grad():
            pred_to_img = logits.argmax(dim=1)
            img_to_pred = logits.argmax(dim=0)
            acc = 0.5 * ((pred_to_img == targets).float()
                         + (img_to_pred == targets).float())
        out[f"contrastive_acc_{i_branch}"] = acc.mean()

        contrastive_loss_per_source = defaultdict(list)
        sources = batch['source']
        if has_image is not None:
            mask_list = has_image.detach().cpu().tolist()
            sources = [s for s, m in zip(sources, mask_list) if m]
        for i, src in enumerate(sources):
            contrastive_loss_per_source[src].append(acc[i])
        for src, acc_list in contrastive_loss_per_source.items():
            out[f"contrastive_acc_{src}"] = torch.stack(acc_list).mean()
        return out

    def forward(self, batch, mask=None, encoder_only=False, band_idx=None):
        """Encode ``batch`` and (unless ``encoder_only``) compute losses.

        When ``encoder_only=True`` the patch-embed → transformer body
        still runs, but the contrastive, equivariance, and reconstruction
        branches are skipped and the return is
        ``(None, {'global_rep': ..., 'patch_tokens': ...})``. Used by the
        DINO teacher/student crops, the equivariance helper, and the
        world-model future-window encode — all of which need only the
        latent tokens, and several of which sit inside a ``no_grad``
        block where loss computation would be wasted.

        ``band_idx`` selects a band-specialised patch embedding from
        ``self.band_patch_embeddings`` (requires ``contrastive_band=True``).
        When None the default ``self.patch_embedding`` is used.
        """
        x = batch['timeseries'] # (B, n_ch, seq_len, in_dim)
        ch_coords = batch['ch_coords']

        # Save originals before the batch dict gets mutated by global-token
        # concatenation — needed for the equivariance second pass.
        _orig_vcm = batch.get('valid_channel_mask')
        _orig_vlm = batch.get('valid_length_mask')
        _orig_mask = mask

        # --- Apply mask in sensor space, then project to source space ---
        valid_ch = batch.get('valid_channel_mask', None)
        if self.project_to_source:
            if mask is not None:
                # Replace masked patches with a learned sentinel (fix #2) and
                # also exclude them from the projector's cross-attention key
                # set (fix #1, via sensor_mask).
                x_masked = x.clone()
                x_masked[mask == 1] = self.sensor_mask_embed.to(x_masked.dtype)
                x = self.source_projector(
                    x_masked, ch_coords,
                    valid_channel_mask=valid_ch,
                    sensor_mask=mask,
                )
            else:
                x = self.source_projector(x, ch_coords, valid_channel_mask=valid_ch)
            # Don't pass mask to patch_embedding — already applied
            mask = None

        # Remember base N before patch embedding so we can map the
        # interleaved multi-band tokens back to per-patch reconstructions.
        base_N = x.size(2)
        if band_idx is not None:
            pe = self.band_patch_embeddings[band_idx]
        else:
            pe = self.patch_embedding
        patch_emb = pe(x, mask)

        # Fix #3: tell the temporal transformer which time positions were
        # masked in sensor space. Aggregate the per-(channel, patch) mask
        # over channels into a per-patch score and add a learned mask-
        # presence embedding to every source token at that time.
        if self.project_to_source and _orig_mask is not None:
            time_score = _orig_mask.float().mean(dim=1)  # (B, N) in [0, 1]
            if patch_emb.size(2) == time_score.size(1):
                patch_emb = patch_emb + (
                    time_score.unsqueeze(1).unsqueeze(-1)
                    * self.source_mask_embed.view(1, 1, 1, -1)
                )

        if self.patch_embed_type == 'mamba' and 'valid_length_mask' in batch:
            batch['valid_length_mask'] = pe.adapt_valid_length_mask(
                batch['valid_length_mask'])

        contrastive_loss = dict()
        branch_embs = None

        if self.project_to_source:
            # Sources carry a fixed, montage-independent identity, so
            # cross-channel attention is unnecessary: concatenate all K
            # source features at each timestep and apply temporal-only
            # attention (see TemporalConcatEncoder).
            global_rep, patch_tokens, branch_embs, branch_global = \
                self._encode_source(patch_emb, batch)

            has_image = batch.get('has_image', None)
            if (not encoder_only
                    and 'image_encoder_inputs' in batch
                    and has_image is not None and has_image.any()
                    and self.alignment_weight > 0):
                semantic_emb = self.semantic_readout(branch_global[has_image])
                pred_flatten = self.contrastive_proj(semantic_emb)
                contrastive_loss.update(
                    self._image_contrastive(pred_flatten, batch, has_image))
        else:
            coord_pe, finite_coord_mask = self._spherical_positional_encoding(ch_coords)
            coord_emb = self.coord_enhancement(coord_pe)
            patch_emb = patch_emb + coord_emb.unsqueeze(2)

            # --- Build MoE gate inputs and band-prior targets from the raw
            # per-patch FFT magnitude that PatchEmbedding stashed during its
            # forward. Both tensors must be aligned with `patch_emb` after
            # the global-channel/token concatenation below.
            gate_features = None
            band_targets = None
            if self.use_moe:
                mag = getattr(pe, 'last_spectral_magnitude', None)
                assert mag is not None, (
                    "MoE expects PatchEmbedding to expose `last_spectral_magnitude`. "
                    "Did you change the patch embed type away from 'cnn'?"
                )
                gate_features = self.moe_gate_proj(mag)  # (B, C, N, gate_input_dim)
                energy = mag.pow(2)
                band_targets = torch.einsum(
                    'bcnf,ef->bcne', energy, self.moe_band_assign,
                )  # (B, C, N, E)

            if self.add_global:
                patch_emb = torch.cat([self.global_channel.expand(patch_emb.size(0), -1, patch_emb.size(2), -1), patch_emb], dim=1)
                patch_emb = torch.cat([self.global_token.expand(patch_emb.size(0), patch_emb.size(1), -1, -1), patch_emb], dim=2)
                if 'valid_channel_mask' in batch:
                    batch['valid_channel_mask'] = torch.cat([torch.ones_like(batch['valid_channel_mask'][:, :1], dtype=torch.bool), batch['valid_channel_mask']], dim=1)
                if 'valid_length_mask' in batch:
                    batch['valid_length_mask'] = torch.cat([torch.ones_like(batch['valid_length_mask'][:, :1], dtype=torch.bool), batch['valid_length_mask']], dim=1)

                if self.use_moe:
                    B_, C_, N_, _ = patch_emb.shape  # (B, C+1, N+1, d)
                    g_gate = self.moe_global_gate_input.view(1, 1, 1, -1).expand(B_, C_, N_, -1)
                    g_band = self.moe_global_band_target.view(1, 1, 1, -1).expand(B_, C_, N_, -1)
                    # Insert original gate/targets at [1:, 1:]; global channel row
                    # and global time column take the learned default.
                    gate_features_padded = g_gate.clone()
                    gate_features_padded[:, 1:, 1:, :] = gate_features
                    band_targets_padded = g_band.clone()
                    band_targets_padded[:, 1:, 1:, :] = band_targets
                    gate_features = gate_features_padded
                    band_targets = band_targets_padded

            moe_balance_running = patch_emb.new_zeros(())
            moe_band_prior_running = patch_emb.new_zeros(())
            moe_z_loss_running = patch_emb.new_zeros(())
            for layer_idx in range(self.encoder.num_layers):
                patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
                # patch_emb = self.BrainEmbedEEGLayer(patch_emb, self.area_config) + patch_emb
                patch_emb = self.encoder.layers[layer_idx](
                    patch_emb, self.area_config,
                    inter_window_attn_mask=~batch['valid_length_mask'] if 'valid_length_mask' in batch else None,
                    inter_region_attn_mask=~batch['valid_channel_mask'] if 'valid_channel_mask' in batch else None,
                    gate_features=gate_features,
                    band_targets=band_targets,
                )
                if self.use_moe:
                    aux = self.encoder.layers[layer_idx].last_aux_loss
                    moe_balance_running = moe_balance_running + aux['balance']
                    moe_z_loss_running = moe_z_loss_running + aux['z_loss']
                    if 'band_prior' in aux:
                        moe_band_prior_running = moe_band_prior_running + aux['band_prior']

                if layer_idx == self.encoder.num_layers - 3:
                    i_branch = 0
                    has_image = batch.get('has_image', None)
                    if has_image is not None:
                        branch_embs = patch_emb[has_image] # (B_with_image, K, N, d)
                    else:
                        branch_embs = patch_emb
                    if (not encoder_only
                            and 'image_encoder_inputs' in batch
                            and has_image.any()
                            and self.alignment_weight > 0):
                        if self.use_llm_vq:
                            assert self.add_global, "LLM VQ path requires add_global=True."
                            global_branch = branch_embs[:, 0, 0, :]  # (B_img, d_model)
                            k_tokens = self.global_to_k_tokens(global_branch).view(
                                global_branch.size(0), self.num_language_tokens, self.d_model,
                            )
                            _, vq_info = self.llm_vq(k_tokens.unsqueeze(2))  # (B, K, 1, d_model)
                            quantized = vq_info["quantized"].view(
                                global_branch.size(0), self.num_language_tokens, -1,
                            )  # (B_img, K, llama_dim)
                            pred_flatten = self.llm_contrastive_proj(
                                quantized.reshape(quantized.size(0), -1)
                            )
                            contrastive_loss["llm_vq_aux_loss"] = (
                                self.llm_vq_aux_weight, vq_info["aux_loss"],
                            )
                        else:
                            if self.add_global:
                                semantic_emb = self.semantic_readout(branch_embs[:, 0, 0, :])
                            else:
                                semantic_emb = self.semantic_readout(branch_embs.view(branch_embs.size(0), branch_embs.size(1), -1))
                            pred_flatten = self.contrastive_proj(semantic_emb) # (B, n_events, d_model) <- does not work

                        contrastive_loss.update(
                            self._image_contrastive(pred_flatten, batch, has_image, i_branch))

            # Extract global token and patch-level representations
            global_rep = patch_emb[:, 0, 0, :] if self.add_global else patch_emb.mean(dim=(1, 2))
            patch_tokens = patch_emb[:, 1:, 1:, :] if self.add_global else patch_emb

        if encoder_only:
            # Skip reconstruction output and equivariance — callers
            # (DINO teacher/student, equivariance helper, world-model
            # future encode) only need the latent tokens.
            return None, {
                "global_rep": global_rep,
                "patch_tokens": patch_tokens,
            }

        if self.project_to_source:
            # Reconstruct back to sensor space for masked reconstruction loss
            out = self.source_projector.inverse(patch_tokens, ch_coords,
                                                valid_channel_mask=valid_ch)
            out = self.proj_out(out)
        else:
            out = self.proj_out(patch_emb)
            if self.add_global:
                out = out[:, 1:, 1:, :]

        # For mamba patch embed, the token axis is multi-band interleaved.
        # Keep only the base-band (f1) positions so downstream reconstruction
        # indexing (mask has shape (B, C, N)) still lines up.
        if self.patch_embed_type == 'mamba':
            f1_pos = self.patch_embedding.base_band_positions(base_N)
            f1_idx = torch.tensor(f1_pos, device=out.device, dtype=torch.long)
            out = out.index_select(2, f1_idx)

        rep = branch_embs # (B, K, N, d)
        #     rep = torch.cat([self.global_token.expand(out.size(0), -1, -1, -1), out], dim=2)

        # ------------------------------------------------------------------
        # Hemispheric equivariance loss  (only during training)
        # ------------------------------------------------------------------
        equiv_losses = {}
        if self.training and self.equivariance_weight > 0:
            ch_names = batch.get('ch_names')
            if ch_names is not None:
                x_raw = batch['timeseries']   # original (scaled) signal
                B, C, T, d = x_raw.shape

                flip_perm = build_flip_perm_batch(
                    ch_names, _orig_vcm).to(x_raw.device)      # (B, C)
                perm_exp = (flip_perm[:, :C]
                            .unsqueeze(-1).unsqueeze(-1)
                            .expand(-1, -1, T, d))
                x_flip = torch.gather(x_raw, 1, perm_exp)

                global_rep_flip = self._encode_to_global_rep(
                    x_flip, ch_coords, _orig_vcm, _orig_vlm, _orig_mask)

                # Project into the equivariance-specific space so the
                # inv/eq split lives in the projector (discarded at
                # fine-tuning), not in the backbone representation.
                z      = self.equiv_projector(global_rep)
                z_flip = self.equiv_projector(global_rep_flip)

                half = z.size(-1) // 2
                z_inv,   z_eq   = z[:, :half],      z[:, half:]
                z_inv_f, z_eq_f = z_flip[:, :half],  z_flip[:, half:]

                # Invariant half should be identical; equivariant half should negate
                equiv_loss = (F.mse_loss(z_inv_f, z_inv)
                              + F.mse_loss(z_eq_f, -z_eq))
                equiv_losses["equiv_loss"] = (
                    self.equivariance_weight, equiv_loss)

                # VICReg-style collapse prevention
                if self.info_max_weight > 0:
                    info_loss = self._info_max_loss(z_inv, z_eq)
                    equiv_losses["info_max_loss"] = (
                        self.info_max_weight, info_loss)

        moe_losses = {}
        if self.use_moe and not self.project_to_source:
            n_layers = self.encoder.num_layers
            moe_losses["moe_balance_loss"] = (
                self.moe_balance_weight, moe_balance_running / n_layers,
            )
            if self.moe_z_loss_weight > 0:
                moe_losses["moe_z_loss"] = (
                    self.moe_z_loss_weight, moe_z_loss_running / n_layers,
                )
            if self.moe_band_prior_weight > 0:
                moe_losses["moe_band_prior_loss"] = (
                    self.moe_band_prior_weight, moe_band_prior_running / n_layers,
                )

        return out, {
            "rep": rep,
            "global_rep": global_rep,
            "patch_tokens": patch_tokens,
            # "zero_lag_sync_loss": (0.1, zero_lag_sync_loss),
            **contrastive_loss,
            **equiv_losses,
            **moe_losses,
        }
    
    def training_step(self, *args, **kwargs):
        return self.forward(*args, **kwargs)



class MambaPatchEmbedding(nn.Module):
    """Multi-frequency SSM/Mamba tokenizer.

    A stack of Mamba layers encodes the raw per-channel EEG samples to a
    sequence of the same length T. Tokens are extracted at several stride
    intervals ("bands") and interleaved in chronological order:

        X_{f1,1}, X_{f1,2}, X_{f1,3}, X_{f2,1}, X_{f1,4}, X_{f1,5},
        X_{f1,6}, X_{f2,2}, X_{f3,1}, ...

    band_periods holds the sample-level stride for each band (finest
    first). All periods must be integer multiples of band_periods[0]
    so higher-frequency tokens align with coarser-band boundaries.
    Default [in_dim, 3*in_dim, 6*in_dim] gives ratio 1 : 3 : 6 which
    reproduces the user-specified interleaving pattern.
    """

    def __init__(self, in_dim, out_dim, d_model, seq_len,
                 band_periods=None, n_mamba_layers=2,
                 d_state=16, d_conv=4, expand=2, causal=False):
        super().__init__()
        from mamba_ssm import Mamba

        self.d_model = d_model
        self.in_dim = in_dim
        self.causal = causal
        self.bidirectional = not causal

        if band_periods is None:
            band_periods = [in_dim, 3 * in_dim, 6 * in_dim]
        base = band_periods[0]
        for p in band_periods:
            assert p % base == 0, (
                "band_periods must all be integer multiples of "
                "band_periods[0] so tokens interleave cleanly")
        self.band_periods = list(band_periods)
        self.band_ratios = [p // base for p in band_periods]
        self.num_bands = len(band_periods)

        self.sample_proj = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

        self.fwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_mamba_layers)
        ])
        self.fwd_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_mamba_layers)
        ])

        if self.bidirectional:
            self.bwd_mamba = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_mamba_layers)
            ])
            self.bwd_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(n_mamba_layers)
            ])
            self.fuse = nn.Linear(2 * d_model, d_model)

        self.band_embeddings = nn.Parameter(torch.zeros(self.num_bands, d_model))
        nn.init.normal_(self.band_embeddings, std=0.02)

        self.band_proj = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(self.num_bands)
        ])

        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        if causal:
            self.positional_encoding = nn.Conv2d(
                d_model, d_model, kernel_size=(19, 7),
                stride=1, padding=(9, 0), groups=d_model,
            )
            self.pos_time_pad = 6
        else:
            self.positional_encoding = nn.Conv2d(
                d_model, d_model, kernel_size=(19, 7),
                stride=1, padding=(9, 3), groups=d_model,
            )

    def _interleave_events(self, N_base):
        events = []
        for step in range(1, N_base + 1):
            base_idx = step - 1
            for k, r_k in enumerate(self.band_ratios):
                if step % r_k == 0:
                    within_k = step // r_k - 1
                    events.append((base_idx, k, within_k))
        return events

    def expected_seq_len(self, N_base):
        return sum(N_base // r for r in self.band_ratios)

    def adapt_valid_length_mask(self, valid_length_mask):
        if valid_length_mask is None:
            return None
        N_base = valid_length_mask.size(-1)
        events = self._interleave_events(N_base)
        idx = torch.tensor([ev[0] for ev in events],
                           device=valid_length_mask.device, dtype=torch.long)
        return valid_length_mask.index_select(-1, idx)

    def base_band_positions(self, N_base):
        events = self._interleave_events(N_base)
        return [pos for pos, (_, k, _) in enumerate(events) if k == 0]

    def _run_mamba(self, feats):
        h = feats
        for mamba, norm in zip(self.fwd_mamba, self.fwd_norms):
            h = h + mamba(norm(h))
        if not self.bidirectional:
            return h
        h_b = torch.flip(feats, dims=[1])
        for mamba, norm in zip(self.bwd_mamba, self.bwd_norms):
            h_b = h_b + mamba(norm(h_b))
        h_b = torch.flip(h_b, dims=[1])
        return self.fuse(torch.cat([h, h_b], dim=-1))

    def forward(self, x, mask=None):
        B, C, N, P = x.shape
        assert P == self.in_dim, f"Expected patch_size={self.in_dim}, got {P}"
        T = N * P

        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        ts = mask_x.reshape(B * C, 1, T)

        feats = self.sample_proj(ts)
        feats = feats.transpose(1, 2).contiguous()
        feats = self._run_mamba(feats)

        band_tokens = []
        for k, p_k in enumerate(self.band_periods):
            n_k = T // p_k
            positions = torch.arange(n_k, device=feats.device) * p_k + (p_k - 1)
            tokens_k = feats.index_select(1, positions)
            tokens_k = self.band_proj[k](tokens_k) + self.band_embeddings[k]
            band_tokens.append(tokens_k)

        events = self._interleave_events(N)
        offsets = [0]
        for toks in band_tokens[:-1]:
            offsets.append(offsets[-1] + toks.size(1))
        flat = torch.cat(band_tokens, dim=1)
        flat_indices = torch.tensor(
            [offsets[k] + i for (_, k, i) in events],
            device=feats.device, dtype=torch.long)
        interleaved = flat.index_select(1, flat_indices)
        N_total = interleaved.size(1)

        patch_emb = interleaved.view(B, C, N_total, self.d_model)

        pe_input = patch_emb.permute(0, 3, 1, 2)
        if self.causal:
            pe_input = F.pad(pe_input, (self.pos_time_pad, 0))
        pos_emb = self.positional_encoding(pe_input).permute(0, 2, 3, 1)
        patch_emb = patch_emb + pos_emb

        return patch_emb


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len, causal=False,
                 spectral_mode='static', stft_n_fft=64, stft_hop=1):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        if spectral_mode not in ('static', 'instantaneous'):
            raise ValueError(f"Unknown spectral_mode: {spectral_mode}")
        self.spectral_mode = spectral_mode
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

        if self.spectral_mode == 'static':
            self.spectral_proj = nn.Sequential(
                nn.Linear(d_model // 2 + 1, d_model),
                nn.Dropout(0.1),
            )
        else:
            # Instantaneous spectrum: STFT is run on the full per-channel signal
            # (patch_num * patch_size samples) so internal patch boundaries are
            # handled seamlessly, then the time axis is split back into
            # (patch_num, frames_per_patch). A Conv2d head turns each per-patch
            # spectrogram into a d_model vector.
            if in_dim % stft_hop != 0:
                raise ValueError(
                    f"stft_hop ({stft_hop}) must divide patch_size ({in_dim}) so the "
                    f"STFT time axis can be reshaped cleanly into per-patch frames."
                )
            self.stft_n_fft = stft_n_fft
            self.stft_hop = stft_hop
            self.register_buffer(
                'stft_window', torch.hann_window(stft_n_fft), persistent=False,
            )
            F_bins = stft_n_fft // 2 + 1
            T_frames = in_dim // stft_hop  # frames per patch (= patch_size when hop=1)
            c1, c2 = 16, 32

            def _conv_out(L, k, s, p):
                return (L + 2 * p - k) // s + 1

            f1 = _conv_out(F_bins, 7, 4, 3)
            t1 = _conv_out(T_frames, 7, 8, 3)
            f2 = _conv_out(f1, 3, 2, 1)
            t2 = _conv_out(t1, 3, 4, 1)
            flatten_dim = c2 * f2 * t2

            self.spectral_conv = nn.Sequential(
                nn.Conv2d(1, c1, kernel_size=(7, 7), stride=(4, 8), padding=(3, 3)),
                nn.GroupNorm(min(8, c1), c1),
                nn.GELU(),
                nn.Conv2d(c1, c2, kernel_size=(3, 3), stride=(2, 4), padding=(1, 1)),
                nn.GroupNorm(min(8, c2), c2),
                nn.GELU(),
            )
            self.spectral_proj = nn.Sequential(
                nn.Linear(flatten_dim, d_model),
                nn.Dropout(0.1),
            )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        # Keep the masked 4D signal around for the spectral path; the per-branch
        # reshape below may rebind ``mask_x`` for the proj_in path.
        mask_x_4d = mask_x  # (bz, ch, patch_num, patch_size)

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

        # Always compute the per-patch FFT magnitude; we expose it as a
        # side output so an upstream MoE can route on raw spectral content.
        sig_static = mask_x_4d.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral_static = torch.fft.rfft(sig_static, dim=-1, norm='forward')
        spectral_static = torch.abs(spectral_static).contiguous().view(
            bz, ch_num, patch_num, sig_static.shape[1] // 2 + 1,
        )
        self.last_spectral_magnitude = spectral_static

        if self.spectral_mode == 'static':
            spectral_emb = self.spectral_proj(spectral_static)
        else:
            # Run STFT on the full per-channel timeseries so internal patch
            # boundaries don't introduce spurious zero-padding, then split the
            # time axis back into (patch_num, frames_per_patch).
            L = patch_num * patch_size
            full_sig = mask_x_4d.contiguous().view(bz * ch_num, L)
            spec = torch.stft(
                full_sig,
                n_fft=self.stft_n_fft,
                hop_length=self.stft_hop,
                win_length=self.stft_n_fft,
                window=self.stft_window,
                center=True,
                return_complex=True,
                normalized=True,
            )
            # center=True with hop=h gives 1 + L // h frames; drop the trailing
            # frame so the time axis is exactly L // h = patch_num * (patch_size // h).
            frames_per_patch = patch_size // self.stft_hop
            total_frames = patch_num * frames_per_patch
            spec = spec.abs()[..., :total_frames]                       # (B*C, F, total_frames)
            F_bins = spec.shape[-2]
            spec = spec.view(bz, ch_num, F_bins, patch_num, frames_per_patch)
            spec = spec.permute(0, 1, 3, 2, 4).contiguous()             # (bz, ch, patch_num, F, frames_per_patch)
            spec = spec.view(bz * ch_num * patch_num, 1, F_bins, frames_per_patch)
            feats = self.spectral_conv(spec).flatten(start_dim=1)
            spectral_emb = self.spectral_proj(feats).view(bz, ch_num, patch_num, self.d_model)
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
