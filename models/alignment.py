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


class LateralizationSplit(nn.Module):
    """Learned additive split of the raw EEG into bilateral + lateral streams.

    Given raw per-channel patches ``x`` (B, C, N, d) a small per-channel
    network predicts a gate ``g in (0, 1)`` of the *same* shape and splits the
    signal additively::

        x_lat = g * x            # lateralization features
        x_bi  = (1 - g) * x      # bilateral features
        x_bi + x_lat == x        # exact, by construction

    Only ``x_lat`` is acted on by the hemispheric channel-swap downstream, so
    ``g`` learns *which* signal components are lateralized (and therefore swap
    hemispheres when the visual scene is mirrored). The gate is conditioned on
    the per-patch signal and on a spherical channel positional encoding, so
    different scalp locations can carry different lateral fractions. It is
    initialised near zero (``x_lat ~= 0``) so training starts from
    ``x_flip ~= x`` and grows the lateral stream only as the flip-alignment
    objective demands — this is the standard "start as identity" stabiliser.
    """

    def __init__(self, in_dim, coord_pe_dim, hidden=64, gate_init_bias=-2.0):
        super().__init__()
        self.in_dim = in_dim
        self.gate_init_bias = float(gate_init_bias)
        self.sig_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
        )
        self.coord_proj = nn.Linear(coord_pe_dim, hidden)
        self.gate_head = nn.Linear(hidden, in_dim)
        self.reset_gate_init()

    def reset_gate_init(self):
        """(Re)apply the near-zero gate init. Call after the owner's global
        ``apply(_weights_init)`` pass, which would otherwise overwrite it."""
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, self.gate_init_bias)

    def forward(self, x, coord_pe, valid_channel_mask=None):
        """x: (B, C, N, d); coord_pe: (B, C, coord_pe_dim).

        Returns ``(x_lat, gate)`` both (B, C, N, d). ``x_bi`` is just
        ``x - x_lat`` for the caller; we return the gate so the caller can
        regularise the lateral fraction.
        """
        h = self.sig_proj(x)                                   # (B, C, N, hidden)
        h = h + self.coord_proj(coord_pe).unsqueeze(2)         # +channel context
        gate = torch.sigmoid(self.gate_head(h))                # (B, C, N, d)
        if valid_channel_mask is not None:
            m = valid_channel_mask.to(x.dtype).view(
                *valid_channel_mask.shape, 1, 1)
            gate = gate * m
        return gate * x, gate


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


class VolumeConductionEncoding(nn.Module):
    """Volume-conduction-aware channel positional encoding (arXiv 2601.06134).

    Given per-sample 3D electrode positions, builds a learnable, row-
    normalised exponential distance-decay kernel and uses it to smooth
    each electrode's position into a convex combination of its physical
    neighbours. The smoothed Cartesian position is projected to the
    transformer width and added to the patch embedding as a channel-
    level positional encoding.

    Inputs to ``forward``
    ---------------------
    ch_coords : (B, C, 3)
        Either Cartesian (x, y, z) in metres or spherical (r, theta, phi)
        as produced by :func:`datasets.cached_dataset._to_spherical`.
        Controlled by ``coord_kind``. NaN rows are tolerated and treated
        as invalid channels (their contribution to the kernel is zeroed
        and their output is set to zero).
    valid_channel_mask : (B, C) bool, optional
        ``False`` for padded / dropped channels.

    Returns
    -------
    (B, C, d_model) channel-level embedding.
    """

    def __init__(self, d_model: int, coord_kind: str = 'spherical',
                 tau_init: float = 0.08, eps: float = 1e-6):
        super().__init__()
        if coord_kind not in ('spherical', 'cartesian'):
            raise ValueError(
                f"coord_kind must be 'spherical' or 'cartesian', got {coord_kind}"
            )
        self.coord_kind = coord_kind
        self.eps = eps

        # tau = softplus(alpha) + eps. Initialise alpha so tau ~= tau_init.
        # softplus(a) = tau -> a = log(exp(tau) - 1).
        tau_init = max(tau_init, 10.0 * eps)
        alpha_init = math.log(math.expm1(tau_init))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.proj = nn.Linear(3, d_model)

    @staticmethod
    def _spherical_to_cartesian(coords: torch.Tensor) -> torch.Tensor:
        """(B, C, 3) spherical (r, theta=atan2(y,x), phi=acos(z/r)) -> Cartesian."""
        r = coords[..., 0]
        theta = coords[..., 1]
        phi = coords[..., 2]
        sin_phi = torch.sin(phi)
        x = r * sin_phi * torch.cos(theta)
        y = r * sin_phi * torch.sin(theta)
        z = r * torch.cos(phi)
        return torch.stack([x, y, z], dim=-1)

    def forward(self, ch_coords: torch.Tensor,
                valid_channel_mask: torch.Tensor = None) -> torch.Tensor:
        if self.coord_kind == 'spherical':
            cart = self._spherical_to_cartesian(ch_coords)
        else:
            cart = ch_coords

        # Treat NaN/inf rows as invalid; combine with any user-supplied mask.
        finite_mask = torch.isfinite(cart).all(dim=-1)            # (B, C)
        if valid_channel_mask is not None:
            valid = finite_mask & valid_channel_mask.bool()
        else:
            valid = finite_mask
        safe_cart = torch.where(
            finite_mask.unsqueeze(-1), cart, torch.zeros_like(cart),
        )

        # Pairwise Euclidean distances over the channel axis.
        diff = safe_cart.unsqueeze(2) - safe_cart.unsqueeze(1)    # (B, C, C, 3)
        dist = torch.linalg.vector_norm(diff, dim=-1)             # (B, C, C)

        tau = F.softplus(self.alpha) + self.eps
        kernel = torch.exp(-dist / tau)                           # (B, C, C)

        # Zero out columns belonging to invalid channels so they cannot
        # contribute mass to any row's smoothed position.
        col_mask = valid.unsqueeze(1).to(kernel.dtype)            # (B, 1, C)
        kernel = kernel * col_mask

        # Row-normalise. Rows with no valid neighbours fall back to a
        # safe identity (smoothed = own coord) so output stays finite;
        # those rows are zeroed below via ``valid`` anyway when they are
        # themselves invalid.
        row_sum = kernel.sum(dim=-1, keepdim=True)                # (B, C, 1)
        kernel_norm = kernel / row_sum.clamp_min(self.eps)

        smoothed = torch.bmm(kernel_norm, safe_cart)              # (B, C, 3)

        emb = self.proj(smoothed)                                 # (B, C, d_model)
        emb = emb * valid.unsqueeze(-1).to(emb.dtype)
        return emb


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


class LearnableFilterbank(nn.Module):
    """SincNet-style learnable bandpass filterbank with INDEPENDENT per-band bounds.

    Each band k learns its own lower/upper cutoff directly, so bands are free to
    overlap (or not) as the data demands — there is no shared ordered edge set
    and no global overlap hyperparameter. Validity is guaranteed by construction
    via a bounded reparameterization:

        low_k  = f_min + sigmoid(low_logits_k) * (nyq - f_min - min_bw)
        high_k = low_k + min_bw + sigmoid(width_logits_k) * (nyq - low_k - min_bw)

    so for every band, independently:  f_min <= low_k  and
    low_k + min_bw <= high_k <= nyquist.  Bands are *initialized* to tile the
    spectrum into equal slices (to break the K-fold symmetry — identical bands
    would otherwise receive identical gradients and never separate), but each
    band may move and overlap any other freely thereafter.

    For each band a SincNet bandpass FIR kernel is built (difference of two
    normalized-sinc low-pass filters, Hamming-windowed) and the **same** kernel
    is convolved over every channel. The model is not told which band is which;
    downstream layers learn the band semantics.
    """

    def __init__(self, num_bands, fs, kernel_size=101, min_bw_hz=1.0, f_min_hz=1.0):
        super().__init__()
        assert kernel_size % 2 == 1, (
            "kernel_size must be odd for a symmetric zero-phase FIR filter")
        self.num_bands = int(num_bands)
        self.fs = float(fs)
        self.kernel_size = int(kernel_size)
        self.min_bw = float(min_bw_hz)
        self.f_min = float(f_min_hz)
        self.nyquist = self.fs / 2.0
        assert self.f_min + self.min_bw < self.nyquist, (
            f"f_min ({f_min_hz}) + min_bw ({min_bw_hz}) must be < nyquist "
            f"({self.nyquist}); reduce them or raise fs.")

        # Per-band, independent low/width logits. Initialize to an equal-width
        # tiling of the spectrum by inverting the sigmoid reparameterization.
        span = self.nyquist - self.f_min - self.min_bw
        tile = (self.nyquist - self.f_min) / self.num_bands
        low_logits = torch.empty(self.num_bands)
        width_logits = torch.empty(self.num_bands)

        def _logit(p):
            p = min(max(p, 1e-3), 1.0 - 1e-3)
            return math.log(p / (1.0 - p))

        for k in range(self.num_bands):
            lo = self.f_min + k * tile
            hi = min(lo + tile, self.nyquist)
            wid = max(hi - lo, self.min_bw)
            low_logits[k] = _logit((lo - self.f_min) / span)
            width_logits[k] = _logit(
                (wid - self.min_bw) / max(self.nyquist - lo - self.min_bw, 1e-6))
        self.low_logits = nn.Parameter(low_logits)
        self.width_logits = nn.Parameter(width_logits)

        # Fixed (non-learned) tap indices and Hamming window.
        half = (self.kernel_size - 1) / 2.0
        n = torch.arange(self.kernel_size, dtype=torch.float32) - half
        idx = torch.arange(self.kernel_size, dtype=torch.float32)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * idx / (self.kernel_size - 1))
        self.register_buffer('tap_idx', n, persistent=False)      # (L,)
        self.register_buffer('window', window, persistent=False)  # (L,)

    def band_bounds(self):
        """Return (low (K,), high (K,)) independent per-band cutoffs in Hz."""
        span = self.nyquist - self.f_min - self.min_bw
        low = self.f_min + torch.sigmoid(self.low_logits) * span
        width = self.min_bw + torch.sigmoid(self.width_logits) * (self.nyquist - low - self.min_bw)
        high = low + width
        return low, high                                    # each (K,)

    def _build_kernels(self, dtype):
        low, high = self.band_bounds()                       # (K,), (K,)
        f1 = (low / self.fs).unsqueeze(1)                    # (K,1) cycles/sample
        f2 = (high / self.fs).unsqueeze(1)
        n = self.tap_idx.to(low.device).unsqueeze(0)         # (1,L)
        # Normalized-sinc low-pass (torch.sinc(x) = sin(pi x)/(pi x); value 1 at 0).
        lp2 = 2 * f2 * torch.sinc(2 * f2 * n)
        lp1 = 2 * f1 * torch.sinc(2 * f1 * n)
        bp = (lp2 - lp1) * self.window.to(low.device).unsqueeze(0)  # (K,L)
        return bp.unsqueeze(1).to(dtype)                     # (K,1,L)

    def forward(self, x):
        """x: (B, C, N, in_dim) raw patches -> (B, K, C, N, in_dim) band signals."""
        B, C, N, d = x.shape
        T = N * d
        sig = x.reshape(B * C, 1, T)
        kernels = self._build_kernels(x.dtype)               # (K,1,L)
        bands = F.conv1d(sig, kernels, padding=self.kernel_size // 2)  # (B*C, K, T)
        bands = bands.reshape(B, C, self.num_bands, T)
        bands = bands.permute(0, 2, 1, 3).contiguous()       # (B, K, C, T)
        return bands.reshape(B, self.num_bands, C, N, d)


class CrossBandAttention(nn.Module):
    """Mix information across the K band axis at each (channel, time) token.

    Models cross-frequency coupling without the full band x scale Cartesian
    product: a single multi-head attention over the length-K band sequence,
    shared across all (channel, time) positions. ``out_proj`` is zero-init so
    the layer is an identity at the start of training (the caller re-zeroes it
    after the model's global ``apply(_weights_init)`` pass).
    """

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, x, B, K):
        # x: (B*K, Cp, Np, d) -> attend over the K axis -> (B*K, Cp, Np, d)
        BK, Cp, Np, d = x.shape
        h = (x.view(B, K, Cp, Np, d)
               .permute(0, 2, 3, 1, 4)              # (B, Cp, Np, K, d)
               .reshape(B * Cp * Np, K, d))
        h_n = self.norm(h)
        h = h + self.attn(h_n, h_n, h_n, need_weights=False)[0]
        return (h.reshape(B, Cp, Np, K, d)
                 .permute(0, 3, 1, 2, 4)            # (B, K, Cp, Np, d)
                 .reshape(B * K, Cp, Np, d))


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
                 image_pool_heads=4,
                 use_volume_conduction=False, vc_tau_init=0.08,
                 use_spectral_bands=False, num_spectral_bands=4,
                 fs=200, filterbank_kernel_size=101,
                 filterbank_min_bw_hz=1.0, use_cross_band_attn=True,
                 cross_band_every=1, use_band_type_embedding=True,
                 num_visual_levels=3, band_decorr_weight=0.01,
                 lateralization_flip=False, flip_align_weight=1.0,
                 lat_sparsity_weight=0.01, flip_split_hidden=64,
                 flip_n_col_bands=2, flip_motion_ref=0.0, flip_motion_min=0.0,
                 frame_averaging=False, frame_avg_flip_prob=0.5,
                 frame_avg_align_weight=1.0, frame_avg_recon_weight=1.0):
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

        # --- Spectral-band config ---
        # When enabled, a learnable filterbank splits the raw EEG into K bands;
        # the bands are run (folded into the batch) through the SHARED encoder,
        # mixed by an optional cross-band attention, fused for reconstruction,
        # and aligned to multiple visual levels via a learned soft assignment.
        self.use_spectral_bands = use_spectral_bands
        self.num_spectral_bands = int(num_spectral_bands)
        self.use_cross_band_attn = use_cross_band_attn
        self.cross_band_every = max(int(cross_band_every), 1)
        self.use_band_type_embedding = use_band_type_embedding
        self.num_visual_levels = int(num_visual_levels) if use_spectral_bands else 1
        self.band_decorr_weight = band_decorr_weight

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

        # Volume-conduction channel positional encoding (arXiv 2601.06134).
        # Replaces the sinusoidal coord_enhancement output when enabled.
        self.use_volume_conduction = use_volume_conduction
        if use_volume_conduction:
            self.volume_conduction = VolumeConductionEncoding(
                d_model=d_model, coord_kind='spherical', tau_init=vc_tau_init,
            )

        self.features_by_layer = []
        self.input_features = []

        # --- Learnable spectral-band stream (optional) ---------------------
        if self.use_spectral_bands:
            if self.use_moe:
                raise ValueError(
                    "use_spectral_bands is mutually exclusive with use_moe in "
                    "v1: the MoE frequency band-prior conflicts with an "
                    "explicit learnable filterbank.")
            if patch_embed_type != 'cnn':
                raise ValueError(
                    "use_spectral_bands currently requires patch_embed_type='cnn'.")
            if not self.add_global:
                raise ValueError(
                    "use_spectral_bands requires add_global=True (per-band "
                    "global tokens drive the multi-level alignment).")
            K = self.num_spectral_bands
            self.filterbank = LearnableFilterbank(
                num_bands=K, fs=fs, kernel_size=filterbank_kernel_size,
                min_bw_hz=filterbank_min_bw_hz,
            )
            # Per-band additive "type" embedding (zero-init => first step ~ baseline).
            if self.use_band_type_embedding:
                self.band_type_embed = nn.Parameter(torch.zeros(K, d_model))
            # Cross-band attention at a subset of encoder layers.
            if self.use_cross_band_attn:
                self.cross_band_attn = nn.ModuleList([
                    CrossBandAttention(d_model, nhead)
                    if ((layer_idx + 1) % self.cross_band_every == 0) else None
                    for layer_idx in range(n_layer)
                ])
            # Learned soft band->visual-level assignment, normalized over levels.
            self.band_level_logits = nn.Parameter(
                torch.zeros(K, self.num_visual_levels))
            # Per-band projection into the (shared) image feature space.
            self.band_align_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model), nn.GELU(),
                    nn.Linear(d_model, self.image_feature_dim),
                ) for _ in range(K)
            ])

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

        # --- Bilateralization prior: learned x = x_bi + x_lat split + a
        # flip-equivariant image-alignment objective (see _flip_alignment).
        # This is a *separate* mechanism from the legacy equivariance_weight
        # path above; it does not touch the main encode of ``x`` (the split is
        # the identity there) and only adds a second, channel-swapped encode.
        self.lateralization_flip = bool(lateralization_flip)
        self.flip_align_weight = float(flip_align_weight)
        self.lat_sparsity_weight = float(lat_sparsity_weight)
        # Image-side flip target: the global CLS is ~horizontal-flip-invariant
        # (probe: cos(img, flip(img)) ~ 0.97 on EgoBrain) so it makes the
        # objective vacuous. Instead we align to a CENTERED column-band spatial
        # descriptor: split the patch grid into ``flip_n_col_bands`` vertical
        # bands, mean-pool each, and remove the cross-band mean. This is
        # strongly flip-sensitive (cos ~ 0.09 on motion frames) so the gate is
        # forced to engage. ``flip_n_col_bands=2`` is the left/right descriptor.
        self.flip_n_col_bands = int(flip_n_col_bands)
        # Optional per-sample motion weighting of the flip loss: EgoBrain has
        # static stretches where a horizontal flip changes little; weight each
        # row by clamp(motion / flip_motion_ref, flip_motion_min, 1) where
        # ``motion`` is the t->t+1 frame difference supplied by the wrapper.
        # 0 disables (uniform weights). Rows with no motion info -> weight 1.
        self.flip_motion_ref = float(flip_motion_ref)
        self.flip_motion_min = float(flip_motion_min)
        if self.lateralization_flip:
            assert self.flip_n_col_bands >= 1, "flip_n_col_bands must be >= 1"
            coord_pe_dim = 3 * 2 * self.spherical_num_freqs
            self.lateralization_split = LateralizationSplit(
                in_dim=in_dim, coord_pe_dim=coord_pe_dim,
                hidden=flip_split_hidden,
            )
            # Maps the global token into the (band-concatenated) image-feature
            # space; shared between the un-flipped and flipped halves so the
            # equivariance (input flip <-> image flip) is measured in one
            # consistent space.
            self.flip_align_proj = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, self.flip_n_col_bands * self.image_feature_dim),
            )

        # --- Equivariant frame-averaging frontend (plans/eeg-wm.md). Upgrades
        # the raw-space bilateral/lateral split into a proper invariance/
        # equivariance decomposition under the homologous-channel swap P
        # (P^2 = I). A channel-independent frontend f produces z = z_bi + z_lat
        # (split in *feature* space, on the patch embedding), and the transformer
        # body T is wrapped by frame averaging over the 2-element group {I, P}:
        #   h_bi  = (T(z) + T(P z)) / 2          (P-invariant  -> bilateral half)
        #   h_lat = (T(z) - P T(P z)) / 2        (P-anti-equivariant -> lateral half)
        # Tokens are split along the feature dim: h = [h_bi[:d/2] ; h_lat[d/2:]].
        # A random per-step ``flip`` chooses whether to PRESENT z or P(z) (and the
        # original vs horizontally-mirrored video frame); presenting P(z) only
        # negates the lateral half, so the architecture is exactly equivariant.
        self.frame_averaging = bool(frame_averaging)
        self.frame_avg_flip_prob = float(frame_avg_flip_prob)
        self.frame_avg_recon_weight = float(frame_avg_recon_weight)
        if self.frame_averaging:
            assert not self.project_to_source and not self.use_spectral_bands \
                and not self.use_moe and not self.contrastive_band, (
                "frame_averaging requires the plain CNN encoder path "
                "(no project_to_source / use_spectral_bands / use_moe / "
                "contrastive_band).")
            assert self.add_global, (
                "frame_averaging needs add_global for the global-token rep.")
            assert self.patch_embed_type == 'cnn', (
                "frame_averaging requires patch_embed_type='cnn'.")
            # f must commute with P -> drop the cross-channel conv positional
            # encoding so the patch embedding is channel-independent.
            self.patch_embedding.drop_pos_conv = True
            coord_pe_dim = 3 * 2 * self.spherical_num_freqs
            # Feature-space split: a per-channel gate on the d_model patch
            # embedding (not the raw signal). z_lat = g * z, z_bi = (1-g) * z.
            self.frame_split = LateralizationSplit(
                in_dim=d_model, coord_pe_dim=coord_pe_dim, hidden=flip_split_hidden,
            )
            assert self.flip_n_col_bands >= 1, "flip_n_col_bands must be >= 1"
            # Projects the global token into the centered column-band image
            # descriptor space for the same-sample hard-negative alignment (the
            # presented vs opposite-orientation global rep must match the
            # presented vs MIRRORED frame). Mirrors flip_align_proj but for the
            # frame-averaging path.
            self.frame_flip_align_proj = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, self.flip_n_col_bands * self.image_feature_dim),
            )

        self.apply(_weights_init)

        # Restore the near-zero gate init that the generic apply() above
        # overwrote, so the lateral stream starts ~0 (x_flip ~= x).
        if self.lateralization_flip:
            self.lateralization_split.reset_gate_init()
        if self.frame_averaging:
            # Start near-identity: z_lat ~ 0 so P(z) ~ z; the lateral stream
            # grows only as the flipped-prediction objective demands.
            self.frame_split.reset_gate_init()

        # Zero-init cross-band attention out_proj AFTER the generic init pass so
        # the band stream starts as an identity mix (controlled, near-baseline).
        if self.use_spectral_bands and self.use_cross_band_attn:
            for mod in self.cross_band_attn:
                if mod is not None:
                    nn.init.zeros_(mod.attn.out_proj.weight)
                    if mod.attn.out_proj.bias is not None:
                        nn.init.zeros_(mod.attn.out_proj.bias)

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

    @staticmethod
    def _level_indices(num_blocks, n_levels):
        """Pick ``n_levels`` shallow->deep block indices into ``hidden_states``.

        ``hidden_states`` has ``num_blocks + 1`` entries (index 0 = embeddings,
        1..num_blocks = transformer blocks). n_levels=1 returns the final block
        (matching the original ``hidden_states[12]`` behavior for DINOv2-base).
        """
        if n_levels <= 1:
            return [num_blocks]
        return [max(1, round((i + 1) * num_blocks / n_levels)) for i in range(n_levels)]

    @torch.inference_mode()
    def _dinov2_cls_token(self, image_encoder_inputs, n_levels=1):
        """Frozen DINOv2-style ViT CLS tokens at ``n_levels`` shallow->deep
        depths. Returns (B, n_levels, d_img)."""
        outputs = self.pretrained_image_encoder(
            **image_encoder_inputs, output_hidden_states=True)
        hs = outputs.hidden_states
        idxs = self._level_indices(len(hs) - 1, n_levels)
        return torch.stack([hs[i][:, 0] for i in idxs], dim=1)  # (B, L, d_img)

    @torch.no_grad()
    def _vjepa2_patch_tokens(self, pixel_values, n_levels=1):
        """Run frozen V-JEPA 2 and attention-pool patch tokens at ``n_levels``
        depths with the shared learned query. Returns (B, n_levels, d_img).

        ``pixel_values``: (B, 3, H, W). V-JEPA 2 expects a video tensor; we
        replicate the frame T=2 times so the temporal patch count is 1.
        """
        pv = pixel_values.unsqueeze(1).expand(-1, 2, -1, -1, -1).contiguous()
        outputs = self.pretrained_image_encoder(
            pixel_values_videos=pv, output_hidden_states=(n_levels > 1))
        B = pv.size(0)
        q = self.image_pool_query.expand(B, -1, -1)
        if n_levels <= 1:
            sources = [outputs.last_hidden_state]
        else:
            hs = outputs.hidden_states
            sources = [hs[i] for i in self._level_indices(len(hs) - 1, n_levels)]
        pooled = []
        for tok in sources:
            p, _ = self.image_pool_attn(q, tok, tok, need_weights=False)
            pooled.append(self.image_pool_norm(p.squeeze(1)))  # (B, d_img)
        return torch.stack(pooled, dim=1)  # (B, L, d_img)

    def get_image_hidden_states(self, n_levels=1, **image_encoder_inputs):
        """Return (B, n_levels, image_feature_dim) for either encoder.

        Dispatches on ``self.encoder_kind``: V-JEPA 2 (no CLS) -> attention
        pool over patch tokens with the learned query; DINOv2-style ViTs ->
        CLS token at the selected depths. ``n_levels=1`` reproduces the
        original single-level behavior byte-for-byte.
        """
        if self.encoder_kind == 'vjepa2':
            pixel_values = image_encoder_inputs.get('pixel_values')
            assert pixel_values is not None, (
                "V-JEPA 2 image encoder expects `pixel_values` in image_encoder_inputs"
            )
            return self._vjepa2_patch_tokens(pixel_values, n_levels=n_levels)
        # DINOv2-style fallback (also covers any future ViT with a CLS token).
        return self._dinov2_cls_token(image_encoder_inputs, n_levels=n_levels)

    @torch.no_grad()
    def vjepa2_grid_tokens(self, pixel_values):
        """Frozen V-JEPA 2 *un-pooled* spatial patch-token grid for a single
        static frame — the "world state" used by the action-conditioned world
        model (see ``models/action_world_model.py``).

        ``pixel_values``: (B, 3, H, W). V-JEPA 2 is natively a video tubelet
        model; we replicate the single frame T=2 times so the temporal patch
        count collapses to 1, leaving a purely *spatial* grid
        ``(B, P, image_feature_dim)`` with ``P = (H/patch)*(W/patch)`` (256 for
        a 256x256 frame at patch=16). Unlike :meth:`_vjepa2_patch_tokens` this
        returns the full token grid rather than the attention-pooled vector.

        The encoder is frozen at construction; we additionally force ``eval()``
        for the duration of the call so that an enclosing ``.train()`` on the
        owning module cannot re-enable any dropout inside the ViT.
        """
        assert self.encoder_kind == 'vjepa2', (
            "vjepa2_grid_tokens requires a V-JEPA 2 vision encoder, got "
            f"'{self.encoder_kind}'")
        was_training = self.pretrained_image_encoder.training
        self.pretrained_image_encoder.eval()
        try:
            pv = pixel_values.unsqueeze(1).expand(-1, 2, -1, -1, -1).contiguous()
            outputs = self.pretrained_image_encoder(pixel_values_videos=pv)
            return outputs.last_hidden_state  # (B, P, image_feature_dim)
        finally:
            if was_training:
                self.pretrained_image_encoder.train()

    # ------------------------------------------------------------------
    # Learnable spectral-band stream
    # ------------------------------------------------------------------
    def _encode_spectral_bands(self, x, batch, mask, ch_coords):
        """Decompose -> per-band embed (+band-type) -> shared encoder folded
        over (B*K) with optional cross-band mixing -> fuse.

        Returns:
            global_rep:   (B, d)        mean over bands of the per-band globals
            patch_tokens: (B, C, N, d)  fused (sum over bands) content tokens
            band_global:  (B, K, d)     per-band global tokens (alignment / rep)
        """
        B, C, N, in_dim = x.shape
        K, d = self.num_spectral_bands, self.d_model

        # 1. Decompose raw EEG into K ordered bands.
        band_sig = self.filterbank(x)                          # (B, K, C, N, in_dim)

        # 2. Coordinate PE (shared across bands), computed once.
        if self.use_volume_conduction:
            coord_emb = self.volume_conduction(
                ch_coords, valid_channel_mask=batch.get('valid_channel_mask', None))
        else:
            coord_pe, _ = self._spherical_positional_encoding(ch_coords)
            coord_emb = self.coord_enhancement(coord_pe)        # (B, C, d)

        # 3. Per-band patch embedding (shared) + coord PE + band-type embedding.
        band_embs = []
        for k in range(K):
            emb_k = self.patch_embedding(band_sig[:, k], mask)  # (B, C, N, d)
            emb_k = emb_k + coord_emb.unsqueeze(2)
            if self.use_band_type_embedding:
                emb_k = emb_k + self.band_type_embed[k].view(1, 1, 1, -1)
            band_embs.append(emb_k)
        patch_emb = torch.stack(band_embs, dim=1)               # (B, K, C, N, d)

        # 4. Prepend global channel row + global time column (per band).
        gch = self.global_channel.view(1, 1, 1, 1, -1).expand(B, K, 1, N, d)
        patch_emb = torch.cat([gch, patch_emb], dim=2)          # (B, K, C+1, N, d)
        Cp = patch_emb.size(2)
        gtok = self.global_token.view(1, 1, 1, 1, -1).expand(B, K, Cp, 1, d)
        patch_emb = torch.cat([gtok, patch_emb], dim=3)         # (B, K, C+1, N+1, d)
        Np = patch_emb.size(3)

        # 5. B-sized masks WITH prepended global rows (do NOT mutate batch — the
        #    equivariance second pass reuses the originals).
        def _expand(m, size):
            m = torch.cat([torch.ones_like(m[:, :1], dtype=torch.bool), m], dim=1)
            return m.unsqueeze(1).expand(B, K, size).reshape(B * K, size)
        vcm_bk = _expand(batch['valid_channel_mask'], Cp) if 'valid_channel_mask' in batch else None
        vlm_bk = _expand(batch['valid_length_mask'], Np) if 'valid_length_mask' in batch else None

        # 6. Fold bands into the batch and run the shared encoder + cross-band mix.
        h = patch_emb.reshape(B * K, Cp, Np, d)
        h = self._run_band_encoder(
            h, B, K,
            iw_mask=~vlm_bk if vlm_bk is not None else None,
            ir_mask=~vcm_bk if vcm_bk is not None else None,
        )                                                       # (B*K, Cp, Np, d)

        # 7. Unfold and split global / content.
        h = h.view(B, K, Cp, Np, d)
        band_global = h[:, :, 0, 0, :]                          # (B, K, d)
        band_content = h[:, :, 1:, 1:, :]                       # (B, K, C, N, d)
        global_rep = band_global.mean(dim=1)                    # (B, d)
        patch_tokens = band_content.sum(dim=1)                  # (B, C, N, d)
        return global_rep, patch_tokens, band_global

    def _run_band_encoder(self, h, B, K, iw_mask, ir_mask):
        """Run the shared encoder over folded (B*K) tokens, applying cross-band
        attention at the configured subset of layers."""
        for layer_idx in range(self.encoder.num_layers):
            h = self.TemEmbedEEGLayer(h) + h
            h = self.encoder.layers[layer_idx](
                h, self.area_config,
                inter_window_attn_mask=iw_mask,
                inter_region_attn_mask=ir_mask,
            )
            if (self.use_cross_band_attn
                    and self.cross_band_attn[layer_idx] is not None):
                h = self.cross_band_attn[layer_idx](h, B, K)
        return h

    def _band_multilevel_align(self, band_global, batch, has_image):
        """Align per-band globals to multiple visual levels via a learned soft
        band->level assignment, with a band-decorrelation regularizer.

        band_global: (B, K, d) full batch; ``has_image`` selects aligned rows.
        Returns a loss/metric dict following the trainer's key conventions
        ('loss'->(coef, tensor), 'acc'->tensor, 'diag_'->diagnostic).
        """
        img_inputs = {k: v[has_image] for k, v in batch['image_encoder_inputs'].items()}
        image_hs = self.get_image_hidden_states(
            n_levels=self.num_visual_levels, **img_inputs)      # (B_img, L, d_img)
        bg = band_global[has_image]                             # (B_img, K, d)
        B_img, K, _ = bg.shape
        L = image_hs.size(1)

        z = torch.stack([self.band_align_proj[k](bg[:, k]) for k in range(K)], dim=1)
        z = F.normalize(z, dim=-1)                              # (B_img, K, d_img)
        img = F.normalize(image_hs, dim=-1)                     # (B_img, L, d_img)

        A = F.softmax(self.band_level_logits, dim=1)            # (K, L) over levels
        temperature = 0.07
        targets = torch.arange(B_img, device=bg.device)
        out = {}
        total = bg.new_zeros(())
        total_acc = bg.new_zeros(())
        for l in range(L):
            eeg_l = F.normalize((A[:, l].view(1, K, 1) * z).sum(dim=1), dim=-1)
            logits = (eeg_l @ img[:, l].t()) / temperature
            loss_l = 0.5 * (F.cross_entropy(logits, targets)
                            + F.cross_entropy(logits.t(), targets))
            total = total + loss_l
            with torch.no_grad():
                acc = 0.5 * ((logits.argmax(1) == targets).float().mean()
                             + (logits.argmax(0) == targets).float().mean())
            total_acc = total_acc + acc
            out[f"band_align_acc_level{l}"] = acc
        out["band_align_loss"] = (self.alignment_weight, total / L)
        out["band_align_acc_total"] = total_acc / L

        if self.band_decorr_weight > 0:
            zc = F.normalize(z - z.mean(dim=0, keepdim=True), dim=-1)
            gram = torch.einsum('bkd,bjd->kj', zc, zc) / B_img
            eye = torch.eye(K, device=zc.device)
            decorr = (gram * (1 - eye)).pow(2).sum() / max(K * (K - 1), 1)
            out["band_decorr_loss"] = (self.band_decorr_weight, decorr)

        with torch.no_grad():
            low, high = self.filterbank.band_bounds()
            out["diag_band_level_assignment"] = A.detach()
            out["diag_band_low_hz"] = low.detach()
            out["diag_band_high_hz"] = high.detach()
        return out

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

    @staticmethod
    def _symmetric_infonce(pred, target, temperature=0.07, weight=None):
        """Symmetric InfoNCE between L2-normalised ``pred`` and ``target``
        (both (M, D)). ``weight`` (M,) optionally down-weights each row's
        positive term (used for motion weighting) — every row still serves as
        a negative for the others. Returns ``(loss, acc)``."""
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        logits = torch.matmul(pred, target.t()) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        if weight is None:
            loss = 0.5 * (F.cross_entropy(logits, labels)
                          + F.cross_entropy(logits.t(), labels))
        else:
            ce = 0.5 * (F.cross_entropy(logits, labels, reduction='none')
                        + F.cross_entropy(logits.t(), labels, reduction='none'))
            loss = (weight * ce).sum() / weight.sum().clamp(min=1e-6)
        with torch.no_grad():
            acc = 0.5 * ((logits.argmax(1) == labels).float().mean()
                         + (logits.argmax(0) == labels).float().mean())
        return loss, acc

    @torch.no_grad()
    def _image_patch_grid(self, pixel_values):
        """Frozen vision-encoder patch tokens reshaped to (B, s, s, d_img).

        Dispatches on ``encoder_kind``: V-JEPA 2 -> un-pooled tubelet grid via
        :meth:`vjepa2_grid_tokens`; DINOv2-style ViTs -> ``last_hidden_state``
        with the CLS (and any register) tokens dropped. The width axis (last
        spatial dim) is the horizontal/left-right axis.
        """
        if self.encoder_kind == 'vjepa2':
            tok = self.vjepa2_grid_tokens(pixel_values)          # (B, P, d_img)
        else:
            was_training = self.pretrained_image_encoder.training
            self.pretrained_image_encoder.eval()
            try:
                out = self.pretrained_image_encoder(pixel_values=pixel_values)
            finally:
                if was_training:
                    self.pretrained_image_encoder.train()
            hs = out.last_hidden_state                           # (B, T, d_img)
            n_reg = getattr(self.pretrained_image_encoder.config,
                            'num_register_tokens', 0) or 0
            tok = hs[:, 1 + n_reg:, :]                           # drop CLS(+reg)
        B, P, d = tok.shape
        s = int(round(math.sqrt(P)))
        assert s * s == P, (
            f"vision patch grid is not square (P={P}); cannot build the "
            f"column-band flip descriptor")
        return tok.reshape(B, s, s, d)

    def _image_lateral_descriptor(self, pixel_values):
        """Centered column-band spatial descriptor — the flip-sensitive image
        target for the bilateralization prior. Returns (B, n_bands * d_img).

        Splits the patch grid into ``flip_n_col_bands`` equal vertical (column)
        bands, mean-pools each, and (for n_bands>=2) removes the cross-band
        mean so the flip-invariant common component is gone, leaving the
        spatial left-right structure that a horizontal flip actually changes.
        """
        grid = self._image_patch_grid(pixel_values)              # (B, s, s, d)
        B, s, _, d = grid.shape
        nb = self.flip_n_col_bands
        assert s % nb == 0, (
            f"flip_n_col_bands={nb} must divide the patch-grid width {s}")
        w = s // nb
        bands = torch.stack(
            [grid[:, :, k * w:(k + 1) * w, :].mean(dim=(1, 2)) for k in range(nb)],
            dim=1)                                               # (B, nb, d)
        if nb >= 2:
            bands = bands - bands.mean(dim=1, keepdim=True)
        return bands.reshape(B, nb * d)

    def build_lateral_flip(self, x, ch_coords, ch_names,
                           valid_channel_mask=None):
        """Bilateralization flip of the raw input: x_flip = x_bi + flip(x_lat).

        Learns the additive split x = x_bi + x_lat (gate), then hemisphere-swaps
        only the lateral stream (C3<->C4, ...). ``x``: (B, C, N, in_dim).
        Returns ``(x_flip, gate)`` — the gate is returned so callers can
        regularise the lateral fraction. Shared by the image flip-alignment and
        the world-model flipped prediction so both use one flip construction.
        """
        B, C, N, d = x.shape
        coord_pe, _ = self._spherical_positional_encoding(ch_coords)
        x_lat, gate = self.lateralization_split(x, coord_pe, valid_channel_mask)
        perm = build_flip_perm_batch(ch_names, valid_channel_mask).to(x.device)
        perm_exp = perm[:, :C].view(B, C, 1, 1).expand(B, C, N, d)
        return x - x_lat + torch.gather(x_lat, 1, perm_exp), gate

    def _flip_alignment(self, batch, global_rep, has_image,
                        orig_vcm, orig_vlm, orig_mask):
        """Bilateralization-prior flip-equivariant alignment.

        Decompose the raw signal ``x = x_bi + x_lat`` with a learned gate,
        build ``x_flip = x_bi + flip(x_lat)`` by swapping homologous channels
        (C3<->C4, ...) of the *lateral* stream only, and encode ``x_flip``
        through the shared backbone. The objective ties an *input* flip to an
        *image* flip via ONE stacked InfoNCE: queries ``[pred_i ; pred_flip_i]``
        match targets ``[d_i ; d_flip_i]`` (original / mirrored frame
        descriptors), so the same-sample opposite-flip frame is an explicit
        hard negative. This forces the gate to engage — with ``g=0`` a sample's
        two queries are identical yet need different targets. An optional
        minimality penalty on the gate keeps ``x_lat`` small.

        Operates only on the ``has_image`` rows (alignment needs a frame), so
        the extra encode is restricted to those samples.
        """
        idx = has_image.nonzero(as_tuple=True)[0]
        x = batch['timeseries'].index_select(0, idx)            # (M, C, N, d)
        ch_coords = batch['ch_coords'].index_select(0, idx)     # (M, C, 3)
        ch_names = [batch['ch_names'][i] for i in idx.tolist()]
        vcm = orig_vcm.index_select(0, idx) if orig_vcm is not None else None
        vlm = orig_vlm.index_select(0, idx) if orig_vlm is not None else None
        mask = orig_mask.index_select(0, idx) if orig_mask is not None else None
        M, C, N, d = x.shape

        # 1-2. Learned split x = x_bi + x_lat, then hemispheric swap of the
        #      lateral stream only -> x_flip = x_bi + flip(x_lat).
        x_flip, gate = self.build_lateral_flip(x, ch_coords, ch_names, vcm)

        # 3. Encode the flipped signal through the shared backbone.
        flip_batch = {'timeseries': x_flip, 'ch_coords': ch_coords}
        if vcm is not None:
            flip_batch['valid_channel_mask'] = vcm
        if vlm is not None:
            flip_batch['valid_length_mask'] = vlm
        _, info_flip = self.forward(flip_batch, mask=mask, encoder_only=True)
        global_rep_flip = info_flip['global_rep']                     # (M, d)

        # 4. Flip-sensitive image target: centered column-band spatial
        #    descriptor of the original + horizontally-flipped frame. (The
        #    global CLS is ~flip-invariant, which makes the objective vacuous.)
        pixel_values = batch['image_encoder_inputs']['pixel_values'].index_select(0, idx)
        img_emb = self._image_lateral_descriptor(pixel_values)        # (M, nb*d_img)
        img_emb_flip = self._image_lateral_descriptor(
            torch.flip(pixel_values, dims=[-1]))

        # 4b. Per-sample motion weight (down-weights static frames whose flip
        #     is near-vacuous). Sentinel <0 (no motion info, e.g. single-image
        #     Alljoined rows) -> weight 1.
        weight = None
        motion = batch.get('flip_motion')
        if motion is not None and self.flip_motion_ref > 0:
            m = motion.index_select(0, idx).to(global_rep.dtype)      # (M,)
            weight = torch.where(
                m >= 0, (m / self.flip_motion_ref), torch.ones_like(m))
            weight = weight.clamp(min=self.flip_motion_min, max=1.0)

        # 5. Stacked symmetric InfoNCE with the same-sample opposite-flip frame
        #    as an explicit HARD NEGATIVE. Queries [pred_i ; pred_flip_i] must
        #    match targets [d_i ; d_flip_i]. With g=0 a sample's two queries are
        #    the SAME vector yet need DIFFERENT targets -> impossible, so the
        #    gate is forced to engage (un-flipped rep -> original image, flipped
        #    rep -> mirrored image). Cross-sample rows remain negatives too.
        #    (Two *separate* matrices instead let g=0 survive via a bisector
        #    between d_i and d_flip_i — observed gate collapse, run vocstdci.)
        pred = self.flip_align_proj(global_rep.index_select(0, idx))   # (M, D)
        pred_flip = self.flip_align_proj(global_rep_flip)              # (M, D)
        Q = torch.cat([pred, pred_flip], dim=0)                       # (2M, D)
        T = torch.cat([img_emb, img_emb_flip], dim=0)                 # (2M, D)
        W = torch.cat([weight, weight], dim=0) if weight is not None else None
        loss, acc = self._symmetric_infonce(Q, T, weight=W)

        out = {
            'flip_align_loss': (self.flip_align_weight, loss),
            'flip_align_acc': acc,
        }
        # Per-sample flip DISCRIMINATION (chance 0.5): does each rep prefer its
        # own orientation's image over the opposite one? Direct signal that the
        # gate is being used — if this stays ~0.5 the flip is being ignored.
        with torch.no_grad():
            pn = F.normalize(pred, dim=-1); pfn = F.normalize(pred_flip, dim=-1)
            dn = F.normalize(img_emb, dim=-1); dfn = F.normalize(img_emb_flip, dim=-1)
            out['diag_flip_discrim_acc'] = 0.5 * (
                ((pn * dn).sum(-1) > (pn * dfn).sum(-1)).float().mean()
                + ((pfn * dfn).sum(-1) > (pfn * dn).sum(-1)).float().mean())
        if weight is not None:
            out['diag_flip_motion_weight'] = weight.mean().detach()

        # 6. Minimality of the lateral stream -> most signal stays bilateral.
        if vcm is not None:
            m = vcm.to(gate.dtype).view(M, C, 1, 1)
            lat_frac = (gate * m).sum() / m.expand_as(gate).sum().clamp(min=1)
        else:
            lat_frac = gate.mean()
        if self.lat_sparsity_weight > 0:
            out['lat_sparsity_loss'] = (self.lat_sparsity_weight, lat_frac)
        out['diag_lat_gate_mean'] = lat_frac.detach()
        return out

    # ------------------------------------------------------------------
    # Equivariant frame-averaging frontend (plans/eeg-wm.md)
    # ------------------------------------------------------------------

    @staticmethod
    def _swap_channels(t, perm, channel_offset=0):
        """Homologous-channel swap P on the channel axis of ``t`` (B, Ctot, N, d).

        ``perm`` (B, C) is the per-sample real-channel permutation from
        :func:`build_flip_perm_batch`. ``channel_offset`` is the number of
        leading rows P must leave fixed (1 when a global channel sits at index 0,
        0 on the raw per-channel tokens). P is an involution (perm is)."""
        B, Ctot, N, d = t.shape
        C = perm.size(1)
        idx = torch.arange(Ctot, device=t.device).unsqueeze(0).expand(B, -1).clone()
        idx[:, channel_offset:channel_offset + C] = perm + channel_offset
        idx_exp = idx.view(B, Ctot, 1, 1).expand(B, Ctot, N, d)
        return torch.gather(t, 1, idx_exp)

    def _frame_transformer_body(self, z, ch_coords, vcm=None, vlm=None):
        """The transformer block ``T`` over per-channel tokens ``z`` (B, C, N, d).

        Adds the coord positional embedding, prepends the global channel/token
        rows, and runs the encoder layer loop. Returns the full token grid
        (B, C+1, N+1, d) (with the global rows) when ``add_global`` else
        (B, C, N, d). Pure: it does NOT mutate ``batch`` masks (frame averaging
        calls it twice per forward), and it computes no inline contrastive/recon
        loss — those live in :meth:`_forward_frame_averaging` on the
        frame-averaged representation.
        """
        B, C, N, d = z.shape
        coord_pe, _ = self._spherical_positional_encoding(ch_coords)
        coord_emb = self.coord_enhancement(coord_pe)            # (B, C, d)
        patch_emb = z + coord_emb.unsqueeze(2)                  # (B, C, N, d)

        if self.add_global:
            patch_emb = torch.cat(
                [self.global_channel.expand(B, -1, patch_emb.size(2), -1),
                 patch_emb], dim=1)
            patch_emb = torch.cat(
                [self.global_token.expand(B, patch_emb.size(1), -1, -1),
                 patch_emb], dim=2)
            if vcm is not None:
                vcm = torch.cat(
                    [torch.ones_like(vcm[:, :1], dtype=torch.bool), vcm], dim=1)
            if vlm is not None:
                vlm = torch.cat(
                    [torch.ones_like(vlm[:, :1], dtype=torch.bool), vlm], dim=1)

        inter_window = ~vlm if vlm is not None else None
        inter_region = ~vcm if vcm is not None else None
        for layer_idx in range(self.encoder.num_layers):
            patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
            patch_emb = self.encoder.layers[layer_idx](
                patch_emb, self.area_config,
                inter_window_attn_mask=inter_window,
                inter_region_attn_mask=inter_region,
            )
        return patch_emb

    def _build_Pz(self, z, coord_pe, perm, vcm):
        """Apply the operator P in feature space: P(z) = z_bi + flip(z_lat),
        where z_lat = gate * z (learned per-channel split) and ``flip`` is the
        homologous channel swap. Returns ``(P(z), gate)``. The body T is run on
        BOTH z and P(z); ``flip`` only picks which combination is presented."""
        z_lat, gate = self.frame_split(z, coord_pe, vcm)        # z_lat = g * z
        z_lat_swapped = self._swap_channels(z_lat, perm, channel_offset=0)
        Pz = z - z_lat + z_lat_swapped                          # z_bi + flip(z_lat)
        return Pz, gate

    def _forward_frame_averaging(self, batch, mask=None, encoder_only=False):
        """Equivariant frame-averaging forward (see plans/eeg-wm.md).

        1. Channel-independent frontend f: z = patch_embedding(x) (conv pos-enc
           dropped) split additively into z_bi + z_lat in feature space.
        2. P(z) = z_bi + flip(z_lat) (homologous channel swap of z_lat only).
        3. Frame averaging over {I, P}: run T on z and P(z), combine into the
           invariant (bilateral) and anti-equivariant (lateral) parts; the token
           feature dim is split half-bilateral / half-lateral.
        4. A random per-step ``flip`` presents z or P(z) and (downstream) the
           original or horizontally-mirrored frame.
        """
        x = batch['timeseries']                                 # (B, C, N, in_dim)
        ch_coords = batch['ch_coords']
        ch_names = batch['ch_names']
        vcm = batch.get('valid_channel_mask')
        vlm = batch.get('valid_length_mask')
        B, C, N, in_dim = x.shape

        # Decide the presentation. The world-model wrapper sets a shared ``flip``
        # so the current and future windows are presented consistently; a bare
        # encoder samples per forward in train and is canonical (no flip) in eval
        # so the equivariance test is deterministic.
        if 'flip' in batch:
            flip = bool(batch['flip'])
        elif self.training:
            flip = bool(torch.rand(()) < self.frame_avg_flip_prob)
        else:
            flip = False

        # 1-2. Frontend + feature-space split + P operator.
        z = self.patch_embedding(x, mask)                       # (B, C, N, d)
        coord_pe, _ = self._spherical_positional_encoding(ch_coords)
        perm = build_flip_perm_batch(ch_names, vcm).to(z.device)   # (B, C)
        Pz, gate = self._build_Pz(z, coord_pe, perm, vcm)

        # 3. Frame averaging: run T on z and P(z) (always both, independent of
        #    `flip`), combine. Both orientations are available from this single
        #    pair of body passes — `flip` only selects which combination is the
        #    presented lateral half vs the opposite (hard-negative) one.
        a = self._frame_transformer_body(z,  ch_coords, vcm, vlm)
        b = self._frame_transformer_body(Pz, ch_coords, vcm, vlm)
        off = 1 if self.add_global else 0
        Pa = self._swap_channels(a, perm, channel_offset=off)
        Pb = self._swap_channels(b, perm, channel_offset=off)
        inv = 0.5 * (a + b)                                     # P-invariant
        # P-anti-equivariant. Presenting P(z) swaps (a,b) -> the lateral half
        # negates: eq(P z) = (b - P a)/2 = -P( (a - P b)/2 ).
        eq = 0.5 * (b - Pa) if flip else 0.5 * (a - Pb)
        # The OPPOSITE orientation's lateral half (free: just the other combo).
        # Its global token is the present one with the lateral half negated, so
        # it serves as the same-sample hard negative for image alignment.
        eq_opp = 0.5 * (a - Pb) if flip else 0.5 * (b - Pa)
        half = inv.size(-1) // 2
        h = torch.cat([inv[..., :half], eq[..., half:]], dim=-1)   # (B, C+1, N+1, d)

        if self.add_global:
            global_rep = h[:, 0, 0, :]
            patch_tokens = h[:, 1:, 1:, :]
            opp_global = torch.cat(
                [inv[:, 0, 0, :half], eq_opp[:, 0, 0, half:]], dim=-1)
        else:
            global_rep = h.mean(dim=(1, 2))
            patch_tokens = h
            opp_global = torch.cat(
                [inv.mean((1, 2))[:, :half], eq_opp.mean((1, 2))[:, half:]], dim=-1)

        if encoder_only:
            return None, {"global_rep": global_rep, "patch_tokens": patch_tokens}

        # Reconstruction head.
        out = self.proj_out(h)
        if self.add_global:
            out = out[:, 1:, 1:, :]                             # (B, C, N, out_dim)

        info = {"global_rep": global_rep, "patch_tokens": patch_tokens, "rep": None}

        # Alignment loss: present-orientation global token -> present-orientation
        # frame (original / horizontally-mirrored). Standard symmetric InfoNCE,
        # reusing the image-contrastive machinery (self.alignment_weight).
        has_image = batch.get('has_image')
        if (self.alignment_weight > 0 and 'image_encoder_inputs' in batch
                and has_image is not None and has_image.any()):
            semantic_emb = self.semantic_readout(global_rep[has_image])
            pred_flatten = self.contrastive_proj(semantic_emb)
            align_batch = batch
            if flip:
                pv = batch['image_encoder_inputs']['pixel_values']
                pv = torch.flip(pv, dims=[-1])                  # mirror horizontally
                align_batch = {**batch, 'image_encoder_inputs': {
                    **batch['image_encoder_inputs'], 'pixel_values': pv}}
            info.update(self._image_contrastive(pred_flatten, align_batch, has_image))

        # Same-sample HARD-NEGATIVE alignment on the global rep (restored from the
        # legacy _flip_alignment). The presented and opposite-orientation global
        # reps differ ONLY in the sign of the lateral half; they must align to the
        # presented vs MIRRORED frame's CENTERED column-band descriptor (the CLS
        # is ~flip-invariant -> vacuous). With a trivial (L-R symmetric) split the
        # two reps coincide yet need different targets, so this directly penalises
        # the trivial flip and forces the lateral half (and z_lat) to carry
        # laterality. Cheap here: the opposite rep is already computed (eq_opp);
        # no second EEG encode is needed.
        if (self.flip_align_weight > 0 and 'image_encoder_inputs' in batch
                and has_image is not None and has_image.any()):
            idx = has_image.nonzero(as_tuple=True)[0]
            pv = batch['image_encoder_inputs']['pixel_values'].index_select(0, idx)
            desc_orig = self._image_lateral_descriptor(pv)
            desc_mirror = self._image_lateral_descriptor(torch.flip(pv, dims=[-1]))
            present_desc, opposite_desc = (
                (desc_mirror, desc_orig) if flip else (desc_orig, desc_mirror))
            pred_present = self.frame_flip_align_proj(global_rep.index_select(0, idx))
            pred_opp = self.frame_flip_align_proj(opp_global.index_select(0, idx))
            # Optional per-sample motion weighting (static frames flip ~vacuously).
            weight = None
            motion = batch.get('flip_motion')
            if motion is not None and self.flip_motion_ref > 0:
                mm = motion.index_select(0, idx).to(global_rep.dtype)
                weight = torch.where(
                    mm >= 0, mm / self.flip_motion_ref, torch.ones_like(mm)
                ).clamp(min=self.flip_motion_min, max=1.0)
            Q = torch.cat([pred_present, pred_opp], dim=0)
            T = torch.cat([present_desc, opposite_desc], dim=0)
            W = torch.cat([weight, weight], dim=0) if weight is not None else None
            floss, facc = self._symmetric_infonce(Q, T, weight=W)
            info['flip_align_loss'] = (self.flip_align_weight, floss)
            info['flip_align_acc'] = facc
            with torch.no_grad():
                pn = F.normalize(pred_present, dim=-1)
                po = F.normalize(pred_opp, dim=-1)
                dp = F.normalize(present_desc, dim=-1)
                do = F.normalize(opposite_desc, dim=-1)
                info['diag_flip_discrim_acc'] = 0.5 * (
                    ((pn * dp).sum(-1) > (pn * do).sum(-1)).float().mean()
                    + ((po * do).sum(-1) > (po * dp).sum(-1)).float().mean())
            if weight is not None:
                info['diag_flip_motion_weight'] = weight.mean().detach()

        # Reconstruction loss on flip steps. ALWAYS skip the trainer's external
        # masked MSE(out, x): ``out`` reconstructs the PRESENTED (flipped) signal,
        # so comparing it to the original ``x`` is wrong. The frontend-space
        # self-consistency (no ground-truth flipped timeseries exists) — f(out)
        # should match the presented CLEAN latent P(z_clean) on masked patches —
        # is ALWAYS computed and logged so it stays monitorable, but it is only
        # added to the optimised loss when --frame_avg_recon_weight > 0. Weight 0
        # therefore disables recon's CONTRIBUTION on flip steps while still
        # logging diag_frame_recon (compute is detached, so no backward cost).
        if flip:
            info['skip_external_recon'] = True
            want_grad = self.frame_avg_recon_weight > 0
            with torch.set_grad_enabled(want_grad):
                z_recon = self.patch_embedding(out, None)       # f(reconstruction)
                with torch.no_grad():
                    z_clean = self.patch_embedding(x, None)
                    zlat_clean, _ = self.frame_split(z_clean, coord_pe, vcm)
                    Pz_clean = (z_clean - zlat_clean
                                + self._swap_channels(zlat_clean, perm, 0))
                if mask is not None and (mask == 1).any():
                    m = (mask == 1)
                    recon = F.mse_loss(z_recon[m], Pz_clean[m])
                else:
                    recon = F.mse_loss(z_recon, Pz_clean)
            if want_grad:
                info['frame_recon_loss'] = (self.frame_avg_recon_weight, recon)
            else:
                info['diag_frame_recon'] = recon.detach()

        # Diagnostics.
        with torch.no_grad():
            if vcm is not None:
                m = vcm.to(gate.dtype).view(B, C, 1, 1)
                lat_frac = (gate * m).sum() / m.expand_as(gate).sum().clamp(min=1)
            else:
                lat_frac = gate.mean()
            info['diag_frame_lat_gate_mean'] = lat_frac.detach()
            info['diag_frame_flip'] = torch.tensor(float(flip), device=x.device)
            info['diag_frame_lat_half_norm'] = (
                eq[..., half:].norm(dim=-1).mean().detach())
            info['diag_frame_bi_half_norm'] = (
                inv[..., :half].norm(dim=-1).mean().detach())

        return out, info

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
        if self.frame_averaging:
            return self._forward_frame_averaging(
                batch, mask=mask, encoder_only=encoder_only)

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
        elif self.use_spectral_bands:
            # Learnable-filterbank band stream: decompose -> per-band embed
            # (+ band-type) -> shared encoder folded over (B*K) with optional
            # cross-band mixing -> fuse. Replaces the standard encoder path.
            global_rep, patch_tokens, band_global = self._encode_spectral_bands(
                x, batch, mask, ch_coords)
            branch_embs = band_global  # (B, K, d) -> stored as info['rep']
            has_image = batch.get('has_image', None)
            if (not encoder_only
                    and 'image_encoder_inputs' in batch
                    and has_image is not None and has_image.any()
                    and self.alignment_weight > 0):
                contrastive_loss.update(
                    self._band_multilevel_align(band_global, batch, has_image))
        else:
            if self.use_volume_conduction:
                coord_emb = self.volume_conduction(
                    ch_coords,
                    valid_channel_mask=batch.get('valid_channel_mask', None),
                )
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
        elif self.use_spectral_bands:
            # patch_tokens is already the fused (B, C, N, d) content (no global rows).
            out = self.proj_out(patch_tokens)
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

        # ------------------------------------------------------------------
        # Bilateralization-prior flip-equivariant alignment (only when images
        # are present). Skipped for the source-projector path (its ``x`` is in
        # source space, not sensor space, so a channel swap is ill-defined).
        # ------------------------------------------------------------------
        flip_losses = {}
        has_image = batch.get('has_image')
        if (self.lateralization_flip
                and self.flip_align_weight > 0
                and not self.project_to_source
                and batch.get('ch_names') is not None
                and 'image_encoder_inputs' in batch
                and has_image is not None and has_image.any()):
            flip_losses = self._flip_alignment(
                batch, global_rep, has_image,
                _orig_vcm, _orig_vlm, _orig_mask,
            )

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
            **flip_losses,
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
                 spectral_mode='static', stft_n_fft=64, stft_hop=1,
                 drop_pos_conv=False):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        # ``drop_pos_conv`` skips the depthwise Conv2d positional encoding (kernel
        # 19 along the channel axis) — the only cross-channel mixer in this
        # module. With it off the patch embedding is channel-independent and so
        # commutes with the homologous-channel swap P (up to GroupNorm pooling),
        # which is what the equivariant frame-averaging frontend f needs.
        self.drop_pos_conv = bool(drop_pos_conv)
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

        if not self.drop_pos_conv:
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
