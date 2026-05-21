import copy
import math
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
import warnings
from torch import Tensor
from torch.nn import functional as F
import numpy as np


class MoEFeedForward(nn.Module):
    """Top-k MoE FFN sized to preserve the dense FFN weight budget.

    Per-expert hidden = ``dim_feedforward // num_experts`` so the combined
    weight count of ``w1`` (E, d, h_e) + ``w2`` (E, h_e, d) equals the dense
    ``2 * d * dim_feedforward``. Bias and gate parameters add a small
    residual (≤ (E-1)*d_model + E*g) that is negligible at the scales here.
    The gate is conditioned on an external ``gate_input`` — typically a
    per-patch spectral descriptor — so experts can specialize by frequency
    rather than by token identity.
    """

    def __init__(self, d_model: int, dim_feedforward: int, num_experts: int,
                 top_k: int = 2, gate_input_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 activation: Callable[[Tensor], Tensor] = F.gelu,
                 bias: bool = True):
        super().__init__()
        assert dim_feedforward % num_experts == 0, (
            f"dim_feedforward ({dim_feedforward}) must be divisible by "
            f"num_experts ({num_experts}) so per-expert hidden size matches "
            f"the dense FFN parameter budget."
        )
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.expert_hidden = dim_feedforward // num_experts
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, self.expert_hidden))
        self.w2 = nn.Parameter(torch.empty(num_experts, self.expert_hidden, d_model))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        if bias:
            self.b1 = nn.Parameter(torch.zeros(num_experts, self.expert_hidden))
            self.b2 = nn.Parameter(torch.zeros(num_experts, d_model))
        else:
            self.register_parameter('b1', None)
            self.register_parameter('b2', None)

        self.gate_input_dim = gate_input_dim if gate_input_dim is not None else d_model
        self.gate = nn.Linear(self.gate_input_dim, num_experts)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: Tensor,
                gate_input: Optional[Tensor] = None,
                band_targets: Optional[Tensor] = None):
        """x: (..., d_model). Returns (out, aux) with same leading shape.

        ``gate_input`` (..., gate_input_dim) drives expert selection; if
        ``None``, gate falls back to the token embedding. ``band_targets``
        (..., num_experts) is an optional soft prior over experts (e.g.,
        per-band energy) used to compute a KL alignment loss.
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        N = x_flat.size(0)

        if gate_input is None:
            gate_in = x_flat
        else:
            gate_in = gate_input.reshape(-1, self.gate_input_dim)
        gate_logits = self.gate(gate_in)
        # log_softmax for numerically stable KL and z-loss.
        log_probs = F.log_softmax(gate_logits, dim=-1)
        gate_probs_full = log_probs.exp()

        topk_vals, topk_idx = gate_probs_full.topk(self.top_k, dim=-1)
        topk_norm = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        gate_weights = torch.zeros_like(gate_probs_full).scatter(
            dim=-1, index=topk_idx, src=topk_norm,
        )

        h_all = torch.einsum('nd,edh->neh', x_flat, self.w1)
        if self.b1 is not None:
            h_all = h_all + self.b1.unsqueeze(0)
        h_all = self.activation(h_all)
        h_all = self.dropout(h_all)
        out_all = torch.einsum('neh,ehd->ned', h_all, self.w2)
        if self.b2 is not None:
            out_all = out_all + self.b2.unsqueeze(0)
        out = (out_all * gate_weights.unsqueeze(-1)).sum(dim=1)
        out = out.reshape(orig_shape)

        top1 = topk_idx[:, 0]
        f = torch.zeros(self.num_experts, device=x_flat.device, dtype=gate_probs_full.dtype)
        f.scatter_add_(0, top1, torch.ones(N, device=x_flat.device, dtype=gate_probs_full.dtype))
        f = f / max(N, 1)
        P = gate_probs_full.mean(dim=0)
        balance_loss = self.num_experts * (f * P).sum()

        # Router z-loss (Zoph et al. ST-MoE). Penalises unbounded growth of
        # gate logits — the standard fix for MoE training going NaN late.
        # log Z = logsumexp(logits) per token; loss = mean((log Z)^2).
        log_Z = torch.logsumexp(gate_logits, dim=-1)
        z_loss = log_Z.pow(2).mean()

        aux = {'balance': balance_loss, 'z_loss': z_loss}
        if band_targets is not None:
            tgt = band_targets.reshape(-1, self.num_experts)
            tgt_sum = tgt.sum(dim=-1, keepdim=True)
            # Masked patches (zero signal) have zero spectral energy in every
            # band → tgt_sum == 0. Fall back to uniform there so the prior
            # stays a valid distribution and the gate isn't pushed anywhere.
            uniform = torch.full_like(tgt, 1.0 / self.num_experts)
            tgt = torch.where(
                tgt_sum > 1e-6,
                tgt / tgt_sum.clamp_min(1e-6),
                uniform,
            )
            # F.kl_div(input=log_probs, target=probs, log_target=False) uses
            # xlogy semantics, so tgt = 0 components contribute 0 cleanly.
            aux['band_prior'] = F.kl_div(
                log_probs, tgt, reduction='batchmean', log_target=False,
            )
        return out, aux


class CSBrain_TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, bias: bool = True,
                 area_config: dict = {}, sorted_indices: list = [], causal: bool = False,
                 use_moe: bool = False, num_experts: int = 4, moe_top_k: int = 2,
                 moe_gate_input_dim: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.causal = causal

        self.inter_region_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                       bias=bias, batch_first=batch_first)

        self.inter_window_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                       bias=bias, batch_first=batch_first)

        self.global_fc = nn.Linear(d_model, d_model, bias=bias)

        self.use_moe = use_moe
        if use_moe:
            # Resolve activation now so MoE uses the same nonlinearity as the
            # dense FFN it replaces.
            if isinstance(activation, str):
                act_fn = getattr(F, activation, F.relu)
            else:
                act_fn = activation
            self.moe = MoEFeedForward(
                d_model=d_model, dim_feedforward=dim_feedforward,
                num_experts=num_experts, top_k=moe_top_k,
                gate_input_dim=moe_gate_input_dim,
                dropout=dropout, activation=act_fn, bias=bias,
            )
            self.dropout = nn.Dropout(dropout)  # kept for state-dict symmetry; unused in MoE path
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        # Populated by the most recent forward when ``use_moe`` is True; read
        # by the caller (e.g. CSBrainAlign) to assemble auxiliary losses.
        self.last_aux_loss: dict = {}

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = getattr(F, activation, F.relu)
        self.activation = activation

        self.area_config = area_config
        self.mask_builder = None
        self.region_attn_mask = None
        self.region_indices_dict = None

        if area_config is not None:
            total_channels = sum(len(range(info['slice'].start or 0, info['slice'].stop, info['slice'].step or 1))
                                 if isinstance(info['slice'], slice) else len(info['slice'])
                                 for info in area_config.values())

            self.mask_builder = RegionAttentionMaskBuilder(total_channels, area_config)
            self.region_attn_mask = self.mask_builder.get_mask()
            self.region_indices_dict = self.mask_builder.get_region_indices()

    def forward(
        self,
        src: torch.Tensor,
        area_config: Optional[dict] = None,
        src_mask: Optional[torch.Tensor] = None,
        inter_window_attn_mask: Optional[torch.Tensor] = None,
        inter_region_attn_mask: Optional[torch.Tensor] = None,
        gate_features: Optional[torch.Tensor] = None,
        band_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = src
        x = x + self._inter_window_attention(self.norm1(x), src_mask, inter_window_attn_mask)
        if self.region_attn_mask is None and area_config is not None:
            x = x + self._inter_region_attention_dynamic(self.norm2(x), area_config, src_mask, inter_region_attn_mask)
            raise
        else:
            x = x + self._inter_region_attention_static(self.norm2(x), src_mask, inter_region_attn_mask)

        x = x + self._ff_block(self.norm3(x), gate_features=gate_features, band_targets=band_targets)
        return x

    def _inter_region_attention_static(self, x: torch.Tensor,
                                       attn_mask: Optional[torch.Tensor] = None,
                                       key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # if self.region_attn_mask is None or self.region_indices_dict is None:
        #     raise ValueError("no initialized region attention mask or region indices dictionary")

        batch, chans, T, F = x.shape

        x_reshaped = x.permute(0, 2, 1, 3)
        x_flat = x_reshaped.reshape(batch * T, chans, F)
        key_padding_mask = key_padding_mask.unsqueeze(2).expand(-1, -1, T).permute(0, 2, 1).reshape(batch * T, chans) if key_padding_mask is not None else None

        if self.region_attn_mask is None or self.region_indices_dict is None:
            x_enhanced = x_flat
            region_attn_mask = None
        else:
            region_global_features = {}
            for region_name, region_indices in self.region_indices_dict.items():
                region_x = x[:, region_indices, :, :]
                region_global = region_x.mean(dim=1, keepdim=True)
                region_global_features[region_name] = region_global

            global_features = torch.zeros_like(x_flat)

            for region_name, region_indices in self.region_indices_dict.items():
                region_global = region_global_features[region_name]
                region_global = region_global.permute(0, 2, 1, 3)
                region_global = region_global.reshape(batch * T, 1, F)

                for idx in region_indices:
                    global_features[:, idx:idx + 1, :] = region_global

            global_features = self.global_fc(global_features)
            x_enhanced = x_flat + global_features

            region_attn_mask = self.region_attn_mask.to(x.device)

        attn_output = self.inter_region_attn(
            x_enhanced, x_enhanced, x_enhanced,
            attn_mask=region_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]

        attn_output = attn_output.reshape(batch, T, chans, F).permute(0, 2, 1, 3)

        return self.dropout1(attn_output)

    def _inter_window_attention(self, x: torch.Tensor,
                                attn_mask: Optional[torch.Tensor] = None,
                                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, chans, T, Fea = x.shape
        window_size = min(T, 5)

        num_windows = T // window_size
        original_T = T

        if T % window_size != 0:
            pad_length = window_size - (T % window_size)
            x = F.pad(x, (0, 0, 0, pad_length))
            key_padding_mask = F.pad(key_padding_mask, (0, pad_length), value=True) if key_padding_mask is not None else None
            T = T + pad_length
            num_windows = T // window_size

        x = x.view(batch, chans, num_windows, window_size, Fea)

        x = x.permute(0, 3, 1, 2, 4)
        x = x.reshape(batch * window_size * chans, num_windows, Fea)

        key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, chans, -1).view(batch, chans, num_windows, window_size).permute(0, 3, 1, 2).reshape(batch * window_size * chans, num_windows) if key_padding_mask is not None else None

        temporal_attn_mask = None
        if self.causal:
            temporal_attn_mask = torch.triu(
                torch.ones(num_windows, num_windows, device=x.device) * float('-inf'),
                diagonal=1
            )
        elif attn_mask is not None:
            if isinstance(attn_mask, torch.Tensor) and attn_mask.dim() == 2:
                temporal_attn_mask = torch.triu(
                    torch.ones(num_windows, num_windows, device=x.device) * float('-inf'),
                    diagonal=1
                )

        x = self.inter_window_attn(
            x, x, x,
            attn_mask=temporal_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]

        x = x.reshape(batch, window_size, chans, num_windows, Fea)
        x = x.permute(0, 2, 3, 1, 4)

        x = x.reshape(batch, chans, T, Fea)
        if T != original_T:
            x = x[:, :, :original_T, :]

        return self.dropout2(x)

    def _ff_block(self, x: torch.Tensor,
                  gate_features: Optional[torch.Tensor] = None,
                  band_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, T, F = x.shape

        if self.use_moe:
            x_ff, aux = self.moe(x, gate_input=gate_features, band_targets=band_targets)
            self.last_aux_loss = aux
        else:
            x_reshaped = x.permute(0, 2, 1, 3).reshape(B * T, C, F)
            x_ff = self.linear2(self.dropout(self.activation(self.linear1(x_reshaped))))
            x_ff = x_ff.reshape(B, T, C, F).permute(0, 2, 1, 3)

        return self.dropout3(x_ff)


class RegionAttentionMaskBuilder:
    def __init__(self, num_channels: int, area_config: dict, device=None):
        self.num_channels = num_channels
        self.area_config = area_config
        self.device = device

        self.region_indices_dict = self._process_region_indices()

        self.attention_mask = self._build_attention_mask()

    def _process_region_indices(self):
        region_indices_dict = {}

        for region_name, region_info in self.area_config.items():
            region_slice = region_info['slice']
            if isinstance(region_slice, slice):
                start = region_slice.start or 0
                stop = region_slice.stop
                step = region_slice.step or 1
                region_indices = list(range(start, stop, step))
            else:
                region_indices = list(region_slice)

            region_indices_dict[region_name] = region_indices

        return region_indices_dict

    def _build_attention_mask(self):
        device = self.device if self.device is not None else torch.device('cpu')
        region_attn_mask = torch.ones(self.num_channels, self.num_channels, device=device) * float('-inf')

        num_groups = max(len(indices) for indices in self.region_indices_dict.values())

        groups = [[] for _ in range(num_groups)]

        for g in range(num_groups):
            for region_name, region_indices in self.region_indices_dict.items():
                n_electrodes = len(region_indices)
                if n_electrodes == 0:
                    continue

                electrode_idx = region_indices[g % n_electrodes]
                groups[g].append(electrode_idx)

        for g, group_electrodes in enumerate(groups):
            for idx1 in group_electrodes:
                for idx2 in group_electrodes:
                    region_attn_mask[idx1, idx2] = 0

        return region_attn_mask

    def get_mask(self):
        return self.attention_mask

    def get_region_indices(self):
        return self.region_indices_dict


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            return src_size[0]
        else:
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


if __name__ == '__main__':
    pass