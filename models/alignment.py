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
from transformers import Dinov2Model


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
                 source_projector_ckpt=None, freeze_source_projector=True,
                 adversarial_weight=0.0, num_sessions=256,
                 equivariance_weight=0.0, info_max_weight=0.0,
                 alignment_weight=1.0,
                 patch_embed_type='cnn',
                 mamba_band_periods=None, n_mamba_layers=2,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 use_llm_vq=False, num_language_tokens=8,
                 max_llm_codebook_size=4096, llm_vq_aux_weight=0.1):
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

        if patch_embed_type == 'mamba':
            self.patch_embedding = MambaPatchEmbedding(
                in_dim, out_dim, d_model, seq_len,
                band_periods=mamba_band_periods,
                n_mamba_layers=n_mamba_layers,
                d_state=mamba_d_state, d_conv=mamba_d_conv,
                expand=mamba_expand, causal=causal,
            )
        elif patch_embed_type == 'cnn':
            self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len, causal=causal)
        else:
            raise ValueError(f"Unknown patch_embed_type: {patch_embed_type}")

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
            nn.Linear(d_model, 768),
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


    # ------------------------------------------------------------------
    # Encode — lightweight forward returning CLS + patch tokens
    # ------------------------------------------------------------------

    def encode(self, batch, mask=None):
        """Encoder-only pass returning CLS token and patch-level tokens.

        Mirrors the main ``forward`` pipeline but skips contrastive
        learning, reconstruction output, equivariance, and adversarial
        losses.  Used by the DINOv2 teacher and by the equivariance
        loss helper.

        Returns:
            cls_token:    (B, d_model)
            patch_tokens: (B, C, N, d_model) — global tokens stripped
        """
        x = batch['timeseries']
        ch_coords = batch['ch_coords']
        valid_channel_mask = batch.get('valid_channel_mask')
        valid_length_mask = batch.get('valid_length_mask')

        # --- source projection ---
        if self.project_to_source:
            if mask is not None:
                x_in = x.clone()
                x_in[mask == 1] = 0.0
                x = self.source_projector(x_in, ch_coords,
                                          valid_channel_mask=valid_channel_mask)
            else:
                x = self.source_projector(x, ch_coords,
                                          valid_channel_mask=valid_channel_mask)
            mask = None  # already applied

        # --- spectral filtering ---
        if not self.causal:
            ts = x.view(x.size(0), x.size(1), -1)
            spectral = torch.fft.rfft(ts, dim=-1, norm='forward')
            freq_mask = torch.sigmoid(
                self.freq_mask_logits[:1, :spectral.size(-1)])
            ts = torch.fft.irfft(spectral * freq_mask.unsqueeze(0),
                                 norm='forward')
            x = ts.view_as(x)

        # --- patch embedding ---
        patch_emb = self.patch_embedding(x, mask)

        # --- coordinate enhancement / source-space bookkeeping ---
        if self.project_to_source:
            b = patch_emb.size(0)
            vcm = torch.ones(b, self.source_projector.num_sources,
                             dtype=torch.bool, device=patch_emb.device)
        else:
            coord_pe, _ = self._spherical_positional_encoding(ch_coords)
            coord_emb = self.coord_enhancement(coord_pe)
            patch_emb = patch_emb + coord_emb.unsqueeze(2)
            vcm = valid_channel_mask
        vlm = valid_length_mask
        if self.patch_embed_type == 'mamba':
            vlm = self.patch_embedding.adapt_valid_length_mask(vlm)

        # --- global tokens ---
        if self.add_global:
            patch_emb = torch.cat([
                self.global_channel.expand(
                    patch_emb.size(0), -1, patch_emb.size(2), -1),
                patch_emb], dim=1)
            patch_emb = torch.cat([
                self.global_token.expand(
                    patch_emb.size(0), patch_emb.size(1), -1, -1),
                patch_emb], dim=2)
            if vcm is not None:
                vcm = torch.cat([
                    torch.ones_like(vcm[:, :1], dtype=torch.bool), vcm
                ], dim=1)
            if vlm is not None:
                vlm = torch.cat([
                    torch.ones_like(vlm[:, :1], dtype=torch.bool), vlm
                ], dim=1)

        # --- transformer encoder (no contrastive branch) ---
        for layer_idx in range(self.encoder.num_layers):
            patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
            patch_emb = self.encoder.layers[layer_idx](
                patch_emb, self.area_config,
                inter_window_attn_mask=(
                    ~vlm if vlm is not None else None),
                inter_region_attn_mask=(
                    ~vcm if vcm is not None else None),
            )

        cls_token = (patch_emb[:, 0, 0, :]
                     if self.add_global
                     else patch_emb.mean(dim=(1, 2)))
        patch_tokens = (patch_emb[:, 1:, 1:, :]
                        if self.add_global
                        else patch_emb)
        return cls_token, patch_tokens

    # ------------------------------------------------------------------
    # Hemispheric equivariance helpers
    # ------------------------------------------------------------------

    def _encode_to_global_rep(self, x, ch_coords,
                              valid_channel_mask=None,
                              valid_length_mask=None,
                              mask=None):
        """Return only the global-token representation (backward compat)."""
        _batch = {'timeseries': x, 'ch_coords': ch_coords}
        if valid_channel_mask is not None:
            _batch['valid_channel_mask'] = valid_channel_mask
        if valid_length_mask is not None:
            _batch['valid_length_mask'] = valid_length_mask
        cls_token, _ = self.encode(_batch, mask=mask)
        return cls_token

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

    def forward(self, batch, mask=None):
        x = batch['timeseries'] # (B, n_ch, seq_len, in_dim)
        ch_coords = batch['ch_coords']

        # Save originals before the batch dict gets mutated by global-token
        # concatenation — needed for the equivariance second pass.
        _orig_vcm = batch.get('valid_channel_mask')
        _orig_vlm = batch.get('valid_length_mask')
        _orig_mask = mask

        # --- Apply mask in sensor space, then project to source space ---
        if self.project_to_source:
            raise NotImplementedError("Source projection with patch-based masking is not currently supported.")
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

        ## DEPRECATED spectral branch
        # if not self.causal:
        #     # Spectral filtering mixes all time steps — skip in causal mode
        #     timeseries = x.view(x.size(0), x.size(1), -1)
        #     spectral = torch.fft.rfft(timeseries, dim=-1, norm='forward')[..., :self.freq_mask_logits.size(-1)]
        #     masks = torch.sigmoid(self.freq_mask_logits[:1, :spectral.size(-1)]) # (num_visual_levels, n_freq_bins)
        #     branch_spectral = spectral * masks.unsqueeze(0) # (n_branch, B, n_ch, n_freq_bins)
        #     branch_time = torch.fft.irfft(branch_spectral, norm='forward', n=timeseries.size(-1)) # (n_branch, B, n_ch, seq_len)
        #     x = branch_time.view_as(x)

        # Remember base N before patch embedding so we can map the
        # interleaved multi-band tokens back to per-patch reconstructions.
        base_N = x.size(2)
        patch_emb = self.patch_embedding(x, mask)

        if self.patch_embed_type == 'mamba' and 'valid_length_mask' in batch:
            batch['valid_length_mask'] = self.patch_embedding.adapt_valid_length_mask(
                batch['valid_length_mask'])

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
                if 'image_encoder_inputs' in batch and has_image.any() and self.alignment_weight > 0:
                    image_encoder_inputs = {k: v[has_image] for k, v in batch['image_encoder_inputs'].items()}
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

                    contrastive_loss[f"contrastive_loss_{i_branch}"] = (self.alignment_weight, branch_loss)
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

        # Extract global token and patch-level representations
        global_rep = patch_emb[:, 0, 0, :] if self.add_global else patch_emb.mean(dim=(1, 2))
        patch_tokens = patch_emb[:, 1:, 1:, :] if self.add_global else patch_emb

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

        return out, {
            "rep": rep,
            "global_rep": global_rep,
            "patch_tokens": patch_tokens,
            # "zero_lag_sync_loss": (0.1, zero_lag_sync_loss),
            **contrastive_loss,
            **equiv_losses,
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
