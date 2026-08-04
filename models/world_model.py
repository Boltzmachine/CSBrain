"""EEG world-model / video-prediction extension of ``CSBrainAlign``.

See ``plans/world_model.md`` for the design rationale. Two modules live here:

* :class:`LatentPredictor` — a small transformer that predicts the latent
  patch tokens of the next EEG window from the current latent tokens.
* :class:`WorldModelWrapper` — wraps an existing ``CSBrainAlign`` encoder,
  composes alignment + masked reconstruction + latent prediction into a
  single loss dict, and follows the ``(weight, tensor)`` convention that the
  pretraining trainer already understands.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import generate_mask
from models.lewm_modules import SIGReg


# Dataset-level mean per-patch L1 motion between the window-0 anchor grid and each
# of the H future frame grids — the PER-HORIZON reference that normalises the
# per-patch weighting of the dense frame-prediction loss
# (``WorldModelWrapper._motion_weight``). Entry ``τ-1`` is the mean of
# ``mean_d |grid[k+τ·train_step] - grid[k]|`` over all anchors/subjects.
#
# Why PER-STEP: motion grows monotonically with the horizon (scene drift
# accumulates), so a single scalar would make the "static" floor threshold
# (``floor * ref``) too strict at short lags and too loose at long lags. A
# per-horizon reference makes that threshold lag-appropriate. Why FIXED: the
# per-batch mean jitters with batch composition + motion_resample, so the
# static/dynamic boundary would drift during training.
#
# Precomputed by scripts/compute_frame_motion_ref.py over ALL 24 EgoBrain subjects
# (60k anchor×horizon samples) for the facebook/dinov2-base g0.2 sz224 grid cache
# with stride 0.2s / max_horizon 5 (train_step=1). RECOMPUTE and update this if
# you change the vision encoder, grid_s, stride_s, or max_horizon — the scale is
# encoder- and window-config-specific. Index by horizon τ=1..H; a run with H<5
# uses the first H entries. (scalar mean over all steps ≈ 0.564.) [2026-07-04]
EGOBRAIN_FRAME_MOTION_REF_PER_STEP = (0.4215, 0.5227, 0.5842, 0.6280, 0.6628)


# ``frame_clean_cond`` two-view split (see WorldModelWrapper.training_step). The
# MASKED view is the reconstruction pretext ONLY; the CLEAN (unmasked) view
# carries everything downstream-facing (alignment, flip-align, hand, prediction).
# Batch keys stripped from the recon (masked) forward so the encoder SKIPS the
# alignment / flip-align / hand branches (they gate on these keys) — recon needs
# none of them, and dropping them saves that compute on the masked pass.
_RECON_ONLY_STRIP = frozenset({
    'image_encoder_inputs', 'has_image', 'frame_cls', 'frame_cls_flip',
    'frame_grid', 'frame_grid_flip', 'hand_targets', 'hand_valid', 'flip_motion'})
# info keys taken from the MASKED view (reconstruction-related); everything else
# in the returned info comes from the CLEAN view.
_RECON_VIEW_INFO_KEYS = frozenset({
    'frame_recon_loss', 'skip_external_recon', 'flip_row'})


# ---------------------------------------------------------------------------
# Latent predictor
# ---------------------------------------------------------------------------

class LatentPredictor(nn.Module):
    """Predict ``ŝ_{t+k}^{patch}`` from ``s_t`` latent tokens.

    Latents are flattened into a token sequence with a learned horizon
    embedding added. We keep the network intentionally narrow (V-JEPA
    design choice) so the encoder — not the predictor — carries the
    representational load.
    """

    def __init__(
        self,
        d_model: int,
        predictor_d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_horizon: int = 8,
        max_tokens: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.predictor_d_model = predictor_d_model

        self.in_proj_latent = nn.Linear(d_model, predictor_d_model)

        self.horizon_embed = nn.Embedding(max_horizon + 1, predictor_d_model)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_tokens, predictor_d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(predictor_d_model)

        # Two readout heads: one for the patch-token target (primary), one
        # for the CLS-token target used in the auxiliary cross-modal term.
        self.out_proj_patch = nn.Linear(predictor_d_model, d_model)
        self.out_proj_cls = nn.Linear(predictor_d_model, d_model)

    def forward(
        self,
        s_t_patch: torch.Tensor,   # (B, C, N, D)
        horizon: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (ŝ_{t+k}^{patch}, ŝ_{t+k}^{cls})."""
        B, C, N, D = s_t_patch.shape

        tokens = self.in_proj_latent(s_t_patch.reshape(B, C * N, D))
        seq_len = tokens.size(1)
        assert seq_len <= self.pos_embed.size(1), (
            f"LatentPredictor received sequence length {seq_len}, larger "
            f"than max_tokens={self.pos_embed.size(1)}")
        tokens = tokens + self.pos_embed[:, :seq_len]

        k = torch.tensor(
            min(horizon, self.horizon_embed.num_embeddings - 1),
            device=tokens.device, dtype=torch.long)
        tokens = tokens + self.horizon_embed(k).view(1, 1, -1)

        h = self.encoder(tokens)
        h = self.norm_out(h)

        pred_patch = self.out_proj_patch(h).reshape(B, C, N, D)
        # CLS readout: mean-pool over the latent tokens.
        pred_cls = self.out_proj_cls(h.mean(dim=1))
        return pred_patch, pred_cls


# ---------------------------------------------------------------------------
# Frame predictor (cross-modal objective)
# ---------------------------------------------------------------------------

class FramePredictor(nn.Module):
    """Dense temporal frame decoder (cross-modal world-model objective).

    Predicts the per-patch grids of the next ``H = max_horizon`` frames — the
    co-occurring 0.2 s frame grid spanning the current 1 s EEG window — from the
    anchor (window-0) frame grid, conditioned on the current window's EEG
    tokens. ONE forward emits all ``H`` grids so every 0.2 s step is supervised
    (dense), replacing the old single-``horizon`` predictor.

    Tokens fed to a plain (bidirectional) transformer encoder:

      * EEG      ``(B, M, eeg_dim)``    — current-window tokens (``M = C*N``) or a
                                          single global vector (``M = 1``)
      * anchor   ``(B, P, frame_dim)``  — the window-0 frozen vision grid
      * queries  ``(B, H*P, .)``        — one learned slot per (step τ, patch p)

    The EEG backbone is NOT causal, so there is no temporal mask: every query
    may attend to every EEG token of the segment (and to the anchor and the
    other queries). Query slot ``(τ, p)`` carries a per-step temporal embedding
    ``τ`` and a spatial embedding ``p`` shared with the anchor grid, so it reads
    off the predicted grid at that time/patch. Targets are frozen vision-encoder
    grids, so there is no collapse and no EMA teacher.

    Output: ``(B, H, P, frame_dim)``. With ``pred_residual`` (default) the head
    predicts the delta from the anchor grid (``ŝ_τ = s_0 + Δ_τ``), so a zero
    output copies the scene and the EEG only has to supply motion.
    """

    def __init__(
        self,
        frame_dim: int,
        eeg_dim: int,
        predictor_d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_horizon: int = 5,
        max_patches: int = 1024,
        pred_residual: bool = True,
    ):
        super().__init__()
        self.frame_dim = frame_dim
        self.eeg_dim = eeg_dim
        self.predictor_d_model = predictor_d_model
        self.max_horizon = int(max_horizon)
        self.pred_residual = bool(pred_residual)

        self.in_proj_frame = nn.Linear(frame_dim, predictor_d_model)
        self.in_proj_eeg = nn.Linear(eeg_dim, predictor_d_model)

        # Learned spatial position embedding, SHARED by the anchor grid and the
        # query grid (same P-patch layout) so query patch p aligns to anchor p.
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, max_patches, predictor_d_model))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        # Per-step (0.2 s) temporal embedding for the H query timesteps.
        self.step_pos = nn.Parameter(
            torch.zeros(1, self.max_horizon, 1, predictor_d_model))
        nn.init.trunc_normal_(self.step_pos, std=0.02)
        # Learned base query token, broadcast to every (step, patch) slot.
        self.query_token = nn.Parameter(
            torch.zeros(1, 1, 1, predictor_d_model))
        nn.init.trunc_normal_(self.query_token, std=0.02)
        # Type embeddings distinguishing the three token groups in the shared
        # attention stack.
        self.eeg_type = nn.Parameter(torch.zeros(1, 1, predictor_d_model))
        self.anchor_type = nn.Parameter(torch.zeros(1, 1, predictor_d_model))
        self.query_type = nn.Parameter(torch.zeros(1, 1, 1, predictor_d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(predictor_d_model)
        self.out_proj = nn.Linear(predictor_d_model, frame_dim)

    def forward(
        self,
        s_anchor: torch.Tensor,    # (B, P, frame_dim) — window-0 anchor grid
        eeg_emb: torch.Tensor,     # (B, eeg_dim) global OR (B, M, eeg_dim) tokens
        eeg_key_padding_mask: Optional[torch.Tensor] = None,  # (B, M) True=ignore
    ) -> torch.Tensor:
        """Return the dense grid stack ``(B, H, P, frame_dim)``.

        ``eeg_emb`` may be a single global vector ``(B, eeg_dim)`` or a set of
        per-patch EEG tokens ``(B, M, eeg_dim)``; both are handled.
        ``eeg_key_padding_mask`` (``True`` == padded/ignore) drops invalid EEG
        tokens from the attention. Anchor + query tokens are never padded, so no
        attention row is ever fully masked.
        """
        B, P, _ = s_anchor.shape
        assert P <= self.spatial_pos.size(1), (
            f"FramePredictor received {P} patches, larger than "
            f"max_patches={self.spatial_pos.size(1)}")

        # Normalise the EEG conditioning to a token set (global -> 1 token).
        if eeg_emb.dim() == 2:
            eeg_emb = eeg_emb.unsqueeze(1)                        # (B, 1, eeg_dim)
        M = eeg_emb.size(1)
        H, D = self.max_horizon, self.predictor_d_model

        anchor = (self.in_proj_frame(s_anchor)
                  + self.spatial_pos[:, :P] + self.anchor_type)  # (B, P, D)
        eeg = self.in_proj_eeg(eeg_emb) + self.eeg_type          # (B, M, D)

        # Query grid: (1, H, P, D) via broadcast of the base token + step + patch
        # embeddings, then flattened to (B, H*P, D).
        q = (self.query_token
             + self.spatial_pos[:, :P].unsqueeze(1)              # (1, 1, P, D)
             + self.step_pos                                     # (1, H, 1, D)
             + self.query_type)                                  # (1, 1, 1, D)
        q = q.expand(B, H, P, D).reshape(B, H * P, D)            # (B, H*P, D)

        tokens = torch.cat([eeg, anchor, q], dim=1)              # (B, M+P+H*P, D)

        kpm = None
        if eeg_key_padding_mask is not None:
            rest = torch.zeros(
                B, P + H * P, dtype=torch.bool, device=tokens.device)
            kpm = torch.cat([eeg_key_padding_mask, rest], dim=1)  # (B, M+P+H*P)

        h = self.norm_out(self.encoder(tokens, src_key_padding_mask=kpm))
        q_out = h[:, M + P:].reshape(B, H, P, D)                 # drop EEG + anchor
        out = self.out_proj(q_out)                               # (B, H, P, frame_dim)
        if self.pred_residual:
            out = out + s_anchor.unsqueeze(1)                    # skip from anchor
        return out


# ---------------------------------------------------------------------------
# Frame predictor with AdaLN-zero EEG conditioning (wm_predictor='frame_adaln')
# ---------------------------------------------------------------------------

class FrameAdaLNPredictor(nn.Module):
    """Dense multi-frame predictor with LeWM-style AdaLN-zero EEG conditioning.

    Same OBJECTIVE and I/O contract as :class:`FramePredictor` — predict the next
    ``H = max_horizon`` per-patch frame grids from the anchor (current) frame grid,
    ``(B, H, P, frame_dim)`` — so it drops straight into ``_frame_prediction_step``
    (dense, from-the-current-frame, NOT autoregressive). The difference is HOW the
    EEG is injected:

      * FramePredictor : the EEG tokens are CONCATENATED with anchor + learned
        query tokens and everything attends to everything (query-transformer style).
      * this predictor : the EEG is a CONDITIONING signal that modulates each
        transformer block via AdaLN-zero (shift / scale / gate), exactly like
        LeWorldModel's ``ConditionalBlock`` (there the action does this). The EEG
        token set is reduced to one vector per sample (masked mean over the valid
        tokens, or the global rep) and combined with a per-horizon-step embedding,
        so step tau's patch tokens get EEG-derived modulation specific to that step.

    The frame tokens are the anchor grid replicated over the H steps (+ spatial and
    step position embeddings) and attention is BIDIRECTIONAL (every query patch may
    attend to every other). AdaLN-zero init => at start the blocks are the identity
    and, with the anchor residual, the prediction copies the anchor; the EEG-driven
    modulation then has to supply the motion (mirrors FramePredictor's residual).
    """

    def __init__(
        self,
        frame_dim: int,
        eeg_dim: int,
        predictor_d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_horizon: int = 5,
        max_patches: int = 1024,
        pred_residual: bool = True,
    ):
        super().__init__()
        from models.lewm_modules import ConditionalBlock
        D = predictor_d_model
        self.frame_dim = frame_dim
        self.eeg_dim = eeg_dim
        self.predictor_d_model = D
        self.max_horizon = int(max_horizon)
        self.pred_residual = bool(pred_residual)

        self.in_proj_frame = nn.Linear(frame_dim, D)
        self.eeg_proj = nn.Linear(eeg_dim, D)                # EEG summary -> cond

        # Spatial (patch) + step (horizon) position embeddings for the frame
        # tokens, and a per-step conditioning offset added to the EEG cond vector.
        self.spatial_pos = nn.Parameter(torch.zeros(1, max_patches, D))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        self.step_pos = nn.Parameter(torch.zeros(1, self.max_horizon, 1, D))
        nn.init.trunc_normal_(self.step_pos, std=0.02)
        self.step_cond = nn.Parameter(torch.zeros(1, self.max_horizon, D))
        nn.init.trunc_normal_(self.step_cond, std=0.02)

        dim_head = max(1, D // n_heads)
        self.blocks = nn.ModuleList([
            ConditionalBlock(D, heads=n_heads, dim_head=dim_head,
                             mlp_dim=dim_feedforward, dropout=dropout, causal=False)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, frame_dim)

    def forward(
        self,
        s_anchor: torch.Tensor,    # (B, P, frame_dim) — window-0 anchor grid
        eeg_emb: torch.Tensor,     # (B, eeg_dim) global OR (B, M, eeg_dim) tokens
        eeg_key_padding_mask: Optional[torch.Tensor] = None,  # (B, M) True=ignore
    ) -> torch.Tensor:
        """Return the dense grid stack ``(B, H, P, frame_dim)``."""
        B, P, _ = s_anchor.shape
        assert P <= self.spatial_pos.size(1), (
            f"FrameAdaLNPredictor received {P} patches, larger than "
            f"max_patches={self.spatial_pos.size(1)}")
        H, D = self.max_horizon, self.predictor_d_model

        # EEG -> single conditioning vector (masked mean over valid tokens, or the
        # global vector), then a per-step conditioning cond[:, tau] = eeg + step_cond.
        if eeg_emb.dim() == 3:
            if eeg_key_padding_mask is not None:
                v = (~eeg_key_padding_mask).to(eeg_emb.dtype).unsqueeze(-1)  # (B,M,1)
                eeg_vec = (eeg_emb * v).sum(1) / v.sum(1).clamp(min=1.0)
            else:
                eeg_vec = eeg_emb.mean(1)
        else:
            eeg_vec = eeg_emb
        eeg_c = self.eeg_proj(eeg_vec)                        # (B, D)
        cond = eeg_c.unsqueeze(1) + self.step_cond           # (B, H, D)
        cond = cond.unsqueeze(2).expand(B, H, P, D).reshape(B, H * P, D)

        # Frame tokens: anchor replicated over H steps + spatial + step pos.
        anchor = self.in_proj_frame(s_anchor)                # (B, P, D)
        x = anchor.unsqueeze(1).expand(B, H, P, D)
        x = x + self.spatial_pos[:, :P].unsqueeze(1) + self.step_pos  # (B,H,P,D)
        x = x.reshape(B, H * P, D)

        for blk in self.blocks:
            x = blk(x, cond)                                  # AdaLN-zero EEG cond
        x = self.norm(x)
        out = self.out_proj(x).reshape(B, H, P, self.frame_dim)
        if self.pred_residual:
            out = out + s_anchor.unsqueeze(1)                 # copy-the-anchor prior
        return out


# ---------------------------------------------------------------------------
# LeWM autoregressive world-model head (wm_predictor='ar')
# ---------------------------------------------------------------------------

class ARWorldModel(nn.Module):
    """LeWorldModel (LeWM) autoregressive head — the faithful port.

    Bundles the LeWM building blocks (see models/lewm_modules.py) for the
    causal next-embedding objective:

      * ``projector``      — frame CLS (trainable-ViT hidden) -> embed_dim (BN-MLP)
      * ``action_encoder`` — per-window EEG global rep -> action embedding
      * ``predictor``      — causal AR transformer, AdaLN-zero conditioned on the
                             action (LeWM ``ARPredictor``)
      * ``pred_proj``      — predictor output -> embed_dim (BN-MLP)

    The EEG plays LeWM's *action* role: window ``t``'s EEG conditions the
    prediction of frame ``t+1`` from frame ``t``, so the AR prediction loss
    trains the EEG foundation model jointly with the world model. The wrapper
    drives these in ``WorldModelWrapper._ar_prediction_step``.
    """

    def __init__(self, frame_dim, eeg_dim, embed_dim=192, act_dim=None,
                 history=3, num_preds=1, depth=6, heads=16, mlp_dim=2048,
                 dim_head=64, dropout=0.1, proj_hidden=2048):
        super().__init__()
        from models.lewm_modules import ARPredictor, Embedder, MLP
        act_dim = int(act_dim) if act_dim else int(embed_dim)
        self.embed_dim = int(embed_dim)
        self.history = int(history)
        self.num_preds = int(num_preds)
        # LeWM projectors use BatchNorm1d (config/train/model/lewm.yaml).
        self.projector = MLP(frame_dim, proj_hidden, embed_dim,
                             norm_fn=nn.BatchNorm1d)
        self.pred_proj = MLP(embed_dim, proj_hidden, embed_dim,
                             norm_fn=nn.BatchNorm1d)
        self.action_encoder = Embedder(
            input_dim=eeg_dim, smoothed_dim=eeg_dim, emb_dim=act_dim)
        self.predictor = ARPredictor(
            num_frames=history, depth=depth, heads=heads, mlp_dim=mlp_dim,
            input_dim=embed_dim, hidden_dim=embed_dim, output_dim=embed_dim,
            dim_head=dim_head, dropout=dropout)
        # Uniform interface: AR consumes T = history + num_preds frame windows;
        # expose max_horizon so the wrapper's W>=max_horizon+1 style checks hold.
        self.max_horizon = self.history + self.num_preds - 1

    def project_frames(self, cls):
        """(B, T, frame_dim) -> (B, T, embed_dim)."""
        B, T, d = cls.shape
        return self.projector(cls.reshape(B * T, d)).reshape(B, T, self.embed_dim)

    def encode_action(self, eeg_global):
        """(B, T, eeg_dim) -> (B, T, act_dim)."""
        return self.action_encoder(eeg_global)

    def predict(self, ctx_emb, ctx_act):
        """(B, H, embed), (B, H, act) -> (B, H, embed)."""
        pred = self.predictor(ctx_emb, ctx_act)
        B, H, d = pred.shape
        return self.pred_proj(pred.reshape(B * H, d)).reshape(B, H, self.embed_dim)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class WorldModelWrapper(nn.Module):
    """Compose ``CSBrainAlign`` with a latent predictor and its losses.

    The wrapper is invoked by the pretraining trainer exactly like a
    standard model: ``training_step(batch, mask=None)`` returns
    ``(out, info)`` where ``out`` is the masked-patch reconstruction and
    ``info`` carries every loss term with a ``(weight, tensor)`` tuple so
    the trainer can sum them.
    """

    def __init__(
        self,
        encoder: nn.Module,                      # CSBrainAlign instance
        predictor: Optional[LatentPredictor],    # None when max_horizon=0
        latent_pred_weight: float = 1.0,
        cls_pred_weight: float = 0.1,
        max_horizon: int = 1,
        ramp_epochs: int = 2,
        target_momentum: float = 0.998,
        flip_pred_weight: float = 1.0,
        objective: str = 'eeg',
        frame_eeg_cond: str = 'global',
        frame_motion_alpha: float = 0.0,
        frame_motion_floor: float = 0.1,
        frame_motion_ref: float = -1.0,
        frame_clean_cond: bool = False,
        frame_contrast_weight: float = 0.0,
        frame_contrast_n_neg: int = 1,
        frame_contrast_temp: float = 0.1,
        frame_contrast_mode: str = 'infonce',
        frame_contrast_margin: float = 0.1,
        frame_contrast_detach_neg: bool = True,
        frame_contrast_batched: bool = False,
        frame_contrast_excl_samples: int = 0,
        scratch_vit: bool = False,
        sigreg_weight: float = 0.0,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
        wm_predictor: str = 'frame',
    ):
        super().__init__()
        assert objective in ('eeg', 'frame'), (
            f"objective must be 'eeg' or 'frame', got {objective!r}")
        assert wm_predictor in ('frame', 'ar', 'frame_adaln'), (
            f"wm_predictor must be 'frame', 'frame_adaln' or 'ar', got {wm_predictor!r}")
        assert frame_eeg_cond in ('global', 'tokens'), (
            f"frame_eeg_cond must be 'global' or 'tokens', got {frame_eeg_cond!r}")
        assert frame_motion_alpha >= 0.0, (
            f"frame_motion_alpha must be >= 0, got {frame_motion_alpha}")
        assert 0.0 <= frame_motion_floor <= 1.0, (
            f"frame_motion_floor must be in [0, 1], got {frame_motion_floor}")
        assert frame_contrast_weight >= 0.0, (
            f"frame_contrast_weight must be >= 0, got {frame_contrast_weight}")
        assert frame_contrast_mode in ('infonce', 'margin'), (
            f"frame_contrast_mode must be 'infonce' or 'margin', got "
            f"{frame_contrast_mode!r}")
        assert frame_contrast_temp > 0.0, (
            f"frame_contrast_temp must be > 0, got {frame_contrast_temp}")
        assert frame_contrast_n_neg >= 1, (
            f"frame_contrast_n_neg must be >= 1, got {frame_contrast_n_neg}")
        # Per-patch motion weighting for the DENSE frame objective (see
        # ``_motion_weight`` / ``_frame_prediction_step``). ``alpha == 0`` keeps
        # the legacy uniform mean over patches (byte-identical); ``alpha > 0``
        # up-weights each patch by how far its target moved from the anchor grid,
        # so the loss concentrates on EEG-explainable motion instead of the
        # near-static scene the copy-the-anchor shortcut already reproduces.
        self.frame_motion_alpha = float(frame_motion_alpha)
        self.frame_motion_floor = float(frame_motion_floor)
        # Reference scale for the weight normalisation ``clip(motion/ref, floor)``.
        # ``< 0`` (default) -> the FIXED baked-in per-horizon
        # ``EGOBRAIN_FRAME_MOTION_REF_PER_STEP`` (stable, lag-appropriate floor
        # threshold); ``== 0`` -> per-batch per-step mean (legacy dynamic,
        # jitters); ``> 0`` -> this explicit scalar applied to every step
        # (override for a different encoder/window config). Resolved by
        # ``_resolved_motion_ref``.
        self.frame_motion_ref = float(frame_motion_ref)
        # Frame-objective two-view split. The masked forward (needed for the recon
        # objective) feeds EEG tokens that are half mask-token reconstruction
        # guesses AND intermittently masked, so the predictor / alignment learn to
        # ignore the EEG (diag_frame_eeg_gap ~0). When True, run a second UNMASKED
        # forward and route the objectives by view (MAE-world-model style): the
        # masked view is the reconstruction pretext ONLY (mask_loss + spectral aux
        # + frame_recon); the clean view carries everything downstream-facing —
        # image alignment, flip-align, hand-pred, and the frame prediction's EEG
        # conditioning — on the complete signal. Masked recon is untouched. Costs
        # one extra (cheap) EEG-encoder forward. Default off (legacy: single masked
        # forward carries all losses; predictor conditions on the masked tokens).
        self.frame_clean_cond = bool(frame_clean_cond)
        # For the frame objective, condition the predictor on either the EEG
        # window-level global rep (``'global'``) or the full per-patch EEG token
        # set (``'tokens'``, M = C*N tokens). Ignored for the EEG objective.
        self.frame_eeg_cond = frame_eeg_cond
        # Negative-EEG contrastive term (see ``_add_frame_contrast``). The dense
        # frame predictor learns to COPY the anchor grid and ignore the EEG
        # (diag_frame_eeg_gap ~0). This term re-runs the predictor on the SAME
        # anchor grid conditioned on OTHER rows' EEG (in-batch negatives) and
        # penalises the predictor for reproducing the true future from the wrong
        # EEG — so the copy shortcut can no longer win and the predictor is forced
        # to read the motion out of the EEG. ``weight == 0`` (default) is a no-op:
        # the whole block (and its K extra predictor forwards) is skipped, so
        # existing runs are byte-identical. Frame objective only.
        #   * mode 'infonce' — softmax over {positive, K negatives} on -distance;
        #     the correct EEG must give the smallest prediction distance.
        #   * mode 'margin'  — softplus(margin + d_pos - d_neg): each negative's
        #     distance must exceed the positive's by ``margin``.
        # Negatives are OTHER rows' EEG conditioning drawn from rows sharing the
        # anchor's frame-averaging flip (so the predictor can't discriminate on
        # orientation instead of motion; see _same_flip_neg_indices). With
        # ``detach_neg`` (default) the negative EEG is detached so ONLY the
        # predictor (not the EEG encoder) learns the discrimination from the
        # negatives, while the encoder is still pushed — through the positive term
        # of the softmax — to make its OWN EEG the most predictive. This keeps the
        # arbitrary (eeg_j, scene_i) pairings from injecting noise into the
        # downstream-facing EEG backbone. NOTE the negatives are still DIFFERENT
        # scenes, so the term can be partly satisfied by scene identity rather than
        # motion — BEST PAIRED with frame_motion_alpha > 0:
        # the motion weighting focuses the contrastive distance on DYNAMIC patches,
        # otherwise the objective can be satisfied by encoding static SCENE IDENTITY
        # into the EEG (the anchor already carries that, but a scene-match detector
        # would still lower the loss without learning motion).
        self.frame_contrast_weight = float(frame_contrast_weight)
        self.frame_contrast_n_neg = int(frame_contrast_n_neg)
        self.frame_contrast_temp = float(frame_contrast_temp)
        self.frame_contrast_mode = frame_contrast_mode
        self.frame_contrast_margin = float(frame_contrast_margin)
        self.frame_contrast_detach_neg = bool(frame_contrast_detach_neg)
        # Run all K negative predictor forwards as ONE batched call over K*Bv rows
        # (fewer/larger kernels) instead of a K-step Python loop. Numerically
        # identical to the loop (up to dropout RNG in train), but raises peak
        # attention memory ~K x — this flag exists to test whether the GPU has the
        # capacity for it. Default off = the memory-safe loop.
        self.frame_contrast_batched = bool(frame_contrast_batched)
        # Temporal exclusion (window-0 start samples) between a subject-block row
        # and its in-batch negatives; 0 = no exclusion (block gating off or the
        # caller supplies no ``anchor``). See ``_same_flip_neg_indices``.
        self.frame_contrast_excl_samples = int(frame_contrast_excl_samples)
        # ``'eeg'`` (default): predict the next EEG window's latent from the
        # current EEG latent (the original world-model objective). ``'frame'``:
        # predict the next video frame's per-patch embedding from the current
        # frame's per-patch embedding, conditioned on the current EEG embedding.
        # The choice only swaps the predictor objective; the encoder's masked
        # reconstruction + image alignment + aux terms are untouched either way.
        self.objective = objective
        self.encoder = encoder
        # ---- LeWM from-scratch world model (scratch_vit) ----
        # ``scratch_vit`` == the encoder owns a TRAINABLE from-scratch ViT
        # (CSBrainAlign.vision_trainable). On this path the frame-prediction
        # target is the ViT's OWN online embedding (no stop-grad, no EMA — see
        # _frame_prediction_step / _ar_prediction_step), and collapse is held
        # off SOLELY by SIGReg, the single LeWM regularizer. ``wm_predictor``
        # selects the predictor: 'frame' reuses the dense EEG-conditioned
        # FramePredictor; 'ar' uses the LeWM causal ARWorldModel head.
        self.scratch_vit = bool(scratch_vit)
        self.wm_predictor = wm_predictor
        self.sigreg_weight = float(sigreg_weight)
        # Build SIGReg ONLY on the scratch path. It is never used off it (the
        # SIGReg loss terms are gated on the encoder's vision_trainable / the AR
        # predictor), and gating construction here keeps the frozen path's
        # state_dict + parameters byte-identical even though --wm_sigreg_weight
        # defaults to 0.09 (so a scratch run is collapse-safe by default).
        self.sigreg = (SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)
                       if (sigreg_weight > 0 and self.scratch_vit) else None)
        # Multiplier on the bilateralization flipped-prediction terms (predict
        # the flipped-future EEG latent from the flipped-current). Only active
        # when the encoder has the learned x_bi/x_lat split (lateralization_flip)
        # and a predictor; 0 disables it.
        self.flip_pred_weight = float(flip_pred_weight)
        # ``max_horizon == 0`` reduces the wrapper to the plain CSBrainAlign
        # pipeline (masked recon + image alignment). In that mode no
        # predictor is built at all, so no extra parameters enter the
        # optimiser and no future windows are consumed.
        self.predictor = predictor
        self.latent_pred_weight = latent_pred_weight
        self.cls_pred_weight = cls_pred_weight
        self.max_horizon = max_horizon
        self.ramp_epochs = ramp_epochs
        self.target_momentum = target_momentum
        # The trainer writes ``current_epoch`` before each epoch so the
        # wrapper can ramp its loss weights without needing its own hook.
        self.register_buffer(
            'current_epoch', torch.tensor(0.0), persistent=False)

        # EMA target encoder — used only when the predictor is active.
        # Why: the regression target ``s_{t+k}`` is a function of the
        # encoder weights; if it's computed by the online encoder, every
        # optimizer step shifts the target in the same direction as the
        # online net just moved, so there's no fixed point for the
        # predictor to chase. The EMA copy keeps the target quasi-static
        # across steps, which is what actually anchors the dynamic
        # (mask-recon prevents trivial-constant collapse, but leaves a
        # drift direction in the null-space of reconstruction).
        #
        # Only the EEG-latent objective needs it: the ``'frame'`` objective
        # regresses against the *frozen* vision encoder, which is already a
        # fixed target, so no EMA copy is built (and none of its parameters
        # enter the optimiser).
        if (self.objective == 'eeg'
                and self.predictor is not None and self.max_horizon >= 1):
            self.target_encoder = copy.deepcopy(encoder)
            # ``encode()`` does not invoke the DINOv2 image encoder, so
            # drop it from the target to avoid doubling that memory.
            if hasattr(self.target_encoder, 'pretrained_image_encoder'):
                del self.target_encoder.pretrained_image_encoder
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            self.target_encoder.eval()
        else:
            self.target_encoder = None

    # ------------------------------------------------------------------

    def _pred_weight_scale(self) -> float:
        if self.ramp_epochs <= 0:
            return 1.0
        return float(min(1.0, self.current_epoch.item() / self.ramp_epochs))

    def _encode_future(self, batch_future: dict) -> tuple[torch.Tensor, torch.Tensor]:
        # forward() mutates its batch argument (global-token concat); pass
        # a shallow copy so the caller's ``batch`` dict is unaffected.
        # Route through the EMA target_encoder in eval() mode with
        # ``encoder_only=True`` — this (a) decouples the target from
        # same-step online updates, (b) suppresses dropout/BN updates in
        # the target path so the regression target is deterministic, and
        # (c) reuses the main forward pipeline so changes to the encoder
        # don't need a parallel update to a separate ``encode`` path.
        target = self.target_encoder if self.target_encoder is not None else self.encoder
        was_training = target.training
        target.eval()
        try:
            with torch.no_grad():
                _, info = target({**batch_future}, encoder_only=True)
            return info['global_rep'], info['patch_tokens']
        finally:
            if was_training:
                target.train()

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None) -> None:
        """EMA update of ``target_encoder`` from ``encoder``. Call after
        ``optimizer.step()``. No-op when no target encoder is configured.

        Iterates by name because ``target_encoder`` drops
        ``pretrained_image_encoder`` — zipping by positional order would
        mis-align parameters past that module.
        """
        if self.target_encoder is None:
            return
        m = self.target_momentum if momentum is None else float(momentum)
        online_params = dict(self.encoder.named_parameters())
        for name, p_target in self.target_encoder.named_parameters():
            p_online = online_params.get(name)
            if p_online is None:
                continue
            p_target.data.mul_(m).add_(p_online.data, alpha=1.0 - m)
        # Buffers (e.g. BN running stats if present) should track the
        # online encoder — copy directly, momentum is unnecessary.
        online_bufs = dict(self.encoder.named_buffers())
        for name, b_target in self.target_encoder.named_buffers():
            b_online = online_bufs.get(name)
            if b_online is None:
                continue
            b_target.data.copy_(b_online.data)

    def _build_future_subbatch(
        self,
        batch: dict,
        cb_idx: torch.Tensor,
        window_idx: int,
    ) -> dict:
        """Build an encoder batch for a future window, restricted to the
        rows in ``cb_idx`` (samples that carry a future stack).

        ``timeseries_future`` and friends in ``batch`` are expected to be
        already filtered to those rows (M = len(cb_idx)) — this matches
        ``collate_cached_with_future`` (mix mode) and ``collate_cinebrain``
        (pure CineBrain mode, where every row is a CineBrain row).

        Per-sample fields keyed on the full batch dimension B
        (``ch_coords``, masks, lists) are sliced via ``cb_idx``.
        """
        ts_f = batch['timeseries_future']  # (M, W, C, N, d)
        out = {
            'timeseries': ts_f[:, window_idx] / 100.0,
            'ch_coords': batch['ch_coords'][cb_idx],
        }
        for k in ('valid_channel_mask', 'valid_length_mask'):
            if k in batch:
                out[k] = batch[k][cb_idx]
        cb_list = cb_idx.tolist()
        if 'ch_names' in batch:
            out['ch_names'] = [batch['ch_names'][i] for i in cb_list]
        if 'source' in batch:
            out['source'] = [batch['source'][i] for i in cb_list]
        pv_f = batch.get('pixel_values_future')
        has_f = batch.get('has_image_future')
        if pv_f is not None and has_f is not None:
            out['image_encoder_inputs'] = {'pixel_values': pv_f[:, window_idx]}
            out['has_image'] = has_f[:, window_idx]
        return out

    def _build_alignment_batch(
        self,
        batch: dict,
        window_idx: int,
    ) -> dict:
        """Slice ``batch`` so the inner encoder sees window ``window_idx``.

        ``timeseries_future`` has shape (B, W, C, N, d) and
        ``pixel_values_future`` has shape (B, W, 3, H, W).

        The pretraining trainer rescales ``batch['timeseries']`` by /100
        before calling the model but doesn't know about
        ``timeseries_future``; we apply the same rescale here so the
        encoder sees a consistent value range on every window.
        """
        ts_f = batch.get('timeseries_future')
        if window_idx == 0 and 'timeseries' in batch:
            # Already /100'd by the trainer.
            ts = batch['timeseries']
        else:
            # If the caller asked for window_idx > 0 but no future stack is
            # present, that's a real bug — surface it loudly.
            assert ts_f is not None, (
                f"window_idx={window_idx} requires 'timeseries_future' in batch")
            ts = ts_f[:, window_idx] / 100.0
        out = {
            'timeseries': ts,
            'ch_coords': batch['ch_coords'],
        }
        for k in ('ch_names', 'valid_channel_mask', 'valid_length_mask',
                  'source', 'hand_targets', 'hand_valid'):
            if k in batch:
                out[k] = batch[k]
        if window_idx == 0:
            # Every collate that feeds this wrapper (collate_cached,
            # collate_cached_with_future, collate_cinebrain) populates
            # ``image_encoder_inputs`` / ``has_image`` at full batch B,
            # aligned with ``batch['timeseries']``. The future stacks
            # (``pixel_values_future``, ``has_image_future``) may be
            # sliced to only the M CineBrain rows in mix mode, so do NOT
            # use them here — that would mismatch the B rows of ``ts``.
            if 'image_encoder_inputs' in batch:
                out['image_encoder_inputs'] = batch['image_encoder_inputs']
            if 'has_image' in batch:
                out['has_image'] = batch['has_image']
            # Window-0 cached vision-encoder embeddings (egobrain_extract_
            # embeddings): forwarded so the frame-averaging encoder uses them
            # for image alignment + the flip-align descriptor instead of the
            # frozen-encoder forward. Full batch B, aligned with timeseries.
            for k in ('frame_cls', 'frame_cls_flip', 'frame_grid',
                      'frame_grid_flip'):
                if k in batch:
                    out[k] = batch[k]
        else:
            pv_f = batch.get('pixel_values_future')
            has_f = batch.get('has_image_future')
            if pv_f is not None and has_f is not None:
                out['image_encoder_inputs'] = {'pixel_values': pv_f[:, window_idx]}
                out['has_image'] = has_f[:, window_idx]
        return out

    # ------------------------------------------------------------------

    def _compute_flip_motion(self, batch: dict) -> Optional[torch.Tensor]:
        """Per-sample motion = mean|frame_{t+1} - frame_t| from the future
        frame stack, for the encoder's flip-alignment motion weighting.

        Returns a (B,) tensor aligned with ``batch['timeseries']`` rows; rows
        without a future frame stack get a -1 sentinel (the encoder maps that
        to weight 1). Returns None when no future frames are present.
        """
        pvf = batch.get('pixel_values_future')          # (M, W, 3, H, W) or None
        if pvf is None or pvf.size(1) < 2:
            return None
        if 'cinebrain_idx' in batch:
            rows = batch['cinebrain_idx']
        else:
            rows = torch.arange(pvf.size(0), device=pvf.device)
        B = batch['timeseries'].size(0)
        motion = batch['timeseries'].new_full((B,), -1.0)
        m = (pvf[:, 1] - pvf[:, 0]).abs().mean(dim=(1, 2, 3))   # (M,)
        motion[rows] = m.to(motion.dtype)
        return motion

    @staticmethod
    def _eeg_token_padding_mask(batch, cb_idx, valid, C, N):
        """Key-padding mask for the flattened ``(C*N)`` EEG token set.

        Returns ``(Bv, C*N)`` bool with ``True`` == padded/ignore (an invalid
        channel OR an invalid time patch), restricted to the (cb -> valid) rows,
        or ``None`` when the batch carries no validity masks (nothing to pad).
        """
        vcm = batch.get('valid_channel_mask')
        vlm = batch.get('valid_length_mask')
        if vcm is None and vlm is None:
            return None
        ref = vcm if vcm is not None else vlm
        Bv = int(valid.sum().item())
        if vcm is not None:
            vcm = vcm[cb_idx][valid][:, :C].bool()             # (Bv, C)
        else:
            vcm = torch.ones(Bv, C, dtype=torch.bool, device=ref.device)
        if vlm is not None:
            vlm = vlm[cb_idx][valid][:, :N].bool()             # (Bv, N)
        else:
            vlm = torch.ones(Bv, N, dtype=torch.bool, device=ref.device)
        valid_tok = vcm.unsqueeze(2) & vlm.unsqueeze(1)        # (Bv, C, N)
        return ~valid_tok.reshape(valid_tok.size(0), C * N)    # True == ignore

    @staticmethod
    def _motion_weight(s_anchor: torch.Tensor, s_tgt: torch.Tensor,
                       tgt_valid: torch.Tensor, alpha: float, floor: float,
                       ref=None):
        """Per-(row, step, patch) weight for the dense frame-prediction loss.

        The dense frame target ``s_tgt`` (next 0.2 s DINOv2/V-JEPA grids) is
        ~static in feature space, so a uniform mean over patches is dominated by
        patches the anchor already explains — the predictor minimises it by
        copying the anchor and never uses the EEG (``diag_frame_eeg_gap`` ~0).
        Weighting each patch by how far its target MOVED from the anchor (the
        residual the copy leaves unexplained) concentrates the loss on the
        dynamic patches, the only place the EEG can lower it, so the copy
        shortcut stops scoring well and gradient flows into the EEG embedding.

        ``w = clip(motion / ref_τ, floor, inf) ** alpha`` where
        ``motion[b,τ,p] = mean_d |s_tgt[b,τ,p] - s_anchor[b,p]|`` and ``ref_τ`` is
        a PER-HORIZON reference (motion grows with the horizon, so a per-step
        reference keeps the floor threshold ``floor*ref_τ`` lag-appropriate).
        ``ref`` may be:
          * ``None``     — per-batch per-step mean over VALID (row, patch) at each
                           step (legacy dynamic; jitters step-to-step);
          * a scalar     — the same reference for every step;
          * a length-≥H sequence/tensor — the per-horizon reference (the fixed
                           baked-in constant; recommended).
        The reference only sets the floor threshold: because the reducer
        normalises by ``w.sum()`` it otherwise cancels, so the loss magnitude
        stays comparable to the uniform mean and ``latent_pred_weight`` needs no
        retuning. ``alpha <= 0`` returns ``(None, None)`` and the caller falls
        back to a plain mean (the legacy, byte-identical path). All motion is
        computed under ``no_grad`` on the frozen-encoder targets, so no gradient
        flows through the weight.

        Returns ``(w, ref_per_step)`` with ``w`` of shape ``(Bv, H, P)`` and
        ``ref_per_step`` of shape ``(H,)``, or ``(None, None)``.
        """
        if alpha <= 0:
            return None, None
        with torch.no_grad():
            motion = (s_tgt - s_anchor.unsqueeze(1)).abs().mean(dim=-1)  # (Bv,H,P)
            H = motion.size(1)
            if ref is None:
                vexp = tgt_valid.unsqueeze(-1)                           # (Bv,H,1)
                num = (motion * vexp).sum(dim=(0, 2))                    # (H,)
                den = vexp.expand_as(motion).sum(dim=(0, 2)).clamp(min=1.0)
                ref_ps = num / den                                      # (H,)
            else:
                ref_ps = torch.as_tensor(
                    ref, device=motion.device, dtype=motion.dtype)
                if ref_ps.ndim == 0:
                    ref_ps = ref_ps.expand(H)
                assert ref_ps.numel() >= H, (
                    f"frame_motion_ref has {ref_ps.numel()} entries < H={H}; "
                    f"recompute EGOBRAIN_FRAME_MOTION_REF_PER_STEP for this "
                    f"max_horizon")
                ref_ps = ref_ps[:H]
            w = (motion / ref_ps.view(1, H, 1).clamp(min=1e-6)
                 ).clamp(min=floor).pow(alpha)
        return w, ref_ps

    def _resolved_motion_ref(self):
        """Resolve ``self.frame_motion_ref`` to the reference passed to
        ``_motion_weight``: ``< 0`` -> the fixed per-horizon
        ``EGOBRAIN_FRAME_MOTION_REF_PER_STEP`` tuple (stable, default); ``== 0``
        -> ``None`` (per-batch per-step mean); ``> 0`` -> that scalar (applied to
        every step)."""
        r = self.frame_motion_ref
        if r < 0:
            return EGOBRAIN_FRAME_MOTION_REF_PER_STEP
        if r > 0:
            return r
        return None

    @staticmethod
    def _reduce_over_patches(x_bhp: torch.Tensor,
                             w: Optional[torch.Tensor]) -> torch.Tensor:
        """Reduce a per-(row, step, patch) tensor ``(Bv, H, P)`` over patches to
        ``(Bv, H)``. ``w is None`` -> plain mean (legacy). Otherwise a weighted
        mean by the motion weight ``w`` (same shape), normalised by ``w.sum()`` so
        the scale matches the uniform mean regardless of the weight magnitude."""
        if w is None:
            return x_bhp.mean(dim=-1)
        return (x_bhp * w).sum(dim=-1) / w.sum(dim=-1).clamp(min=1e-6)

    def _row_frame_distance(self, pred: torch.Tensor, s_tgt: torch.Tensor,
                            w_motion: Optional[torch.Tensor],
                            tgt_valid: torch.Tensor,
                            row_denom: torch.Tensor) -> torch.Tensor:
        """Per-row motion-weighted L1 distance between a dense prediction and the
        target grids, reduced over patches (same weighting as the main loss) and
        averaged over each row's VALID horizon steps.

        ``pred`` / ``s_tgt``: ``(Bv, H, P, d)``; ``w_motion``: ``(Bv, H, P)`` or
        None; ``tgt_valid``: ``(Bv, H)``; ``row_denom``: ``(Bv,)`` = per-row valid
        step count (clamped >= 1). Returns ``(Bv,)``. This is the SAME reduction
        the frame_pred_loss uses, just kept per-row (not summed) so it can serve as
        the contrastive energy.
        """
        err = F.l1_loss(pred, s_tgt, reduction='none').mean(dim=-1)   # (Bv,H,P)
        per_step = self._reduce_over_patches(err, w_motion)          # (Bv,H)
        return (per_step * tgt_valid).sum(dim=1) / row_denom         # (Bv,)

    def _frame_neg_distances(self, neg_idx: torch.Tensor,
                             s_anchor: torch.Tensor, s_tgt: torch.Tensor,
                             eeg_emb: torch.Tensor,
                             eeg_kpm: Optional[torch.Tensor],
                             w_motion: Optional[torch.Tensor],
                             tgt_valid: torch.Tensor,
                             row_denom: torch.Tensor) -> torch.Tensor:
        """Per-(row, slot) negative distances ``d_neg`` ``(Bv, K)``: the
        motion-weighted distance from ``predictor(anchor_i, eeg_{neg_idx[i, m]})``
        to ``s_tgt_i`` for each of the ``K`` negative slots.

        Two paths, selected by ``frame_contrast_batched``:
          * loop (default) — ``K`` predictor forwards of ``Bv`` rows each; lowest
            peak memory (one negative's activations live at a time).
          * batched — ONE predictor forward over all ``K * Bv`` (anchor, neg-EEG)
            pairs (slot-major: block ``m`` = every row's slot-``m`` negative); fewer
            + larger kernels, but ~``K x`` peak attention memory. The predictor is
            batched over dim 0 with no cross-row attention, so this is numerically
            identical to the loop (up to dropout RNG in ``train()``).
        Negatives are detached from the EEG-encoder graph iff
        ``frame_contrast_detach_neg`` in BOTH paths.
        """
        Bv, K = neg_idx.shape
        if not self.frame_contrast_batched:
            d_negs = []
            for m in range(K):
                idx = neg_idx[:, m]                                # (Bv,)
                eeg_neg = eeg_emb.index_select(0, idx)
                if self.frame_contrast_detach_neg:
                    eeg_neg = eeg_neg.detach()
                kpm_neg = (eeg_kpm.index_select(0, idx)
                           if eeg_kpm is not None else None)
                pred_neg = self.predictor(
                    s_anchor, eeg_neg, eeg_key_padding_mask=kpm_neg)  # (Bv,H,P,d)
                d_negs.append(self._row_frame_distance(
                    pred_neg, s_tgt, w_motion, tgt_valid, row_denom))
            return torch.stack(d_negs, dim=1)                      # (Bv, K)

        # Batched: stack all K negatives along the batch dim -> one forward. Row
        # ``m * Bv + i`` pairs anchor_i with slot-m negative EEG of row i.
        P, d = s_anchor.size(1), s_anchor.size(2)
        H = s_tgt.size(1)
        flat = neg_idx.t().reshape(-1)                             # (K*Bv,) slot-major
        anchor_rep = (s_anchor.unsqueeze(0).expand(K, -1, -1, -1)
                      .reshape(K * Bv, P, d))
        eeg_all = eeg_emb.index_select(0, flat)                    # (K*Bv, ...)
        if self.frame_contrast_detach_neg:
            eeg_all = eeg_all.detach()
        kpm_all = (eeg_kpm.index_select(0, flat)
                   if eeg_kpm is not None else None)
        pred_all = self.predictor(
            anchor_rep, eeg_all, eeg_key_padding_mask=kpm_all)     # (K*Bv,H,P,d)
        tgt_rep = (s_tgt.unsqueeze(0).expand(K, -1, -1, -1, -1)
                   .reshape(K * Bv, H, P, d))
        w_rep = (w_motion.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * Bv, H, P)
                 if w_motion is not None else None)
        tv_rep = tgt_valid.unsqueeze(0).expand(K, -1, -1).reshape(K * Bv, H)
        rd_rep = row_denom.unsqueeze(0).expand(K, -1).reshape(K * Bv)
        d_flat = self._row_frame_distance(
            pred_all, tgt_rep, w_rep, tv_rep, rd_rep)              # (K*Bv,)
        return d_flat.reshape(K, Bv).t()                          # (Bv, K)

    @staticmethod
    def _same_flip_neg_indices(flip_valid: torch.Tensor, K: int,
                               block_id: Optional[torch.Tensor] = None,
                               anchor: Optional[torch.Tensor] = None,
                               excl_samples: int = 0):
        """Per-row negative row indices drawn ONLY from rows sharing the anchor's
        frame-averaging flip state (and, when subject-block loading is on, its
        block + a temporal exclusion zone).

        Under frame averaging the anchor + target grids AND the positive EEG for
        row ``i`` are all presented in row ``i``'s orientation ``flip_i``. A
        negative from a row with the OPPOSITE flip is orientation-mismatched vs the
        anchor, letting the predictor win the contrastive task on orientation
        bookkeeping instead of motion. Restricting negatives to the SAME flip group
        removes that shortcut. ``flip_valid`` is the per-row flip (``(Bv,)`` bool);
        pass an all-equal tensor when frame averaging is off (any row is then a
        valid negative).

        ``block_id`` (``(Bv,)`` int, optional): when given, negatives must ALSO
        come from the row's own subject-block, so every negative shares the
        anchor's subject + scene and the only thing that differs is the future
        motion — the hard negative that closes the subject/scene-identity
        shortcut the cross-clip in-batch negatives leak. The eligibility group
        becomes ``(flip, block)`` jointly.

        ``anchor`` (``(Bv,)`` int window-0 start sample) + ``excl_samples`` > 0:
        a candidate whose anchor is within ``excl_samples`` of the row's anchor
        is masked out (``neg_valid`` False) — it would share EEG samples with the
        positive (a near-duplicate false negative). Almost never fires under
        temporal jitter (anchors are spread across the subject) but guards
        overlaps and the no-jitter path.

        Returns ``(neg_idx, neg_valid)`` both ``(Bv, K)``. ``neg_idx[i, m]`` is the
        row supplying the slot-``m`` negative EEG for row ``i``; ``neg_valid[i, m]``
        is False when row ``i``'s eligibility group has too few members to fill
        slot ``m`` OR the candidate is inside the exclusion zone — the caller masks
        those slots out of the loss and the diagnostics. A row whose group is a
        singleton gets no valid negative. The realised per-row negative count
        therefore varies with the group sizes, which is why the caller must never
        assume a fixed ``K`` per row.
        """
        Bv = flip_valid.numel()
        device = flip_valid.device
        neg_idx = torch.zeros(Bv, K, dtype=torch.long, device=device)
        neg_valid = torch.zeros(Bv, K, dtype=torch.bool, device=device)
        ar = torch.arange(Bv, device=device)
        # Eligibility key = flip, jointly with block when block gating is on. Rows
        # sharing a key are each other's negative candidates. With block_id=None
        # this reduces to the two (False/True) flip groups exactly as before.
        key = flip_valid.to(torch.long)
        if block_id is not None:
            key = block_id.to(torch.long) * 2 + key
        for g in torch.unique(key):
            members = ar[key == g]                         # rows in this group
            n_g = int(members.numel())
            if n_g < 2:
                continue                                   # no same-group partner
            r = torch.arange(n_g, device=device)
            # Slot m (1-indexed) -> the member m positions ahead cyclically; valid
            # only while m <= n_g - 1 (m == n_g would alias the positive itself).
            for m in range(1, min(K, n_g - 1) + 1):
                cand = members[(r + m) % n_g]
                neg_idx[members, m - 1] = cand
                ok = torch.ones(n_g, dtype=torch.bool, device=device)
                if anchor is not None and excl_samples > 0:
                    ok = (anchor[members] - anchor[cand]).abs() >= excl_samples
                neg_valid[members, m - 1] = ok
        return neg_idx, neg_valid

    def _add_frame_contrast(self, info: dict, per_step_pos: torch.Tensor,
                            s_anchor: torch.Tensor, s_tgt: torch.Tensor,
                            eeg_emb: torch.Tensor,
                            eeg_kpm: Optional[torch.Tensor],
                            w_motion: Optional[torch.Tensor],
                            tgt_valid: torch.Tensor, scale: float,
                            flip_valid: Optional[torch.Tensor] = None,
                            block_id: Optional[torch.Tensor] = None,
                            anchor: Optional[torch.Tensor] = None) -> None:
        """Negative-EEG contrastive term for the dense frame objective.

        For each row ``i`` (anchor grid ``s_anchor[i]``, target grids ``s_tgt[i]``,
        EEG conditioning ``eeg_emb[i]``) we re-run the predictor on the SAME anchor
        conditioned on up to ``K`` OTHER rows' EEG (in-batch negatives) and compare
        each prediction's distance to ``s_tgt[i]``:

          * ``d_pos``   = distance(predictor(anchor_i, eeg_i), s_tgt_i)
          * ``d_neg_m`` = distance(predictor(anchor_i, eeg_j), s_tgt_i), j != i

        Holding the anchor fixed and swapping ONLY the EEG isolates the EEG's
        contribution: the copy-the-anchor shortcut produces the same prediction for
        every EEG, so it CANNOT make ``d_pos < d_neg`` — the only way to lower the
        loss is to read the future motion out of the EEG. ``d_pos`` reuses the
        already-computed positive ``per_step_pos`` (no extra forward for it).

        Negatives are restricted to rows sharing the anchor's frame-averaging flip
        (``flip_valid``) so the predictor cannot discriminate on ORIENTATION
        bookkeeping instead of motion; with frame averaging off, ``flip_valid`` is
        all-equal and any other row is a valid negative. When ``block_id`` is
        given (subject-block loading), negatives are ADDITIONALLY restricted to
        the row's own block — same subject + scene — so the discrimination can no
        longer ride on subject/scene identity, only on the future motion; and
        ``anchor`` + ``self.frame_contrast_excl_samples`` mask any candidate too
        close in time to be a genuine negative. The per-row realised negative
        count varies (a small group yields fewer than ``K``); invalid slots are
        masked out of the loss and diagnostics, and a row with no valid negative
        is dropped from the reduction.

        Writes ``info['frame_contrast_loss']`` and ``diag_frame_contrast_*``.
        Negatives are detached from the EEG-encoder graph iff
        ``frame_contrast_detach_neg`` (default True) so only the predictor learns
        the discrimination from them; the encoder is still pushed via the positive.
        """
        Bv = s_anchor.size(0)
        K = min(self.frame_contrast_n_neg, Bv - 1)
        if K < 1:
            return
        if flip_valid is None:
            flip_valid = torch.zeros(Bv, dtype=torch.bool, device=s_anchor.device)
        neg_idx, neg_valid = self._same_flip_neg_indices(
            flip_valid, K, block_id=block_id, anchor=anchor,
            excl_samples=self.frame_contrast_excl_samples)             # (Bv, K)

        row_denom = tgt_valid.sum(dim=1).clamp(min=1.0)            # (Bv,)
        # A row is usable only if it has >= 1 valid target step AND >= 1 same-flip
        # negative. Rows failing either are dropped from every reduction below.
        rv = ((tgt_valid.sum(dim=1) > 0) & neg_valid.any(dim=1)).to(s_anchor.dtype)
        rv_sum = rv.sum().clamp(min=1.0)
        if float(rv.sum()) == 0.0:
            return

        # Positive per-row distance from the shared positive per-step reduction.
        d_pos = (per_step_pos * tgt_valid).sum(dim=1) / row_denom  # (Bv,)

        # K negative distances via a loop or a single batched forward (equivalent;
        # ``frame_contrast_batched`` trades peak memory for kernel efficiency).
        d_neg = self._frame_neg_distances(
            neg_idx, s_anchor, s_tgt, eeg_emb, eeg_kpm,
            w_motion, tgt_valid, row_denom)                       # (Bv, K)
        nv = neg_valid.to(d_neg.dtype)                            # (Bv, K) 1=real slot

        if self.frame_contrast_mode == 'infonce':
            # logits = -distance / temp; positive is column 0. Invalid negative
            # slots get -inf so they drop out of the softmax denominator (fewer
            # negatives for that row) without touching the finite positive logit.
            neg_logits = (-d_neg / self.frame_contrast_temp).masked_fill(
                ~neg_valid, float('-inf'))
            pos_logit = (-d_pos / self.frame_contrast_temp).unsqueeze(1)
            logits = torch.cat([pos_logit, neg_logits], dim=1)    # (Bv, K+1)
            target = torch.zeros(Bv, dtype=torch.long, device=logits.device)
            ce = F.cross_entropy(logits, target, reduction='none')  # (Bv,)
            contrast_loss = (ce * rv).sum() / rv_sum
        else:  # 'margin' — each valid negative must beat the positive by ``margin``.
            gap = self.frame_contrast_margin + d_pos.unsqueeze(1) - d_neg   # (Bv,K)
            per_row = (F.softplus(gap) * nv).sum(dim=1) / nv.sum(dim=1).clamp(min=1.0)
            contrast_loss = (per_row * rv).sum() / rv_sum

        info['frame_contrast_loss'] = (
            self.frame_contrast_weight * scale, contrast_loss)

        with torch.no_grad():
            # Accuracy: is the true EEG STRICTLY closer than every valid negative?
            # Strict '<' makes the copy-the-anchor tie (pred_neg == pred_pos ->
            # d_pos == d_neg) count as INCORRECT, so acc reports chance-level in the
            # EEG-ignoring failure mode instead of a misleading 1.0. ->1 means the
            # predictor genuinely discriminates on the EEG.
            min_neg = d_neg.masked_fill(~neg_valid, float('inf')).min(dim=1).values
            correct = (d_pos < min_neg).to(s_anchor.dtype)
            info['diag_frame_contrast_acc'] = (correct * rv).sum() / rv_sum
            # Mean (d_neg - d_pos) over valid negatives; > 0 & growing = EEG used.
            gap_diag = (((d_neg - d_pos.unsqueeze(1)) * nv).sum(dim=1)
                        / nv.sum(dim=1).clamp(min=1.0))
            info['diag_frame_contrast_gap'] = (gap_diag * rv).sum() / rv_sum
            # Mean realised negatives per usable row (varies with flip-group sizes).
            info['diag_frame_contrast_n_neg'] = (nv.sum(dim=1) * rv).sum() / rv_sum
            info['diag_frame_contrast_n_rows'] = rv.sum()

    def _frame_prediction_step(self, out, info: dict, batch: dict,
                               cb_idx: torch.Tensor,
                               flip: Optional[torch.Tensor] = None,
                               cond_info: Optional[dict] = None):
        """Cross-modal DENSE frame-prediction objective (``objective='frame'``).

        Predicts the per-patch grids of the next ``H = max_horizon`` frames —
        the co-occurring 0.2 s frame grid over the current 1 s EEG window —
        from the anchor (window-0) frame grid, conditioned on the current
        window's EEG tokens (``patch_tokens`` / ``global_rep``). Every
        0.2 s step is supervised in one predictor call. All grids come from the
        FROZEN vision encoder, so the targets are fixed (no EMA, no collapse) and
        only the EEG embedding + predictor carry gradient. Only the *current*
        EEG segment is used; the loaded future EEG windows are ignored.

        ``out``/``info`` are the window-0 reconstruction + loss dict already
        produced by the shared (masked) encoder forward; we only add the
        prediction terms and return them. ``cond_info`` optionally supplies the
        EEG CONDITIONING tokens from a separate CLEAN (unmasked) encode
        (``frame_clean_cond``); when None the masked forward's ``info`` is used.
        """
        cond = cond_info if cond_info is not None else info
        has_f = batch.get('has_image_future')     # (M, W) bool
        gfut = batch.get('frame_grid_future')     # (M, W, P, d) cached grids or None
        pv_f = batch.get('pixel_values_future')   # (M, W, 3, H, W) raw frames or None
        if has_f is None or (gfut is None and pv_f is None):
            return out, info
        if getattr(self.encoder, 'vision_trainable', False):
            # The trainable ViT must run LIVE on raw pixels; cached grids would
            # bypass it (zero gradient). Guard against a mis-set data path.
            assert gfut is None, (
                "scratch_vit needs raw pixels; disable --egobrain_use_grid_embeddings")

        # Frame-window count W comes from whichever frame source is present. With
        # cached grid embeddings the raw pixel stack is a 1x1 placeholder
        # (egobrain _getitem_grid skips full frames when the encoder is cached),
        # so prefer frame_grid_future for the shape.
        W = gfut.size(1) if gfut is not None else pv_f.size(1)
        # Drive the dense-target count off the predictor's own output width so
        # the two can never desync (the predictor emits exactly max_horizon
        # grids). In production this equals the wrapper's max_horizon.
        H = self.predictor.max_horizon
        assert W >= H + 1, (
            f"future frame stack has W={W} windows but max_horizon={H} needs "
            f"at least {H + 1} (window 0 = anchor, windows 1..{H} = targets)")

        # Supervise every row whose ANCHOR (window 0) frame exists; missing
        # future frames (clip edges) are masked per-(row, step) in the loss.
        valid = has_f[:, 0]                                    # (M,)
        if int(valid.sum().item()) == 0:
            return out, info

        # Current-window EEG conditioning for the (cb -> valid) rows.
        # ``global_rep`` is (B, d_model), ``patch_tokens`` is (B, C, N, d_model);
        # index by cb_idx (M rows, same order as pv_f) then by valid.
        # ``cond`` is the clean unmasked encode when frame_clean_cond is set,
        # else the masked forward's ``info`` (see caller).
        eeg_kpm = None
        if self.frame_eeg_cond == 'tokens':
            pt = cond['patch_tokens'][cb_idx][valid]           # (Bv, C, N, d_model)
            eeg_emb = pt.reshape(pt.size(0), -1, pt.size(-1))  # (Bv, C*N, d_model)
            # Mask padded channels / invalid time so they cannot leak into the
            # predictor's attention.
            eeg_kpm = self._eeg_token_padding_mask(
                batch, cb_idx, valid, pt.size(1), pt.size(2))  # (Bv, C*N) or None
        else:
            eeg_emb = cond['global_rep'][cb_idx][valid]        # (Bv, d_model)

        if 'frame_grid_future' in batch:
            # Cached path: presented per-row grid stack straight from the cache
            # (the mirrored-frame grid iff that row is flipped) — NEVER
            # torch.flip of the grid (a feature-space flip is wrong). Row order
            # matches pixel_values_future, so flip[valid] lines up.
            assert 'frame_grid_flip_future' in batch, (
                "frame_grid_future present but frame_grid_flip_future missing "
                "— both orientations are required for exact flip semantics")
            gffut = batch['frame_grid_flip_future']
            assert gfut.size(0) == has_f.size(0) and gfut.size(1) == W, (
                f"frame_grid_future {tuple(gfut.shape)} must match has_image_"
                f"future rows/windows ({has_f.size(0)}, {W})")
            gv, gfv = gfut[valid], gffut[valid]                # (Bv, W, P, d)
            if flip is not None and flip.any():
                fv = flip[valid].view(-1, 1, 1, 1)             # (Bv,1,1,1) bool
                present = torch.where(fv, gfv, gv)
            else:
                present = gv
            s_anchor = present[:, 0].contiguous()              # (Bv, P, d)
            s_tgt = present[:, 1:H + 1].contiguous().detach()  # (Bv, H, P, d)
        else:
            pv = pv_f[valid]                                   # (Bv, W, 3, Hh, Ww)
            if flip is not None and flip.any():
                # Per-sample frame-averaging flip: mirror each row's frames iff
                # that row's EEG was presented mirrored, so the EEG conditioning
                # and the prediction targets share one orientation per row.
                fv = flip[valid].view(-1, 1, 1, 1, 1)          # (Bv,1,1,1,1) bool
                pv = torch.where(fv, torch.flip(pv, dims=[-1]), pv)
            if getattr(self.encoder, 'vision_trainable', False):
                # LeWM from-scratch ViT: encode anchor (0) + targets (1..H) WITH
                # gradient and do NOT detach the target — the prediction target
                # is the ONLINE ViT's own future grid (faithful LeWM: no
                # stop-grad, no EMA). Collapse is held off by SIGReg (added
                # after the loss below), not by a frozen target.
                d_img = self.encoder.image_feature_dim
                grids = [self.encoder._scratch_patch_grid(pv[:, w]).reshape(
                             pv.size(0), -1, d_img)                # (Bv, P, d)
                         for w in range(H + 1)]
                s_anchor = grids[0]
                s_tgt = torch.stack(grids[1:H + 1], dim=1)         # (Bv,H,P,d) online
            else:
                # Frozen vision-encoder grids for anchor (0) + targets (1..H).
                # Detached: the encoder has no trainable parameters here, so
                # gradients flow only through eeg_emb + the predictor.
                with torch.no_grad():
                    grids = []
                    for w in range(H + 1):
                        g = self.encoder._image_patch_grid(pv[:, w])   # (Bv, s, s, d)
                        grids.append(g.reshape(g.size(0), -1, g.size(-1)))  # (Bv,P,d)
                s_anchor = grids[0]
                s_tgt = torch.stack(grids[1:H + 1], dim=1).detach()    # (Bv, H, P, d)

        # Per-(row, step) target validity: future frame exists at that step.
        tgt_valid = has_f[valid][:, 1:H + 1].to(s_tgt.dtype)   # (Bv, H)
        if float(tgt_valid.sum()) == 0.0:
            return out, info
        denom = tgt_valid.sum().clamp(min=1.0)

        pred = self.predictor(s_anchor, eeg_emb,
                              eeg_key_padding_mask=eeg_kpm)    # (Bv, H, P, d)
        # Optional per-patch motion weighting (kills the copy-the-anchor
        # shortcut; ``alpha == 0`` -> uniform mean, byte-identical to legacy).
        # ``ref`` is the fixed per-horizon constant by default (stable floor
        # threshold) — see ``_resolved_motion_ref``.
        w_motion, motion_ref = self._motion_weight(
            s_anchor, s_tgt, tgt_valid,
            self.frame_motion_alpha, self.frame_motion_floor,
            ref=self._resolved_motion_ref())
        err = F.l1_loss(pred, s_tgt, reduction='none').mean(dim=-1)  # (Bv, H, P)
        per_step = self._reduce_over_patches(err, w_motion)         # (Bv, H)
        pred_loss = (per_step * tgt_valid).sum() / denom

        scale = self._pred_weight_scale()
        # Reuse ``latent_pred_weight`` as the predictor's loss weight (the
        # objective is a swap-in, not an addition), ramped the same way.
        info['frame_pred_loss'] = (self.latent_pred_weight * scale, pred_loss)

        # SIGReg — LeWM's sole collapse guard on the trainable-ViT path. Since
        # the target grids are the ONLINE ViT's own (non-detached) embeddings,
        # a constant collapse would trivially minimise the L1; SIGReg pushes the
        # grid-embedding batch toward an isotropic Gaussian so that solution is
        # penalised. Fed as (N, Bv, d) with N=(H+1)*P (the batch axis Bv is the
        # distribution SIGReg tests). Only on the scratch path.
        if (self.sigreg is not None and self.sigreg_weight > 0
                and getattr(self.encoder, 'vision_trainable', False)):
            all_emb = torch.cat([s_anchor.unsqueeze(1), s_tgt], dim=1)  # (Bv,H+1,P,d)
            Bvv, Hp1, Pp, dd = all_emb.shape
            sig_in = all_emb.permute(1, 2, 0, 3).reshape(Hp1 * Pp, Bvv, dd)
            info['sigreg_loss'] = (self.sigreg_weight, self.sigreg(sig_in))
            with torch.no_grad():
                # Direct collapse monitor: batch-wise std of the ViT grid
                # embeddings -> 0 iff they collapse to a constant. Complements
                # sigreg_loss (which SIGReg is meant to keep small WITHOUT collapse).
                info['diag_frame_emb_std'] = all_emb.detach().float().std(dim=0).mean()

        # Negative-EEG contrastive term: penalise the predictor for reproducing the
        # true future from OTHER rows' EEG on the same anchor, so the copy shortcut
        # stops winning and the EEG conditioning becomes load-bearing. Negatives are
        # restricted to rows sharing this row's frame-averaging flip (``flip`` is the
        # per-cb-row flip; restrict to the ``valid`` anchor rows) so orientation
        # can't be used to discriminate; None when frame averaging is off. Skipped
        # (no extra forwards) when the weight is 0, the ramp scale is 0 (loss would
        # be dropped anyway), or the batch has < 2 valid rows. Reuses the positive
        # ``per_step`` so ``d_pos`` costs no forward.
        if self.frame_contrast_weight > 0 and scale > 0 and s_anchor.size(0) >= 2:
            flip_valid = flip[valid] if flip is not None else None
            # Subject-block gating (optional): restrict negatives to the row's own
            # block (same subject/scene) + a temporal exclusion zone. ``block_id``
            # / ``anchor_sample`` are per-row full-batch fields (present only when
            # SubjectBlockBatchSampler is active), so index them by cb_idx then the
            # ``valid`` anchor mask exactly like ``flip``. Absent -> plain in-batch
            # same-flip negatives (byte-identical to before).
            block_id = anchor = None
            blk = batch.get('block_id')
            if blk is not None:
                block_id = blk.index_select(0, cb_idx)[valid].to(s_anchor.device)
                anc = batch.get('anchor_sample')
                if anc is not None and self.frame_contrast_excl_samples > 0:
                    anchor = anc.index_select(0, cb_idx)[valid].to(s_anchor.device)
            self._add_frame_contrast(
                info, per_step, s_anchor, s_tgt, eeg_emb, eeg_kpm,
                w_motion, tgt_valid, scale, flip_valid,
                block_id=block_id, anchor=anchor)

        # Diagnostics. The dominant failure is the predictor IGNORING the EEG and
        # copying the anchor frame (short-horizon frames are ~static in DINOv2/
        # V-JEPA space). ``diag_frame_eeg_gap`` must be > 0 (EEG beats zero-EEG)
        # and ``diag_frame_pred_cos`` should beat ``diag_frame_copy_cos``.
        with torch.no_grad():
            anchor_rep = s_anchor.unsqueeze(1).expand_as(s_tgt)   # (Bv,H,P,d)
            pred_zero = self.predictor(
                s_anchor, torch.zeros_like(eeg_emb), eeg_key_padding_mask=eeg_kpm)
            # eeg_gap / copy_l1 reduce over patches with the SAME motion weight as
            # the loss, so ``eeg_gap`` measures whether the EEG helps on the
            # objective the encoder actually trains on (the whole point of the
            # weighting). The cosine diagnostics stay unweighted — they
            # characterise the raw prediction/data, not the objective.
            per_zero = self._reduce_over_patches(
                F.l1_loss(pred_zero, s_tgt, reduction='none').mean(dim=-1), w_motion)
            copy_l1 = self._reduce_over_patches(
                F.l1_loss(anchor_rep, s_tgt, reduction='none').mean(dim=-1), w_motion)
            pred_cos = F.cosine_similarity(pred, s_tgt, dim=-1).mean(dim=-1)   # (Bv,H)
            copy_cos = F.cosine_similarity(anchor_rep, s_tgt, dim=-1).mean(dim=-1)

            def _m(x):  # mask + mean over valid (row, step)
                return (x * tgt_valid).sum() / denom

            info['diag_frame_eeg_gap'] = _m(per_zero) - pred_loss
            info['diag_frame_copy_l1'] = _m(copy_l1)
            info['diag_frame_pred_cos'] = _m(pred_cos)
            info['diag_frame_copy_cos'] = _m(copy_cos)
            if w_motion is not None:
                info['diag_frame_motion_ref'] = motion_ref.mean()
                info['diag_frame_motion_w_mean'] = w_motion.mean()
            info['diag_frame_s_t_norm'] = s_anchor.flatten(end_dim=-2).norm(dim=-1).mean()
            info['diag_frame_pred_norm'] = pred.flatten(end_dim=-2).norm(dim=-1).mean()
            info['diag_frame_eeg_norm'] = eeg_emb.norm(dim=-1).mean()
            info['diag_n_frame_pairs'] = tgt_valid.sum()
            info['diag_frame_horizon'] = torch.tensor(float(H), device=pred.device)
            info['diag_pred_ramp_scale'] = torch.tensor(scale, device=pred.device)
            # First vs last 0.2 s step L1 — is far-horizon harder?
            step_denom = tgt_valid.sum(dim=0).clamp(min=1.0)       # (H,)
            step_l1 = (per_step * tgt_valid).sum(dim=0) / step_denom
            info['diag_frame_l1_step1'] = step_l1[0]
            info['diag_frame_l1_stepH'] = step_l1[-1]

        return out, info

    def _encode_windows_global(self, batch, cb_idx, valid, n_win):
        """Per-window EEG global rep for the first ``n_win`` future windows,
        restricted to the (cb -> valid) rows. Encodes with the ONLINE encoder
        (gradient flows) so the AR 'action' trains the EEG foundation model.
        Returns (Bv, n_win, d_model).

        Windows are folded into the batch dim (Bv*n_win rows) for a single
        encoder forward; per-sample fields (ch_coords / masks / ch_names) are
        repeat-interleaved so row order matches the folded timeseries.
        """
        ts = batch['timeseries_future'][valid][:, :n_win] / 100.0   # (Bv,n,C,N,d)
        Bv, n, C, N, d = ts.shape
        sub = {
            'timeseries': ts.reshape(Bv * n, C, N, d),
            'ch_coords': batch['ch_coords'][cb_idx][valid].repeat_interleave(n, dim=0),
        }
        for k in ('valid_channel_mask', 'valid_length_mask'):
            if k in batch:
                sub[k] = batch[k][cb_idx][valid].repeat_interleave(n, dim=0)
        if 'ch_names' in batch:
            names = [batch['ch_names'][i] for i in cb_idx.tolist()]
            names = [nm for nm, v in zip(names, valid.tolist()) if v]
            sub['ch_names'] = [nm for nm in names for _ in range(n)]
        _, finfo = self.encoder(sub, encoder_only=True)
        return finfo['global_rep'].reshape(Bv, n, -1)               # (Bv,n,d_model)

    def _ar_prediction_step(self, out, info: dict, batch: dict,
                            cb_idx: torch.Tensor,
                            cond_info: Optional[dict] = None):
        """LeWM autoregressive frame-prediction (``wm_predictor='ar'``).

        Encodes the first ``T = history + num_preds`` frames' CLS with the
        trainable ViT (``e``) and the first ``history`` EEG windows' global rep
        (the per-step 'action'), then predicts each next-frame embedding
        autoregressively conditioned on the EEG. The target is the ONLINE ViT's
        own shifted embedding (no stop-grad, no EMA — faithful LeWM); SIGReg on
        ``e`` is the sole collapse guard.
        """
        pv_f = batch.get('pixel_values_future')   # (M, W, 3, Hh, Ww)
        has_f = batch.get('has_image_future')     # (M, W)
        ts_f = batch.get('timeseries_future')     # (M, W, C, N, d)
        if pv_f is None or has_f is None or ts_f is None:
            return out, info
        assert 'frame_grid_future' not in batch, (
            "AR world model needs raw pixels; disable --egobrain_use_grid_embeddings")
        ar = self.predictor
        W = pv_f.size(1)
        T = ar.history + ar.num_preds
        assert W >= T, (
            f"AR world model needs W>={T} frame windows "
            f"(history={ar.history}+num_preds={ar.num_preds}), got W={W}")
        # The AR 'action' needs per-window EEG, but the frame-objective collate
        # trims timeseries_future to window 0 (egobrain_dataset.py: ts_all[:, :1]).
        # Fail loudly rather than silently broadcasting window-0 EEG across every
        # step: the AR path needs the collate to keep >= history EEG windows.
        assert ts_f.size(1) >= ar.history, (
            f"--wm_predictor ar needs >= history={ar.history} EEG windows in "
            f"timeseries_future, but it has {ts_f.size(1)} (the frame-objective "
            f"collate trims timeseries_future to window 0). Keep the full EEG-window "
            f"stack in the collate before using the AR predictor.")

        # Rows whose first T frames ALL exist (the AR chain needs a contiguous
        # span; missing frames at clip edges / blacklisted subjects are dropped).
        valid = has_f[:, :T].all(dim=1)                    # (M,)
        Bv = int(valid.sum().item())
        if Bv == 0:
            return out, info

        pv = pv_f[valid][:, :T]                            # (Bv, T, 3, Hh, Ww)
        Hh, Ww = pv.shape[-2], pv.shape[-1]
        # Frame CLS via the trainable ViT (grad ON, NOT detached — the target is
        # the online embedding).
        cls = self.encoder._scratch_cls(
            pv.reshape(Bv * T, 3, Hh, Ww))[:, 0]           # (Bv*T, d_vit)
        cls = cls.reshape(Bv, T, -1)
        e = ar.project_frames(cls)                         # (Bv, T, embed)

        # Per-window EEG global rep as the 'action' (grad ON: trains the EEG FM).
        eeg_g = self._encode_windows_global(batch, cb_idx, valid, ar.history)
        act = ar.encode_action(eeg_g)                      # (Bv, history, act)

        ctx_emb = e[:, :ar.history]                        # (Bv, history, embed)
        tgt = e[:, ar.num_preds:ar.num_preds + ar.history]  # online, NO detach
        pred = ar.predict(ctx_emb, act)                    # (Bv, history, embed)

        pred_loss = (pred - tgt).pow(2).mean()
        scale = self._pred_weight_scale()
        info['ar_pred_loss'] = (self.latent_pred_weight * scale, pred_loss)

        if self.sigreg is not None and self.sigreg_weight > 0:
            # SIGReg over the T frame embeddings; (T, Bv, embed) — Bv is the
            # distribution axis SIGReg tests toward isotropic Gaussian.
            info['sigreg_loss'] = (self.sigreg_weight,
                                   self.sigreg(e.transpose(0, 1)))
            with torch.no_grad():
                # Direct collapse monitor (batch-wise std -> 0 iff collapsed).
                info['diag_ar_emb_std'] = e.detach().float().std(dim=0).mean()

        with torch.no_grad():
            info['diag_ar_n_rows'] = torch.tensor(float(Bv), device=e.device)
            info['diag_ar_emb_norm'] = e.flatten(end_dim=-2).norm(dim=-1).mean()
            info['diag_ar_pred_norm'] = pred.flatten(end_dim=-2).norm(dim=-1).mean()
            a = F.normalize(pred.flatten(end_dim=-2), dim=-1)
            b = F.normalize(tgt.flatten(end_dim=-2), dim=-1)
            info['diag_ar_pred_cos'] = (a * b).sum(-1).mean()
            # Does the EEG 'action' help vs a zero action? >0 == EEG is used.
            # Run the predictor in EVAL for this extra forward: pred_proj has a
            # BatchNorm1d whose running stats update even under no_grad in train
            # mode, so logging this diagnostic would otherwise mutate the model
            # with the artificial zero-action distribution.
            was_training = ar.training
            ar.eval()
            try:
                pred_zero = ar.predict(ctx_emb, torch.zeros_like(act))
            finally:
                if was_training:
                    ar.train()
            info['diag_ar_eeg_gap'] = (pred_zero - tgt).pow(2).mean() - pred_loss
            info['diag_pred_ramp_scale'] = torch.tensor(scale, device=e.device)
        return out, info

    def training_step(self, batch: dict, mask: Optional[torch.Tensor] = None):
        # 1. Primary forward on window t — produces the existing
        #    reconstruction output + alignment/recon loss terms. Runs on
        #    the full mixed batch (CineBrain + Alljoined) so masked recon
        #    and image alignment train on every sample.
        encoder_batch = self._build_alignment_batch(batch, window_idx=0)
        # Equivariant frame-averaging frontend (plans/eeg-wm.md): sample a
        # PER-SAMPLE ``flip`` mask (B,) — each row independently presents z or
        # P(z) (and downstream the original vs mirrored frame). Deciding flip
        # per row instead of one scalar for the whole step keeps a coin flip from
        # swinging the entire batch into one orientation, which stabilises
        # training. The SAME row's mask is shared across its current and future
        # windows below so the presentation stays consistent per row. Eval is
        # canonical (all-False mask, no flip).
        frame_avg = getattr(self.encoder, 'frame_averaging', False)
        flip = None
        if frame_avg:
            Bsz = encoder_batch['timeseries'].size(0)
            dev = encoder_batch['timeseries'].device
            if self.training:
                p = getattr(self.encoder, 'frame_avg_flip_prob', 0.5)
                flip = torch.rand(Bsz, device=dev) < p          # (B,) bool
            else:
                flip = torch.zeros(Bsz, dtype=torch.bool, device=dev)
            encoder_batch['flip'] = flip
        # Supply a per-sample motion score (t->t+1 frame diff) so the encoder's
        # flip-alignment can down-weight static frames (whose horizontal flip is
        # near-vacuous). Only when that weighting is enabled on the encoder.
        if getattr(self.encoder, 'flip_motion_ref', 0.0) > 0:
            motion = self._compute_flip_motion(batch)
            if motion is not None:
                encoder_batch['flip_motion'] = motion
        # ``frame_clean_cond`` two-view split: the masked view is the
        # reconstruction pretext ONLY; the clean (unmasked) view carries alignment
        # + flip-align + hand + the tokens that condition the predictor. Masking
        # (needed for recon) feeds the predictor/alignment half mask-token
        # reconstruction guesses, so they learn to ignore the EEG (eeg_gap ~0);
        # the clean view gives them the complete signal. The masked forward runs on
        # a RECON-ONLY batch (alignment/hand inputs stripped, so those branches are
        # skipped) and its ``out`` drives the trainer's mask_loss + spectral aux.
        clean_cond_info = None
        two_view = (self.frame_clean_cond and self.objective == 'frame'
                    and self.predictor is not None and self.max_horizon >= 1)
        if two_view:
            # Clean view (full inputs, no mask): alignment + flip-align + hand +
            # conditioning tokens. Its own reconstruction is discarded. Shallow
            # copy — forward mutates its batch dict.
            _, clean_info = self.encoder({**encoder_batch}, mask=None)
            # Masked view (recon only): strip the downstream-facing inputs so the
            # encoder skips those branches. ``out`` feeds the trainer's recon.
            recon_batch = {k: v for k, v in encoder_batch.items()
                           if k not in _RECON_ONLY_STRIP}
            out, masked_info = self.encoder(recon_batch, mask=mask)
            # Non-recon losses/diags come from the clean view; recon-related keys
            # (frame_recon_loss, skip_external_recon, flip_row) from the masked one.
            info = {k: v for k, v in clean_info.items()
                    if k not in _RECON_VIEW_INFO_KEYS}
            for k in _RECON_VIEW_INFO_KEYS:
                if k in masked_info:
                    info[k] = masked_info[k]
            clean_cond_info = clean_info
        else:
            out, info = self.encoder(encoder_batch, mask=mask)

        # 2. Latent prediction on a random horizon.
        # ``max_horizon=0`` (predictor=None) reduces the wrapper to the
        # plain CSBrainAlign pipeline — skip everything below.
        if self.predictor is None or self.max_horizon < 1:
            return out, info

        # Determine which rows in the batch carry a future-window stack:
        #   - ``collate_cached_with_future`` (mix mode) sets cinebrain_idx
        #     and packs only those rows into ``timeseries_future``.
        #   - ``collate_cinebrain`` (pure CineBrain mode) packs every row
        #     into ``timeseries_future`` and does not set cinebrain_idx;
        #     in that case the prediction loss applies to the whole batch.
        if 'cinebrain_idx' in batch:
            cb_idx = batch['cinebrain_idx']
        elif 'timeseries_future' in batch:
            cb_idx = torch.arange(
                batch['timeseries_future'].size(0),
                device=batch['timeseries_future'].device,
                dtype=torch.long,
            )
        else:
            # Mixed batch with no CineBrain rows this step (or a config
            # that supplies no futures at all) — no prediction to compute.
            return out, info

        if cb_idx.numel() == 0:
            return out, info

        # ``wm_objective='frame'`` swaps the EEG-latent predictor for the
        # video-frame predictor: predict the next frame's per-patch embedding
        # from the current frame's per-patch embedding conditioned on the
        # current EEG embedding. Everything above (window-0 encoder forward,
        # masked recon, alignment, aux terms) is shared and unchanged.
        if self.objective == 'frame':
            # ``flip`` is a (B,) per-sample mask over the full batch; restrict it
            # to the cb rows (the frame stack's row order) so each frame is
            # mirrored iff its own EEG row was presented mirrored.
            flip_cb = flip.index_select(0, cb_idx) if flip is not None else None
            if self.wm_predictor == 'ar':
                # LeWM causal AR head (scratch ViT). Canonical orientation only
                # (frame-averaging is not combined with the AR path), so flip is
                # ignored here.
                return self._ar_prediction_step(
                    out, info, batch, cb_idx, cond_info=clean_cond_info)
            return self._frame_prediction_step(
                out, info, batch, cb_idx, flip_cb, cond_info=clean_cond_info)

        ts_future = batch['timeseries_future']  # (M, W, C, N, d)
        W = ts_future.size(1)
        assert W >= self.max_horizon + 1, (
            f"timeseries_future has W={W} windows but max_horizon="
            f"{self.max_horizon} needs at least {self.max_horizon + 1}")
        k = int(torch.randint(1, self.max_horizon + 1, ()).item())

        # CSBrainAlign.forward always populates ``patch_tokens``; if it
        # doesn't, the contract is broken and we want to know.
        s_t_patch = info['patch_tokens'][cb_idx]

        future_batch = self._build_future_subbatch(
            batch, cb_idx, window_idx=k)
        if frame_avg:
            # Present the future window with each row's SAME per-sample flip so
            # the regression target is the (per-row) flipped-future EEG latent on
            # that row's flip — predict flipped future from flipped current, the
            # bilateral control signal that makes the lateral decomposition
            # non-trivial. ``flip`` is (B,); the future sub-batch is the cb rows.
            future_batch['flip'] = flip.index_select(0, cb_idx)
        s_tpk_cls, s_tpk_patch = self._encode_future(future_batch)
        s_tpk_patch = s_tpk_patch.detach()
        s_tpk_cls = s_tpk_cls.detach()

        pred_patch, pred_cls = self.predictor(s_t_patch, horizon=k)

        # CineBrain uses a fixed montage with no channel padding, so a
        # plain L1 is correct. If a future dataset needs per-channel
        # masking, add it here — with an explicit shape assertion, not a
        # silent fallback.
        pred_loss = F.l1_loss(pred_patch, s_tpk_patch)

        # Auxiliary: match the predicted CLS to the future frame CLS from
        # DINOv2, piggybacking on the encoder's contrastive projector so
        # the 768-d space matches.
        aux_cls_loss = F.l1_loss(pred_cls, s_tpk_cls)

        scale = self._pred_weight_scale()
        info['latent_pred_loss'] = (
            self.latent_pred_weight * scale, pred_loss)
        info['latent_cls_loss'] = (
            self.cls_pred_weight * scale, aux_cls_loss)

        # Diagnostics. ``diag_`` prefix lets the trainer pick them up
        # without confusing them for loss terms. Needed to distinguish the
        # "scale runaway" failure mode (all norms grow together) from the
        # "predictor-only" failure mode (norms stable, cosine collapses).
        with torch.no_grad():
            a = F.normalize(pred_patch.flatten(end_dim=-2), dim=-1)
            b = F.normalize(s_tpk_patch.flatten(end_dim=-2), dim=-1)
            info['diag_latent_pred_cos'] = (a * b).sum(-1).mean()
            info['diag_s_t_patch_norm'] = s_t_patch.flatten(end_dim=-2).norm(dim=-1).mean()
            info['diag_s_tpk_patch_norm'] = s_tpk_patch.flatten(end_dim=-2).norm(dim=-1).mean()
            info['diag_pred_patch_norm'] = pred_patch.flatten(end_dim=-2).norm(dim=-1).mean()
            info['diag_pred_cls_norm'] = pred_cls.flatten(end_dim=-2).norm(dim=-1).mean()
            info['diag_s_tpk_cls_norm'] = s_tpk_cls.flatten(end_dim=-2).norm(dim=-1).mean()
            info['diag_predictor_out_w_norm'] = (
                self.predictor.out_proj_patch.weight.detach().norm()
            )
            info['diag_pred_ramp_scale'] = torch.tensor(
                scale, device=pred_patch.device)

        # ------------------------------------------------------------------
        # Bilateralization flipped prediction: predict the flipped-FUTURE EEG
        # latent from the flipped-CURRENT latent. The flipped stream is built
        # by applying the encoder's learned x_bi + flip(x_lat) split to each
        # raw window, so the next-step scene is mirrored (the prediction target
        # is the EEG latent of the flipped future window — symmetric with the
        # main prediction). Trains encoder + split + predictor; the split now
        # also gets a temporal (dynamics) gradient, not just image alignment.
        # ------------------------------------------------------------------
        # The legacy (raw-space split) flipped-prediction branch. Superseded by
        # the frame-averaging path, where the shared per-step ``flip`` already
        # presents both windows flipped — so the standard prediction loss above
        # IS the flipped-prediction loss on flip steps. Skip it then.
        if (getattr(self.encoder, 'lateralization_flip', False)
                and self.flip_pred_weight > 0
                and not frame_avg):
            cb_list = cb_idx.tolist()

            def _sub(src, ts_key='timeseries'):
                sub = {
                    'timeseries': src[ts_key].index_select(0, cb_idx),
                    'ch_coords': src['ch_coords'].index_select(0, cb_idx),
                    'ch_names': [src['ch_names'][i] for i in cb_list],
                }
                for mk in ('valid_channel_mask', 'valid_length_mask'):
                    if mk in src:
                        sub[mk] = src[mk].index_select(0, cb_idx)
                return sub

            # Current flipped window (online, with grad) -> s_flip_t latent.
            # Use the un-mutated wrapper ``batch`` (the window-0 forward above
            # mutated only its own encoder_batch copy's masks).
            cur = _sub(batch)
            x_flip_t, _ = self.encoder.build_lateral_flip(
                cur['timeseries'], cur['ch_coords'], cur['ch_names'],
                cur.get('valid_channel_mask'))
            _, info_flip_t = self.encoder(
                {**cur, 'timeseries': x_flip_t}, encoder_only=True)
            s_flip_t_patch = info_flip_t['patch_tokens']

            # Flipped future window (EMA target, detached). ``future_batch``
            # already holds the k-th future window for the cb rows. Build the
            # flip with the SAME (EMA) split that ``_encode_future`` encodes
            # with — using the online split here would make the regression
            # target move every step (the predictor could chase it instead of
            # learning stable dynamics).
            flip_target_enc = (self.target_encoder
                               if self.target_encoder is not None
                               else self.encoder)
            with torch.no_grad():
                x_flip_tpk, _ = flip_target_enc.build_lateral_flip(
                    future_batch['timeseries'], future_batch['ch_coords'],
                    future_batch['ch_names'],
                    future_batch.get('valid_channel_mask'))
            s_flip_tpk_cls, s_flip_tpk_patch = self._encode_future(
                {**future_batch, 'timeseries': x_flip_tpk})
            s_flip_tpk_patch = s_flip_tpk_patch.detach()
            s_flip_tpk_cls = s_flip_tpk_cls.detach()

            pred_flip_patch, pred_flip_cls = self.predictor(
                s_flip_t_patch, horizon=k)
            flip_pred_loss = F.l1_loss(pred_flip_patch, s_flip_tpk_patch)
            flip_cls_loss = F.l1_loss(pred_flip_cls, s_flip_tpk_cls)
            info['flip_latent_pred_loss'] = (
                self.latent_pred_weight * self.flip_pred_weight * scale,
                flip_pred_loss)
            info['flip_latent_cls_loss'] = (
                self.cls_pred_weight * self.flip_pred_weight * scale,
                flip_cls_loss)
            with torch.no_grad():
                a = F.normalize(pred_flip_patch.flatten(end_dim=-2), dim=-1)
                b = F.normalize(s_flip_tpk_patch.flatten(end_dim=-2), dim=-1)
                info['diag_flip_latent_pred_cos'] = (a * b).sum(-1).mean()

        return out, info

    def forward(self, *args, **kwargs):  # convenience
        return self.training_step(*args, **kwargs)
