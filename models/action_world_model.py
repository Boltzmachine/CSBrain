"""Action-conditioned world model: EEG = action, V-JEPA 2 = world.

The inverse of :mod:`models.world_model`. There, EEG *is* the world and the
predictor forecasts the next EEG latent. Here the **world** is the video,
encoded by a *frozen* V-JEPA 2 into a per-frame spatial patch-token grid
``s_t``, and the **EEG model decodes a "moving intention"** — a learned
*action* latent ``a_t``. A small predictor rolls the world forward,

    ŝ_{t+k} = p(s_t, a_t),

supervised against the frozen ``sg[VJEPA2(frame_{t+k})]``. See
``plans/world_model.md`` (Action-Conditioned variant) for the rationale.

Because the regression target comes from a *frozen* encoder (not an EMA of the
online EEG net), there is no representation-collapse risk and no EMA teacher —
a deliberate simplification over :class:`WorldModelWrapper`.

Two modules live here:

* :class:`ActionPredictor` — a narrow transformer that predicts the next
  world-state token grid from the current grid + the EEG action tokens.
* :class:`ActionWorldModelWrapper` — wraps a ``CSBrainAlign`` EEG encoder
  (which already holds the frozen V-JEPA 2 as ``pretrained_image_encoder``)
  plus an action head and the predictor, and emits the loss dict using the
  ``(weight, tensor)`` convention the pretraining trainer understands.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import generate_mask


# ---------------------------------------------------------------------------
# Action head: pool EEG patch tokens into K learned "intention" tokens
# ---------------------------------------------------------------------------

class ActionHead(nn.Module):
    """Attention-pool the EEG encoder's patch tokens into ``K`` action tokens.

    Inputs are the ``(B, C, N, d)`` patch tokens returned by
    ``CSBrainAlign``'s ``encoder_only`` path. ``K`` learned queries attend over
    the flattened ``C*N`` token set (with invalid channel/length positions
    masked out) and produce ``a_t`` of shape ``(B, K, action_dim)``.
    """

    def __init__(self, d_model: int, action_dim: int, n_action_tokens: int = 8,
                 n_heads: int = 4):
        super().__init__()
        self.n_action_tokens = n_action_tokens
        self.query = nn.Parameter(
            torch.randn(1, n_action_tokens, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, action_dim)

    def forward(self, patch_tokens: torch.Tensor,
                valid_channel_mask: Optional[torch.Tensor] = None,
                valid_length_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        B, C, N, d = patch_tokens.shape
        kv = patch_tokens.reshape(B, C * N, d)

        key_padding_mask = None  # True == ignore
        if valid_channel_mask is not None or valid_length_mask is not None:
            vcm = (valid_channel_mask if valid_channel_mask is not None
                   else patch_tokens.new_ones(B, C, dtype=torch.bool))
            vlm = (valid_length_mask if valid_length_mask is not None
                   else patch_tokens.new_ones(B, N, dtype=torch.bool))
            valid = vcm[:, :C].bool().unsqueeze(2) & vlm[:, :N].bool().unsqueeze(1)
            key_padding_mask = ~valid.reshape(B, C * N)

        q = self.query.expand(B, -1, -1)
        a, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask,
                         need_weights=False)
        return self.proj(self.norm(a))  # (B, K, action_dim)


# ---------------------------------------------------------------------------
# Predictor: ŝ_{t+k} = p(s_t, a_t)
# ---------------------------------------------------------------------------

class ActionPredictor(nn.Module):
    """Predict the next world-state token grid from the current grid + action.

    Action tokens are *prepended* as conditioning (V-JEPA 2-AC injects the
    action the same way); after encoding we drop them and read out the world
    tokens. Kept intentionally narrow so the V-JEPA 2 encoder — not the
    predictor — defines the representation.
    """

    def __init__(
        self,
        world_dim: int,
        action_dim: int,
        predictor_d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_horizon: int = 1,
        max_tokens: int = 512,
        pred_residual: bool = False,
    ):
        super().__init__()
        self.world_dim = world_dim
        self.pred_residual = pred_residual

        self.in_proj_world = nn.Linear(world_dim, predictor_d_model)
        self.in_proj_action = nn.Linear(action_dim, predictor_d_model)
        # Direct FiLM-style broadcast: a pooled action vector is added to every
        # world token *before* the transformer. Without this, the action only
        # reaches the world tokens through attention from the prepended action
        # tokens, which at init is far too weak — the predictor collapses to
        # the trivial copy (ŝ_{t+1} ~= s_t) and the action gradient vanishes.
        # This gives the action a strong, direct path so it is used from step 1.
        self.action_to_world_bias = nn.Linear(action_dim, predictor_d_model)
        # Distinguishes prepended action tokens from world tokens.
        self.action_type_embed = nn.Parameter(
            torch.zeros(1, 1, predictor_d_model))
        self.world_pos_embed = nn.Parameter(
            torch.zeros(1, max_tokens, predictor_d_model))
        nn.init.trunc_normal_(self.world_pos_embed, std=0.02)
        self.horizon_embed = nn.Embedding(max_horizon + 1, predictor_d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation=F.gelu, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(predictor_d_model)
        self.out_proj = nn.Linear(predictor_d_model, world_dim)

    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor,
                horizon: int = 1) -> torch.Tensor:
        # s_t: (B, P, world_dim)   a_t: (B, K, action_dim)
        B, P, _ = s_t.shape
        assert P <= self.world_pos_embed.size(1), (
            f"ActionPredictor got {P} world tokens > max_tokens="
            f"{self.world_pos_embed.size(1)}")

        # Broadcast a pooled action vector onto every world token (FiLM-style)
        # so the action directly conditions the prediction, then also prepend
        # the action tokens for richer per-token routing.
        action_bias = self.action_to_world_bias(a_t.mean(dim=1, keepdim=True))
        world = self.in_proj_world(s_t) + self.world_pos_embed[:, :P] + action_bias
        action = self.in_proj_action(a_t) + self.action_type_embed

        k = torch.tensor(
            min(horizon, self.horizon_embed.num_embeddings - 1),
            device=s_t.device, dtype=torch.long)
        h_emb = self.horizon_embed(k).view(1, 1, -1)
        tokens = torch.cat([action + h_emb, world + h_emb], dim=1)

        h = self.norm_out(self.encoder(tokens))
        world_out = h[:, a_t.size(1):]          # drop the K action tokens
        pred = self.out_proj(world_out)         # (B, P, world_dim)
        if self.pred_residual:
            pred = pred + s_t
        return pred


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class ActionWorldModelWrapper(nn.Module):
    """Compose EEG encoder + action head + predictor + frozen V-JEPA 2 world.

    Invoked by the pretraining trainer like any model:
    ``training_step(batch, mask=None)`` returns ``(out, info)``.

    The action is **always** decoded from the *full, unmasked* window. Masked
    patch reconstruction is an *optional* co-objective done as a SEPARATE
    masked forward so it never corrupts the action signal:

    * **wrapper-driven** (``--need_mask ''`` + ``--recon_aux_weight > 0``): the
      wrapper masks internally and adds an MSE ``recon_aux_loss`` to ``info``.
    * **trainer-driven** (leave ``--need_mask`` on): the trainer passes a
      ``mask``; the wrapper returns the reconstruction ``out`` and the
      trainer's existing ``mask_loss`` handles it (band-split / freq options
      for free). ``recon_aux_weight`` is ignored on this route to avoid
      double-counting.

    With both off (``--need_mask ''``, ``recon_aux_weight == 0``) only the
    world-prediction terms enter ``info`` and ``out`` is ``None``.

    No EMA / no ``update_target_encoder`` (the target encoder is frozen), so
    the trainer's EMA hook is skipped automatically.
    """

    def __init__(
        self,
        eeg: nn.Module,                 # CSBrainAlign (holds frozen V-JEPA 2)
        predictor: ActionPredictor,
        action_head: ActionHead,
        pred_l1_weight: float = 1.0,
        pred_cos_weight: float = 0.0,
        max_horizon: int = 1,
        recon_aux_weight: float = 0.0,
        recon_mask_ratio: float = 0.5,
    ):
        super().__init__()
        self.eeg = eeg
        self.predictor = predictor
        self.action_head = action_head
        self.pred_l1_weight = pred_l1_weight
        self.pred_cos_weight = pred_cos_weight
        self.max_horizon = max_horizon
        self.recon_aux_weight = recon_aux_weight
        self.recon_mask_ratio = recon_mask_ratio
        assert getattr(eeg, 'encoder_kind', None) == 'vjepa2', (
            "ActionWorldModel needs a V-JEPA 2 vision encoder; build the EEG "
            "CSBrainAlign with vision_encoder='facebook/vjepa2-vitl-fpc64-256'")

    def _eeg_batch(self, batch: dict) -> dict:
        # Deliberately exclude image_encoder_inputs/has_image so the EEG
        # forward never triggers the (unused) image-alignment branch.
        out = {'timeseries': batch['timeseries'], 'ch_coords': batch['ch_coords']}
        for k in ('ch_names', 'valid_channel_mask', 'valid_length_mask', 'source'):
            if k in batch:
                out[k] = batch[k]
        return out

    def training_step(self, batch: dict, mask: Optional[torch.Tensor] = None):
        eeg_batch = self._eeg_batch(batch)

        # 1. EEG window t -> action latent a_t. ALWAYS decoded from the FULL,
        #    unmasked window so the optional masked-recon below never corrupts
        #    the intention signal.
        _, enc_info = self.eeg({**eeg_batch}, encoder_only=True)
        patch_tokens = enc_info['patch_tokens']            # (B, C, N, d)
        a_t = self.action_head(
            patch_tokens,
            batch.get('valid_channel_mask'),
            batch.get('valid_length_mask'),
        )                                                  # (B, K, action_dim)

        # Carry any encoder auxiliary losses (e.g. MoE) through untouched.
        info = {k: v for k, v in enc_info.items() if 'loss' in k}

        # 2. Optional masked-patch reconstruction (a SEPARATE masked forward).
        out = None
        if mask is not None:
            # Trainer-driven (--need_mask on): return the reconstruction so the
            # trainer's mask_loss handles it; don't also add recon_aux_loss.
            out, _ = self.eeg({**eeg_batch}, mask=mask)
        elif self.recon_aux_weight > 0:
            # Wrapper-driven (--need_mask '' + --recon_aux_weight>0).
            x = batch['timeseries']
            B, C, N, _ = x.shape
            rmask = generate_mask(
                B, C, N, mask_ratio=self.recon_mask_ratio, device=x.device)
            out, _ = self.eeg({**eeg_batch}, mask=rmask)
            recon_loss = F.mse_loss(out[rmask == 1], x[rmask == 1])
            info['recon_aux_loss'] = (self.recon_aux_weight, recon_loss)

        # 3. World prediction on a random horizon k.
        pv_f = batch.get('pixel_values_future')            # (B, W, 3, H, W)
        has_f = batch.get('has_image_future')              # (B, W)
        if pv_f is None or has_f is None or self.max_horizon < 1:
            return out, info

        W = pv_f.size(1)
        assert W >= self.max_horizon + 1, (
            f"pixel_values_future has W={W} windows but max_horizon="
            f"{self.max_horizon} needs at least {self.max_horizon + 1}")
        k = int(torch.randint(1, self.max_horizon + 1, ()).item())

        # Supervise only rows whose frame exists at BOTH t and t+k.
        valid = has_f[:, 0] & has_f[:, k]
        if int(valid.sum().item()) == 0:
            return out, info

        s_t = self.eeg.vjepa2_grid_tokens(pv_f[valid, 0])      # (Bv, P, Dw) frozen
        s_tpk = self.eeg.vjepa2_grid_tokens(pv_f[valid, k]).detach()
        a_v = a_t[valid]

        pred = self.predictor(s_t, a_v, horizon=k)             # (Bv, P, Dw)
        l1 = F.l1_loss(pred, s_tpk)
        info['world_pred_loss'] = (self.pred_l1_weight, l1)
        if self.pred_cos_weight > 0:
            cos_loss = 1.0 - F.cosine_similarity(pred, s_tpk, dim=-1).mean()
            info['world_cos_loss'] = (self.pred_cos_weight, cos_loss)

        # Diagnostics — the key failure mode is the predictor IGNORING the
        # action (consecutive frames look alike). ``diag_action_gap`` must be
        # > 0 and ``diag_pred_cos`` must beat ``diag_copy_cos``.
        with torch.no_grad():
            pred_zero = self.predictor(s_t, torch.zeros_like(a_v), horizon=k)
            info['diag_action_gap'] = F.l1_loss(pred_zero, s_tpk) - l1
            info['diag_pred_cos'] = F.cosine_similarity(pred, s_tpk, dim=-1).mean()
            info['diag_copy_cos'] = F.cosine_similarity(s_t, s_tpk, dim=-1).mean()
            info['diag_s_t_norm'] = s_t.flatten(end_dim=-2).norm(dim=-1).mean()
            info['diag_pred_norm'] = pred.flatten(end_dim=-2).norm(dim=-1).mean()
            info['diag_action_norm'] = a_v.flatten(end_dim=-2).norm(dim=-1).mean()
            info['diag_n_valid_pairs'] = torch.tensor(
                float(valid.sum().item()), device=pred.device)
            info['diag_horizon'] = torch.tensor(float(k), device=pred.device)

        return out, info

    def forward(self, *args, **kwargs):  # convenience
        return self.training_step(*args, **kwargs)
