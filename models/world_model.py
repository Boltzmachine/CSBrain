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
    ):
        super().__init__()
        self.encoder = encoder
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
        if self.predictor is not None and self.max_horizon >= 1:
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

    def training_step(self, batch: dict, mask: Optional[torch.Tensor] = None):
        # 1. Primary forward on window t — produces the existing
        #    reconstruction output + alignment/recon loss terms. Runs on
        #    the full mixed batch (CineBrain + Alljoined) so masked recon
        #    and image alignment train on every sample.
        encoder_batch = self._build_alignment_batch(batch, window_idx=0)
        # Equivariant frame-averaging frontend (plans/eeg-wm.md): sample ONE
        # ``flip`` for the whole step and share it across the current and future
        # window encodes so the presentation (z vs P(z), original vs mirrored
        # frame) is consistent. Eval is canonical (no flip).
        frame_avg = getattr(self.encoder, 'frame_averaging', False)
        flip = False
        if frame_avg:
            flip = (bool(torch.rand(()) < getattr(
                        self.encoder, 'frame_avg_flip_prob', 0.5))
                    if self.training else False)
            encoder_batch['flip'] = flip
        # Supply a per-sample motion score (t->t+1 frame diff) so the encoder's
        # flip-alignment can down-weight static frames (whose horizontal flip is
        # near-vacuous). Only when that weighting is enabled on the encoder.
        if getattr(self.encoder, 'flip_motion_ref', 0.0) > 0:
            motion = self._compute_flip_motion(batch)
            if motion is not None:
                encoder_batch['flip_motion'] = motion
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
            # Present the future window with the SAME flip so the regression
            # target is the flipped-future EEG latent when this is a flip step
            # (predict flipped future from flipped current — the bilateral
            # control signal that makes the lateral decomposition non-trivial).
            future_batch['flip'] = flip
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
