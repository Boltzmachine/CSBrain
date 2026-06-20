# EEG World-Model / Video-Prediction Extension of CSBrainAlign

## 1. Motivation

Cognitive neuroscience suggests the brain continuously predicts upcoming
sensory states; recent World-Action Models (WAM) and V-JEPA are
instantiations of this idea that learn a forward predictor in a latent space
rather than reconstructing pixels. We want to carry that inductive bias into
EEG: the EEG recorded *now* should (a) explain the current video frame and
(b) carry information to predict the *next* latent state. That same
forward-looking representation should then transfer to motor-imagery
decoding, where anticipatory preparatory signals are the thing we want to
read out.

## 2. Notation

| Symbol | Meaning |
|---|---|
| `x_t ∈ R^{C×N×d}` | EEG segment aligned to time `t` (current patching) |
| `y_t` | Video frame at time `t` |
| `s_t = (s_t^{cls}, s_t^{patch})` | Latent "world state" at time `t`. CLS token `s_t^{cls} ∈ R^{D}` and patch tokens `s_t^{patch} ∈ R^{C×N×D}` — both come from `CSBrainAlign.encode`. |
| `f_θ` | EEG encoder (the current `CSBrainAlign`, re-purposed) |
| `g_ξ` | Frame encoder — this is `CSBrainAlign.pretrained_image_encoder`, already loaded and frozen today |
| `p_φ` | Latent predictor: `(s_t, x_t) → ŝ_{t+1}^{patch}` (predicts patch tokens; see §4) |
| `h_ψ` | Contrastive / alignment projector (reuses `contrastive_proj`) |

## 3. Architecture

```
                          s_t^{cls}  ───▶ h_ψ ───▶ align(·, g_ξ(y_t)[CLS])
        ┌──────────────┐       │
 x_t ──▶│  f_θ (EEG)   │───────┤
        └──────────────┘       │
                          s_t^{patch} ───┐
                                         ▼
                                ┌────────────────┐
                                │ p_φ predictor  │── ŝ^{patch}_{t+1} ──▶ L1 vs.
                                │ inputs:        │    sg[f_θ(x_{t+1})^{patch}]
                                │   s_t (all     │
                                │   tokens), x_t │
                                └────────────────┘
```

Three modules:

1. **EEG encoder `f_θ`** — `CSBrainAlign` returns a CLS token we
   treat as `s_t`. Keep the patch-embed / transformer stack as-is. We are
   *not* touching the equivariance branch at all.
2. **Frame encoder `g_ξ`** — the `self.pretrained_image_encoder` already
   inside `CSBrainAlign` (it happens to be DINOv2-base, but that is just the
   choice of weights; we use it as a black-box frozen feature extractor, the
   same way the current alignment loss already does).
3. **Predictor `p_φ`** — new, small transformer (≈4–6 layers, `d_model=512`),
   lives in `models/world_model.py`. Inputs: `s_t` (CLS + patch tokens from
   `f_θ`) concatenated with patch-embedded `x_t` tokens; output:
   `ŝ_{t+1}^{patch}`. Kept intentionally narrow so the encoder carries the
   representational load (V-JEPA design choice).

### CLS vs patch tokens for `s_t`

We keep **both**. The two heads use different slices:

- **Alignment** operates on the CLS token only: `h_ψ(s_t^{cls}) ↔ g_ξ(y_t)[CLS]`.
  This matches the current `CSBrainAlign` code; CLS is the natural summary
  to match a single frame-level ViT feature.
- **Prediction** operates on the *patch tokens*: `p_φ` predicts
  `ŝ_{t+1}^{patch}` against `sg[f_θ(x_{t+1})^{patch}]`. This follows V-JEPA
  (which predicts patch embeddings, not a scalar summary) and — crucially
  for the downstream motor-imagery story — preserves per-channel / per-time
  structure. MI relies on spatially-localised mu/β patterns over C3/C4;
  collapsing to a single CLS destroys that information.

`CSBrainAlign.encode` at [alignment.py:596-602](models/alignment.py#L596-L602)
already returns both `cls_token` and `patch_tokens`, so no plumbing change.

### Why both `s_t` and `x_t` feed the predictor

`s_t` encodes the brain's model of the *current* world state — the part of
the latent that is explicitly tied to `y_t` via the alignment loss. `x_t`,
on the other hand, carries the raw anticipatory/preparatory signals the
brain produces *before* the next frame is observed. Those signals are
exactly what we want the predictor to exploit: the human's intuition about
what comes next is already physically present in the EEG, but the
frame-aligned bottleneck `s_t` may squeeze it out. Feeding `x_t` (via the
encoder's `patch_embedding` tokens, not raw samples) gives `p_φ` direct
access to that predictive EEG content.

## 4. Objectives

Total loss = α·L_align + β·L_recon + γ·L_pred .

No EMA teacher, no DINO-style self-distillation on the EEG side, no
VICReg on the predictor — those were tried and did not help in this
codebase.

1. **Alignment (existing)** — symmetric InfoNCE between `h_ψ(s_t)` and
   `g_ξ(y_t)[CLS]`. Already implemented in
   [alignment.py:742-754](models/alignment.py#L742-L754); kept verbatim.
2. **Masked patch reconstruction (existing)** — the current pretraining
   objective in `pretrain_trainer.py` that reconstructs masked EEG patches
   from `CSBrainAlign.forward`'s `out`. Kept verbatim. This is what keeps
   `f_θ` grounded in raw EEG content; the alignment loss alone is too weak
   to train a foundation-quality encoder.
3. **Latent prediction** — **L1 loss** between `p_φ(s_t, x_t)` and
   `sg[f_θ(x_{t+1})^{patch}]` (patch-token target — see §3). Same online
   encoder, stop-grad on the target only (no EMA). Averaged over valid
   channel × time-patch positions. Also regress the predicted
   CLS to `g_ξ(y_{t+1})` as an auxiliary cross-modal term.

Multi-step rollout: at training time pick a random horizon `k ∈ {1,…,K_max}`
and supervise `ŝ_{t+k}^{patch}` against `sg[f_θ(x_{t+k})^{patch}]`. Start
with `K_max=1`, ramp to 4 once losses are stable.

## 5. Data Pipeline — CineBrain

- Add `datasets/cinebrain_dataset.py` producing
  `{timeseries (C,N,d), ch_coords, frame_t, frame_{t+1..t+K}, valid_*}`.
- A single `__getitem__` returns temporally-consecutive EEG windows *and*
  the raw frame tensors they are aligned to. Frames are fed through the
  in-model `self.pretrained_image_encoder` at train time, the same path
  the current alignment loss uses — no separate cache file.

(update: change to egobrain)

### Window size / stride (reconsidered with EEG domain knowledge)

Constraints from EEG:
- Visual ERPs peak in the 100–300 ms range post-stimulus (P100, N170,
  P300); object-category information is maximal around 150–250 ms. So a
  "per-frame" window must *at least* cover ~400 ms after the nominal
  frame onset for the response to be represented.
- CineBrain presents continuous video (24–30 fps, i.e. one frame every
  33–42 ms), so frame-locked epoching is not possible in the ERP sense;
  we need a sliding window over a continuous stream.
- The existing `CSBrainAlign` uses `event_window_len = 240`, which at the
  repo's typical 200 Hz sampling rate is **1.2 s**. That is the effective
  "alignment unit" the current model was designed around — we should stay
  close to it.

Proposed defaults:
- **Window**: 2.0 s (≈400 samples @ 200 Hz). Long enough to span multiple
  frame responses and to give the transformer enough patches (`N ≥ 10`
  with `d = 40`, or kept compatible with `in_dim=200` by using larger
  windows). Shorter than the 30 s global-spectral context the backbone
  was trained on, but the alignment loss already operates on
  sub-second-scale content.
- **Stride**: 1.0 s (50 % overlap). Gives a 1 s horizon between
  consecutive windows — well matched to the timescale over which
  anticipatory EEG is informative.
- **Frame alignment inside a window**: the target frame `y_t` is the frame
  displayed at the window's *center*, shifted by +150 ms to roughly
  account for visual-ERP latency. The exact offset should be swept once
  we have Stage-A training running; this is a hyperparameter, not a
  load-bearing design choice.
- **Prediction horizon**: default `k=1` (predict the next 1 s window).
  Ramp `K_max → 4` (2–4 s ahead) later.

These values should be revisited once we inspect the CineBrain metadata
(sampling rate, exact video timing) — nothing above is so firmly justified
that it shouldn't move.

## 6. Training Recipe

One pretraining stage (no warmup split, no EMA, no equivariance):

- Loss: `L_align + L_recon + γ · L_pred`, all three live from step 0.
- `γ` schedule: linear ramp 0 → 1 over the first 2 epochs so the
  untrained predictor cannot push noise gradients back through `f_θ`
  while the encoder is still finding the alignment+recon basin.
- Optimiser / LR / batch size: inherit from the existing
  `pretrain_main.py` configuration for `CSBrainAlign` (no reason to
  change; the added parameters are in a small side-network).
- Logging: track `L_align`, `L_recon`, `L_pred`, predictor target-cosine
  similarity on the patch tokens, and a rank-based next-window retrieval
  metric (mean-pooled `ŝ_{t+1}^{patch}` vs a batch bank of true
  `s_{t+1}^{patch}`).

### Downstream

Fine-tuning for motor imagery is **deliberately deferred** until the
pretraining behaviour looks right. Plan to revisit once we have
Stage-B checkpoints and per-epoch curves. When we do, `f_θ` is the only
thing to load; `p_φ` and `h_ψ` are discarded.

## 7. Concrete Code Changes

| File | Change |
|---|---|
| `models/world_model.py` *(new)* | `LatentPredictor` (transformer), `WorldModelWrapper` wrapping `CSBrainAlign` + `p_φ` and producing the loss dict. |
| `models/alignment.py` | Expose `patch_embedding` output + the final `encode` pathway in a way the predictor can reuse without re-running the full transformer. No change to the existing `forward` behaviour. |
| `datasets/cinebrain_dataset.py` *(new)* | Loader returning `(EEG window, frame_t, frame_{t+1..t+K_max})` tuples. No caching layer in v1. Apply necessary preprocessing. |
| `datasets/__init__.py` | Register CineBrain. |
| `pretrain_main.py` / `pretrain_trainer.py` | Add a `world_model` branch; the loss dict already supports multiple weighted terms via the `(weight, tensor)` tuple convention, so no trainer surgery is needed. |
| `sh/pretrain_worldmodel.sh` *(new)* | Launch script (`conda run -n cbramod`). |

Fine-tuning files are untouched for now.


## Other Important Things
1. Keep your code back-compatible.
2. For some hyperparameters, especially the timing alignment or offset, window or stride size, you should **carefully** investigate the dataset's paper or the corresponding method proposed along with the paper to find out the optimal and reasonable choice.
3. Preprocessing should be conducted including some band filtering and resampling (EEG should be resampled to 200Hz if necessary). For example, in `tueg_dataset.py`, we have
```
raw.resample(200)
raw.filter(l_freq=0.3, h_freq=75)
raw.notch_filter((60))
```
4. Add the training script to `sh.pretrain_CSBrain.sh`.
5. Conduct necessary unit test and end-to-end-test if necessary.


## Dataset
The dataset is downloaded into `data/CineBrain`. Refer to the official codebase `./CineBrain` for potential data loading pipeline.
---

## Action-Conditioned variant (EEG = action, V-JEPA 2 = world)

The design above treats **EEG as the world** (predict the next EEG latent;
video only for CLS alignment). This variant is the **inverse** and a faithful
**V-JEPA 2-AC analog** — implemented separately as model type
`ActionWorldModel` (`models/action_world_model.py`), leaving the `WorldModel`
path untouched.

- **World `g`** = a *frozen* V-JEPA 2 (`facebook/vjepa2-vitl-fpc64-256`,
  hidden 1024). A single static frame is replicated to the T=2 tubelet →
  **1 temporal patch × 16×16 = 256 spatial tokens × 1024**, i.e. a purely
  *spatial* grid `s_t` (no temporal axis). Exposed via
  `CSBrainAlign.vjepa2_grid_tokens` (un-pooled `last_hidden_state`).
- **Action `a_t`** = a learned "moving intention" latent the EEG model decodes
  (`ActionHead`: K attention-pooled tokens over the EEG patch tokens).
  *Unsupervised* — no IMU/motion targets; trained end-to-end by the prediction
  loss only.
- **Predictor `p`** (`ActionPredictor`) prepends the action tokens to the
  projected world grid (+pos +horizon embeds), encodes, and reads out
  `ŝ_{t+k}` (optionally as a residual `s_t + Δ`).

```
 frame_t   ─▶ frozen V-JEPA2 ─▶ s_t (B,256,1024) ─┐
 x_t (EEG) ─▶ CSBrainAlign ─▶ ActionHead ─▶ a_t ──┤
                                                   ▼
                                       ActionPredictor ─▶ ŝ_{t+1}
                                                   │
                       L1(+cos) vs sg[VJEPA2(frame_{t+1})] ◀┘  (frozen target)
```

**Objective:** `L = L_pred` (L1 over the 256 tokens, optional `+ (1−cos)`).
Frozen target ⇒ **no EMA teacher, no loss ramp, no collapse risk** (the major
simplification over `WorldModel`).

**Optional masked-patch reconstruction co-objective.** The action is *always*
decoded from the full, unmasked window; recon (when enabled) is a *separate*
masked EEG forward so it never corrupts the action signal. Two routes:
- `--recon_aux_weight W` (with `--need_mask ''`): the wrapper masks internally
  (ratio `--mask_ratio`) and adds an MSE `recon_aux_loss` to the loss dict.
- leave `--need_mask` on: the trainer passes a mask, the wrapper returns the
  reconstruction `out`, and the trainer's existing `mask_loss` handles it
  (band-split / freq-mask options for free). `recon_aux_weight` is ignored on
  this route to avoid double-counting.

Both default off (`--need_mask ''`, `recon_aux_weight=0`).

**EEG warm-start:** `--eeg_ckpt <recon.pt>` loads a masked-recon backbone
(shape-matched, vision/alignment heads skipped) via `_load_eeg_ckpt` in
`models/__init__.py`.

**Data:** EgoBrain only (video subjects P0001–P0024); egocentric self-motion is
the natural match for "moving intention". Reuses the existing `egobrain`
dataset branch + `collate_egobrain` (already emits `timeseries_future`,
`pixel_values_future`, `has_image_future`). One-time prerequisite: build the
V-JEPA 2 256px frame cache (`datasets.egobrain_extract_frames
--vision_encoder facebook/vjepa2-vitl-fpc64-256`).

**Action-utility diagnostics (critical):** the classic failure is the
predictor ignoring the action (consecutive frames look alike). The wrapper logs
`diag_action_gap` = `L_pred(zeroed a) − L_pred(a)` (must be > 0) and
`diag_pred_cos` vs `diag_copy_cos` (must beat the trivial-copy baseline).

Launch: `sh/pretrain_actionworldmodel.sh`.
