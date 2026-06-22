# EEG-WM: equivariance/invariance frontend (from eeg-wm.drawio.xml, Page-2)

Goal: Upgrade the current lateral/bilateral design to an **explicit
even/odd decomposition under the homologous-channel swap `P`** (P = swap C3↔C4…,
an involution `P²=I`), so `x_bi` is P-invariant and `x_lat` is P-equivariant **by
construction**.

## Forward (prediction)
1. Inputs: `x` and its mirror `P(x)` (P = homologous-channel swap, `P²=I`), where `P(x) = x_bi + flip(x_lat)`.
2. **Channel-wise frontend `f`** (depthwise — *no* cross-channel mixing, so it
   commutes with `P`). Learned; discovers `x_lat` from `x` like the current
   `LateralizationSplit`. (Candidate names: channel-wise / depthwise / P-equivariant
   frontend.)
   - **Reuse the CNN `PatchEmbedding`**, but add an argument to drop its conv
     positional encoding (the depthwise Conv2d with kernel 19 along the channel
     axis — that conv is the only cross-channel mixer; with it off the patch
     embed is channel-independent and commutes with `P`).
   - Caveat: its `GroupNorm` still pools normalization stats across the channel
     axis, so it's not strictly channel-independent — check whether that breaks
     the `P`-equivariance enough to matter.
3. **Frame averaging** (always part of the forward, *not* a flip-only branch):
   run `f` on both frames and combine into the invariant + equivariant parts —
   - `x_bi  = (f(x) + f(P(x))) / 2`        (P-invariant  → bilateral)
   - `x_lat = (f(x) − P(f(P(x)))) / 2`      (P-equivariant → lateral)
4. **Transformer** over the bi/lat streams → `bi features`, `lat features`;
   Tokens are **split by feature dimension** into a bi-half and a lat-half.

## Training
- **`flip?` = a RANDOM per-step signal**: with some probability use `flip(x_lat)`
  so the reconstruction target becomes `P(x) = x_bi + flip(x_lat)` instead of `x`.
- **Reconstruction loss**: recon head (from bi features) → `x̂`, matched to the
  (possibly flipped) input — `x`, or `P(x)` when flipped.
- **Alignment loss**: bi & lat global tokens → image (as current implementation).


## Important tests
- Check the equivariance / invariance of the architecture.
