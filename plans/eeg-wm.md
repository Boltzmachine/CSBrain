# EEG-WM: equivariance/invariance frontend

Goal: upgrade the current lateral/bilateral design into an equivariance/invariance decomposition under an operation `P` (the homologous-channel swap C3↔C4…, an involution with `P² = I`).

## NOTE
- Two distinct meanings of *frame*: the `frame` in *frame averaging* vs. the `frame` in *video frames* — don't conflate them.
- Two distinct uses of *flip*: (1) the flip of the bilateral features (its operator `P`), which enforces the equivariance/invariance of the transformer block; and (2) the flip used to match the flipped video frames. They are the same operation, but they are not coupled.

## Forward (prediction)
1. **Input.** A frontend `f` encodes `x` into lateral and bilateral information: `z = z_bi + z_lat = f(x)`. Define `P(z) = z_bi + flip(z_lat)`, where `flip` is the homologous-channel swap and `P² = I`.
2. **Channel-wise frontend `f`** (depthwise — *no* cross-channel mixing, so it commutes with `P`). Learned; it discovers `z_bi` and `z_lat` from `x`, like the current `LateralizationSplit`. (Candidate names: channel-wise / depthwise / P-equivariant frontend.)
   - **Reuse the CNN `PatchEmbedding`**, but add an option to drop its
     convolutional positional encoding (the depthwise Conv2d with kernel 19 along the channel axis). That conv is the only cross-channel mixer; with it off, the patch embedding is channel-independent and commutes with `P`.
   - Caveat: its `GroupNorm` still pools normalization statistics across the channel axis, so the frontend is not strictly channel-independent — check whether this breaks `P`-equivariance enough to matter.
3. **Frame averaging** (always part of the forward pass, *not* a flip-only branch): run the transformer block `T` on both frames and combine them into the invariant and equivariant parts —
   - `h_bi  = (T(z) + T(P(z))) / 2`        (P-invariant  → bilateral)
   - `h_lat = (T(z) − P(T(P(z)))) / 2`      (P-equivariant → lateral)
4. **Transformer** over the bi/lat streams → `bi features`, `lat features`.
   Tokens are **split along the feature dimension** into a bi-half and a
   lat-half: `h = [h_bi; h_lat]`.

## Training
- **`flip?` is a RANDOM per-step signal**: with some probability, use
  horizontally-flipped video frames; in that case the input timeseries is treated as if it were the one that produces `P(z)`. Either run one flip or non-flip in one forward pass.
- **Reconstruction loss**: the recon head is matched to the input `x`. On a flipped forward, run the frontend on the reconstructed series again and apply the reconstruction loss between its output and the previous output of itself (because there is no ground-truth timeseries for the one that produces the flipped video frame).
- **Alignment loss**: bi & lat global tokens → image (as in the current
  implementation).

The non-triviality of the learned decomposition is guaranteed by predicting the flipped video frames (when you horizontally flip a video, left-hand movements become right-hand movements).

## Important tests
- Check the equivariance/invariance of the architecture.
