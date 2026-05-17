# Online BCI Evaluation with the EEG World Model

## 1. Motivation

The world model in [models/world_model.py](models/world_model.py) was trained
to predict the next latent EEG state `ŝ_{t+k}` from the current state `s_t`
(see [plans/world_model.md](plans/world_model.md)). The pretraining
hypothesis is that anticipatory / preparatory EEG content is exactly what
motor-imagery (MI) decoding needs to read out. This document plans a
closed-loop application that **tests that hypothesis in vivo**: an OpenBCI
headset streams EEG while the subject controls a first-person game first
passively (cued imagery) and then actively (free imagery), and we evaluate
how well the pretrained encoder / predictor transfer.

## 2. Scope & success criterion

- **Task**: closed-loop BCI demo controlling **Minecraft Java Edition**
  (first-person view) with **2 discrete actions {left, right}**, realised
  as ~200 ms presses of `A` / `D` (strafe). Subject auto-walks forward
  (held `W`); each decoder emission nudges them sideways. Single subject,
  single session per evaluation.
- **Why Minecraft is workable here** (despite the timing concerns flagged
  for a pixel-grounded paradigm): the EEG decoder does not consume game
  frames as input — it only consumes EEG. Minecraft's rendering jitter
  therefore does not enter the EEG ↔ label pipeline. The only thing that
  has to be tight is the **cue → imagery-onset marker**, which is pushed
  by the paradigm process (overlay + LSL), not by Minecraft itself. So MC
  is fine as the display surface as long as the marker stream stays
  decoupled from the game.
- **Control**: vanilla Java Edition driven by **pynput** key/mouse
  injection — no mods, no Malmo/MineRL dependency. Subject plays in a
  prebuilt flat-world corridor so "go left / right" has unambiguous
  semantics and there are no enemies / inventory distractions.
- **Why 2-class**: standard first-time online MI configuration (chance
  = 50 %). Once 2-class works, extend to 3-class (add no-op) or 4-class
  (add jump / forward-burst).
- **Primary metric**: per-trial classification accuracy in Phase C
  (active imagery → predicted command). Secondary: ITR (bits/min),
  per-window inference latency, false-positive rate during rest.
- **Go/no-go gate**: a fast frozen-encoder linear probe (§6.5) on
  Phase A data must beat 65 % accuracy 2-class **before** running the
  full fine-tune in §6.1–6.4. If it doesn't, the pretrained encoder
  either isn't generalising to OpenBCI or the montage map (§5) is
  wrong; fix that upstream issue first — running the full fine-tune
  won't rescue a broken signal pipeline.

## 3. Hardware, deployment, and streaming

- **Headset**: OpenBCI Cyton (8 ch @ 250 Hz, 24-bit). Montage (motor focus):
  **C3, C4, Cz, FC1, FC2, CP1, CP2, Fz**. C3/C4 are the mu/β
  lateralisation channels; the rest provide enough spatial context for the
  encoder's `ch_coords` lookup to behave like it does on training data.
- **Deployment**: single GPU PC. Headset dongle plugs into this machine;
  OpenBCI GUI, the Pygame paradigm, and inference all run locally. LSL is
  localhost, so there are no firewall / discovery concerns.
- **OpenBCI GUI**: enable the **Networking → LSL** widget, push one EEG
  stream `OpenBCI_EEG` (float32, n_channels × samples) at 250 Hz.
- **Markers**: the paradigm process pushes a second LSL stream
  `BCI_Markers` (int32) carrying `(trial_id, class_label, event_code)` per
  cue/event. `pylsl.local_clock()` gives sub-millisecond alignment between
  the two streams — do NOT rely on Python wall-clock timestamps.
- **Pretrained checkpoint**: a pretrained world-model checkpoint already
  exists on disk (per user, e.g. the one referenced in
  [sh/finetune_CSBrain_PhysioNet.sh](sh/finetune_CSBrain_PhysioNet.sh):
  `outputs/worldmodel-mix-cinebrain-v8-seed43/epoch4_*.pth`). The
  pipeline loads it as `--foundation_dir`, **discards `p_φ` and `h_ψ`**,
  fine-tunes the encoder + a classifier head on Phase A data (§6), and
  ships that fine-tuned model to Phase C inference.

## 4. Paradigm

Two cooperating processes on the GPU PC:

- **Minecraft Java Edition** — vanilla, launched into a flat-world
  corridor save. The subject sees this fullscreen on the primary monitor.
- **`bci_app/paradigm.py`** — a Python process that (a) draws a
  **transparent always-on-top overlay window** (PyQt5 or tkinter
  with `-topmost` + alpha) showing the fixation cross and cue arrows on
  top of Minecraft, (b) pushes LSL markers, and (c) drives Minecraft via
  **pynput** key injection (`W` held during phases, `A`/`D` pressed for
  ~200 ms on cue / on decoder emission).

The overlay timing — not Minecraft's frame timing — defines the
EEG-relevant moments. The cue marker is pushed at the same instant the
overlay arrow is drawn (single LSL `push_sample` call), so the
EEG ↔ marker alignment is independent of Minecraft's render loop.

### Phase A — passive cued imagery (training data collection)

Graz-MI inspired trial structure:

```
| 2.0 s fixation cross | 1.25 s cue (arrow overlay + paradigm presses A or D on MC) | 4.0 s imagery window | 1.0–2.0 s rest (jitter) |
```

- **Fixation**: overlay shows a `+` at screen centre; `W` is released, so
  the subject stands still in MC. Marker `fixation_on`.
- **Cue**: overlay shows a large translucent arrow (← or →). The
  paradigm presses the corresponding strafe key in MC for the full
  1.25 s so the subject **sees the avatar move in the cued direction**.
  Marker `cue_left` / `cue_right`. The visual feedback during the cue is
  what makes "cued imagery" different from cold imagination — the
  subject's motor cortex is primed by watching the action play out
  before being asked to imagine it.
- **Imagery window**: arrow overlay fades; MC continues to display the
  static scene (no key input). Subject **imagines performing the strafe**
  in the cued direction. Marker `imagery_on` at start, `imagery_off` at
  end.
- **Rest**: blank overlay, jittered 1–2 s.

The 4 s imagery window yields **three overlapping 2 s EEG samples**
(stride 1 s), matching the world-model's training windowing in
[plans/world_model.md:138-147](plans/world_model.md#L138-L147).

~40 trials per class × 2 classes = 80 trials ≈ 13 min per block. Plan
**3 blocks**: 240 trials total. The third block is held out as the
Phase B test set.

### Phase B — calibration check (offline)

Same trial structure as Phase A. The held-out block is scored against the
decoders trained in §6 to confirm offline accuracy before unlocking active
control. **No model update happens here** — this is the go/no-go gate.

### Phase C — active control (closed loop)

- Subject auto-walks forward in MC (`W` held by the paradigm process for
  the full phase). No cue arrows by default.
- Sliding-window decoding every 250 ms (§7). On emission, the paradigm
  presses `A` or `D` in MC for 200 ms — the avatar strafes.
- Two evaluation conditions:
  1. **Free run** (5 min): subject is told to just navigate the
     corridor; we measure ITR and qualitative controllability.
  2. **Cued evaluation** (5 min, ~20 cued targets): the overlay shows
     an arrow telling the subject which direction to imagine, but the
     paradigm does **not** press the strafe key itself — only the
     decoder's emission does. We measure cued-action accuracy. **This
     is the headline number.**

### LSL marker codes

Keep the table sparse and documented in `bci_app/markers.py`:

```
0   trial_start             10  cue_left
1   fixation_on             11  cue_right
2   imagery_on              20  game_action_executed
3   imagery_off             30  inference_emitted_left
4   trial_end               31  inference_emitted_right
                            32  inference_emitted_noop
```

## 5. Real-time signal pipeline

New files: `bci_app/stream.py` (LSL → ring buffer), `bci_app/preprocess.py`.

Two threads: one pulls from LSL into a thread-safe ring buffer (`numpy`
backed, ~10 s capacity); the inference thread consumes 2 s windows every
250 ms.

Preprocessing **must match the training distribution**. Per
[plans/world_model.md:200-205](plans/world_model.md#L200-L205) and
`datasets/tueg_dataset.py`:

```
resample      250 → 200 Hz      (scipy / mne)
bandpass      0.3 – 75 Hz       (zero-phase IIR, applied causally online)
notch         60 Hz             (US line frequency)
re-reference  common average    (optional; only if the model was trained CAR)
z-score       per-channel, running stats from a 30 s eyes-open baseline
scale         / 100             ← matches the trainer's scaling in models/world_model.py:242
patch         (C, N, d=200)     ← C=8, N=2 for a 2 s window
```

**Causal vs zero-phase filtering**: training used `mne.filter` which is
zero-phase; online must be causal to avoid lookahead. Use a Butterworth
order-4 IIR applied with `scipy.signal.lfilter` (not `filtfilt`). This
introduces a small phase shift relative to training — flag for sweep if
linear-probe accuracy is borderline.

**N=2 patches is small**. If the encoder's CSBrain blocks expect more
temporal patches than the 2 s window provides, lengthen the inference
window to 3 s (`N=3`) and re-record Phase A with the same window length.
Verify against the `patch_embedding` configuration in
[models/alignment.py](models/alignment.py) before recording.

## 6. Training the decoder — mirror the PhysioNet finetune recipe

End-to-end workflow:

```
[pretrained world-model checkpoint]
            │  load (foundation_dir)
            ▼
[Phase A on local PC]  ──▶  windows + labels saved to data/preprocessed/live_session/
            │
            ▼
[finetune_main.py --downstream_dataset LiveSession --num_of_classes 2]
            │  produces a fine-tuned classifier checkpoint
            ▼
[Phase C inference]  loads the fine-tuned checkpoint and decodes online
```

The fine-tune step mirrors
[sh/finetune_CSBrain_PhysioNet.sh](sh/finetune_CSBrain_PhysioNet.sh) — same
`finetune_main.py` entrypoint, same `--model Align`, same
`--use_pretrained_weights`, same hyperparameters
(`--dropout 0.3 --weight_decay 0.01 --lr 5e-5`), same
`--use_initial_segment_only` flag. The only differences are
`--num_of_classes 2` (left vs right) and a new
`--downstream_dataset LiveSession` branch.

### 6.1 New `LiveSession` branch in `finetune_main.py`

Add after the existing `PhysioNet-MI` branch:

```python
elif params.downstream_dataset == 'LiveSession':
    load_dataset = live_session_dataset.LoadDataset(params)
    data_loader = load_dataset.get_data_loader()
    model = model_for_live_session.Model(params)
    t = Trainer(params, data_loader, model)
    results = t.train_for_binaryclass()
```

`train_for_binaryclass` follows the SHU-MI pattern in
[finetune_main.py](finetune_main.py) — appropriate for the 2-class L/R
configuration. When we extend to 3+ classes later, switch to
`train_for_multiclass` (same call as PhysioNet-MI).

### 6.2 New `datasets/live_session_dataset.py`

Mirror [datasets/physio_dataset.py](datasets/physio_dataset.py):
- `LoadDataset(params)` reads windows + labels from
  `data/preprocessed/live_session/<session_id>/`.
- Windows are the **same shape** the encoder was pretrained on
  (`C × N × d`, per the resolution open question in §12 about 2 s vs 3 s).
- Train / val / test split = Phase A blocks 1–2 (train) / block 3 (val)
  / Phase C cued evaluation trials (test, populated later).
- Apply the per-channel z-score and `/100` scaling here, mirroring how
  `physio_dataset.py` handles its scaling (do NOT re-z-score in the
  online loop if the dataset already does it — pick one path and stick
  with it).

Register in [datasets/__init__.py](datasets/__init__.py).

### 6.3 New `models/model_for_live_session.py`

Mirror [models/model_for_physio.py](models/model_for_physio.py):
- Loads the pretrained backbone from `--foundation_dir` (the world-model
  checkpoint).
- Wraps it with a classifier head sized to `params.num_of_classes`.
- Honors `--use_pretrained_weights` and `--use_initial_segment_only`.

If the only delta from `model_for_physio.py` is the number of classes
and the input shape, **import and reuse** rather than copy-pasting —
add an optional `num_of_classes` constructor arg if needed.

### 6.4 New `sh/finetune_live_session.sh`

Mirror [sh/finetune_CSBrain_PhysioNet.sh](sh/finetune_CSBrain_PhysioNet.sh)
but without SLURM directives (this runs on the local GPU PC after Phase A,
not on the cluster):

```bash
FOUNDATION_DIR="outputs/<world_model_ckpt>.pth"
SESSION_ID=$(date +%Y%m%d_%H%M%S)

conda run -n cbramod python finetune_main.py \
    --model Align \
    --downstream_dataset LiveSession \
    --datasets_dir data/preprocessed/live_session/${SESSION_ID} \
    --num_of_classes 2 \
    --model_dir outputs/live_session/${SESSION_ID} \
    --foundation_dir "$FOUNDATION_DIR" \
    --wandb_run_name "live_${SESSION_ID}" \
    --results_csv outputs/live_session_results.csv \
    --use_pretrained_weights \
    --use_initial_segment_only \
    --dropout 0.3 \
    --weight_decay 0.01 \
    --lr 0.00005
```

The `${SESSION_ID}` directory is written by `bci_app/paradigm.py` at the
end of Phase A; the fine-tune script picks it up. End of Phase A → run
this script → Phase C uses the new checkpoint.

### 6.5 Optional fast sanity check (CLS linear probe)

Before running the full fine-tune, run a 1-minute frozen-encoder linear
probe on `s_t^{cls}` against the Phase A labels. This is the **go/no-go
gate** in §2: if a linear probe can't beat 65 % 2-class, the full
fine-tune almost certainly won't either, and the failure is upstream
(montage map, preprocessing, electrode contact). Implement as a small
utility in `bci_app/sanity_probe.py` — runs in <1 min so it doesn't
delay the session.

## 7. Online inference loop

New file: `bci_app/decode.py`. Runs in a separate thread from the LSL
puller.

- Sliding 2 s window, stride 250 ms (≈ 4 predictions per second).
- Per step: preprocess → fine-tuned model (encoder + classifier head
  from §6) → softmax. Log every decision (timestamp, class probs, raw
  window hash) to `runs/<session>/decisions.parquet` for offline
  analysis.
- **Smoothing**: 1 s EMA over class probabilities. Raw single-window MI
  predictions are noisy; smoothing trades latency for stability.
- **Dual emission gate** (the difference between a twitchy demo and a
  usable one):

  ```
  emit class c  iff   p(c) > 0.6   AND   c was argmax in ≥ 3 of the last 4 windows
  ```

  Otherwise emit no-op. Both thresholds get a quick sweep on the Phase B
  set before going live.
- Latency budget on local GPU: pull (~5 ms) + preprocess (~3 ms) +
  encoder (~30 ms with a 768-d ViT-base scale model on a recent GPU) +
  decoder (<1 ms) + pynput key dispatch (~5 ms) ≈ **45 ms per window**,
  well under the 250 ms stride. Add a benchmark in `bci_app/decode.py`
  that asserts total < 200 ms and aborts the session if it ever exceeds
  that.
- **Key dispatch contract**: on emission of class `c`, the decode thread
  enqueues `(c, t_emit)` onto a queue consumed by the paradigm process,
  which then injects the corresponding key (`A` or `D`) for 200 ms. Keep
  injection out of the decode thread — pynput can block briefly on X11
  / Wayland depending on the focus state.

## 8. Evaluation protocol

### Offline (Phase A blocks 1–2 → train, block 3 → val)

- **Split by trial, not by window** — windows from the same trial are not
  independent; splitting by window inflates accuracy by 10–20 pp. The
  `LiveSession` dataset class in §6.2 enforces this.
- Report: per-class accuracy, confusion matrix, training / val curves
  (logged to W&B via `finetune_main.py`'s existing hooks).
- **Per-channel ablation**: zero out each channel in turn, rerun the
  fast linear probe (§6.5), report accuracy drop. C3/C4 should be the
  dominant channels for left/right MI; if instead Fp1/Fp2 dominate, the
  model is riding EOG artifacts and the result is invalid.

### Online (Phase C)

- **Cued evaluation**: accuracy vs cued direction across ~20 cued
  targets. Primary headline number.
- **Free run**: ITR (Wolpaw formula), commands/minute, qualitative
  controllability notes from the subject.
- **False-positive rate during rest**: measured during the inter-trial
  rest in Phase A blocks (no imagery should be happening). Should be
  near 50 % (chance) for a well-calibrated decoder; if heavily biased
  toward one class, recalibrate the decision threshold per class.

### Causal sanity check

- Shuffle the time axis within each test trial and rerun the fine-tuned
  model. Accuracy must collapse to chance. Catches the classic BCI
  failure where the model is locked onto a slow drift / electrode-pop
  / EOG artifact rather than imagery-related band power.

## 9. Channel-coordinate mapping

The encoder takes `ch_coords` per channel; a wrong coord mapping silently
destroys spatial generalisation.

- New file: `bci_app/montage.py` mapping OpenBCI electrode labels →
  the project's `ch_coords` convention.
- **Reuse** whatever standard 10-20 lookup is already in
  `datasets/tueg_dataset.py` and `datasets/cinebrain_dataset.py` — do not
  re-derive coordinates by hand.
- **Unit test**: feed a known-good CineBrain sample through both
  `datasets/cinebrain_dataset.py` and `bci_app/montage.py` (with the same
  channel labels) and assert the resulting `ch_coords` tensors are
  bit-identical. This test runs before every Phase A recording.

## 10. File plan

| File | Purpose |
|---|---|
| `bci_app/__init__.py` *(new)* | Package marker |
| `bci_app/markers.py` *(new)* | LSL marker codes, single source of truth |
| `bci_app/montage.py` *(new)* | OpenBCI label → `ch_coords` + unit test |
| `bci_app/stream.py` *(new)* | LSL puller, ring buffer |
| `bci_app/preprocess.py` *(new)* | Resample, filter, z-score, patch — causal |
| `bci_app/overlay.py` *(new)* | Transparent always-on-top cue overlay (PyQt5 or tkinter) |
| `bci_app/mc_control.py` *(new)* | pynput wrappers: hold W, timed A/D press, focus check |
| `bci_app/paradigm.py` *(new)* | State machine for Phase A / B / C, marker push, drives overlay + mc_control |
| `bci_app/decode.py` *(new)* | Online inference, smoothing, emission queue |
| `datasets/live_session_dataset.py` *(new)* | `LoadDataset` for Phase A windows + labels, mirrors `physio_dataset.py` |
| `models/model_for_live_session.py` *(new)* | Classifier head on top of pretrained backbone, mirrors `model_for_physio.py` |
| `finetune_main.py` *(modify)* | Add `LiveSession` branch (§6.1) — small edit, no surgery |
| `datasets/__init__.py` *(modify)* | Register `live_session_dataset` |
| `bci_app/sanity_probe.py` *(new)* | Fast frozen-encoder linear probe — go/no-go gate (§6.5) |
| `sh/finetune_live_session.sh` *(new)* | Local-PC launcher mirroring `sh/finetune_CSBrain_PhysioNet.sh` (§6.4) |
| `sh/run_bci_session.sh` *(new)* | One-command launcher (calibration → A → sanity probe → finetune → B → C) |
| `tests/test_bci_montage.py` *(new)* | Coord-mapping parity test |
| `tests/test_bci_preprocess.py` *(new)* | Causal-filter shape + frequency-response test |
| `plans/online_bci_eval.md` | This document |

## 11. Build order (don't waste a recording session)

1. **Minecraft side ready, no EEG**: install Java MC, build the flat
   corridor world, verify pynput can `W`-hold and inject `A`/`D` presses
   into the focused MC window. Confirm the overlay window draws on top
   without stealing focus from MC (this is the most common gotcha — on
   X11 use `Qt.WindowStaysOnTopHint | Qt.WindowTransparentForInput`).
2. **Fake-EEG end-to-end**: paradigm + overlay + MC + a synthetic LSL
   stream (random noise). Run a full Phase A trial sequence with the
   LSL recorder open — verify markers, key presses, and overlay
   transitions all line up. No headset on. Catches pynput / pylsl /
   focus issues without burning subject time.
3. **Real OpenBCI stream into the ring buffer**: visualise filtered
   signal in real time; confirm preprocessing matches `tueg_dataset.py`
   by feeding the same offline file through both paths and asserting
   numerical equivalence (up to filter direction).
4. **Encoder forward pass on a live 2 s window**: assert output shapes
   and that `ch_coords` is non-degenerate. Log `s_t^{cls}` norms — if
   they're dramatically different from training-time norms, the
   preprocessing pipeline does not match.
5. **Phase A recording** (3 blocks, 240 trials total). Save windows +
   labels under `data/preprocessed/live_session/<session_id>/`.
6. **Fast sanity probe** (§6.5). **Go/no-go gate** — see §2. If this
   fails, stop and debug §5/§9 before fine-tuning.
7. **Run `sh/finetune_live_session.sh`** — the full fine-tune mirroring
   the PhysioNet recipe. Produces `outputs/live_session/<session_id>/`.
8. **Phase B offline scoring** on held-out block 3.
9. **Online control (Phase C)** — `bci_app/decode.py` loads the
   fine-tuned checkpoint.

## 12. Open questions to resolve before recording

- **Window length**: 2 s (matches training) or 3 s (more temporal
  patches)? Decided by `patch_embedding` config in
  [models/alignment.py](models/alignment.py); answer is a one-line read
  but blocks Phase A.
- **Re-referencing**: did pretraining use CAR? Check
  `datasets/tueg_dataset.py` / `datasets/cinebrain_dataset.py` and match.
- **Eye-blink artifact handling**: at minimum, reject trials where any
  channel exceeds ±100 µV during the imagery window. ICA-based cleanup
  is out of scope for v1.
- **Subject preparation**: skin prep, electrode impedance check (target
  < 10 kΩ via OpenBCI GUI). Add a checklist to `sh/run_bci_session.sh`.
- **MC corridor world**: build (or download) a flat-world save with a
  long straight corridor and visible left/right side-passages, so cued
  strafing has unambiguous semantics. Commit the world seed / save path
  to the repo (not the world data itself).
- **MC version pin**: pynput-injected keystrokes behave the same across
  recent Java versions, but commit a specific version (e.g. 1.20.x) so
  the corridor world and key bindings are reproducible.
- **Display server**: on Linux, pynput's reliability differs between X11
  and Wayland (Wayland sandboxes synthetic input). If the recording
  station is Wayland, fall back to `xdotool` via subprocess or
  switch the session to X11.
- **Focus loss**: if the subject Alt-Tabs or the overlay accidentally
  steals focus, key injection silently no-ops. `mc_control.py` should
  verify the focused window is MC before each injection and pause the
  paradigm with a visible warning otherwise.

## 13. Other notes

1. Use `conda run -n cbramod` for all training / inference commands
   (matches the project convention).
2. Keep all online-stack code in `bci_app/` so it can be deleted or
   moved out of the research repo if it ever becomes a separate
   product.
3. **Do not commit subject data** — Phase A recordings go to
   `runs/<session>/` which should be in `.gitignore`. Anonymise filenames
   if data leaves the recording machine.
4. Frame this as evaluation, not deployment: the encoder is a research
   artifact and the demo's purpose is to **measure** whether it
   transfers to live MI, not to be a robust assistive device.
