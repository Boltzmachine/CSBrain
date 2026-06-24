"""GPU-node segfault probe for the hand-label pipeline. Run on l40s.
Each step prints (flushed) before doing the work, so the last printed line +
the faulthandler dump localise the crash."""
import faulthandler, os, sys, json
faulthandler.enable()
P = lambda *a: print(*a, flush=True)

P("STEP 0: python", sys.version.split()[0])
import numpy as np
P("STEP 1: numpy", np.__version__)
import torch
P("STEP 2: torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
  "dev", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

import cv2
P("STEP 3: cv2", cv2.__version__)
cv2.setNumThreads(1)
a = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
b = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY); c = cv2.resize(a, (640, 360))
P("STEP 4: cv2 ops ok", b.shape, c.shape)

import decord
P("STEP 5: import decord", decord.__version__)
decord.bridge.set_bridge('native')

# Resolve P0001's first video chapter from the EEG cache metadata.
sys.path.insert(0, 'datasets')
from egobrain_hand_labels import build_chapters
meta = json.load(open('data/EgoBrain/cache_eeg_200hz/P0001/clips.json'))
chapters = build_chapters(meta['video'], 'data/EgoBrain')
vpath = chapters[0]['path']
P("STEP 6: video path", vpath, "exists", os.path.exists(vpath))

vr = decord.VideoReader(vpath)
P("STEP 7: VideoReader opened; fps", float(vr.get_avg_fps()), "n", len(vr))
frames = vr.get_batch([100, 130, 160, 190]).asnumpy()
P("STEP 8: decord get_batch ok", frames.shape, frames.dtype)
sm = np.stack([cv2.resize(f, (1280, int(f.shape[0]*1280/f.shape[1]))) for f in frames])
P("STEP 9: resized frames", sm.shape)

from egobrain_hand_labels import HandEstimator, parse_wilor_frame
est = HandEstimator('wilor', 'cuda', 'float16')
P("STEP 10: WiLoR init ok")
outs = est.predict_frame(sm[0])
P("STEP 11: predict on real frame ok; n_raw_hands", len(outs))
l, r = parse_wilor_frame(outs, 'hybrid', img_w=sm.shape[2])
P("STEP 12: parse ok; left", l is not None, "right", r is not None)
P("ALL STEPS PASSED")
