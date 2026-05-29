"""Compare single-window vs sliding-window-ensemble eval for the PhysioNet
finetune checkpoint.

The new model_for_physio behaviour (eval-only) averages classifier logits
over every patch-aligned start window. This script:

  1. loads the finetune checkpoint;
  2. runs the same 4 per-class samples as test_shift_inference.py
     under both modes for an easy side-by-side;
  3. evaluates a configurable slice of the test split to report
     balanced accuracy + Cohen's kappa for both modes.
"""
import os
import sys
from argparse import Namespace

import lmdb
import numpy as np
import pickle
import torch
import torch.nn.functional as F

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score  # noqa: E402

from utils.util import load_pretrain_checkpoint, apply_arch_params  # noqa: E402
from datasets.cached_dataset import _to_spherical  # noqa: E402
from models import model_for_physio  # noqa: E402


FOUNDATION_DIR = 'outputs/Align-time-volume/epoch5_loss0.03542475029826164.pth'
CKPT_PATH = 'outputs/CSBrain/finetune_CSBrain_PhysioNet/epoch35_acc_0.50913_kappa_0.34553_f1_0.50915.pth'
DATA_DIR = 'data/preprocessed/physionet_mi'
CLASS_NAMES = ['L_fist', 'R_fist', 'both_fists', 'both_feet']


def build_params(foundation_dir):
    p = Namespace(
        model='Align', downstream_dataset='PhysioNet-MI',
        datasets_dir=DATA_DIR, num_of_classes=4,
        foundation_dir=foundation_dir,
        use_pretrained_weights=False,
        use_initial_segment_only=True, segment_forward=False,
        dropout=0.3, linear_probe=False, use_finetune_weights=False,
        use_lora=False, image_mode='raw', image_size=0,
        stft_n_fft=64, stft_hop_length=16,
        in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
        seq_len=30, n_layer=12, nhead=8,
        need_mask=True, mask_ratio=0.5,
        TemEmbed_kernel_sizes="[(1,), (3,), (5,),]",
        use_CrossTemEmbed=False, use_SmallerToken=False,
        use_CSBrainTF=False, use_CSBrainTF_Tep_Spa=False,
        use_CSBrainTF_Tep_Bra=False, use_CSBrainTF_Tep_Bra_Tiny=False,
        use_CSBrainTF_Tep_Bra_Pal=False, use_IntraBraEmbed=False,
        project_to_source=False, num_sources=32,
        patch_embed_type='cnn', mamba_band_periods=None,
        n_mamba_layers=2, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        vision_encoder='facebook/dinov2-base', image_pool_heads=4,
        use_volume_conduction=True, vc_tau_init=0.08,
        spectral_mode='instantaneous', stft_hop=16,
        use_llm_vq=False, num_language_tokens=0, max_llm_codebook_size=0,
        hemisphere_flip_aug=False,
    )
    _, saved = load_pretrain_checkpoint(foundation_dir)
    apply_arch_params(p, saved)
    return p


def make_batch(samples, device):
    """Stack a list of LMDB sample dicts into a batch."""
    if not isinstance(samples, list):
        samples = [samples]
    x = np.stack([s['sample'] / 100.0 for s in samples], axis=0).astype(np.float32)
    y = np.array([s['label'] for s in samples], dtype=np.int64)
    ch_coords = torch.stack([_to_spherical(s['ch_coords']) for s in samples], dim=0)
    return {
        'x': torch.from_numpy(x).to(device),
        'y': torch.from_numpy(y).to(device),
        'ch_coords': ch_coords.to(device),
    }


@torch.no_grad()
def predict_single_window(model, batch_template):
    """Run the model on only the first seg_len patches (legacy eval behaviour)."""
    batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch_template.items()}
    x = batch.pop('x')
    x = x.reshape(x.size(0), x.size(1), -1, model.param.in_dim)
    bz, ch_num, _, patch_size = x.shape
    seg_len = model.param.seq_len
    x = x[:, :, :seg_len, :].contiguous()
    batch['timeseries'] = x
    feats = model.backbone(batch)
    feats = feats[1]['rep']
    out = feats.contiguous().view(bz, -1)
    return model.classifier(out)


@torch.no_grad()
def predict_ensemble(model, batch_template):
    """Use the model's default eval path (sliding-window ensemble)."""
    batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch_template.items()}
    return model(batch)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = build_params(FOUNDATION_DIR)
    print(f"Device: {device}")
    print(f"Encoder seq_len={params.seq_len}, in_dim={params.in_dim} "
          f"-> visible window = {params.seq_len*params.in_dim} samples")

    model = model_for_physio.Model(params).to(device).eval()

    db = lmdb.open(DATA_DIR, readonly=True, lock=False)
    with db.begin(write=False) as txn:
        test_keys = pickle.loads(txn.get('__keys__'.encode()))['test']

    # --- Per-class headline sample (matches test_shift_inference.py) ----------
    with db.begin(write=False) as txn:
        head = {}
        for k in test_keys:
            s = pickle.loads(txn.get(k.encode()))
            if s['label'] not in head:
                head[s['label']] = s
            if len(head) == 4:
                break
    head_samples = [head[i] for i in sorted(head.keys())]

    # Materialize LazyLinear then load checkpoint
    head_batch = make_batch(head_samples, device)
    with torch.no_grad():
        _ = predict_ensemble(model, head_batch)
    sd = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load] missing={len(missing)}")
    if unexpected:
        print(f"[load] unexpected={len(unexpected)}")
    model.eval()

    # --- Side-by-side on the 4 headline samples ------------------------------
    print("\n--- 4 per-class samples (probs at single-window vs ensemble) ---")
    p_single = F.softmax(predict_single_window(model, head_batch), dim=-1).cpu().numpy()
    p_ens = F.softmax(predict_ensemble(model, head_batch), dim=-1).cpu().numpy()
    header = "true | mode      | pred              | " + "  ".join(f"{n:>10}" for n in CLASS_NAMES)
    print(header)
    print('-' * len(header))
    for i, s in enumerate(head_samples):
        true = s['label']
        for name, p in [('single', p_single[i]), ('ensemble', p_ens[i])]:
            pred = int(p.argmax())
            ok = '*' if pred == true else ' '
            print(f"  {true}  | {name:<9} | {pred} ({CLASS_NAMES[pred]:>10}) {ok} | "
                  + "  ".join(f"{v:.3f}     " for v in p))

    # --- Aggregate over a slice of the test split ----------------------------
    # Set TEST_LIMIT=0 to use the full test split (slower on CPU).
    n_test = len(test_keys)
    limit = int(os.environ.get('TEST_LIMIT', 200))
    if limit <= 0 or limit > n_test:
        limit = n_test
    print(f"\n--- Evaluating {limit}/{n_test} test samples ---")

    y_all, single_pred, ens_pred = [], [], []
    bs = int(os.environ.get('EVAL_BS', 16))
    rng = np.random.default_rng(0)
    idxs = rng.permutation(n_test)[:limit].tolist()

    with db.begin(write=False) as txn:
        for batch_start in range(0, limit, bs):
            chunk = idxs[batch_start:batch_start + bs]
            samples = [pickle.loads(txn.get(test_keys[i].encode())) for i in chunk]
            batch = make_batch(samples, device)
            logits_s = predict_single_window(model, batch)
            logits_e = predict_ensemble(model, batch)
            y_all.extend(batch['y'].cpu().tolist())
            single_pred.extend(logits_s.argmax(dim=-1).cpu().tolist())
            ens_pred.extend(logits_e.argmax(dim=-1).cpu().tolist())
            print(f"  processed {batch_start + len(chunk)}/{limit}")

    y_all = np.array(y_all); single_pred = np.array(single_pred); ens_pred = np.array(ens_pred)
    def metrics(p):
        return (balanced_accuracy_score(y_all, p),
                cohen_kappa_score(y_all, p),
                f1_score(y_all, p, average='weighted'))
    s_acc, s_kap, s_f1 = metrics(single_pred)
    e_acc, e_kap, e_f1 = metrics(ens_pred)
    print(f"\nsingle  | bal-acc={s_acc:.4f}  kappa={s_kap:.4f}  f1={s_f1:.4f}")
    print(f"ensemble| bal-acc={e_acc:.4f}  kappa={e_kap:.4f}  f1={e_f1:.4f}")
    print(f"agreement between modes: {(single_pred == ens_pred).mean():.3f}")


if __name__ == '__main__':
    main()
