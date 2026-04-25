"""Linear-probe a (frozen) DINO encoder on EEG classification.

Canonical SSL linear-probe evaluation:
    1. Build DINO encoder + EEG->image transform (both frozen, eval mode).
    2. Extract the CLS-token feature for every train/val/test sample once.
       Cache to disk so the expensive backbone pass isn't repeated per epoch.
    3. Train a single ``nn.Linear`` on the cached features, sweeping a small
       list of learning rates (classical DINOv2 protocol).
    4. For each LR, pick the epoch with best val accuracy and record its test
       metrics. Report the LR whose val accuracy is highest globally.

Run:
    python scripts/linear_probe_dinov3_eeg.py \\
        --downstream_dataset PhysioNet-MI \\
        --datasets_dir data/preprocessed/physionet_mi \\
        --num_of_classes 4 \\
        --vision_encoder facebook/dinov2-base
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel

# Make sibling packages importable when the script is launched from the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets import physio_dataset  # noqa: E402
from models.dinov3_eeg import (  # noqa: E402
    EEGRawImage, EEGSpectrogramImage, _resolve_preprocessing,
)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

_DATASET_REGISTRY = {
    'PhysioNet-MI': physio_dataset.LoadDataset,
}


def build_loaders(args):
    if args.downstream_dataset not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset: {args.downstream_dataset}. "
            f"Available: {list(_DATASET_REGISTRY)}"
        )
    loader_cls = _DATASET_REGISTRY[args.downstream_dataset]
    return loader_cls(args).get_data_loader()


# ----------------------------------------------------------------------------
# Feature extraction (cached)
# ----------------------------------------------------------------------------

@torch.inference_mode()
def extract_split(dino, eeg_to_image, loader, device, desc):
    feats, labels = [], []
    for batch in tqdm(loader, mininterval=5, desc=desc):
        x = batch['x'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        if x.dim() == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
        images = eeg_to_image(x)
        out = dino(pixel_values=images)
        feats.append(out.last_hidden_state[:, 0].cpu())
        labels.append(y.cpu())
    return torch.cat(feats), torch.cat(labels)


def extract_or_load_features(args, device):
    if args.feature_cache and os.path.exists(args.feature_cache):
        print(f'[cache] loading features from {args.feature_cache}')
        blob = torch.load(args.feature_cache, map_location='cpu', weights_only=False)
        return blob['features'], blob['labels']

    dino = AutoModel.from_pretrained(args.vision_encoder).to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False

    crop, resize, mean, std = _resolve_preprocessing(args.vision_encoder)
    if args.image_size:
        crop = resize = args.image_size
    if args.image_mode == 'raw':
        eeg_to_image = EEGRawImage(
            crop_size=crop, image_mean=mean, image_std=std,
        ).to(device).eval()
    elif args.image_mode == 'spectrogram':
        eeg_to_image = EEGSpectrogramImage(
            crop_size=crop, resize_size=resize,
            image_mean=mean, image_std=std,
            n_fft=args.stft_n_fft, hop_length=args.stft_hop_length,
        ).to(device).eval()
    else:
        raise ValueError(f"image_mode must be 'raw' or 'spectrogram', got {args.image_mode!r}")

    loaders = build_loaders(args)
    features, labels = {}, {}
    for split in ('train', 'val', 'test'):
        f, l = extract_split(dino, eeg_to_image, loaders[split], device, desc=split)
        features[split], labels[split] = f, l
        print(f'  {split}: feats {tuple(f.shape)}  labels {tuple(l.shape)}')

    if args.feature_cache:
        os.makedirs(os.path.dirname(args.feature_cache) or '.', exist_ok=True)
        torch.save({'features': features, 'labels': labels}, args.feature_cache)
        print(f'[cache] wrote features to {args.feature_cache}')
    return features, labels


# ----------------------------------------------------------------------------
# Linear-probe training
# ----------------------------------------------------------------------------

def _metrics(y_true, y_pred):
    return {
        'acc': accuracy_score(y_true, y_pred),
        'bacc': balanced_accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
    }


@torch.inference_mode()
def evaluate(head, loader, device):
    head.eval()
    preds, truths = [], []
    for f, y in loader:
        f = f.to(device, non_blocking=True)
        preds.append(head(f).argmax(dim=-1).cpu())
        truths.append(y)
    preds = torch.cat(preds).numpy()
    truths = torch.cat(truths).numpy()
    return _metrics(truths, preds)


def train_logreg_probe(features, labels, args, C):
    """Paper's probe: sklearn multinomial LogisticRegression on L2-normalised features."""
    Xtr = F.normalize(features['train'].float(), dim=-1).numpy()
    Xva = F.normalize(features['val'].float(), dim=-1).numpy()
    Xte = F.normalize(features['test'].float(), dim=-1).numpy()
    ytr = labels['train'].numpy()
    yva = labels['val'].numpy()
    yte = labels['test'].numpy()

    clf = LogisticRegression(
        C=C, penalty='l2', solver='lbfgs', multi_class='multinomial',
        max_iter=args.logreg_max_iter, n_jobs=-1,
    )
    clf.fit(Xtr, ytr)

    val = _metrics(yva, clf.predict(Xva))
    test = _metrics(yte, clf.predict(Xte))
    return {
        'val_acc': val['acc'], 'val_bacc': val['bacc'],
        'val_kappa': val['kappa'], 'val_f1': val['f1'],
        'test_acc': test['acc'], 'test_bacc': test['bacc'],
        'test_kappa': test['kappa'], 'test_f1': test['f1'],
        'epoch': -1,
    }


def train_linear_probe(features, labels, args, lr, device):
    hidden = features['train'].size(-1)
    train_ds = TensorDataset(features['train'], labels['train'])
    val_ds = TensorDataset(features['val'], labels['val'])
    test_ds = TensorDataset(features['test'], labels['test'])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    head = nn.Linear(hidden, args.num_of_classes).to(device)
    if args.optimizer == 'SGD':
        opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9,
                              weight_decay=args.weight_decay, nesterov=True)
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=lr,
                                weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=0.0,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best = {'val_acc': -1.0, 'epoch': -1, 'state': None}
    for epoch in range(args.epochs):
        head.train()
        for f, y in train_loader:
            f = f.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = head(f)
            loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        val_metrics = evaluate(head, val_loader, device)
        if val_metrics['acc'] > best['val_acc']:
            best = {
                'val_acc': val_metrics['acc'],
                'val_bacc': val_metrics['bacc'],
                'val_kappa': val_metrics['kappa'],
                'val_f1': val_metrics['f1'],
                'epoch': epoch + 1,
                'state': {k: v.detach().clone() for k, v in head.state_dict().items()},
            }

    head.load_state_dict(best.pop('state'))
    test_metrics = evaluate(head, test_loader, device)
    return {
        **best,
        'test_acc': test_metrics['acc'],
        'test_bacc': test_metrics['bacc'],
        'test_kappa': test_metrics['kappa'],
        'test_f1': test_metrics['f1'],
    }


# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------

def _parse_lrs(s):
    s = s.strip()
    if s.startswith('['):
        return [float(x) for x in eval(s)]
    return [float(x) for x in s.split(',')]


def main():
    parser = argparse.ArgumentParser(description='Linear-probe a frozen DINO encoder on EEG.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--downstream_dataset', type=str, default='PhysioNet-MI')
    parser.add_argument('--datasets_dir', type=str, required=True)
    parser.add_argument('--num_of_classes', type=int, required=True)

    # Encoder + EEG-to-image
    parser.add_argument('--vision_encoder', type=str, default='facebook/dinov2-base')
    parser.add_argument('--image_mode', type=str, default='raw',
                        choices=['raw', 'spectrogram'],
                        help="'raw' = (C,T) 2D layout (paper default); "
                             "'spectrogram' = per-channel STFT grid.")
    parser.add_argument('--image_size', type=int, default=0,
                        help='override DINO crop size; 0 = use processor default')
    parser.add_argument('--stft_n_fft', type=int, default=64)
    parser.add_argument('--stft_hop_length', type=int, default=16)

    # Linear-probe training
    parser.add_argument('--probe', type=str, default='logreg',
                        choices=['logreg', 'sgd'],
                        help="'logreg' = sklearn multinomial LogisticRegression "
                             "(paper); 'sgd' = gradient-descent nn.Linear.")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lrs', type=str, default='[0.001, 0.01, 0.1, 1.0]',
                        help='LR sweep (sgd probe only); python-list or comma-separated.')
    parser.add_argument('--logreg_Cs', type=str,
                        default='[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]',
                        help='L2 inverse-strength sweep for logreg probe.')
    parser.add_argument('--logreg_max_iter', type=int, default=5000)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    parser.add_argument('--feature_cache', type=str, default=None,
                        help='if set, save/load extracted features to this .pt file')
    parser.add_argument('--results_csv', type=str, default=None)

    # Passed through to dataset loaders; disabled by default since augmentation
    # changes features between epochs, which defeats caching.
    parser.add_argument('--hemisphere_flip_aug', action='store_true', default=False)

    args = parser.parse_args()
    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    features, labels = extract_or_load_features(args, device)

    if args.probe == 'logreg':
        sweep = _parse_lrs(args.logreg_Cs)
        sweep_name = 'C'
    else:
        sweep = _parse_lrs(args.lrs)
        sweep_name = 'lr'

    best_overall = None
    per_run = []
    for hp in sweep:
        setup_seed(args.seed)  # identical init across sweep points
        if args.probe == 'logreg':
            result = train_logreg_probe(features, labels, args, hp)
        else:
            result = train_linear_probe(features, labels, args, hp, device)
        result[sweep_name] = hp
        per_run.append(result)
        print(
            f'[{sweep_name}={hp:<7g}]  val_acc={result["val_acc"]:.4f}  '
            f'val_bacc={result["val_bacc"]:.4f}  val_kappa={result["val_kappa"]:.4f}  '
            f'val_f1={result["val_f1"]:.4f}  ||  test_acc={result["test_acc"]:.4f}  '
            f'test_bacc={result["test_bacc"]:.4f}  test_kappa={result["test_kappa"]:.4f}  '
            f'test_f1={result["test_f1"]:.4f}  (ep={result["epoch"]})'
        )
        if best_overall is None or result['val_acc'] > best_overall['val_acc']:
            best_overall = result

    print('\n*** Best linear-probe result (selected by val_acc) ***')
    for k in (sweep_name, 'epoch', 'val_acc', 'val_bacc', 'val_kappa', 'val_f1',
              'test_acc', 'test_bacc', 'test_kappa', 'test_f1'):
        print(f'  {k}: {best_overall[k]}')

    if args.results_csv:
        import csv
        os.makedirs(os.path.dirname(args.results_csv) or '.', exist_ok=True)
        write_header = not os.path.exists(args.results_csv)
        with open(args.results_csv, 'a', newline='') as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow(['vision_encoder', 'dataset', 'image_mode', 'probe',
                            'sweep_name', 'sweep_value', 'best_epoch',
                            'val_acc', 'val_bacc', 'val_kappa', 'val_f1',
                            'test_acc', 'test_bacc', 'test_kappa', 'test_f1'])
            w.writerow([
                args.vision_encoder, args.downstream_dataset,
                args.image_mode, args.probe,
                sweep_name, best_overall[sweep_name], best_overall['epoch'],
                best_overall['val_acc'], best_overall['val_bacc'],
                best_overall['val_kappa'], best_overall['val_f1'],
                best_overall['test_acc'], best_overall['test_bacc'],
                best_overall['test_kappa'], best_overall['test_f1'],
            ])
        print(f'results appended to {args.results_csv}')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
