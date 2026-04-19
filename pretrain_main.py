import argparse
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.pretraining_dataset import PretrainingDataset
from models.CSBrain import *
from models import get_model
from pretrain_trainer import Trainer
import wandb
import os
import logging
logger = logging.getLogger(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number')
    parser.add_argument('--parallel', type=bool, default=False, help='parallel')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight_decay')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', help='lr_scheduler')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--in_dim', type=int, default=200, help='in_dim')
    parser.add_argument('--out_dim', type=int, default=200, help='out_dim')
    parser.add_argument('--d_model', type=int, default=200, help='d_model')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='dim_feedforward')
    parser.add_argument('--seq_len', type=int, default=30, help='seq_len')
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--nhead', type=int, default=8, help='nhead')
    parser.add_argument('--need_mask', type=bool, default=True, help='need_mask')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio')
    parser.add_argument('--dataset_dir', type=str, default='path/to/dataset', help='dataset_dir')
    parser.add_argument('--model_dir', type=str, default='outputs', help='model_dir') # eg. 'CSBrain/pth'
    parser.add_argument('--TemEmbed_kernel_sizes', type=str, default="[(1,), (3,), (5,)]")
    parser.add_argument('--use_SmallerToken', type=bool, default=False, help='SmallerToken->dataset.py')
    parser.add_argument('--model', type=str, default='CSBrain', help='CSBrain')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--causal', action='store_true', default=False, help='use causal (next-patch prediction) instead of masked reconstruction')
    parser.add_argument('--project_to_source', action='store_true', default=False, help='project sensors to source space before transformer')
    parser.add_argument('--num_sources', type=int, default=32, help='number of brain sources')
    parser.add_argument('--decorr_weight', type=float, default=0.1, help='decorrelation loss weight for source projector pretraining')
    parser.add_argument('--source_projector_ckpt', type=str, default=None, help='path to pre-trained source projector checkpoint')
    parser.add_argument('--freeze_source_projector', action='store_true', default=False, help='freeze source projector weights during stage 2')
    parser.add_argument('--adversarial_weight', type=float, default=0.0, help='weight for adversarial session-agnostic loss (0 = disabled)')
    parser.add_argument('--samples_per_session', type=int, default=8, help='samples per session in session-grouped batching')
    parser.add_argument('--sessions_per_batch', type=int, default=16, help='number of distinct sessions per batch')
    parser.add_argument('--equivariance_weight', type=float, default=0.0, help='weight for hemispheric equivariance loss (0 = disabled)')
    parser.add_argument('--info_max_weight', type=float, default=0.0, help='weight for VICReg-style info-max regulariser on inv/eq subspaces (0 = disabled)')
    parser.add_argument('--alignment_weight', type=float, default=1.0, help='weight for EEG-image contrastive alignment loss (0 = disabled)')

    # --- SSM/Mamba multi-frequency patch embedding ---
    parser.add_argument('--patch_embed_type', type=str, default='cnn', choices=['cnn', 'mamba'], help='patch embedder: CNN (default) or multi-frequency Mamba SSM')
    parser.add_argument('--mamba_band_periods', type=str, default=None, help='list of band sample periods, e.g. "[200,600,1200]"; default [in_dim, 3*in_dim, 6*in_dim]')
    parser.add_argument('--n_mamba_layers', type=int, default=2, help='number of stacked Mamba blocks in the patch embedder')
    parser.add_argument('--mamba_d_state', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--mamba_d_conv', type=int, default=4, help='Mamba depthwise conv width')
    parser.add_argument('--mamba_expand', type=int, default=2, help='Mamba expansion factor')

    # --- DINOv2 self-distillation ---
    parser.add_argument('--dino_mode', action='store_true', default=False, help='enable DINOv2 self-distillation (replaces masked reconstruction)')
    parser.add_argument('--n_prototypes', type=int, default=4096, help='number of prototype vectors in DINO heads')
    parser.add_argument('--dino_head_hidden_dim', type=int, default=256, help='DINO head MLP hidden dimension')
    parser.add_argument('--dino_head_bottleneck_dim', type=int, default=256, help='DINO head bottleneck dimension before prototype layer')
    parser.add_argument('--dino_head_n_layers', type=int, default=3, help='number of MLP layers in DINO head')
    parser.add_argument('--student_temp', type=float, default=0.1, help='student softmax temperature')
    parser.add_argument('--teacher_temp_base', type=float, default=0.04, help='teacher temperature (initial)')
    parser.add_argument('--teacher_temp_final', type=float, default=0.07, help='teacher temperature (after warmup)')
    parser.add_argument('--teacher_temp_warmup_epochs', type=int, default=30, help='epochs to linearly warm up teacher temperature')
    parser.add_argument('--ema_momentum_base', type=float, default=0.992, help='EMA momentum for teacher (initial)')
    parser.add_argument('--ema_momentum_final', type=float, default=0.9995, help='EMA momentum for teacher (final)')
    parser.add_argument('--dino_loss_weight', type=float, default=1.0, help='weight for DINO CLS-token distillation loss')
    parser.add_argument('--ibot_loss_weight', type=float, default=1.0, help='weight for iBOT masked-patch distillation loss')
    parser.add_argument('--koleo_loss_weight', type=float, default=0.1, help='weight for KoLeo diversity loss')
    parser.add_argument('--use_freq_subband', action='store_true', default=False, help='enable frequency sub-band view augmentation for student')
    parser.add_argument('--freq_n_bands', type=int, default=5, help='number of frequency bands to divide the spectrum into')
    parser.add_argument('--freq_min_bands', type=int, default=1, help='minimum number of bands to keep in sub-band view')
    parser.add_argument('--freq_max_bands', type=int, default=None, help='maximum number of bands to keep (default: all)')
    parser.add_argument('--last_layer_freeze_iters', type=int, default=1250, help='freeze weight-norm magnitude of prototype layer for this many initial iterations (0 = never freeze)')
    parser.add_argument('--lr_warmup_iters', type=int, default=0, help='linear LR warmup iterations (0 = disabled)')
    # Multi-crop
    parser.add_argument('--n_local_crops', type=int, default=4, help='number of local EEG crops for DINO CLS loss (0 = single-view)')
    parser.add_argument('--local_crop_time_scale', type=str, default='(0.3, 0.7)', help='(min, max) fraction of time patches in local crops')
    parser.add_argument('--local_crop_channel_scale', type=str, default='(0.5, 1.0)', help='(min, max) fraction of channels in local crops')

    # --- World model (EEG video-prediction) ---
    parser.add_argument('--cinebrain_subjects', type=str, default='sub-0001', help='comma-separated list of CineBrain subject directories')
    parser.add_argument('--cinebrain_root', type=str, default='data/CineBrain', help='root directory for the CineBrain dataset')
    parser.add_argument('--cinebrain_n_windows', type=int, default=3, help='number of windows to emit per 4 s CineBrain clip')
    parser.add_argument('--cinebrain_window_s', type=float, default=2.0, help='EEG window length in seconds (must give window_samples divisible by in_dim)')
    parser.add_argument('--cinebrain_stride_s', type=float, default=1.0, help='stride between consecutive EEG windows in seconds')
    parser.add_argument('--cinebrain_erp_latency_s', type=float, default=0.0, help='time shift added to the frame-lookup timestamp (paper default: 0; sweep +/-0.15 to probe visual-ERP lag)')
    parser.add_argument('--cinebrain_load_frames', type=int, default=1, help='whether to decode raw video frames (0 disables the alignment term)')
    parser.add_argument('--predictor_d_model', type=int, default=512)
    parser.add_argument('--predictor_n_layers', type=int, default=4)
    parser.add_argument('--predictor_n_heads', type=int, default=8)
    parser.add_argument('--predictor_dim_feedforward', type=int, default=1024)
    parser.add_argument('--max_horizon', type=int, default=1, help='maximum prediction horizon k (in windows)')
    parser.add_argument('--latent_pred_weight', type=float, default=1.0)
    parser.add_argument('--cls_pred_weight', type=float, default=0.1)
    parser.add_argument('--pred_ramp_epochs', type=int, default=2, help='linearly ramp latent-prediction weight 0→1 over this many epochs')

    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)
    params.model_dir = os.path.join(params.model_dir, params.run_name)

    if os.environ.get('DEBUG', '0') == '1':
        params.batch_size = 4

    if params.dataset_dir == 'mix':
        from datasets.cached_dataset import get_webdataset, collate_cached, SessionGroupedLoader
        from torch.utils.data import ConcatDataset
        dataset_names = [
            # 'tueg/*.tar',
            # 'siena_scalp/*.tar',
            # 'physionet_2018/*.tar',
            # 'raw_eeg/*.tar',
            # 'ds006171/*.tar', 
            # 'ds006317/*.tar',
            # 'ds006367/*.tar', 
            # 'ds006370/*.tar', 
            # 'ds006437/*.tar', 
            # 'ds006446/*.tar', 
            # 'ds006466/*.tar', 
            # 'ds006480/*.tar', 
            # 'ds006525/*.tar', 
            # 'ds006547/*.tar',
            "Alljoined-1.6M/*.tar",
            # "things_eeg2/*.tar",
        ]
        n_samples_per_epoch = 1109545
        pretrained_dataset = get_webdataset(
            dataset_names,
            params
        ) #resampled=True

        num_workers = 8 if os.environ.get('DEBUG', '0') == '0' else 0

        if params.adversarial_weight > 0:
            # Session-grouped batching.
            # WebDataset yields individual samples (no .batched());
            # multi-worker DataLoader interleaves samples from different
            # shards (different sessions); SessionGroupedLoader in the
            # main process collects them into session-balanced batches.
            effective_batch_size = params.samples_per_session * params.sessions_per_batch
            n_batches_per_epoch = n_samples_per_epoch // effective_batch_size
            samples_per_worker = math.ceil(n_samples_per_epoch / max(1, num_workers))

            pretrained_dataset = (
                pretrained_dataset
                .shuffle(2000)
                .with_epoch(samples_per_worker)
                .with_length(n_samples_per_epoch)
            )

            raw_loader = DataLoader(
                pretrained_dataset,
                batch_size=None,
                num_workers=num_workers,
                pin_memory=False,
            )
            data_loader = SessionGroupedLoader(
                raw_loader,
                samples_per_session=params.samples_per_session,
                sessions_per_batch=params.sessions_per_batch,
                collation_fn=collate_cached,
                batches_per_epoch=n_batches_per_epoch,
            )
        else:
            n_batches_per_epoch = (n_samples_per_epoch + params.batch_size - 1) // params.batch_size
            batches_per_worker = math.ceil(n_batches_per_epoch / max(1, num_workers))
            total_yielded_batches = batches_per_worker * max(1, num_workers)

            pretrained_dataset = (
                pretrained_dataset
                .shuffle(5000)
                .batched(params.batch_size, partial=True, collation_fn=collate_cached)
                .with_epoch(batches_per_worker)
                .with_length(total_yielded_batches)
            )

            data_loader = DataLoader(
                pretrained_dataset,
                batch_size=None,
                num_workers=num_workers,
                pin_memory=True,
            )
       
    elif params.dataset_dir == 'cinebrain':
        from datasets.cinebrain_dataset import CineBrainDataset, collate_cinebrain
        subjects = [s.strip() for s in params.cinebrain_subjects.split(',') if s.strip()]
        pretrained_dataset = CineBrainDataset(
            data_dir=params.cinebrain_root,
            subjects=subjects,
            in_dim=params.in_dim,
            n_windows=params.cinebrain_n_windows,
            window_s=params.cinebrain_window_s,
            stride_s=params.cinebrain_stride_s,
            erp_latency_s=params.cinebrain_erp_latency_s,
            load_frames=bool(params.cinebrain_load_frames),
        )
        print('CineBrain clips:', len(pretrained_dataset))
        num_workers = 8 if os.environ.get('DEBUG', '0') == '0' else 0
        # Match the 'mix' branch's epoch size so runtime + LR schedules stay
        # comparable. CineBrain has far fewer unique clips, so we sample with
        # replacement — each clip is visited ~n_samples_per_epoch/len(ds)
        # times per epoch, which is fine since windows/frames are
        # deterministic per clip.
        n_samples_per_epoch = 1109545
        sampler = torch.utils.data.RandomSampler(
            pretrained_dataset,
            replacement=True,
            num_samples=n_samples_per_epoch,
        )
        data_loader = DataLoader(
            pretrained_dataset,
            batch_size=params.batch_size,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=collate_cinebrain,
            pin_memory=True,
            drop_last=True,
        )
    else:
        pretrained_dataset = PretrainingDataset(
            dataset_dir=params.dataset_dir,
            SmallerToken=params.use_SmallerToken,
        )
        print(len(pretrained_dataset))
        data_loader = DataLoader(
            pretrained_dataset,
            batch_size=params.batch_size,
            num_workers=8,
            shuffle=True,
        )

    brain_regions = [
        0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 1
    ]
    electrode_labels = [
        "FP1-REF", "FP2-REF", "F3-REF", "F4-REF",
        "C3-REF", "C4-REF", "P3-REF", "P4-REF",
        "O1-REF", "O2-REF", "F7-REF", "F8-REF",
        "T3-REF", "T4-REF", "T5-REF", "T6-REF",
        "FZ-REF", "CZ-REF", "PZ-REF"
    ]
    topology = {
        0: ["FP1-REF", "F7-REF", "F3-REF", "FZ-REF", "F4-REF", "F8-REF", "FP2-REF"], 
        4: ["C3-REF", "CZ-REF", "C4-REF"],
        1: ["P3-REF", "PZ-REF", "P4-REF"],
        3: ["O1-REF", "O2-REF"],
        2: ["T3-REF", "T5-REF", "T6-REF", "T4-REF"],
        -1: ["A1-REF"]
    }
    region_groups = {}
    for i, region in enumerate(brain_regions):
        if region not in region_groups:
            region_groups[region] = []
        region_groups[region].append((i, electrode_labels[i]))
    sorted_indices = []
    for region in sorted(region_groups.keys()):
        region_electrodes = region_groups[region]
        sorted_electrodes = sorted(region_electrodes, key=lambda x: topology[region].index(x[1]))
        sorted_indices.extend([e[0] for e in sorted_electrodes])

    print("Sorted Indices:", sorted_indices)

    model = get_model(params, brain_regions, sorted_indices)
  
    wandb.init(project="EEG", name=params.run_name, config=vars(params))
    wandb.watch(model, log="all", log_freq=500)
    trainer = Trainer(params, data_loader, model)
    trainer.train()

    if hasattr(pretrained_dataset, 'db'):
        pretrained_dataset.db.close()


if __name__ == '__main__':
    main()
