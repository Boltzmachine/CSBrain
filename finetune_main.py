import argparse
import csv
import random

import numpy as np
import torch
import os

from datasets import faced_dataset, seedv_dataset, physio_dataset, shu_dataset, isruc_dataset, chb_dataset, \
    speech_dataset, mumtaz_dataset, seedvig_dataset, stress_dataset, tuev_dataset, tuab_dataset, bciciv2a_dataset, tusl_dataset
from datasets import tusl_dataset, siena_dataset, hmc_dataset
from finetune_trainer import Trainer
from models import model_for_seedv,model_for_bciciv2a, model_for_tuab, model_for_tuev,model_for_faced,model_for_chb,model_for_speech,model_for_tusl,model_for_shu,model_for_seedvig,model_for_physio,model_for_isruc
from models import model_for_siena, model_for_hmc,model_for_stress,model_for_mumtaz
from utils.util import load_pretrain_checkpoint, apply_arch_params
import wandb

def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)') # 42
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD, Muon)')
    parser.add_argument('--muon_lr', type=float, default=0.02,
                        help='learning rate for the Muon parameter group (only used when --optimizer Muon); --lr is used for the AdamW group')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
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
    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='FACED',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a]')
    parser.add_argument('--datasets_dir', type=str,
                        default='',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=9, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='', help='model_dir')
    """############ Downstream dataset settings ############"""
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=bool, default=False,
                        help='multi_lr')  
    parser.add_argument('--frozen', type=bool,
                        default=False, help='frozen')
    parser.add_argument('--linear_probe', action='store_true', default=False,
                        help='linear probing: freeze the backbone and replace the '
                             'classifier head with a single linear layer '
                             '(currently wired for PhysioNet-MI)')
    parser.add_argument('--use_pretrained_weights', action='store_true', help='Use pretrained weights')
    parser.add_argument('--foundation_dir', type=str,default='pth/CSBrain.pth',help='foundation_dir')
    parser.add_argument('--model', type=str, default='CSBrain', help='CBraMod CSBrain CSBrain_new CSBrain_I CSBrain_II')
    parser.add_argument('--use_CrossTemEmbed', type=bool, default=False, help='CrossTemEmbedEEGLayer')
    parser.add_argument('--use_SmallerToken', type=bool, default=False, help='SmallerToken->dataset.py')
    parser.add_argument('--TemEmbed_kernel_sizes', type=str, default="[(1,), (3,), (5,),]")
    parser.add_argument('--use_CSBrainTF', action='store_true', default=False, help='use_CSBrainTF')
    parser.add_argument('--use_CSBrainTF_Tep_Spa', action='store_true', default=False, help='use_CSBrainTF_Tep_Spa')
    parser.add_argument('--use_CSBrainTF_Tep_Bra', action='store_true', default=False, help='use_CSBrainTF_Tep_Bra')
    parser.add_argument('--use_CSBrainTF_Tep_Bra_Tiny', action='store_true', default=False, help='use_CSBrainTF_Tep_Bra_Tiny')
    parser.add_argument('--use_CSBrainTF_Tep_Bra_Pal', action='store_true', default=False, help='use_CSBrainTF_Tep_Bra_Pal')
    parser.add_argument('--use_IntraBraEmbed', action='store_true', default=False, help='use_IntraBraEmbed')
    parser.add_argument('--use_finetune_weights', type=bool,default=False, help='use_finetune_weights')
    parser.add_argument('--project_to_source', action='store_true', default=False, help='project sensors to source space before transformer')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--results_csv', type=str, default=None, help='path to CSV file for logging results')
    parser.add_argument('--lateral_flip_aug', action='store_true', default=False,
                        help='bilateralization-prior flip augmentation (MI): reuse the pretrained learned x_bi/x_lat split to build x_flip = x_bi + flip(x_lat) and remap labels. Requires a --lateralization_flip-pretrained checkpoint')
    parser.add_argument('--lateral_flip_prob', type=float, default=0.5,
                        help='per-sample probability of applying the lateral-flip augmentation (training only)')
    parser.add_argument('--lateral_flip_tta', action='store_true', default=False,
                        help='test-time flip augmentation: average logits(x) with the label-aligned logits(x_flip). Uses the same learned split + flip_label_map')
    parser.add_argument('--symmetrize_aug', action='store_true', default=False,
                        help='思路1 augmentation: synthesize the hard bilateral class (both fists) from single-hand trials via x_sym = 0.5*(x + flip(x)) (raw full channel swap), relabeled to --symmetrize_target_label')
    parser.add_argument('--symmetrize_aug_prob', type=float, default=0.25,
                        help='per-sample probability of symmetrizing an eligible (single-hand) trial')
    parser.add_argument('--symmetrize_src_labels', type=str, default='0,1',
                        help='CSV of eligible source labels to symmetrize (PhysioNet-MI: 0,1 = left/right fist)')
    parser.add_argument('--symmetrize_target_label', type=int, default=2,
                        help='label assigned to symmetrized trials (PhysioNet-MI: 2 = both fists)')
    parser.add_argument('--flip_label_map', type=str, default='1,0,2,3',
                        help='CSV label remap under the flip, index->flipped index. PhysioNet-MI default 1,0,2,3 = left fist<->right fist, both fists/both feet unchanged. Length must equal --num_of_classes')
    parser.add_argument('--flip_split_hidden', type=int, default=64,
                        help='hidden width of the pretrained LateralizationSplit gate (must match the pretraining value)')
    parser.add_argument('--frame_rep_mode', type=str, default='both',
                        choices=['both', 'inv', 'eq', 'inv_split', 'eq_split'],
                        help="frame-averaging finetune readout: which half of the "
                             "backbone token h=[inv_half;eq_half] the classifier reads. "
                             "'both' (default) = full h; 'inv'/'eq' = P-invariant "
                             "(bilateral) / P-anti-equivariant (lateral) half of the PATCH "
                             "tokens, with the global token kept FULL; 'inv_split'/'eq_split' "
                             "= same half but the global token is ALSO split (whole grid h "
                             "sliced). No-op unless the checkpoint is a frame-averaging "
                             "(equi) model.")
    parser.add_argument('--use_euclidean_alignment', action='store_true', default=False,
                        help='apply per-subject Euclidean Alignment whitening before patching. '
                             'Requires a precomputed sidecar at <datasets_dir>/ea_subject.pt '
                             'or --ea_path. Currently wired into PhysioNet.')
    parser.add_argument('--ea_path', type=str, default=None,
                        help='override the EA sidecar path; defaults to '
                             '<datasets_dir>/ea_subject.pt')
    parser.add_argument('--segment_forward', action='store_true', default=False,
                        help='split the input time dim into segments of size seq_len, '
                             'encode each with the pretrained encoder, and concat the '
                             'resulting representations along the time axis. Use when '
                             'the finetune time dim exceeds the pretrained seq_len '
                             '(e.g. world-model pretraining with seq_len=5 applied '
                             'to PhysioNet with time dim 20).')
    parser.add_argument('--use_initial_segment_only', action='store_true', default=False,
                        help='truncate the finetune input to a single seq_len-patch '
                             'segment (i.e. the pretrained model native window) instead '
                             'of processing the full time dim. Which segment is chosen '
                             'is controlled by --segment_index. Takes precedence over '
                             '--segment_forward when both are set.')
    parser.add_argument('--segment_index', type=int, default=0,
                        help='with --use_initial_segment_only, index of the seq_len-patch '
                             'segment to crop along the time axis (0 = first/native window). '
                             'E.g. for a 4s PhysioNet trial with a 1s pretrained window, '
                             'segment_index 0..3 selects seconds 1..4.')

    # --- SSM/Mamba multi-frequency patch embedding ---
    parser.add_argument('--patch_embed_type', type=str, default='cnn', choices=['cnn', 'mamba'],
                        help='patch embedder: CNN (default) or multi-frequency Mamba SSM')
    parser.add_argument('--mamba_band_periods', type=str, default=None,
                        help='list of band sample periods, e.g. "[200,600,1200]"; default [in_dim, 3*in_dim, 6*in_dim]')
    parser.add_argument('--n_mamba_layers', type=int, default=2, help='number of stacked Mamba blocks in the patch embedder')
    parser.add_argument('--mamba_d_state', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--mamba_d_conv', type=int, default=4, help='Mamba depthwise conv width')
    parser.add_argument('--mamba_expand', type=int, default=2, help='Mamba expansion factor')

    # --- Vision encoder for image-alignment branches ---
    # Used by --model DINOv3EEG (image classifier) AND by --model Align/WorldModel
    # (the CSBrainAlign image_alignment head). MUST match the encoder used at
    # pretraining; the checkpoint loader will surface a shape mismatch otherwise.
    parser.add_argument('--vision_encoder', type=str, default='facebook/dinov2-base',
                        help='HF model id of the frozen vision encoder (e.g. facebook/dinov2-base, facebook/vjepa2-vitl-fpc64-256). Used by DINOv3EEG and by Align/WorldModel backbones; must match the pretraining encoder.')
    parser.add_argument('--image_pool_heads', type=int, default=4,
                        help='Heads for the V-JEPA 2 attention-pool head (Align/WorldModel backbones only); ignored for image-encoder backbones with a CLS token.')
    parser.add_argument('--image_mode', type=str, default='raw',
                        choices=['raw', 'spectrogram'],
                        help="EEG->image transform: 'raw' (C,T) 2D layout (paper default) "
                             "or 'spectrogram' per-channel STFT grid")
    parser.add_argument('--image_size', type=int, default=0, help='override DINO image size; 0 means use the processor default')
    parser.add_argument('--stft_n_fft', type=int, default=64, help='STFT n_fft for EEG->image (spectrogram mode only)')
    parser.add_argument('--stft_hop_length', type=int, default=16, help='STFT hop length for EEG->image (spectrogram mode only)')

    # --- Learnable spectral-band backbone (must match the pretrained checkpoint) ---
    parser.add_argument('--fs', type=int, default=200, help='sampling rate (Hz) for the learnable filterbank')
    parser.add_argument('--highpass_hz', type=float, default=0.0,
                        help='zero-phase high-pass cutoff in Hz applied to the input before '
                             'patching (0 = off). E.g. 7 strips delta/theta + slow cue-locked '
                             'evoked drift, keeping mu/beta. Currently wired into PhysioNet.')
    parser.add_argument('--use_spectral_bands', action='store_true', default=False,
                        help='use the learnable SincNet-style filterbank + cross-band + multi-level-alignment backbone (must match pretraining)')
    parser.add_argument('--num_spectral_bands', type=int, default=4, help='number of learnable frequency bands K')
    parser.add_argument('--filterbank_kernel_size', type=int, default=101, help='SincNet bandpass FIR length (odd)')
    parser.add_argument('--filterbank_min_bw_hz', type=float, default=1.0, help='minimum per-band bandwidth in Hz (each band learns its own low/high independently)')
    parser.add_argument('--use_cross_band_attn', action='store_true', default=True, help='enable cross-band attention')
    parser.add_argument('--no_cross_band_attn', dest='use_cross_band_attn', action='store_false', help='disable cross-band attention (ablation)')
    parser.add_argument('--cross_band_every', type=int, default=1, help='apply cross-band attention every N encoder layers')
    parser.add_argument('--use_band_type_embedding', action='store_true', default=True, help='add a learned per-band type embedding')
    parser.add_argument('--no_band_type_embedding', dest='use_band_type_embedding', action='store_false', help='disable band-type embedding (ablation)')
    parser.add_argument('--num_visual_levels', type=int, default=3, help='number of image-encoder hidden levels to align bands against')
    parser.add_argument('--band_decorr_weight', type=float, default=0.01, help='weight of the band-decorrelation regularizer')
    parser.add_argument('--use_lora', action='store_true', help='attach LoRA adapters to DINO attention Q/K/V')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze the DINO encoder (linear probing)')

    # --- gradient analysis ---
    parser.add_argument('--grad_analysis', action='store_true',
                        help='record per-parameter grad norms and weight drift '
                             'across training and dump CSVs + plots')
    parser.add_argument('--grad_analysis_dir', type=str, default=None,
                        help='where to write grad analysis artifacts '
                             '(default: <model_dir>/grad_analysis)')

    params = parser.parse_args()
    if os.environ.get("DEBUG", "0") == "1":
        params.batch_size = 2

    if params.use_pretrained_weights and params.foundation_dir:
        _, saved_params = load_pretrain_checkpoint(params.foundation_dir)
        apply_arch_params(params, saved_params)

    print(params)

    setup_seed(params.seed)
    # torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    wandb.init(project='CSBrain_finetune', group=f"{params.downstream_dataset}", name=params.wandb_run_name)
    results = None
    if params.downstream_dataset == 'FACED':
        load_dataset = faced_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_faced.Model(params)
        t = Trainer(params, data_loader, model)
        results = t.train_for_multiclass()
    elif params.downstream_dataset == 'SEED-V':
        load_dataset = seedv_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedv.Model(params)
        t = Trainer(params, data_loader, model)
        results = t.train_for_multiclass()
    elif params.downstream_dataset == 'PhysioNet-MI':
        load_dataset = physio_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        if params.model == 'DINOv3EEG':
            from models import dinov3_eeg
            model = dinov3_eeg.Model(params)
        else:
            model = model_for_physio.Model(params)
        t = Trainer(params, data_loader, model)
        results = t.train_for_multiclass()
    elif params.downstream_dataset == 'SHU-MI':
        load_dataset = shu_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_shu.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'ISRUC':
        load_dataset = isruc_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_isruc.Model(params)
        t = Trainer(params, data_loader, model)
        results = t.train_for_multiclass()
    elif params.downstream_dataset == 'CHB-MIT':
        load_dataset = chb_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_chb.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC2020-3':
        load_dataset = speech_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_speech.Model(params)
        t = Trainer(params, data_loader, model)
        results = t.train_for_multiclass()
    elif params.downstream_dataset == 'Mumtaz2016':
        load_dataset = mumtaz_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_mumtaz.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'SEED-VIG':
        # load_dataset = seedvig_dataset.LoadDataset(params)
        load_dataset = seedvig_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedvig.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_regression()
    elif params.downstream_dataset == 'MentalArithmetic':
        load_dataset = stress_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_stress.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUEV':
        load_dataset = tuev_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuev.Model(params)
        t = Trainer(params, data_loader, model)
        results = t.train_for_multiclass()
    elif params.downstream_dataset == 'TUAB':
        load_dataset = tuab_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuab.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUSL':
        load_dataset = tusl_dataset.get_data_loader(params)
        data_loader = load_dataset
        model = model_for_tusl.Model(params) # TODO
        t = Trainer(params, data_loader, model)
        results = t.train_for_multiclass()
    elif params.downstream_dataset == 'BCIC-IV-2a':
        load_dataset = bciciv2a_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_bciciv2a.Model(params)
        t = Trainer(params, data_loader, model)
        results = t.train_for_multiclass()
    elif params.downstream_dataset == 'siena': 
        load_dataset = siena_dataset.LoadDataset(params) 
        data_loader = load_dataset.get_data_loader()
        model = model_for_siena.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'HMC':  # 
        load_dataset = hmc_dataset.LoadDataset(params) 
        data_loader = load_dataset.get_data_loader()
        model = model_for_hmc.Model(params)
        t = Trainer(params, data_loader, model)
        results = t.train_for_multiclass()
    print("model:", params.model, "seed:", params.seed, "lr:", params.lr,"weight_decay:",params.weight_decay, "dropout:", params.dropout, "foundation_dir:", params.foundation_dir)

    if params.results_csv and results is not None:
        csv_path = params.results_csv
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['base_model_ckpt', 'finetuned_model_ckpt',
                                 'val_kappa', 'val_acc', 'val_f1',
                                 'test_kappa', 'test_acc', 'test_f1'])
            writer.writerow([
                os.path.join(os.path.basename(os.path.dirname(params.foundation_dir)), os.path.basename(params.foundation_dir)),
                os.path.basename(results['model_path']),
                results['val_kappa'], results['val_acc'], results['val_f1'],
                results['test_kappa'], results['test_acc'], results['test_f1'],
            ])
        print(f"Results appended to {csv_path}")

    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
