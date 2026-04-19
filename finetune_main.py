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
import wandb

def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)') # 42
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
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
    parser.add_argument('--hemisphere_flip_aug', action='store_true', default=False,
                        help='hemisphere-flip data augmentation with label swap (motor imagery)')

    # --- SSM/Mamba multi-frequency patch embedding ---
    parser.add_argument('--patch_embed_type', type=str, default='cnn', choices=['cnn', 'mamba'],
                        help='patch embedder: CNN (default) or multi-frequency Mamba SSM')
    parser.add_argument('--mamba_band_periods', type=str, default=None,
                        help='list of band sample periods, e.g. "[200,600,1200]"; default [in_dim, 3*in_dim, 6*in_dim]')
    parser.add_argument('--n_mamba_layers', type=int, default=2, help='number of stacked Mamba blocks in the patch embedder')
    parser.add_argument('--mamba_d_state', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--mamba_d_conv', type=int, default=4, help='Mamba depthwise conv width')
    parser.add_argument('--mamba_expand', type=int, default=2, help='Mamba expansion factor')

    params = parser.parse_args()
    if os.environ.get("DEBUG", "0") == "1":
        params.batch_size = 2
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
