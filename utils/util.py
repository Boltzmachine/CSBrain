import os
import random
import signal

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import random

def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask

def to_tensor(array):
    return torch.from_numpy(array).float()


# Fields that control the encoder architecture and must match between
# pretraining and finetuning. Stored in the checkpoint so finetune scripts
# no longer need to re-specify them.
ARCH_PARAM_FIELDS = (
    'in_dim', 'out_dim', 'd_model', 'dim_feedforward',
    'seq_len', 'n_layer', 'nhead',
    'TemEmbed_kernel_sizes',
    'project_to_source', 'num_sources',
    'patch_embed_type', 'mamba_band_periods', 'n_mamba_layers',
    'mamba_d_state', 'mamba_d_conv', 'mamba_expand',
    'use_llm_vq', 'num_language_tokens', 'max_llm_codebook_size',
)


def save_pretrain_checkpoint(path, model, params):
    torch.save({
        'state_dict': model.state_dict(),
        'params': vars(params),
    }, path)


def load_pretrain_checkpoint(path, map_location='cpu'):
    """Load a pretraining checkpoint, returning (state_dict, saved_params).

    Handles both the new format ({'state_dict', 'params'}) and legacy
    checkpoints that saved the bare state_dict; in the legacy case
    saved_params is None.
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if (isinstance(obj, dict) and 'state_dict' in obj and 'params' in obj
            and isinstance(obj['params'], dict)):
        return obj['state_dict'], obj['params']
    return obj, None


def build_muon_optimizer(model, lr, weight_decay, muon_lr=None,
                         muon_momentum=0.95, muon_weight_decay=None,
                         extra_aux_keywords=()):
    """Wrap ``model``'s trainable params in a SingleDeviceMuonWithAuxAdam.

    2D hidden weights go to Muon; embeddings, classifier heads, prototypes,
    cls/mask tokens, and any 1D params (gains, biases, norms) go to the
    internal AdamW. Per the Muon paper, embedding and final-output layers
    must not be optimised by Muon.

    ``lr`` is the AdamW lr; ``muon_lr`` the Muon lr (defaults to 0.02 if
    unset, which matches the package default for hidden weights).
    """
    from muon import SingleDeviceMuonWithAuxAdam

    if muon_lr is None:
        muon_lr = 0.02
    if muon_weight_decay is None:
        muon_weight_decay = weight_decay

    # Per the Muon paper, lookup embeddings, the final output layer, and
    # all gains/biases must use AdamW. We match by both module type
    # (nn.Embedding) and by path component (head/classifier/prototype/
    # CLS/mask tokens etc).
    aux_components = {
        'head', 'classifier', 'last_layer', 'prototype', 'prototypes',
        'cls_token', 'mask_token', 'pos_emb', 'position_emb',
        'positional_emb', 'positional_encoding',
    } | set(extra_aux_keywords)

    import torch.nn as nn
    embedding_param_ids = set()
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Embedding):
            for p in mod.parameters(recurse=False):
                embedding_param_ids.add(id(p))

    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split('.')
        is_aux = (
            id(p) in embedding_param_ids
            or any(part in aux_components for part in parts)
        )
        if p.ndim >= 2 and not is_aux:
            muon_params.append(p)
        else:
            adam_params.append(p)

    param_groups = []
    if muon_params:
        param_groups.append(dict(
            params=muon_params, use_muon=True,
            lr=muon_lr, momentum=muon_momentum,
            weight_decay=muon_weight_decay,
        ))
    if adam_params:
        param_groups.append(dict(
            params=adam_params, use_muon=False,
            lr=lr, betas=(0.9, 0.95), eps=1e-10,
            weight_decay=weight_decay,
        ))
    def _safe_numel(ps):
        # Lazy modules may hold UninitializedParameters at this point;
        # .numel() raises on those, so just count tensors instead.
        try:
            return sum(p.numel() for p in ps)
        except ValueError:
            return f"{len(ps)} tensors"
    print(f"[Muon] {_safe_numel(muon_params)} params -> Muon "
          f"(lr={muon_lr}), {_safe_numel(adam_params)} -> AdamW "
          f"(lr={lr})")
    return SingleDeviceMuonWithAuxAdam(param_groups)


def apply_arch_params(params, saved_params):
    """Overlay encoder-architecture fields from a pretraining checkpoint.

    Wrapper flags (dino_mode, WorldModel) are intentionally not overlayed:
    finetuning always builds the plain encoder, and the state_dict loader
    strips wrapper prefixes. ``model`` is mapped WorldModel -> Align for
    the same reason.
    """
    if saved_params is None:
        return
    for field in ARCH_PARAM_FIELDS:
        if field in saved_params:
            setattr(params, field, saved_params[field])
    saved_model = saved_params.get('model')
    if saved_model == 'WorldModel':
        params.model = 'Align'
    elif saved_model is not None:
        params.model = saved_model


if __name__ == '__main__':
    a = generate_mask(192, 32, 15, mask_ratio=0.5, device=None)
    print(a)