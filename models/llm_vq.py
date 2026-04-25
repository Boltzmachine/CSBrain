import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CSBrain_transformerlayer import *
from models.CSBrain_transformer import *
from collections import Counter
from collections import defaultdict

from vector_quantize_pytorch import LFQ, VectorQuantize
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json

class LLMEmbeddingVQ(nn.Module):
    def __init__(
        self,
        input_dim,
        llm_embedding_dim,
        quantizer_method='vq',
        max_codebook_size=4096,
        entropy_loss_weight=0.1,
        diversity_gamma=1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.llm_embedding_dim = llm_embedding_dim
        self.max_codebook_size = max_codebook_size
        self.quantizer_method = quantizer_method
        self.entropy_loss_weight = entropy_loss_weight
        self.diversity_gamma = diversity_gamma

        self.input_proj = nn.Linear(input_dim, llm_embedding_dim)
        self.quantizer = None

        self.out_proj = nn.Linear(llm_embedding_dim, input_dim)

        self.register_buffer(
            "llm_embedding_bank",
            torch.empty(0, llm_embedding_dim),
            persistent=False,
        )
        self.register_buffer(
            "selected_token_ids",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

    def set_llm_embedding_bank(self, llm_embedding_bank, token_frequencies=None):
        if llm_embedding_bank.ndim != 2:
            raise ValueError("llm_embedding_bank must have shape [vocab_size, llm_embedding_dim].")
        if llm_embedding_bank.size(-1) != self.llm_embedding_dim:
            raise ValueError(
                f"Expected LLM embedding dim {self.llm_embedding_dim}, "
                f"got {llm_embedding_bank.size(-1)}."
            )

        vocab_size = llm_embedding_bank.size(0)
        keep_tokens = min(self.max_codebook_size, vocab_size)

        token_frequencies = token_frequencies.reshape(-1)
        if token_frequencies.numel() != vocab_size:
            raise ValueError("token_frequencies must have one value per LLM token.")
        selected_token_ids = torch.topk(token_frequencies, k=keep_tokens).indices

        selected_token_ids = selected_token_ids.long()
        target_device = self.input_proj.weight.device
        target_dtype = self.input_proj.weight.dtype
        reduced_bank = llm_embedding_bank.index_select(0, selected_token_ids).detach()
        reduced_bank = reduced_bank.to(device=target_device, dtype=target_dtype)
        selected_token_ids = selected_token_ids.to(device=target_device)

        self.llm_embedding_bank = reduced_bank
        self.selected_token_ids = selected_token_ids
        if self.quantizer_method == 'vq':
            self.quantizer = VectorQuantize(
                dim = reduced_bank.size(-1),          # dimension of the input vectorsq
                codebook_size = reduced_bank.size(0),     # codebook size
                decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = 1.,   # the weight on the commitment loss
                rotation_trick=True,
                learnable_codebook=False,
            )
        elif self.quantizer_method == 'lfq':
            self.quantizer = LFQ(
                dim=self.llm_embedding_dim,
                codebook_size=reduced_bank.size(0),
                # num_codebooks=1,
                entropy_loss_weight=self.entropy_loss_weight,
                diversity_gamma=self.diversity_gamma,
            )
        else:
            raise ValueError(f"Unsupported quantizer method: {self.quantizer_method}")  
        self.quantizer.to(device=target_device)
        self.quantizer.codebook.data.copy_(self.llm_embedding_bank)
        assert self.quantizer.codebook.requires_grad == False, "LLM embedding codebook should not be updated during training."

    def forward(self, x):
        B, C, N, D = x.shape
        projected_states = self.input_proj(x).view(B, C * N, -1)
        quantized, indices, vq_aux_loss = self.quantizer(projected_states)
        if indices.ndim == projected_states.ndim and indices.size(-1) == 1:
            indices = indices.squeeze(-1)
        if indices.ndim != projected_states.ndim - 1:
            raise ValueError("LFQ indices shape is incompatible with the projected EEG tensor.")

        return self.out_proj(quantized).view(B, C, N, -1), {
            "indices": indices,
            "quantized": quantized.view(B, C, N, -1),
            # "projected_states": projected_states,
            "aux_loss": vq_aux_loss,
        }


class CSBrainLLMVQ(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                 nhead=8, TemEmbed_kernel_sizes=[(1,), (3,), (5,)], brain_regions=[], sorted_indices=[],
                 max_llm_codebook_size=4096,
                 lfq_entropy_loss_weight=0.1, lfq_diversity_gamma=1.0):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)

        self.TemEmbed_kernel_sizes = TemEmbed_kernel_sizes
        kernel_sizes = self.TemEmbed_kernel_sizes
        self.TemEmbedEEGLayer = TemEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim, kernel_sizes=kernel_sizes, stride=1)

        self.brain_regions = brain_regions
        self.area_config = generate_area_config(sorted(brain_regions))
        self.BrainEmbedEEGLayer = BrainEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim)
        self.sorted_indices = sorted_indices

        encoder_layer = CSBrain_TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, area_config=self.area_config, sorted_indices=self.sorted_indices, batch_first=True,
            activation=F.gelu
        )
        self.encoder = CSBrain_TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)

        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )
        self.llm_vq = None

        self.features_by_layer = []
        self.input_features = []
        self.last_vq_info = None

        self.set_llm_embedding_bank(
            max_llm_codebook_size=max_llm_codebook_size,
            lfq_entropy_loss_weight=lfq_entropy_loss_weight,
            lfq_diversity_gamma=lfq_diversity_gamma,
        )

    def set_llm_embedding_bank(
        self,
        max_llm_codebook_size=4096,
        lfq_entropy_loss_weight=0.1,
        lfq_diversity_gamma=1.0,
    ):
        llm_id = "meta-llama/Llama-2-7b-hf"
        #load from huggingface
        from transformers import AutoTokenizer, AutoModel
        model = AutoModel.from_pretrained(
            llm_id,
            device_map='cpu',
        )
        llm_embedding_bank = model.get_input_embeddings().weight.data.to(self.proj_out[0].weight.device)
        del model
        torch.cuda.empty_cache()

        self.llm_vq = LLMEmbeddingVQ(
            input_dim=self.d_model,
            llm_embedding_dim=llm_embedding_bank.size(-1),
            max_codebook_size=max_llm_codebook_size,
            entropy_loss_weight=lfq_entropy_loss_weight,
            diversity_gamma=lfq_diversity_gamma,
        )

        llm_token_frequencies_dict = json.load(open("data/llama2_token_frequencies.json", "r"))
        llm_token_frequencies = torch.zeros(llm_embedding_bank.size(0), dtype=torch.long, device=llm_embedding_bank.device)
        for i, item in enumerate(llm_token_frequencies_dict):
            token_id = int(item["token_id"])
            count = int(item["count"])
            assert i == token_id, "Token IDs in the frequency file should be in order and match their index."
            llm_token_frequencies[token_id] = count

        self.llm_vq.set_llm_embedding_bank(
            llm_embedding_bank=llm_embedding_bank,
            token_frequencies=llm_token_frequencies,
        )

    def forward(self, batch, mask=None, return_vq_info=False):
        x = batch['timeseries']
        x = x[:, self.sorted_indices, :, :]

        patch_emb = self.patch_embedding(x, mask)
        vq_info = None

        for layer_idx in range(self.encoder.num_layers):
            patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
            patch_emb = self.BrainEmbedEEGLayer(patch_emb, self.area_config) + patch_emb

            patch_emb = self.encoder.layers[layer_idx](patch_emb, self.area_config)
            if layer_idx == self.encoder.num_layers - 3:
                patch_emb, vq_info = self.llm_vq(patch_emb)

        out = self.proj_out(patch_emb)
        self.last_vq_info = vq_info

        if return_vq_info:
            return out, vq_info

        return out
    
    def training_step(self, batch, mask=None):
        assert mask is not None
        out, vq_info = self.forward(batch['timeseries'], mask=mask, return_vq_info=True)
        return out, {
            "vq_aux_loss": (0.1, vq_info["aux_loss"]),
        }


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(d_model // 2 + 1, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, mask_x.shape[1] // 2 + 1)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb

def generate_area_config(brain_regions):
    region_to_channels = defaultdict(list)
    for channel_idx, region in enumerate(brain_regions):
        region_to_channels[region].append(channel_idx)

    area_config = {}
    for region, channels in region_to_channels.items():
        area_config[f'region_{region}'] = {
            'channels': len(channels),
            'slice': slice(channels[0], channels[-1] + 1)
        }
    return area_config

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CSBrain(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8).to(device)
    model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth',
                                     map_location=device))
    a = torch.randn((8, 16, 10, 200)).cuda()
    b = model(a)
    print(a.shape, b.shape)
