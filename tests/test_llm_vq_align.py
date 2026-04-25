"""Smoke test for CSBrainAlign with LLM-embedding VQ tokenization."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.alignment import CSBrainAlign
from models.llm_vq import LLMEmbeddingVQ

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_llm_embedding_vq_returns_quantized():
    """LLMEmbeddingVQ.forward should surface the LLM-space quantized tensor."""
    torch.manual_seed(0)
    d_model, llm_dim = 16, 32
    vq = LLMEmbeddingVQ(input_dim=d_model, llm_embedding_dim=llm_dim, max_codebook_size=64)
    bank = torch.randn(128, llm_dim)
    freqs = torch.arange(128, 0, -1, dtype=torch.long)
    vq.set_llm_embedding_bank(bank, token_frequencies=freqs)

    B, C, N = 2, 3, 4
    x = torch.randn(B, C, N, d_model)
    out, info = vq(x)
    assert out.shape == (B, C, N, d_model), out.shape
    assert "quantized" in info
    assert info["quantized"].shape == (B, C, N, llm_dim), info["quantized"].shape
    assert info["indices"].shape == (B, C * N), info["indices"].shape
    print("LLMEmbeddingVQ returns quantized tensor: OK")


def test_align_with_llm_vq_forward():
    """Construct CSBrainAlign with use_llm_vq=True and run a forward pass."""
    torch.manual_seed(0)
    B, C, N, P = 2, 4, 6, 40
    d_model = 40
    model = CSBrainAlign(
        in_dim=P, out_dim=P, d_model=d_model, dim_feedforward=80,
        seq_len=N, n_layer=3, nhead=4,
        TemEmbed_kernel_sizes=[(1,), (3,)],
        brain_regions=None,
        alignment_weight=1.0,
        use_llm_vq=True,
        num_language_tokens=4,
        max_llm_codebook_size=256,
        llm_vq_aux_weight=0.1,
    ).to(DEVICE)

    # Sanity: the VQ path modules exist and the old path is unused.
    assert hasattr(model, 'global_to_k_tokens')
    assert hasattr(model, 'llm_vq')
    assert hasattr(model, 'llm_contrastive_proj')
    llama_dim = model.llm_vq.llm_embedding_dim
    assert model.global_to_k_tokens.out_features == 4 * d_model
    assert model.llm_contrastive_proj[0].in_features == 4 * llama_dim

    x = torch.randn(B, C, N, P, device=DEVICE)
    ch_coords = torch.randn(B, C, 3, device=DEVICE) * 0.1
    vcm = torch.ones(B, C, dtype=torch.bool, device=DEVICE)
    vlm = torch.ones(B, N, dtype=torch.bool, device=DEVICE)
    has_image = torch.ones(B, dtype=torch.bool, device=DEVICE)  # force contrastive branch
    batch = {
        'timeseries': x,
        'ch_coords': ch_coords,
        'valid_channel_mask': vcm,
        'valid_length_mask': vlm,
        'has_image': has_image,
        'image_encoder_inputs': {'pixel_values': torch.zeros(B, 3, 224, 224, device=DEVICE)},
        'source': ['x'] * B,
    }
    mask = torch.zeros(B, C, N, dtype=torch.long, device=DEVICE)
    mask[:, :, 0] = 1
    out, aux = model(batch, mask=mask)

    assert out.shape == (B, C, N, P), out.shape
    assert 'contrastive_loss_0' in aux, list(aux.keys())
    assert 'llm_vq_aux_loss' in aux, list(aux.keys())
    vq_weight, vq_loss = aux['llm_vq_aux_loss']
    assert vq_weight == 0.1
    assert vq_loss.dim() == 0 and torch.isfinite(vq_loss), vq_loss

    # Backprop through the contrastive + VQ path.
    total = aux['contrastive_loss_0'][0] * aux['contrastive_loss_0'][1] + vq_weight * vq_loss
    total.backward()
    assert model.global_to_k_tokens.weight.grad is not None
    # The frozen LLM codebook should not accumulate grad.
    assert model.llm_vq.quantizer.codebook.requires_grad is False
    print(f"CSBrainAlign + LLM VQ forward+backward: OK, vq_loss={vq_loss.item():.4f}")


if __name__ == '__main__':
    test_llm_embedding_vq_returns_quantized()
    test_align_with_llm_vq_forward()
    print("All tests passed.")
