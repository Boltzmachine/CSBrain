"""Sanity test for MambaPatchEmbedding + CSBrainAlign integration."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models.alignment import MambaPatchEmbedding, CSBrainAlign

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_mamba_patch_embedding_shape():
    if DEVICE == 'cpu':
        print("SKIP forward: Mamba requires CUDA — validating shape helpers only")
        mpe = MambaPatchEmbedding(
            in_dim=50, out_dim=50, d_model=32, seq_len=6,
            band_periods=[50, 150, 300],
            n_mamba_layers=1,
        )
        events = mpe._interleave_events(6)
        assert events == [
            (0, 0, 0), (1, 0, 1), (2, 0, 2), (2, 1, 0),
            (3, 0, 3), (4, 0, 4), (5, 0, 5), (5, 1, 1), (5, 2, 0),
        ], events
        assert mpe.expected_seq_len(6) == 9
        assert mpe.base_band_positions(6) == [0, 1, 2, 4, 5, 6]
        print("MambaPatchEmbedding interleave helpers: OK")
        return
    B, C, N, P = 2, 4, 6, 50
    d_model = 32
    mpe = MambaPatchEmbedding(
        in_dim=P, out_dim=P, d_model=d_model, seq_len=N,
        band_periods=[P, 3 * P, 6 * P],
        n_mamba_layers=1, d_state=8, d_conv=4, expand=2,
        causal=False,
    ).to(DEVICE)
    x = torch.randn(B, C, N, P, device=DEVICE)
    out = mpe(x)
    # N_total = N + N//3 + N//6 = 6 + 2 + 1 = 9
    assert out.shape == (B, C, 9, d_model), out.shape
    # Interleave events should match the user example for the first 9 tokens:
    # (0,0,0),(1,0,1),(2,0,2),(2,1,0),(3,0,3),(4,0,4),(5,0,5),(5,1,1),(5,2,0)
    events = mpe._interleave_events(N)
    assert events == [
        (0, 0, 0), (1, 0, 1), (2, 0, 2), (2, 1, 0),
        (3, 0, 3), (4, 0, 4), (5, 0, 5), (5, 1, 1), (5, 2, 0),
    ], events
    f1_positions = mpe.base_band_positions(N)
    assert f1_positions == [0, 1, 2, 4, 5, 6], f1_positions
    print("MambaPatchEmbedding shape & interleave: OK")


def test_valid_length_mask_adaptation():
    mpe = MambaPatchEmbedding(
        in_dim=50, out_dim=50, d_model=32, seq_len=6,
        band_periods=[50, 150, 300],
        n_mamba_layers=1,
    )
    vlm = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    adapted = mpe.adapt_valid_length_mask(vlm)
    # base positions for events: [0,1,2,2,3,4,5,5,5]; vlm [1,1,1,1,0,0]
    expected = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.equal(adapted, expected), (adapted, expected)
    print("valid_length_mask adaptation: OK")


def test_align_forward_with_mamba():
    if DEVICE == 'cpu':
        print("SKIP CSBrainAlign forward test: requires CUDA for Mamba")
        return
    torch.manual_seed(0)
    B, C, N, P = 2, 3, 6, 50
    d_model = 32
    model = CSBrainAlign(
        in_dim=P, out_dim=P, d_model=d_model, dim_feedforward=64,
        seq_len=N, n_layer=2, nhead=4,
        TemEmbed_kernel_sizes=[(1,), (3,)],
        brain_regions=None,
        patch_embed_type='mamba',
        mamba_band_periods=[P, 3 * P, 6 * P],
        n_mamba_layers=1, mamba_d_state=8,
        alignment_weight=0.0,
    ).to(DEVICE)
    x = torch.randn(B, C, N, P, device=DEVICE)
    ch_coords = torch.randn(B, C, 3, device=DEVICE) * 0.1
    vcm = torch.ones(B, C, dtype=torch.bool, device=DEVICE)
    vlm = torch.ones(B, N, dtype=torch.bool, device=DEVICE)
    has_image = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    batch = {
        'timeseries': x, 'ch_coords': ch_coords,
        'valid_channel_mask': vcm, 'valid_length_mask': vlm,
        'has_image': has_image,
        'image_encoder_inputs': {'pixel_values': torch.zeros(B, 3, 224, 224, device=DEVICE)},
        'source': ['x'] * B,
    }
    mask = torch.zeros(B, C, N, dtype=torch.long, device=DEVICE)
    mask[:, :, 0] = 1  # mask first patch on every channel
    out, aux = model(batch, mask=mask)
    # out should be (B, C, N, out_dim) — compatible with mask-indexed recon loss
    assert out.shape == (B, C, N, P), out.shape
    # Reconstruction indexing should work
    masked_y = out[mask == 1]
    masked_x = x[mask == 1]
    assert masked_y.shape == masked_x.shape
    print("CSBrainAlign with Mamba patch embed: OK, out.shape=", out.shape)


if __name__ == '__main__':
    test_mamba_patch_embedding_shape()
    test_valid_length_mask_adaptation()
    test_align_forward_with_mamba()
    print("All tests passed.")
