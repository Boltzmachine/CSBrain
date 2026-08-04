"""From-scratch (randomly initialized) ViT for the LeWM-style world model.

LeWM trains a ViT-tiny end-to-end from raw pixels (``pretrained: false``,
patch14, 224). Their reference builds it via ``stable_pretraining``'s
``vit_hf`` helper, which is not installed here — so we build the equivalent
geometry with a randomly-initialized HuggingFace ``ViTModel``.

The resulting model is a drop-in for CSBrainAlign's ``pretrained_image_encoder``
attribute on the trainable (``scratch_vit``) path:

  * ``forward(pixel_values=...).last_hidden_state`` -> (B, 1 + P, hidden_size)
    with the CLS token at index 0 and NO register tokens, so the existing
    DINOv2-style extraction (``last_hidden_state[:, 1 + n_reg:]``) yields the
    P-patch grid with ``n_reg == 0``.
  * ``.config.hidden_size`` feeds ``image_feature_dim`` / ``alignment_feature_dim``
    and ``FramePredictor.frame_dim``.
  * ``.config.model_type == 'vit'`` routes through CSBrainAlign's DINOv2/CLS
    code paths (the ``!= 'vjepa2'`` branch).

Dropout is disabled by default so train- vs eval-mode forwards are identical
(matching the frozen encoder's behaviour, but WITH gradients).
"""

from transformers import ViTModel, ViTConfig


# ViT size presets: (hidden_size, depth, num_heads). dim_head is hidden/heads.
_VIT_SIZES = {
    "tiny": (192, 12, 3),    # LeWM default (~5.5M params); dim_head 64
    "small": (384, 12, 6),
    "base": (768, 12, 12),
}


def build_scratch_vit(
    size="tiny",
    embed_dim=None,
    depth=None,
    heads=None,
    patch_size=16,
    image_size=224,
    mlp_ratio=4,
):
    """Build a randomly-initialized ``transformers.ViTModel``.

    ``size`` selects a preset; ``embed_dim``/``depth``/``heads`` override
    individual dims when given (embed_dim must be divisible by heads).
    """
    p_embed, p_depth, p_heads = _VIT_SIZES.get(size, _VIT_SIZES["tiny"])
    hidden = int(embed_dim) if embed_dim else p_embed
    n_layers = int(depth) if depth else p_depth
    n_heads = int(heads) if heads else p_heads
    assert hidden % n_heads == 0, (
        f"scratch ViT hidden_size ({hidden}) must be divisible by heads ({n_heads})")

    cfg = ViTConfig(
        hidden_size=hidden,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        intermediate_size=hidden * mlp_ratio,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        qkv_bias=True,
    )
    # add_pooling_layer=False -> forward returns last_hidden_state (CLS at 0)
    # with no extra pooled/tanh head to train.
    return ViTModel(cfg, add_pooling_layer=False)
