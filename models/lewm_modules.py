"""LeWorldModel (LeWM) building blocks — verbatim port of le-wm/module.py.

Source: https://github.com/lucas-maes/le-wm  (Maes, Le Lidec, Scieur, LeCun,
Balestriero 2026, "LeWorldModel: Stable End-to-End Joint-Embedding Predictive
Architecture from Pixels").

Ported for the from-scratch, jointly-trained world model (see
plans/mighty-dazzling-lovelace.md). The two ingredients that make LeWM stable
end-to-end WITHOUT an EMA target or stop-gradient are here:

  * ``SIGReg`` — the Sketched Isotropic Gaussian Regularizer (Epps-Pulley
    normality test over random 1-D projections). It is the SOLE collapse guard:
    minimising it pushes the embeddings toward an isotropic Gaussian, so the
    trivial constant-collapse solution is penalised even though the prediction
    target is the online encoder's own (non-detached) embedding.
  * ``ARPredictor`` — a causal autoregressive transformer whose blocks are
    conditioned (AdaLN-zero) on a per-step "action" embedding. In CSBrain the
    EEG plays the role of LeWM's action.

The einops ``rearrange`` calls in the original are rewritten with native torch
``reshape``/``permute`` so this module has no einops dependency.
"""

import torch
from torch import nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU).

    Verbatim from le-wm/module.py. ``forward`` expects ``proj`` of shape
    ``(T, B, D)`` and reduces over the batch axis ``B`` to estimate the
    empirical characteristic function of the D-dim embeddings projected onto
    ``num_proj`` random unit directions, comparing it (Epps-Pulley statistic)
    to the standard-normal characteristic function ``exp(-t^2/2)``. Returns a
    scalar; minimising it drives the marginals toward N(0, 1) (isotropic
    Gaussian, via Cramer-Wold).
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj: (T, B, D)."""
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device,
                        dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


class FeedForward(nn.Module):
    """FeedForward network used in Transformers."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with (optional) causal masking."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """x: (B, T, D)."""
        b, t, _ = x.shape
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # each (B, T, inner_dim)
        # (B, T, h*d) -> (B, h, T, d)
        q, k, v = (
            u.reshape(b, t, self.heads, self.dim_head).permute(0, 2, 1, 3)
            for u in qkv
        )
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = out.permute(0, 2, 1, 3).reshape(b, t, self.heads * self.dim_head)
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning.

    ``causal`` (default True, as in LeWM's AR predictor) makes the self-attention
    causal; pass ``causal=False`` for a bidirectional block (e.g. a dense
    multi-frame predictor where every query patch may attend to every other).
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=True):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.causal = causal
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), causal=self.causal)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    """Encodes a per-step conditioning sequence (LeWM's action encoder).

    In CSBrain this wraps the per-window EEG global representation into the
    action embedding that conditions the AR predictor.
    """

    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """x: (B, T, D)."""
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation (projector)."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """x: (B*T, D)."""
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction.

    Verbatim from le-wm/module.py. ``forward(x, c)`` adds a learned positional
    embedding to ``x`` (B, T, d) and runs a causal transformer whose blocks are
    AdaLN-zero conditioned on ``c`` (B, T, act_dim), returning the contextual
    next-step predictions (B, T, output_dim).
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """x: (B, T, d)  c: (B, T, act_dim)."""
        t = x.size(1)
        x = x + self.pos_embedding[:, :t]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
