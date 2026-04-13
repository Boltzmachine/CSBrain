import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    """Gradient Reversal Layer (Ganin et al., 2016).
    Forward: identity. Backward: negate gradients scaled by alpha.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


class SessionDiscriminator(nn.Module):
    """Pairwise discriminator that predicts whether two representations
    come from the same recording session.

    Input:  two (N, d_model) representation tensors
    Output: (N,) logits  (positive → same session)
    """

    def __init__(self, d_model, hidden_dim=256):
        super().__init__()
        # Features: |rep_a - rep_b| concatenated with rep_a * rep_b
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rep_a, rep_b):
        diff = torch.abs(rep_a - rep_b)
        prod = rep_a * rep_b
        features = torch.cat([diff, prod], dim=-1)
        return self.net(features).squeeze(-1)


def construct_session_pairs(session_ids, max_pairs=128):
    """Build balanced same-session / different-session pairs from a batch.

    Args:
        session_ids: list[str] of length B
        max_pairs: cap on the number of pairs per class (pos / neg)

    Returns:
        indices_a, indices_b: (N,) long tensors of sample indices
        labels: (N,) float tensor, 1 = same session, 0 = different
        Returns empty tensors if no positive pairs can be formed.
    """
    session_to_indices = defaultdict(list)
    for i, sid in enumerate(session_ids):
        session_to_indices[sid].append(i)

    # --- positive pairs (same session) ---
    pos_pairs = []
    for indices in session_to_indices.values():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pos_pairs.append((indices[i], indices[j]))

    if len(pos_pairs) == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([])

    if len(pos_pairs) > max_pairs:
        pos_pairs = random.sample(pos_pairs, max_pairs)

    # --- negative pairs (different session), same count as positive ---
    sessions = [s for s, idx in session_to_indices.items() if len(idx) >= 1]
    neg_pairs = []
    attempts = 0
    while len(neg_pairs) < len(pos_pairs) and attempts < len(pos_pairs) * 10:
        s1, s2 = random.sample(sessions, 2)
        i = random.choice(session_to_indices[s1])
        j = random.choice(session_to_indices[s2])
        neg_pairs.append((i, j))
        attempts += 1

    neg_pairs = neg_pairs[: len(pos_pairs)]

    all_pairs = pos_pairs + neg_pairs
    labels = [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs)

    # shuffle together
    combined = list(zip(all_pairs, labels))
    random.shuffle(combined)
    all_pairs, labels = zip(*combined)

    indices_a = torch.tensor([p[0] for p in all_pairs], dtype=torch.long)
    indices_b = torch.tensor([p[1] for p in all_pairs], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)
    return indices_a, indices_b, labels
