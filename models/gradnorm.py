"""GradNorm adaptive loss balancing (Chen et al., 2018, arXiv:1711.02257).

The pretraining loss is a sum of very differently-scaled terms (masked recon,
frame prediction, image alignment, band-phase/envelope aux, ...). Their gradient
magnitudes at the shared encoder differ by orders of magnitude, so hand-tuned
static weights are fragile. GradNorm learns a per-term weight ``w_i`` online so
that every term contributes a comparable gradient norm at a chosen shared layer,
scaled by each term's relative training rate.

Per step (see :meth:`GradNormBalancer.update`), for the tasks present this step:

  * ``G_i = w_i * || d L_i / d W ||``   — task i's gradient norm at shared W
  * ``Gbar = mean_i G_i``               — target scale (detached)
  * ``r_i = (L_i / L_i(0)) / mean_j(L_j / L_j(0))`` — relative inverse train rate
  * ``target_i = Gbar * r_i ** alpha``  — desired gradient norm (detached)
  * ``L_grad = sum_i | G_i - target_i |``  — minimised w.r.t. ``w_i`` only

``alpha`` (asymmetry) controls how hard slow-training tasks are pulled up
(``alpha=0`` -> just equalise gradient norms; larger -> respect training rates
more). After each weight step the present weights are renormalised to sum to the
number of present tasks so the average weight stays ~1 and the weights can't drift
to zero. The weight update uses ``autograd.grad`` (not ``.backward``) so it never
touches the model's ``.grad`` — the caller runs the normal model backward with the
weighted loss separately.

Only terms whose caller-side weight is nonzero are balanced, so turning a term OFF
(weight 0) still removes it entirely.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def _san(name: str) -> str:
    """ParameterDict keys can't contain '.'; make a safe key from a loss name."""
    return ''.join(c if (c.isalnum() or c == '_') else '_' for c in name)


class GradNormBalancer(nn.Module):
    def __init__(self, alpha: float = 1.5, lr: float = 0.025,
                 weight_min: float = 1e-3):
        super().__init__()
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.weight_min = float(weight_min)
        self.weights = nn.ParameterDict()   # sanitized-name -> scalar weight
        self.initial: Dict[str, float] = {}  # name -> L_i(0)
        self._opt = None

    # ------------------------------------------------------------------
    def _ensure(self, names: List[str], device) -> None:
        """Lazily create a weight (init 1.0) for any new task; (re)build the
        weight optimiser when the parameter set grows."""
        grew = False
        for n in names:
            k = _san(n)
            if k not in self.weights:
                self.weights[k] = nn.Parameter(torch.ones((), device=device))
                grew = True
        if grew or self._opt is None:
            self._opt = torch.optim.Adam(self.weights.parameters(), lr=self.lr)

    def weight_of(self, name: str) -> float:
        k = _san(name)
        return float(self.weights[k]) if k in self.weights else 1.0

    # ------------------------------------------------------------------
    def weighted_sum(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``sum_i w_i * L_i`` over the present tasks, using the CURRENT weights
        (detached from the GradNorm weight-update graph so the model backward
        doesn't propagate into ``w_i``). Call before the model backward."""
        names = list(losses.keys())
        self._ensure(names, next(iter(losses.values())).device)
        return sum(self.weights[_san(n)].detach() * losses[n] for n in names)

    @torch.no_grad()
    def _renormalize(self, names: List[str]) -> None:
        for n in names:
            self.weights[_san(n)].clamp_(min=self.weight_min)
        s = sum(float(self.weights[_san(n)]) for n in names)
        if s > 1e-8:
            scale = len(names) / s
            for n in names:
                self.weights[_san(n)].mul_(scale)

    def update(self, losses: Dict[str, torch.Tensor],
               shared_params: List[torch.Tensor]) -> Dict[str, float]:
        """Do one GradNorm weight step. Call AFTER the model backward (which must
        use ``retain_graph=True``) so the task graphs are still alive. Uses
        ``autograd.grad`` throughout, so the model's ``.grad`` is untouched.
        Returns a dict of ``gnw_<task>`` (weight) and ``gngrad_<task>`` (gradient
        norm) for logging. No-op returning current weights if <2 tasks."""
        names = [n for n in losses if losses[n].requires_grad]
        if len(names) < 2 or not shared_params:
            return {f'gnw_{n}': self.weight_of(n) for n in names}
        for n in names:
            if n not in self.initial:
                self.initial[n] = max(abs(float(losses[n].detach())), 1e-8)

        # G_i = w_i * ||d L_i / d W|| at the shared layer W. The norm is of the RAW
        # loss's gradient, which is independent of w_i, so it is DETACHED — G_i's
        # only dependence on w_i is the explicit ``w_i *`` factor. No create_graph
        # (no second-order term): ``d L_grad / d w_i = sign(G_i - target_i) *
        # ||d L_i/d W||`` exactly, and building the second-order graph would OOM on
        # a large model for no benefit. First-order grads are freed between tasks.
        Gs = []
        for n in names:
            grads = torch.autograd.grad(
                losses[n], shared_params, retain_graph=True, allow_unused=True)
            gn = torch.cat([g.reshape(-1) for g in grads if g is not None]).norm()
            Gs.append(self.weights[_san(n)] * gn.detach())
        Gs = torch.stack(Gs)
        Gbar = Gs.mean().detach()

        # Relative inverse training rate r_i, target gradient norm (detached).
        loss_ratio = torch.stack([
            losses[n].detach() / self.initial[n] for n in names])
        r = loss_ratio / loss_ratio.mean().clamp(min=1e-8)
        target = (Gbar * r.pow(self.alpha)).detach()
        l_grad = (Gs - target).abs().sum()

        # Weight gradient via autograd.grad -> does NOT touch model .grad.
        wparams = [self.weights[_san(n)] for n in names]
        wgrads = torch.autograd.grad(l_grad, wparams)
        self._opt.zero_grad()
        for p, g in zip(wparams, wgrads):
            p.grad = g
        self._opt.step()
        self._renormalize(names)

        logs = {f'gnw_{n}': self.weight_of(n) for n in names}
        logs.update({f'gngrad_{n}': float(Gs[i].detach())
                     for i, n in enumerate(names)})
        return logs
