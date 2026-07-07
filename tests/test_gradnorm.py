"""Unit tests for GradNorm adaptive loss balancing (models/gradnorm.py)."""
from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn as nn

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models.gradnorm import GradNormBalancer, _san


class TestGradNorm(unittest.TestCase):
    def _shared(self):
        torch.manual_seed(0)
        return nn.Parameter(torch.randn(64))

    def test_balances_mismatched_gradients(self):
        # Two tasks whose gradients w.r.t. the shared param differ ~100x. GradNorm
        # (alpha=0 -> equalise grad norms) should upweight the weak task and drive
        # the weighted grad norms close together.
        W = self._shared()
        bal = GradNormBalancer(alpha=0.0, lr=0.05)

        def losses():
            return {'big': (W * 10.0).pow(2).mean(),   # large grad
                    'small': (W * 1.0).pow(2).mean()}   # ~100x smaller grad

        l = losses()
        raw_ratio = (torch.autograd.grad(l['big'], [W], retain_graph=True)[0].norm()
                     / torch.autograd.grad(l['small'], [W])[0].norm())
        self.assertGreater(float(raw_ratio), 50.0)      # starts badly imbalanced

        for _ in range(400):
            l = losses()
            total = bal.weighted_sum(l)
            if W.grad is not None:
                W.grad = None
            total.backward(retain_graph=True)
            bal.update(l, [W])

        wb, ws = bal.weight_of('big'), bal.weight_of('small')
        l = losses()
        Gb = wb * torch.autograd.grad(l['big'], [W], retain_graph=True)[0].norm()
        Gs = ws * torch.autograd.grad(l['small'], [W])[0].norm()
        bal_ratio = float(Gb / Gs)
        self.assertLess(ws * 1e-6, wb + ws)             # sanity
        self.assertGreater(ws, wb)                       # weak task upweighted
        # balanced within ~2x (down from ~100x) — a >25x improvement
        self.assertLess(max(bal_ratio, 1 / bal_ratio), 2.0)

    def test_no_model_grad_pollution(self):
        # update() uses autograd.grad, so it must never modify the model's .grad.
        W = self._shared()
        bal = GradNormBalancer(alpha=1.5, lr=0.05)
        l = {'a': (W * 3.0).pow(2).mean(), 'b': (W * 0.5).pow(2).mean()}
        total = bal.weighted_sum(l)
        total.backward(retain_graph=True)
        before = W.grad.detach().clone()
        bal.update(l, [W])
        self.assertTrue(torch.equal(W.grad, before))

    def test_weights_renormalize_to_n(self):
        W = self._shared()
        bal = GradNormBalancer(alpha=1.0, lr=0.1)
        for _ in range(20):
            l = {'a': (W * 5.0).pow(2).mean(),
                 'b': (W * 0.2).pow(2).mean(),
                 'c': (W * 1.0).pow(2).mean()}
            total = bal.weighted_sum(l)
            if W.grad is not None:
                W.grad = None
            total.backward(retain_graph=True)
            bal.update(l, [W])
        s = sum(bal.weight_of(n) for n in ('a', 'b', 'c'))
        self.assertAlmostEqual(s, 3.0, places=4)         # sum == n_tasks
        for n in ('a', 'b', 'c'):
            self.assertGreaterEqual(bal.weight_of(n), 1e-3)  # floor respected

    def test_single_task_is_noop(self):
        W = self._shared()
        bal = GradNormBalancer()
        l = {'only': (W * 2.0).pow(2).mean()}
        total = bal.weighted_sum(l)
        total.backward(retain_graph=True)
        logs = bal.update(l, [W])                        # <2 tasks -> no-op
        self.assertIn('gnw_only', logs)
        self.assertNotIn('gngrad_only', logs)            # no grad-norm step taken

    def test_weighted_sum_detaches_weights(self):
        # weighted_sum must detach w_i so the model backward doesn't flow into the
        # GradNorm weights (they're updated only via update()).
        W = self._shared()
        bal = GradNormBalancer()
        l = {'a': (W * 2.0).pow(2).mean(), 'b': (W * 1.0).pow(2).mean()}
        total = bal.weighted_sum(l)
        total.backward()
        for k in bal.weights:
            self.assertIsNone(bal.weights[k].grad)

    def test_sanitize(self):
        self.assertEqual(_san('contrastive_loss_0'), 'contrastive_loss_0')
        self.assertEqual(_san('a.b-c'), 'a_b_c')


if __name__ == '__main__':
    unittest.main(verbosity=2)
