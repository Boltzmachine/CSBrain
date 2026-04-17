"""
DINOv2-style loss functions for EEG self-distillation.

References:
    - DINOv2: https://github.com/facebookresearch/dinov2
    - Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """
    CLS-token cross-entropy loss between student and centered teacher outputs.

    The teacher output is centered (momentum-updated mean subtracted) and
    sharpened with a lower temperature, while the student uses a higher
    temperature.  Loss = -sum(teacher_softmax * log_softmax(student)).
    """

    def __init__(self, n_prototypes, student_temp=0.1,
                 teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp          # updated externally via schedule
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, n_prototypes))

    def forward(self, student_output, teacher_output):
        """
        Args:
            student_output: (B, n_prototypes) raw logits from student head
            teacher_output: (B, n_prototypes) raw logits from teacher head
        Returns:
            scalar loss
        """
        student_lsm = F.log_softmax(student_output / self.student_temp, dim=-1)
        teacher_sm = F.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1
        ).detach()

        loss = -torch.sum(teacher_sm * student_lsm, dim=-1).mean()

        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = (self.center * self.center_momentum
                       + batch_center * (1 - self.center_momentum))

    # --- Multi-crop helpers (compute targets once, reuse across views) ---

    def get_teacher_targets(self, teacher_output):
        """Compute centered + sharpened teacher targets. Updates center once."""
        teacher_sm = F.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1
        ).detach()
        self.update_center(teacher_output)
        return teacher_sm

    def loss_from_targets(self, student_output, teacher_targets):
        """Compute DINO loss from pre-computed teacher targets (no center update)."""
        student_lsm = F.log_softmax(student_output / self.student_temp, dim=-1)
        return -torch.sum(teacher_targets * student_lsm, dim=-1).mean()


class iBOTPatchLoss(nn.Module):
    """
    iBOT-style masked-patch distillation loss.

    Computes cross-entropy between student and teacher patch-token outputs,
    restricted to positions that were masked in the student input.
    """

    def __init__(self, n_prototypes, student_temp=0.1,
                 teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, n_prototypes))

    def forward(self, student_patch_output, teacher_patch_output, mask):
        """
        Args:
            student_patch_output: (B, C, N, n_prototypes)
            teacher_patch_output: (B, C, N, n_prototypes)
            mask: (B, C, N) — 1 at masked positions
        Returns:
            scalar loss (only over masked positions)
        """
        B, C, N, D = student_patch_output.shape
        s = student_patch_output.reshape(B * C * N, D)
        t = teacher_patch_output.reshape(B * C * N, D)
        m = mask.reshape(B * C * N).bool()

        if not m.any():
            return torch.tensor(0.0, device=s.device, requires_grad=True)

        s_masked = s[m]
        t_masked = t[m]

        student_lsm = F.log_softmax(s_masked / self.student_temp, dim=-1)
        teacher_sm = F.softmax(
            (t_masked - self.center) / self.teacher_temp, dim=-1
        ).detach()

        loss = -torch.sum(teacher_sm * student_lsm, dim=-1).mean()

        # Update center from ALL teacher patch tokens (not just masked)
        self.update_center(t)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = (self.center * self.center_momentum
                       + batch_center * (1 - self.center_momentum))

    # --- Multi-crop helpers ---

    def get_teacher_targets(self, teacher_patch_output):
        """Compute centered + sharpened targets for all patches. Updates center once."""
        B, C, N, D = teacher_patch_output.shape
        t = teacher_patch_output.reshape(B * C * N, D)
        teacher_sm = F.softmax(
            (t - self.center) / self.teacher_temp, dim=-1
        ).detach()
        self.update_center(t)
        return teacher_sm.reshape(B, C, N, D)

    def loss_from_targets(self, student_patch_output, teacher_targets, mask):
        """Compute iBOT loss at masked positions from pre-computed targets."""
        B, C, N, D = student_patch_output.shape
        s = student_patch_output.reshape(B * C * N, D)
        t = teacher_targets.reshape(B * C * N, D)
        m = mask.reshape(B * C * N).bool()

        if not m.any():
            return torch.tensor(0.0, device=s.device, requires_grad=True)

        s_masked = s[m]
        t_masked = t[m]

        student_lsm = F.log_softmax(s_masked / self.student_temp, dim=-1)
        return -torch.sum(t_masked * student_lsm, dim=-1).mean()


class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko estimator-based diversity loss.

    Encourages uniform spreading of representations by maximising
    the log distance to the nearest neighbour in the batch.
    Applied to L2-normalised CLS embeddings (before the prototype layer).
    """

    def forward(self, student_output):
        """
        Args:
            student_output: (B, D) — L2-normalised embeddings
        Returns:
            scalar loss
        """
        eps = 1e-8
        dists = torch.cdist(student_output, student_output)
        # Mask diagonal without in-place ops (avoids autograd error)
        diag_mask = torch.eye(dists.size(0), device=dists.device,
                              dtype=torch.bool)
        dists = dists.masked_fill(diag_mask, float('inf'))
        nn_dist = dists.min(dim=1).values
        loss = -torch.log(nn_dist + eps).mean()
        return loss
