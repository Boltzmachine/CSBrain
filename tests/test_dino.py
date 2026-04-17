"""Smoke test for DINOv2 self-distillation EEG model."""
import os, sys, traceback
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import warnings
warnings.filterwarnings('ignore')

import torch

def test_components():
    from models.dino_loss import DINOHead, DINOLoss
    from models.dino_eeg import FrequencySubBandAugmentation

    head = DINOHead(in_dim=40, out_dim=256, hidden_dim=64, bottleneck_dim=32, nlayers=2)
    x = torch.randn(4, 40)
    assert head(x).shape == (4, 256), "DINOHead shape mismatch"
    print("[PASS] DINOHead")

    loss_fn = DINOLoss(out_dim=256, student_temp=0.1, center_momentum=0.9)
    student = [torch.randn(4, 256)]
    teacher_logits = torch.randn(4, 256)
    teacher_probs = loss_fn.softmax_center_teacher(teacher_logits, 0.07)
    loss = loss_fn(student, [teacher_probs])
    assert loss.dim() == 0, "DINOLoss should return scalar"
    loss_fn.update_center(teacher_logits)
    print("[PASS] DINOLoss")

    aug = FrequencySubBandAugmentation(sample_rate=200, min_bands=2, max_bands=4)
    eeg = torch.randn(4, 16, 20, 40)
    eeg_aug = aug(eeg)
    assert eeg_aug.shape == eeg.shape, "FreqAug shape mismatch"
    print("[PASS] FrequencySubBandAugmentation")


def test_full_model():
    from models.dino_eeg import CSBrainDINO

    B, C, T, D = 4, 16, 20, 40
    batch = {
        'timeseries': torch.randn(B, C, T, D),
        'ch_coords': torch.randn(B, C, 3),
        'valid_channel_mask': torch.ones(B, C, dtype=torch.bool),
        'valid_length_mask': torch.ones(B, T, dtype=torch.bool),
        'ch_names': [['FP1-REF','FP2-REF','F3-REF','F4-REF','C3-REF','C4-REF',
                       'P3-REF','P4-REF','O1-REF','O2-REF','F7-REF','F8-REF',
                       'T3-REF','T4-REF','T5-REF','T6-REF'] for _ in range(B)],
        'source': ['test'] * B,
    }

    print("Building CSBrainDINO (n_layer=2, equivariance off)...")
    model = CSBrainDINO(
        in_dim=D, out_dim=D, d_model=D, dim_feedforward=D*4,
        seq_len=T, n_layer=2, nhead=4,
        dino_out_dim=256, dino_hidden_dim=64, dino_bottleneck_dim=32, dino_nlayers=2,
        equivariance_weight=0.0,
    )
    model.train()
    n_student = sum(p.numel() for p in model.student.parameters())
    n_teacher = sum(p.numel() for p in model.teacher.parameters())
    print(f"  Student params: {n_student:,}")
    print(f"  Teacher params: {n_teacher:,}")

    print("Running training_step...")
    out, info = model.training_step(batch, iteration=0, total_iterations=100, data_length=10)
    print(f"  Output shape: {out.shape}")
    losses = {}
    for k, v in info.items():
        if isinstance(v, tuple) and len(v) == 2:
            coef, lss = v
            losses[k] = coef * lss
            print(f"  {k}: w={coef}, loss={lss.item():.4f}")
    assert 'dino_loss' in losses, "dino_loss missing from output"
    print("[PASS] training_step")

    print("Testing backward...")
    total_loss = sum(losses.values())
    total_loss.backward()
    teacher_grads = sum(1 for p in model.teacher.parameters() if p.grad is not None)
    assert teacher_grads == 0, f"Teacher should have 0 grads, got {teacher_grads}"
    student_grads = sum(1 for p in model.student.parameters() if p.grad is not None and p.requires_grad)
    print(f"  Student params with grad: {student_grads}")
    print(f"  Teacher params with grad: {teacher_grads} (expect 0)")
    print("[PASS] backward")

    print("Testing EMA update...")
    t_name, t_param = next(model.teacher.named_parameters())
    before = t_param.data.clone()
    model.update_teacher(0, 100)
    delta = (t_param.data - before).abs().mean().item()
    print(f"  Teacher '{t_name}' delta: {delta:.6f}")
    assert delta > 0, "EMA should change teacher params"
    print("[PASS] EMA update")


if __name__ == '__main__':
    try:
        test_components()
        test_full_model()
        print("\n=== ALL TESTS PASSED ===")
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
