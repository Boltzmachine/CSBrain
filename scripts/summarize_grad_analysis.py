"""Print a human-readable summary of GradientAnalyzer artifacts.

Usage: python scripts/summarize_grad_analysis.py <grad_analysis_dir>
"""
import csv
import os
import sys


def fmt_pct(x):
    return f"{x * 100:7.3f}%"


def fmt_sci(x):
    return f"{x: .3e}"


def main(d):
    pp = os.path.join(d, 'per_param_stats.csv')
    pc = os.path.join(d, 'per_component_summary.csv')
    if not os.path.exists(pp):
        sys.exit(f"missing {pp}")

    rows = []
    with open(pp) as f:
        for r in csv.DictReader(f):
            for k in ('numel',):
                r[k] = int(r[k])
            for k in ('init_norm', 'rms_grad_norm', 'max_grad_norm',
                      'final_abs_drift', 'final_rel_drift'):
                r[k] = float(r[k])
            rows.append(r)

    print(f"\n# Gradient analysis summary ({d})")
    print(f"  total trainable tensors: {len(rows)}")
    print(f"  total params: {sum(r['numel'] for r in rows):,}")

    by_drift = sorted(rows, key=lambda r: r['final_rel_drift'], reverse=True)
    by_grad = sorted(rows, key=lambda r: r['rms_grad_norm'], reverse=True)

    K = 20
    print(f"\n## Top {K} params by relative drift  (||Δθ|| / ||θ₀||)")
    print(f"{'rank':>4} {'rel_drift':>10} {'rms_grad':>11} {'numel':>10}  name")
    for i, r in enumerate(by_drift[:K], 1):
        print(f"{i:>4} {fmt_pct(r['final_rel_drift'])} {fmt_sci(r['rms_grad_norm']):>11} {r['numel']:>10,}  {r['name']}")

    print(f"\n## Bottom {K} params by relative drift  (least changed)")
    for i, r in enumerate(by_drift[-K:], 1):
        print(f"{i:>4} {fmt_pct(r['final_rel_drift'])} {fmt_sci(r['rms_grad_norm']):>11} {r['numel']:>10,}  {r['name']}")

    print(f"\n## Top {K} params by RMS gradient norm")
    print(f"{'rank':>4} {'rms_grad':>11} {'rel_drift':>10} {'numel':>10}  name")
    for i, r in enumerate(by_grad[:K], 1):
        print(f"{i:>4} {fmt_sci(r['rms_grad_norm']):>11} {fmt_pct(r['final_rel_drift'])} {r['numel']:>10,}  {r['name']}")

    if os.path.exists(pc):
        with open(pc) as f:
            comp_rows = list(csv.DictReader(f))
        for r in comp_rows:
            for k in ('numel', 'n_params'):
                r[k] = int(r[k])
            for k in ('rms_grad_norm', 'max_grad_norm', 'final_rel_drift'):
                r[k] = float(r[k])
        print(f"\n## Per-component summary (sorted by relative drift)")
        print(f"{'rel_drift':>10} {'rms_grad':>11} {'numel':>12} {'n_t':>4}  component")
        for r in sorted(comp_rows, key=lambda x: x['final_rel_drift'], reverse=True):
            print(f"{fmt_pct(r['final_rel_drift'])} {fmt_sci(r['rms_grad_norm']):>11} {r['numel']:>12,} {r['n_params']:>4}  {r['component']}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
