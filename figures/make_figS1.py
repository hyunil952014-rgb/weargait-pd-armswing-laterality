"""Supplementary Figure S1 (R1): same-bout step length (left) and, by absolute zero-threshold log-ratio band,
clinical agreement and within-session direction stability with k/n labels and exact 95% CIs (right).

Stability uses the peer-review re-analysis (first vs last two of the four standard-walking passes, A12).
Output: supplementary/Figure_S1_R1.png (300 dpi)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]  # repository root
WIDE = ROOT / "analysis/05_analysis/v1_arm_swing_metrics_wide.csv"
A712 = json.loads((ROOT / "analysis/revision_r1/output_A7_A12/results_A7_A12.json").read_text())
BANDS = [(0, 0.10, "<0.10"), (0.10, 0.25, "0.10–0.25"), (0.25, 0.50, "0.25–0.50"), (0.50, 99, ">0.50")]


def ci(k, n):
    lo, hi = proportion_confint(k, n, method="beta")
    return lo, hi


def main():
    w = pd.read_csv(WIDE)
    sp = w[w.task.eq("SP") & w.clinical_side.isin(["R", "L"]) & w["Days since Part III Clinical Evaluation"].eq(0)].copy()
    assert len(sp) == 73
    sp["absr"] = sp["sensor_log_R_over_L"].abs()
    agree = []
    for lo, hi, name in BANDS:
        s = sp[(sp.absr >= lo) & (sp.absr < hi)]
        agree.append((int(s.agreement.sum()), len(s)))
    stab_src = A712["A12_split_stability"]["first_vs_second_half_of_passes"]
    stab = [(stab_src[name.replace("–", "-")]["k"], stab_src[name.replace("–", "-")]["n"]) for _, _, name in BANDS]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    # left: same-sample comparison (values as in Supplementary Table S2)
    ax = axes[0]
    vals = [(48, 73), (40, 73)]
    for i, (k, n) in enumerate(vals):
        lo, hi = ci(k, n)
        ax.bar(i, k / n, width=0.6, color=["#1f77b4", "#9ca3af"][i])
        ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], color="black", capsize=6, lw=1.5)
        ax.text(i, hi + 0.03, f"{k}/{n}", ha="center", fontsize=10)
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_xticks([0, 1], ["Arm swing", "Step length"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Agreement with clinical laterality")
    ax.set_title("Same participants and walking bouts")
    # right: by magnitude band
    ax = axes[1]
    x = np.arange(len(BANDS))
    for series, color, marker, label, off in ((agree, "#1f77b4", "o", "Clinical agreement", -0.12),
                                              (stab, "#d55e00", "s", "Within-session stability (re-analysed)", 0.12)):
        p = [k / n for k, n in series]
        lo = [p[i] - ci(*series[i])[0] for i in range(len(series))]
        hi = [ci(*series[i])[1] - p[i] for i in range(len(series))]
        ax.errorbar(x + off, p, yerr=[lo, hi], color=color, marker=marker, lw=2, capsize=4, label=label)
        for i, (k, n) in enumerate(series):
            left = color == "#1f77b4"
            ax.text(x[i] + off + (-0.06 if left else 0.06), p[i], f"{k}/{n}", ha="right" if left else "left",
                    va="center", fontsize=8.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9), zorder=5)
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_xticks(x, [b[2] for b in BANDS]); ax.set_xlim(-0.6, 3.75)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Absolute arm-swing log ratio (full session)")
    ax.set_ylabel("Proportion")
    ax.set_title("By asymmetry magnitude")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    out = ROOT / "figures/Figure_S1.png"
    fig.savefig(out, dpi=300)
    print("agreement", agree, "stability", stab, "->", out)


if __name__ == "__main__":
    main()
