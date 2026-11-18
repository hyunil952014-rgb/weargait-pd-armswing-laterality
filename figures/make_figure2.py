"""Figure 2 redrawn from participant-level outputs of the locked n=73 analysis (R1: panel B labels moved).

A: control log(R/L) peak velocity in walking with turns and TUG, with medians (reference thresholds).
B: primary-cohort standard-walking log(R/L) by clinically more affected side, with zero and control thresholds.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A = Path(sys.argv[1])          # .../05_analysis
out = Path(sys.argv[2])
rng = np.random.default_rng(20261026)  # jitter only

w = pd.read_csv(A / "v1_arm_swing_metrics_wide.csv")
sp = w[(w.task == "SP") & w.clinical_side.isin(["L", "R"]) & (w["Days since Part III Clinical Evaluation"] == 0)
       & w.sensor_log_R_over_L.notna()]
assert len(sp) == 73 and int(sp.agreement.sum()) == 48
ctl = pd.read_csv(A / "exploratory_v6" / "control_task_metrics.csv")
thr = {t: float(ctl[ctl.task == t].log_R_over_L.median()) for t in ["SPmT", "TUG"]}

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#888888"
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "Arial", "font.size": 8, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.48, 3.0))

for y, (t, lab) in enumerate([("SPmT", "Walking with turns"), ("TUG", "TUG")]):
    x = ctl[ctl.task == t].log_R_over_L.dropna().to_numpy()
    a.scatter(x, y + rng.uniform(-0.18, 0.18, len(x)), s=9, color=GREY, alpha=0.75, lw=0)
    a.plot([thr[t]] * 2, [y - 0.3, y + 0.3], color="black", lw=2)
    a.text(1.45, y - 0.36, f"right-lower {int((x < 0).sum())}/{len(x)} ({(x < 0).mean() * 100:.0f}%)", ha="right", fontsize=6.5)
a.axvline(0, color=GREY, ls="--", lw=0.8)
a.set_yticks([0, 1], [f"Walking with turns\n(n={ctl[ctl.task == 'SPmT'].log_R_over_L.notna().sum()})",
                      f"TUG\n(n={ctl[ctl.task == 'TUG'].log_R_over_L.notna().sum()})"])
a.set_ylim(1.5, -0.55); a.set_xlim(-1.7, 1.5)
a.set_xlabel("Control log(R/L) peak velocity")
a.set_title("A  Controls without Parkinson disease", loc="left", fontsize=9)

c = thr["SPmT"]
b.axvspan(c, 0, color="#DDDDDD", zorder=0)
for y, (side, col, lab) in enumerate([("L", BLUE, "Left more affected"), ("R", ORANGE, "Right more affected")]):
    s = sp[sp.clinical_side == side].sensor_log_R_over_L.to_numpy()
    b.scatter(s, y + rng.uniform(-0.18, 0.18, len(s)), s=11, color=col, alpha=0.85, lw=0, marker="o" if side == "L" else "s")
    # misclassified: left-worse is correct when s >= threshold (left lower); right-worse when s < threshold
    wrong0 = int((s < 0).sum()) if side == "L" else int((s >= 0).sum())
    wrongc = int((s < c).sum()) if side == "L" else int((s >= c).sum())
    # R1: two-line label in the empty right-hand area so it no longer crosses the threshold lines (reviewer 2)
    b.text(1.45, y - 0.30, f"Misclassified\nzero {wrong0}, referenced {wrongc}", ha="right", va="bottom",
           fontsize=6.5, linespacing=1.15)
b.axvline(0, color=GREY, ls="--", lw=0.8)
b.axvline(c, color="black", lw=1.2)
b.set_yticks([0, 1], [f"Left more affected\n(n={(sp.clinical_side == 'L').sum()})",
                      f"Right more affected\n(n={(sp.clinical_side == 'R').sum()})"])
b.set_ylim(1.5, -0.55); b.set_xlim(-1.7, 1.5)
b.set_xlabel("PD log(R/L) peak velocity, standard walking")
b.set_title("B  Parkinson disease, primary cohort", loc="left", fontsize=9)

fig.tight_layout(w_pad=2.0)
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "Figure_2.pdf")
fig.savefig(out / "Figure_2.tif", dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(out / "Figure_2_preview.png", dpi=300)
fig.savefig(out / "Figure_2_600dpi.png", dpi=600)  # image embedded in the manuscript
print("thresholds", {k: round(v, 3) for k, v in thr.items()})
