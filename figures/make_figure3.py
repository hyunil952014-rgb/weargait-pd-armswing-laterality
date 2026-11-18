"""Figure 3: expected overall agreement by left-right case mix, from the side-specific agreement of the primary cohort.

Illustration only: overall = p_right * agreement_right + (1 - p_right) * agreement_left. Counts are the locked n=73 results
(zero: 24/44 left-worse, 24/29 right-worse; walking-with-turns reference: 33/44, 20/29).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

out = Path(sys.argv[1])
RULES = [("Zero threshold\n(lower-swing side)", (24, 44), (24, 29), "-", "#222222", -1.8),
         ("Control-referenced\nthreshold", (33, 44), (20, 29), "--", "#777777", 1.8)]
OBS = 29 / 73

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "Arial", "font.size": 8, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.7, 2.7))
x = np.linspace(0, 1, 101)
for label, (kl, nl), (kr, nr), ls, col, dx in RULES:
    al, ar = kl / nl, kr / nr
    ax.plot(x * 100, (x * ar + (1 - x) * al) * 100, ls=ls, color=col, lw=1.6)
    for xx, k, n in [(0, kl, nl), (1, kr, nr)]:
        ci = binomtest(k, n).proportion_ci(method="exact")
        ax.errorbar(xx * 100 + dx, k / n * 100, yerr=[[(k / n - ci.low) * 100], [(ci.high - k / n) * 100]],
                    fmt="none", ecolor=col, lw=0.8, capsize=2.5, alpha=0.8)
    yo = (OBS * ar + (1 - OBS) * al) * 100
    ax.plot(OBS * 100, yo, "o", ms=5, mfc="white", mec=col, mew=1.2, zorder=3)
    ax.text(106, ar * 100 + (3 if ls == "-" else -4), label, fontsize=6.5, color=col, va="center")
ax.axvline(OBS * 100, color="#BBBBBB", lw=0.7, ls=":", zorder=0)
ax.text(OBS * 100 + 1.5, 97, "this cohort (39.7% right-worse)", fontsize=6.3, color="#555555", va="top")
ax.hlines(50, 0, 100, color="#BBBBBB", lw=0.7, ls="--", zorder=0)
ax.set_xlim(-4, 140); ax.set_xticks([0, 25, 50, 75, 100])
ax.spines["bottom"].set_bounds(0, 100)
ax.set_ylim(35, 100)
ax.set_xlabel("Right-worse patients in a cohort (%)")
ax.set_ylabel("Expected overall agreement (%)")
fig.tight_layout()
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "Figure_3.pdf")
fig.savefig(out / "Figure_3.tif", dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(Path(__file__).resolve().parent / "Figure_3_preview.png", dpi=300)
print("saved")
