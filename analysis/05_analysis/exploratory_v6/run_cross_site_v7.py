#!/usr/bin/env python3
"""Amendment 003: cross-site (internal-external) validation of the control-referenced threshold.

Sites (WearGait-PD prefixes): NLS = Johns Hopkins (PD), WPD/WHC = VA-Seattle (PD/controls),
HC = control-only site. A threshold estimated from controls at one site is applied to PD
participants at a different site.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
AN = HERE.parent
SEED = 20261008
N_BOOT = 5000
RNG = np.random.default_rng(SEED)
SITE = {"NLS": "JHOC", "WPD": "VA", "WHC": "VA", "HC": "ControlOnly"}


def prop(k, n):
    lo, hi = proportion_confint(k, n, method="beta")
    return {"k": int(k), "n": int(n), "p": k / n, "ci95": [float(lo), float(hi)], "binom_p": float(binomtest(int(k), int(n)).pvalue)}


def evaluate(pdd: pd.DataFrame, ctrl: pd.Series) -> dict:
    theta = float(ctrl.median())
    x = pdd["sensor_log_R_over_L"].to_numpy()
    c = pdd["clinical_side"].to_numpy()
    raw = np.where(x < 0, "R", "L") == c
    ref = np.where(x - theta < 0, "R", "L") == c
    cv = ctrl.to_numpy()
    diffs = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(x), len(x))
        j = RNG.integers(0, len(cv), len(cv))
        t = np.median(cv[j])
        diffs.append(np.mean(np.where(x[i] - t < 0, "R", "L") == c[i]) - np.mean(np.where(x[i] < 0, "R", "L") == c[i]))
    out = {"theta": theta, "n_controls": int(len(cv)),
           "raw": prop(raw.sum(), len(raw)), "referenced": prop(ref.sum(), len(ref)),
           "gained": int((ref & ~raw).sum()), "lost": int((~ref & raw).sum()),
           "change_ci95": np.quantile(diffs, [0.025, 0.975]).tolist()}
    for side in ("L", "R"):
        m = c == side
        out[f"raw_{side}"] = prop(raw[m].sum(), m.sum())
        out[f"ref_{side}"] = prop(ref[m].sum(), m.sum())
    return out


def main():
    wide = pd.read_csv(AN / "v1_arm_swing_metrics_wide.csv")
    ctrl = pd.read_csv(HERE / "control_task_metrics.csv")
    ctrl["site"] = ctrl["participant"].str.extract(r"^([A-Za-z]+)")[0].map(SITE)
    wide["site_name"] = wide["participant"].str.extract(r"^([A-Za-z]+)")[0].map(SITE)
    res = {"seed": SEED, "control_bias_by_site": {}}
    for task in ("SPmT", "TUG"):
        for s, g in ctrl[ctrl["task"].eq(task)].groupby("site"):
            v = g["log_R_over_L"]
            res["control_bias_by_site"][f"{task}_{s}"] = {"n": int(len(v)), "median": float(v.median()),
                                                         "right_lower": prop(int((v < 0).sum()), len(v))}
    schemes = {
        # reference site -> patient sites (always disjoint)
        "ControlOnly_to_allPD": ("ControlOnly", ["JHOC", "VA"]),
        "VA_to_JHOC": ("VA", ["JHOC"]),
        "ControlOnly+JHOC_to_VA": ("ControlOnly|JHOC", ["VA"]),
    }
    res["cross_site"] = {}
    for task, ref_task in (("SP", "SPmT"), ("SPmT", "SPmT"), ("TUG", "TUG")):
        d = wide[wide["task"].eq(task) & wide["clinical_side"].isin(["R", "L"])
                 & wide["Days since Part III Clinical Evaluation"].eq(0)]
        for name, (ref_sites, pd_sites) in schemes.items():
            c = ctrl[ctrl["task"].eq(ref_task) & ctrl["site"].isin(ref_sites.split("|"))]["log_R_over_L"]
            p = d[d["site_name"].isin(pd_sites)]
            res["cross_site"][f"{task}|{name}"] = evaluate(p, c)
    # pooled cross-site estimate for SP: each PD site referenced only to controls from other sites
    d = wide[wide["task"].eq("SP") & wide["clinical_side"].isin(["R", "L"]) & wide["Days since Part III Clinical Evaluation"].eq(0)]
    flags = []
    for s in ("JHOC", "VA"):
        c = ctrl[ctrl["task"].eq("SPmT") & ctrl["site"].ne(s)]["log_R_over_L"]
        p = d[d["site_name"].eq(s)]
        th = c.median()
        flags.append(pd.DataFrame({"ref": np.where(p["sensor_log_R_over_L"] - th < 0, "R", "L") == p["clinical_side"],
                                   "raw": np.where(p["sensor_log_R_over_L"] < 0, "R", "L") == p["clinical_side"],
                                   "side": p["clinical_side"]}))
    f = pd.concat(flags)
    res["SP_leave_site_out_pooled"] = {"raw": prop(f.raw.sum(), len(f)), "referenced": prop(f.ref.sum(), len(f)),
                                       "ref_L": prop(f.ref[f.side.eq("L")].sum(), int(f.side.eq("L").sum())),
                                       "ref_R": prop(f.ref[f.side.eq("R")].sum(), int(f.side.eq("R").sum())),
                                       "gained": int((f.ref & ~f.raw).sum()), "lost": int((~f.ref & f.raw).sum())}
    (HERE / "cross_site_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for k, v in res["cross_site"].items():
        print(f"{k:32s} θ={v['theta']:+.3f} nC={v['n_controls']:3d} raw {v['raw']['k']}/{v['raw']['n']} ({v['raw']['p']:.3f}) -> ref {v['referenced']['k']} ({v['referenced']['p']:.3f}) "
              f"L {v['raw_L']['p']:.2f}->{v['ref_L']['p']:.2f} R {v['raw_R']['p']:.2f}->{v['ref_R']['p']:.2f} +{v['gained']}/-{v['lost']} CI {np.round(v['change_ci95'],3)}")
    print(json.dumps(res["SP_leave_site_out_pooled"], default=float))
    print(json.dumps(res["control_bias_by_site"], default=float))


if __name__ == "__main__":
    main()
