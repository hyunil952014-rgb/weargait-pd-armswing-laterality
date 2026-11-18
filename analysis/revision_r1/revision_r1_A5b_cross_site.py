"""A5b: side-specific agreement for the cross-site control references (Supplementary Table S7).

Independent recomputation from the locked participant table and the cached control metrics,
compared with the stored amendment-003 output (exploratory_v6/cross_site_results.json).
Site prefixes: NLS = JHOC, WPD/WHC = VA-Seattle, HC = control-only site.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

PROJ = Path(__file__).resolve().parents[1] / "05_analysis"
OUT = Path(__file__).resolve().parent / "output_A5b"
SITE = {"NLS": "JHOC", "WPD": "VA", "WHC": "VA", "HC": "ControlOnly"}
SCHEMES = {"ControlOnly_to_allPD": ("ControlOnly", ["JHOC", "VA"]),
           "VA_to_JHOC": ("VA", ["JHOC"]),
           "ControlOnly+JHOC_to_VA": ("ControlOnly|JHOC", ["VA"])}


def prop(k, n):
    lo, hi = proportion_confint(int(k), int(n), method="beta")
    return {"k": int(k), "n": int(n), "pct": round(100 * k / n, 1), "ci95": [round(100 * lo, 1), round(100 * hi, 1)]}


def main():
    OUT.mkdir(exist_ok=True)
    wide = pd.read_csv(PROJ / "v1_arm_swing_metrics_wide.csv")
    ctrl = pd.read_csv(PROJ / "exploratory_v6" / "control_task_metrics.csv")
    stored = json.loads((PROJ / "exploratory_v6" / "cross_site_results.json").read_text())
    ctrl["site"] = ctrl["participant"].str.extract(r"^([A-Za-z]+)")[0].map(SITE)
    wide["site"] = wide["participant"].str.extract(r"^([A-Za-z]+)")[0].map(SITE)
    res, mismatches = {}, []
    for task, ref_task in (("SP", "SPmT"), ("SPmT", "SPmT"), ("TUG", "TUG")):
        d = wide[wide["task"].eq(task) & wide["clinical_side"].isin(["R", "L"]) & wide["Days since Part III Clinical Evaluation"].eq(0)]
        for name, (ref_sites, pd_sites) in SCHEMES.items():
            c = ctrl[ctrl["task"].eq(ref_task) & ctrl["site"].isin(ref_sites.split("|"))]["log_R_over_L"]
            p = d[d["site"].isin(pd_sites)]
            theta = float(c.median())
            x, side = p["sensor_log_R_over_L"].to_numpy(), p["clinical_side"].to_numpy()
            zero = np.where(x < 0, "R", "L") == side
            ref = np.where(x - theta < 0, "R", "L") == side
            row = {"theta": round(theta, 3), "n_controls": int(len(c)), "n_patients": int(len(p))}
            for lab, m in (("left_worse", side == "L"), ("right_worse", side == "R")):
                row[f"zero_{lab}"] = prop(zero[m].sum(), m.sum())
                row[f"ref_{lab}"] = prop(ref[m].sum(), m.sum())
            row["zero_overall"] = prop(zero.sum(), len(zero))
            row["ref_overall"] = prop(ref.sum(), len(ref))
            key = f"{task}|{name}"
            s = stored["cross_site"][key]
            for mine, theirs in ((row["ref_left_worse"]["k"], s["ref_L"]["k"]), (row["ref_right_worse"]["k"], s["ref_R"]["k"]),
                                 (row["zero_left_worse"]["k"], s["raw_L"]["k"]), (row["ref_overall"]["k"], s["referenced"]["k"])):
                if mine != theirs:
                    mismatches.append(key)
            res[key] = row
    # pooled leave-site-out for standard walking (each patient site referenced to other-site controls)
    d = wide[wide["task"].eq("SP") & wide["clinical_side"].isin(["R", "L"]) & wide["Days since Part III Clinical Evaluation"].eq(0)]
    frames = []
    for s in ("JHOC", "VA"):
        th = ctrl[ctrl["task"].eq("SPmT") & ctrl["site"].ne(s)]["log_R_over_L"].median()
        p = d[d["site"].eq(s)]
        frames.append(pd.DataFrame({"ref": (np.where(p["sensor_log_R_over_L"] - th < 0, "R", "L") == p["clinical_side"]),
                                    "side": p["clinical_side"].to_numpy(), "theta": th}))
    f = pd.concat(frames)
    res["SP|pooled_leave_site_out"] = {
        "thetas": {s: round(float(t), 3) for s, t in zip(("JHOC", "VA"), [fr["theta"].iloc[0] for fr in frames])},
        "ref_overall": prop(f["ref"].sum(), len(f)),
        "ref_left_worse": prop(f.loc[f.side.eq("L"), "ref"].sum(), int(f.side.eq("L").sum())),
        "ref_right_worse": prop(f.loc[f.side.eq("R"), "ref"].sum(), int(f.side.eq("R").sum())),
    }
    sp = stored["SP_leave_site_out_pooled"]
    if (res["SP|pooled_leave_site_out"]["ref_left_worse"]["k"], res["SP|pooled_leave_site_out"]["ref_right_worse"]["k"]) != (sp["ref_L"]["k"], sp["ref_R"]["k"]):
        mismatches.append("SP|pooled_leave_site_out")
    res["check_against_stored_amendment003_output"] = {"mismatches": mismatches}
    (OUT / "results_A5b.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
