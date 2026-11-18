"""Revision analyses A1-A6 for sensors-4689297 (added at peer review, exploratory).

Reads the locked participant-level table produced by run_locked_analysis.py
(v1_arm_swing_metrics_wide.csv). The locked code and its outputs are not modified.

A1  Exact McNemar test of directional bias (sensor label vs clinical label), all tasks.
A2  Rigidity vs bradykinesia laterality: concordant, opposite, and tied counts.
A3  Agreement and AUC stratified by clinical asymmetry magnitude (1 point vs >=2 points).
A4  Agreement by reported DBS status, including confirmed non-DBS by clinically worse side.
A5  Site-adjusted side-specific difference (Mantel-Haenszel risk difference; logistic model).
A6  Paired task comparisons in common participant subsets (standard vs tandem, standard vs fast).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

LOCKED = Path(__file__).resolve().parents[1] / "05_analysis" / "v1_arm_swing_metrics_wide.csv"
OUT = Path(__file__).resolve().parent / "output_A1_A6"
SEED = 20261117
N_BOOT = 5000
TASKS = ["SP", "HP", "SPmT", "TUG", "Tandem"]
TASK_NAME = {"SP": "Standard", "HP": "Fast", "SPmT": "Standard with turns", "TUG": "TUG", "Tandem": "Tandem"}
SITE_NAME = {"NLS": "JHOC", "WPD": "VA-Seattle"}


def cohort(wide: pd.DataFrame, task: str) -> pd.DataFrame:
    """Same definition as the locked analysis: non-tied laterality, same-day examination."""
    d = wide[
        wide["task"].eq(task)
        & wide["clinical_side"].isin(["R", "L"])
        & wide["Days since Part III Clinical Evaluation"].eq(0)
    ].copy()
    d["sensor_ref"] = np.where(d["sensor_log_R_over_L"] < 0, "R", "L")
    assert (d["sensor_ref"] == d["sensor_lower_side"]).all()
    d["abs_clin"] = (d["upper_R"] - d["upper_L"]).abs()
    d["right_worse"] = d["clinical_side"].eq("R").astype(int)
    d["agree"] = d["agreement"].astype(bool).astype(int)
    return d


def prop(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "pct": None, "ci95": [None, None]}
    lo, hi = proportion_confint(k, n, alpha=0.05, method="beta")
    return {"k": int(k), "n": int(n), "pct": round(100 * k / n, 1), "ci95": [round(100 * lo, 1), round(100 * hi, 1)]}


def side_agreement(d: pd.DataFrame) -> dict:
    out = {"overall": prop(d["agree"].sum(), len(d))}
    for side, label in (("L", "left_worse"), ("R", "right_worse")):
        s = d[d["clinical_side"].eq(side)]
        out[label] = prop(s["agree"].sum(), len(s))
    lw, rw = out["left_worse"], out["right_worse"]
    if lw["n"] and rw["n"]:
        out["balanced_accuracy_pct"] = round(100 * (lw["k"] / lw["n"] + rw["k"] / rw["n"]) / 2, 1)
    return out


def auc(d: pd.DataFrame, rng: np.random.Generator, boot: bool = True) -> dict:
    y = d["right_worse"].to_numpy()
    s = -d["sensor_log_R_over_L"].to_numpy()  # lower right swing -> right side more affected
    if y.min() == y.max():
        return {"auc": None, "ci95": [None, None], "n": int(len(y))}
    est = roc_auc_score(y, s)
    res = {"auc": round(float(est), 3), "n": int(len(y))}
    if boot:
        vals = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                vals.append(roc_auc_score(y[i], s[i]))
        res["ci95"] = [round(float(v), 3) for v in np.quantile(vals, [0.025, 0.975])]
    return res


def mcnemar_exact(b: int, c: int) -> float:
    return float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0


# ---------------------------------------------------------------- A1
def a1_directional_bias(wide: pd.DataFrame) -> dict:
    res = {}
    for task in TASKS:
        d = cohort(wide, task)
        b = int(((d["clinical_side"] == "L") & (d["sensor_ref"] == "R")).sum())  # clinical L, sensor R-lower
        c = int(((d["clinical_side"] == "R") & (d["sensor_ref"] == "L")).sum())  # clinical R, sensor L-lower
        res[task] = {
            "n": int(len(d)),
            "sensor_right_lower": prop(int((d["sensor_ref"] == "R").sum()), len(d)),
            "clinical_right_worse": prop(int((d["clinical_side"] == "R").sum()), len(d)),
            "discordant_clinL_sensorR": b,
            "discordant_clinR_sensorL": c,
            "mcnemar_exact_p": round(mcnemar_exact(b, c), 4),
            "difference_pct_points": round(100 * (b - c) / len(d), 1),
        }
    return res


# ---------------------------------------------------------------- A2
def a2_domains(wide: pd.DataFrame) -> dict:
    d = cohort(wide, "SP")
    tab = pd.crosstab(d["rigidity_side"], d["brady_side"])
    both = d["rigidity_side"].isin(["L", "R"]) & d["brady_side"].isin(["L", "R"])
    opposite = both & (d["rigidity_side"] != d["brady_side"])
    same = both & (d["rigidity_side"] == d["brady_side"])
    rig_tie = d["rigidity_side"].eq("TIE")
    br_tie = d["brady_side"].eq("TIE")
    # Agreement of the sensor in the domain-concordant subgroup
    conc = d[same]
    res = {
        "n": int(len(d)),
        "crosstab_rigidity_rows_bradykinesia_cols": tab.to_dict(),
        "both_domains_lateralized": int(both.sum()),
        "same_side": int(same.sum()),
        "opposite_sides": int(opposite.sum()),
        "rigidity_tied": int(rig_tie.sum()),
        "bradykinesia_tied": int(br_tie.sum()),
        "rigidity_tied_and_brady_lateralized": int((rig_tie & d["brady_side"].isin(["L", "R"])).sum()),
        "brady_tied_and_rigidity_lateralized": int((br_tie & d["rigidity_side"].isin(["L", "R"])).sum()),
        "composite_matches_rigidity_when_lateralized": int((d["rigidity_side"].isin(["L", "R"]) & (d["rigidity_side"] == d["clinical_side"])).sum()),
        "composite_matches_brady_when_lateralized": int((d["brady_side"].isin(["L", "R"]) & (d["brady_side"] == d["clinical_side"])).sum()),
        "sensor_agreement_domain_concordant": side_agreement(conc),
        "opposite_participants": d.loc[opposite, ["participant", "clinical_side", "rigidity_side", "brady_side", "abs_clin", "agree"]].to_dict("records"),
    }
    return res


# ---------------------------------------------------------------- A3
def a3_magnitude(wide: pd.DataFrame, rng: np.random.Generator) -> dict:
    res = {}
    for task in TASKS:
        d = cohort(wide, task)
        one = d[d["abs_clin"].eq(1)]
        two = d[d["abs_clin"].ge(2)]
        row = {
            "abs_asymmetry_median": float(d["abs_clin"].median()),
            "abs_asymmetry_distribution": {int(k): int(v) for k, v in d["abs_clin"].value_counts().sort_index().items()},
            "one_point": {**side_agreement(one), "auc": auc(one, rng)},
            "two_plus": {**side_agreement(two), "auc": auc(two, rng)},
        }
        table = [[int(one["agree"].sum()), int(len(one) - one["agree"].sum())],
                 [int(two["agree"].sum()), int(len(two) - two["agree"].sum())]]
        row["fisher_p_agreement_1_vs_2plus"] = round(float(fisher_exact(table)[1]), 4)
        # side composition by stratum
        row["right_worse_share_one_point"] = prop(int(one["right_worse"].sum()), len(one))
        row["right_worse_share_two_plus"] = prop(int(two["right_worse"].sum()), len(two))
        # side-specific difference restricted to >=2 points
        lw, rw = row["two_plus"]["left_worse"], row["two_plus"]["right_worse"]
        if lw["n"] and rw["n"]:
            row["two_plus_side_difference_pct_points"] = round(rw["pct"] - lw["pct"], 1)
        # bootstrap AUC difference (2+ minus 1), independent groups
        if task == "SP":
            diffs = []
            y1, s1 = one["right_worse"].to_numpy(), -one["sensor_log_R_over_L"].to_numpy()
            y2, s2 = two["right_worse"].to_numpy(), -two["sensor_log_R_over_L"].to_numpy()
            for _ in range(N_BOOT):
                i = rng.integers(0, len(y1), len(y1))
                j = rng.integers(0, len(y2), len(y2))
                if y1[i].min() != y1[i].max() and y2[j].min() != y2[j].max():
                    diffs.append(roc_auc_score(y2[j], s2[j]) - roc_auc_score(y1[i], s1[i]))
            row["auc_difference_2plus_minus_1_ci95"] = [round(float(v), 3) for v in np.quantile(diffs, [0.025, 0.975])]
        res[task] = row
    return res


# ---------------------------------------------------------------- A4
def a4_dbs(wide: pd.DataFrame) -> dict:
    d = cohort(wide, "SP")
    status = d["DBS?"].astype(str).str.strip().str.upper().map({"YES": "DBS", "NO": "No DBS"}).fillna("Unknown")
    d = d.assign(dbs=status)
    res = {"counts": d["dbs"].value_counts().to_dict()}
    for g in ("No DBS", "DBS", "Unknown"):
        s = d[d["dbs"].eq(g)]
        res[g] = {
            **side_agreement(s),
            "right_worse_share": prop(int(s["right_worse"].sum()), len(s)),
            "sensor_right_lower_share": prop(int((s["sensor_ref"] == "R").sum()), len(s)),
            "years_since_dx_median": float(s["Years since PD diagnosis"].median()),
        }
    excl_reported = d[~d["dbs"].eq("DBS")]
    res["excluding_reported_DBS_(keeps_unknown)"] = side_agreement(excl_reported)
    # Side composition: does DBS status relate to which side is worse?
    known = d[d["dbs"].isin(["DBS", "No DBS"])]
    t = pd.crosstab(known["dbs"], known["clinical_side"])
    res["side_by_dbs_crosstab"] = t.to_dict()
    res["side_by_dbs_fisher_p"] = round(float(fisher_exact(t.to_numpy())[1]), 4)
    # Side-specific difference in confirmed non-DBS
    nd = res["No DBS"]
    res["no_dbs_side_difference_pct_points"] = round(nd["right_worse"]["pct"] - nd["left_worse"]["pct"], 1)
    return res


# ---------------------------------------------------------------- A5
def mh_risk_difference(d: pd.DataFrame, strata: str) -> float:
    num = den = 0.0
    for _, s in d.groupby(strata):
        r = s[s["right_worse"].eq(1)]
        l = s[s["right_worse"].eq(0)]
        if len(r) == 0 or len(l) == 0:
            continue
        w = len(r) * len(l) / len(s)
        num += w * (r["agree"].mean() - l["agree"].mean())
        den += w
    return num / den


def a5_site(wide: pd.DataFrame, rng: np.random.Generator) -> dict:
    d = cohort(wide, "SP")
    d["site_name"] = d["site"].map(SITE_NAME)
    res = {"site_labels": SITE_NAME, "by_site": {}}
    for site, s in d.groupby("site_name"):
        res["by_site"][site] = {**side_agreement(s), "right_worse_share": prop(int(s["right_worse"].sum()), len(s))}
    crude = d.loc[d.right_worse.eq(1), "agree"].mean() - d.loc[d.right_worse.eq(0), "agree"].mean()
    mh = mh_risk_difference(d, "site_name")
    boots = []
    groups = {k: g for k, g in d.groupby(["site_name", "right_worse"])}
    for _ in range(N_BOOT):
        bs = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups.values()])
        boots.append(mh_risk_difference(bs, "site_name"))
    res["crude_difference_pct_points"] = round(100 * crude, 1)
    res["mh_site_adjusted_difference_pct_points"] = round(100 * mh, 1)
    res["mh_bootstrap_ci95_stratified"] = [round(100 * float(v), 1) for v in np.quantile(boots, [0.025, 0.975])]
    m1 = smf.logit("agree ~ right_worse + C(site_name)", data=d).fit(disp=0)
    m2 = smf.logit("agree ~ right_worse * C(site_name)", data=d).fit(disp=0)
    ci = m1.conf_int().loc["right_worse"]
    res["logistic_site_adjusted_OR_right_vs_left"] = {
        "or": round(float(np.exp(m1.params["right_worse"])), 2),
        "ci95": [round(float(np.exp(ci[0])), 2), round(float(np.exp(ci[1])), 2)],
        "p": round(float(m1.pvalues["right_worse"]), 4),
    }
    lr = 2 * (m2.llf - m1.llf)
    from scipy.stats import chi2
    res["side_by_site_interaction_LR_p"] = round(float(chi2.sf(lr, 1)), 4)
    m0 = smf.logit("agree ~ right_worse", data=d).fit(disp=0)
    ci0 = m0.conf_int().loc["right_worse"]
    res["logistic_crude_OR_right_vs_left"] = {
        "or": round(float(np.exp(m0.params["right_worse"])), 2),
        "ci95": [round(float(np.exp(ci0[0])), 2), round(float(np.exp(ci0[1])), 2)],
        "p": round(float(m0.pvalues["right_worse"]), 4),
    }
    return res


# ---------------------------------------------------------------- A6
def paired_tasks(wide: pd.DataFrame, t1: str, t2: str, rng: np.random.Generator) -> dict:
    a = cohort(wide, t1).set_index("participant")
    b = cohort(wide, t2).set_index("participant")
    ids = sorted(set(a.index) & set(b.index))
    a, b = a.loc[ids].reset_index(), b.loc[ids].reset_index()
    assert (a["clinical_side"].to_numpy() == b["clinical_side"].to_numpy()).all()
    res = {"n_common": len(ids), t1: {**side_agreement(a), "auc": auc(a, rng)}, t2: {**side_agreement(b), "auc": auc(b, rng)}}
    # McNemar on agreement (overall and by side)
    for label, mask in (("overall", np.ones(len(a), bool)), ("left_worse", a["clinical_side"].eq("L").to_numpy()),
                        ("right_worse", a["clinical_side"].eq("R").to_numpy())):
        x, y = a["agree"].to_numpy()[mask], b["agree"].to_numpy()[mask]
        b01 = int(((x == 1) & (y == 0)).sum())
        b10 = int(((x == 0) & (y == 1)).sum())
        res[f"mcnemar_{label}"] = {f"{t1}_only_agree": b01, f"{t2}_only_agree": b10,
                                   "difference_pct_points": round(100 * (y.mean() - x.mean()), 1),
                                   "exact_p": round(mcnemar_exact(b01, b10), 4)}
    # paired bootstrap AUC difference (t2 - t1)
    y = a["right_worse"].to_numpy()
    s1, s2 = -a["sensor_log_R_over_L"].to_numpy(), -b["sensor_log_R_over_L"].to_numpy()
    est = roc_auc_score(y, s2) - roc_auc_score(y, s1)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            diffs.append(roc_auc_score(y[i], s2[i]) - roc_auc_score(y[i], s1[i]))
    res[f"auc_difference_{t2}_minus_{t1}"] = {"estimate": round(float(est), 3),
                                              "ci95": [round(float(v), 3) for v in np.quantile(diffs, [0.025, 0.975])]}
    return res


def a6_tasks(wide: pd.DataFrame, rng: np.random.Generator) -> dict:
    res = {"SP_vs_Tandem": paired_tasks(wide, "SP", "Tandem", rng),
           "SP_vs_HP": paired_tasks(wide, "SP", "HP", rng)}
    # all-five-task common subset
    sets = [set(cohort(wide, t)["participant"]) for t in TASKS]
    ids = sorted(set.intersection(*sets))
    res["all_five_tasks_common"] = {"n_common": len(ids)}
    for t in TASKS:
        d = cohort(wide, t)
        d = d[d["participant"].isin(ids)]
        res["all_five_tasks_common"][t] = {**side_agreement(d), "auc": auc(d, rng)}
    return res


def clean(obj):
    """Replace NaN with None so the JSON is standard (allow_nan=False)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def validate(wide: pd.DataFrame) -> dict:
    sp = cohort(wide, "SP")
    checks = {
        "SP_n_73": len(sp) == 73,
        "SP_agree_48": int(sp["agree"].sum()) == 48,
        "SP_left_24_of_44": side_agreement(sp)["left_worse"]["k"] == 24 and side_agreement(sp)["left_worse"]["n"] == 44,
        "SP_right_24_of_29": side_agreement(sp)["right_worse"]["k"] == 24 and side_agreement(sp)["right_worse"]["n"] == 29,
        "task_n_76_71_70_63": [len(cohort(wide, t)) for t in ("HP", "SPmT", "TUG", "Tandem")] == [76, 71, 70, 63],
    }
    if not all(checks.values()):
        print("Validation failed:", checks)
        sys.exit(1)
    return checks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wide = pd.read_csv(LOCKED)
    rng = np.random.default_rng(SEED)
    results = {
        "source": str(LOCKED),
        "seed": SEED,
        "n_boot": N_BOOT,
        "validation_against_reported_values": validate(wide),
        "A1_directional_bias_mcnemar": a1_directional_bias(wide),
        "A2_rigidity_vs_bradykinesia": a2_domains(wide),
        "A3_asymmetry_magnitude": a3_magnitude(wide, rng),
        "A4_dbs": a4_dbs(wide),
        "A5_site": a5_site(wide, rng),
        "A6_paired_tasks": a6_tasks(wide, rng),
    }
    (OUT / "results_A1_A6.json").write_text(json.dumps(clean(results), indent=2, ensure_ascii=False, default=str, allow_nan=False))
    print(json.dumps(results, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
