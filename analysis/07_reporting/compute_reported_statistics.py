"""Statistics computed at the reporting stage (Table 1 flow and QC, Table 3, Figure 1, paired standard-vs-tandem AUC).

Extracted from the authors' manuscript builder (statistics part only; document generation removed).
The bootstrap seeds and the order of random draws are unchanged, so results match the manuscript.
Output: analysis/07_reporting/reported_statistics.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[0]
AN = PROJECT / "05_analysis"
V6 = AN / "exploratory_v6"
RAW = Path(os.environ["WEARGAIT_RAW"])
SEED = 20261007
RNG = np.random.default_rng(SEED)
N_BOOT = 5000


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact(k: int, n: int) -> tuple[float, float, float]:
    lo, hi = proportion_confint(k, n, method="beta")
    return k / n, lo, hi


def auc_boot(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    auc = roc_auc_score(y, s)
    vals = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(auc), float(lo), float(hi)


def pd_task(wide: pd.DataFrame, task: str) -> pd.DataFrame:
    return wide[wide["task"].eq(task) & wide["clinical_side"].isin(["R", "L"])
                & wide["Days since Part III Clinical Evaluation"].eq(0)].copy()


def classify(d: pd.DataFrame, theta: float) -> np.ndarray:
    return np.where(d["sensor_log_R_over_L"].to_numpy() - theta < 0, "R", "L") == d["clinical_side"].to_numpy()


def compute() -> dict:
    S: dict = {}
    wide = pd.read_csv(AN / "v1_arm_swing_metrics_wide.csv")
    ctrl = pd.read_csv(V6 / "control_task_metrics.csv")
    cv = load_json(V6 / "conditional_validity_results.json")
    theta = {t: cv["A2_task_matched"][t]["theta"] for t in ("SPmT", "TUG")}
    theta["SP"] = cv["A3_SP_with_SPmT_theta"]["theta"]
    S["theta"] = theta

    rows = []
    for task in ("SP", "HP", "SPmT", "TUG", "Tandem"):
        d = pd_task(wide, task)
        y = (d["clinical_side"] == "R").astype(int).to_numpy()
        auc = auc_boot(y, -d["sensor_log_R_over_L"].to_numpy())
        raw = classify(d, 0.0)
        row = {"task": task, "n": len(d), "auc": auc}
        for label, agree in (("raw", raw), ("ref", classify(d, theta[task]) if task in theta else None)):
            if agree is None:
                continue
            row[f"{label}_all"] = (int(agree.sum()), len(agree))
            for side in ("L", "R"):
                m = d["clinical_side"].to_numpy() == side
                row[f"{label}_{side}"] = (int(agree[m].sum()), int(m.sum()))
        rows.append(row)
    S["by_task"] = {r["task"]: r for r in rows}

    # Balanced accuracy, added 2026-10-21 after results were known; change CI resamples patients and controls.
    rng_ba = np.random.default_rng(SEED + 1)

    def bal(agree, side):
        return (agree[side == "L"].mean() + agree[side == "R"].mean()) / 2

    for task, ctask in (("SPmT", "SPmT"), ("TUG", "TUG"), ("SP", "SPmT")):
        d = pd_task(wide, task)
        x, side = d["sensor_log_R_over_L"].to_numpy(), d["clinical_side"].to_numpy()
        cvals = ctrl.loc[ctrl["task"].eq(ctask), "log_R_over_L"].to_numpy()
        lab = lambda v, t: np.where(v - t < 0, "R", "L")
        diffs = []
        for _ in range(N_BOOT):
            i = rng_ba.integers(0, len(x), len(x))
            j = rng_ba.integers(0, len(cvals), len(cvals))
            s = side[i]
            if len(set(s)) < 2:
                continue
            diffs.append(bal(lab(x[i], np.median(cvals[j])) == s, s) - bal(lab(x[i], 0.0) == s, s))
        b = S["by_task"][task]
        b["ba_raw"] = bal(lab(x, 0.0) == side, side)
        b["ba_ref"] = bal(lab(x, theta[task]) == side, side)
        b["ba_change_ci95"] = tuple(np.quantile(diffs, [0.025, 0.975]))

    # Standard vs tandem discrimination in participants with both tasks (post hoc, 2026-10-21).
    rng_pt = np.random.default_rng(SEED + 2)
    sp_t, tn_t = pd_task(wide, "SP").set_index("participant"), pd_task(wide, "Tandem").set_index("participant")
    common = sp_t.index.intersection(tn_t.index)
    yc = (sp_t.loc[common, "clinical_side"] == "R").astype(int).to_numpy()
    s_sp, s_tn = -sp_t.loc[common, "sensor_log_R_over_L"].to_numpy(), -tn_t.loc[common, "sensor_log_R_over_L"].to_numpy()
    diffs = []
    for _ in range(N_BOOT):
        i = rng_pt.integers(0, len(yc), len(yc))
        if yc[i].min() != yc[i].max():
            diffs.append(roc_auc_score(yc[i], s_sp[i]) - roc_auc_score(yc[i], s_tn[i]))
    a_sp, a_tn = roc_auc_score(yc, s_sp), roc_auc_score(yc, s_tn)
    S["paired_tandem"] = {"n": len(common), "sp": a_sp, "tandem": a_tn, "diff": a_sp - a_tn,
                          "ci": tuple(np.quantile(diffs, [0.025, 0.975]))}

    # Participant flow for the primary population
    clin = pd.read_csv(RAW / "PD - Demographic+Clinical - datasetV1.csv", header=1, na_values=["-", "", " "])
    items_R = ["MDSUPDRS_3-3-RUE", "MDSUPDRS_3-4-R", "MDSUPDRS_3-5-R", "MDSUPDRS_3-6-R"]
    items_L = ["MDSUPDRS_3-3-LUE", "MDSUPDRS_3-4-L", "MDSUPDRS_3-5-L", "MDSUPDRS_3-6-L"]
    up = clin[items_R + items_L].apply(pd.to_numeric, errors="coerce")
    complete = up.notna().all(axis=1)
    diff = up[items_R].sum(axis=1) - up[items_L].sum(axis=1)
    nontied = complete & diff.ne(0)
    sameday = nontied & pd.to_numeric(clin["Days since Part III Clinical Evaluation"], errors="coerce").eq(0)
    inv = pd.read_csv(PROJECT / "03_exploration/v1_file_inventory.csv")
    sp_files = set(inv.loc[inv["task"].eq("SP"), "participant"])
    S["flow"] = {"pd": len(clin), "complete": int(complete.sum()), "nontied": int(nontied.sum()),
                 "ties": int((complete & diff.eq(0)).sum()), "sameday": int(sameday.sum()),
                 "sameday_with_sp": int(clin.loc[sameday, "Subject ID"].isin(sp_files).sum())}
    qc = pd.read_csv(AN / "v1_arm_swing_metrics_long.csv")
    qc = qc[qc["axes"].eq("XYZ")] if "axes" in qc.columns else qc
    g = qc.groupby("task")
    S["qc"] = pd.DataFrame({"n": g.size(), "eligible": g["eligible"].mean(), "walk_s": g["walk_seconds"].median(),
                            "swings": g["n_swings"].median(), "pc1": g["pc1_variance_share"].median(),
                            "periodic": g["median_periodic_power"].median(),
                            "peak": g["median_peak_velocity_deg_s"].median()}).to_dict(orient="index")
    sp = pd_task(wide, "SP")
    dbs = sp["DBS?"].dropna()
    S["dbs_share"] = (int(dbs.eq("Yes").sum()), len(dbs))
    return S


if __name__ == "__main__":
    out = HERE / "reported_statistics.json"
    out.write_text(json.dumps(compute(), indent=2, default=float), encoding="utf-8")
    print("written", out)
