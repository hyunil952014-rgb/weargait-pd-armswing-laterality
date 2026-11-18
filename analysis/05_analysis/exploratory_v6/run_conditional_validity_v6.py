#!/usr/bin/env python3
"""Amendment 002: conditional validity of wrist arm-swing laterality (post-result exploratory)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest, fisher_exact, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ANALYSIS = PROJECT / "05_analysis"
RAW = Path(os.environ["WEARGAIT_RAW"])  # local copy of the WearGait-PD data
LOCKED = ANALYSIS / "run_locked_analysis.py"
SEED = 20261007
N_BOOT = 5000
RNG = np.random.default_rng(SEED)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


locked = load_module(LOCKED, "locked_analysis_v6")


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def prop(k: int, n: int) -> dict:
    lo, hi = proportion_confint(k, n, method="beta") if n else (np.nan, np.nan)
    return {"k": int(k), "n": int(n), "p": k / n if n else np.nan, "ci95": [float(lo), float(hi)],
            "binom_p_vs_0.5": float(binomtest(int(k), int(n)).pvalue) if n else np.nan}


# ---------------------------------------------------------------- controls
def control_metrics() -> pd.DataFrame:
    cache = HERE / "control_task_metrics.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for suffix, task in (("_SelfPace_matTURN.csv", "SPmT"), ("_TUG.csv", "TUG")):
        for path in sorted((RAW / "control").glob(f"*{suffix}")):
            pid = path.name.removesuffix(suffix).replace(" (control)", "").strip()
            rows.extend(locked.extract_file(path, pid, task, axes="XYZ"))
    m = pd.DataFrame(rows)
    good = m[m["eligible"].eq(True)]
    w = good.pivot_table(index=["participant", "task"], columns="side",
                         values="median_peak_velocity_deg_s", aggfunc="first").dropna().reset_index()
    w["log_R_over_L"] = np.log(w["R"] / w["L"])
    w.to_csv(cache, index=False)
    return w


def turn_direction(path: Path) -> dict:
    """Count turn directions from integrated lower-back angular velocity on the dominant axis."""
    cols = ["GeneralEvent", "LowerBack_Gyr_X", "LowerBack_Gyr_Y", "LowerBack_Gyr_Z"]
    d = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    if any(c not in d.columns for c in cols):
        return {"n_turn": 0, "net_turn": np.nan}
    g = d["GeneralEvent"].fillna("")
    seg = (g != g.shift()).cumsum()
    sums = [x[cols[1:]].sum() for _, x in d.groupby(seg) if x["GeneralEvent"].iloc[0] == "Turn"]
    if not sums:
        return {"n_turn": 0, "net_turn": np.nan}
    s = pd.DataFrame(sums)
    axis = s.abs().sum().idxmax()
    signs = np.sign(s[axis])
    return {"n_turn": int(len(s)), "net_turn": int(signs.sum())}


# ---------------------------------------------------------------- PD
def pd_task(wide: pd.DataFrame, task: str) -> pd.DataFrame:
    return wide[wide["task"].eq(task) & wide["clinical_side"].isin(["R", "L"])
                & wide["Days since Part III Clinical Evaluation"].eq(0)].copy()


def agreement_with_offset(d: pd.DataFrame, theta: float) -> dict:
    adj = d["sensor_log_R_over_L"].to_numpy() - theta
    sensor = np.where(adj < 0, "R", "L")
    agree = sensor == d["clinical_side"].to_numpy()
    out = {"overall": prop(agree.sum(), len(agree))}
    for side in ("L", "R"):
        mask = d["clinical_side"].to_numpy() == side
        out[f"clinical_{side}"] = prop(agree[mask].sum(), mask.sum())
    out["side_gap_R_minus_L"] = out["clinical_R"]["p"] - out["clinical_L"]["p"]
    return out


def boot_offset_diff(pdd: pd.DataFrame, ctrl: pd.Series) -> dict:
    x = pdd["sensor_log_R_over_L"].to_numpy()
    c = pdd["clinical_side"].to_numpy()
    cv = ctrl.to_numpy()
    diffs, gaps = [], []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(x), len(x))
        j = RNG.integers(0, len(cv), len(cv))
        theta = np.median(cv[j])
        raw = np.mean(np.where(x[i] < 0, "R", "L") == c[i])
        adjs = np.where(x[i] - theta < 0, "R", "L") == c[i]
        diffs.append(np.mean(adjs) - raw)
        ci = c[i]
        if (ci == "L").any() and (ci == "R").any():
            gaps.append(adjs[ci == "R"].mean() - adjs[ci == "L"].mean())
    return {"agreement_change_ci95": np.quantile(diffs, [0.025, 0.975]).tolist(),
            "adjusted_side_gap_ci95": np.quantile(gaps, [0.025, 0.975]).tolist()}


def auc_ci(d: pd.DataFrame) -> dict:
    y = (d["clinical_side"] == "R").astype(int).to_numpy()
    s = -d["sensor_log_R_over_L"].to_numpy()  # lower right swing -> right affected
    auc = roc_auc_score(y, s)
    vals = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return {"auc": float(auc), "ci95": np.quantile(vals, [0.025, 0.975]).tolist(), "n": int(len(y))}


def coverage_curve(d: pd.DataFrame, theta: float) -> pd.DataFrame:
    adj = d["sensor_log_R_over_L"].to_numpy() - theta
    agree = np.where(adj < 0, "R", "L") == d["clinical_side"].to_numpy()
    order = np.argsort(-np.abs(adj))
    rows = []
    n = len(adj)
    for keep in range(n, 4, -1):
        idx = order[:keep]
        rows.append({"coverage": keep / n, "n_classified": keep, "agreement": agree[idx].mean(),
                     "abs_threshold": float(np.abs(adj[order[keep - 1]]))})
    return pd.DataFrame(rows)


def logit(df: pd.DataFrame, y: str, xs: list[str]) -> dict:
    use = df[[y] + xs].dropna()
    X = sm.add_constant(use[xs].astype(float))
    fit = sm.Logit(use[y].astype(float), X).fit(disp=0)
    ci = fit.conf_int()
    return {"n": int(len(use)),
            "terms": {k: {"OR": float(np.exp(fit.params[k])), "ci95": [float(np.exp(ci.loc[k, 0])), float(np.exp(ci.loc[k, 1]))],
                          "p": float(fit.pvalues[k])} for k in xs}}


def main() -> None:
    results: dict = {"seed": SEED, "n_boot": N_BOOT}
    wide = pd.read_csv(ANALYSIS / "v1_arm_swing_metrics_wide.csv")
    ctrl = control_metrics()

    # A1 normative bias
    a1 = {}
    for task in ("SPmT", "TUG"):
        c = ctrl[ctrl["task"].eq(task)]["log_R_over_L"]
        thetas = [np.median(c.to_numpy()[RNG.integers(0, len(c), len(c))]) for _ in range(N_BOOT)]
        a1[task] = {"right_lower": prop(int((c < 0).sum()), len(c)),
                    "theta_median": float(c.median()), "theta_ci95": np.quantile(thetas, [0.025, 0.975]).tolist(),
                    "mean": float(c.mean())}
    results["A1_control_bias"] = a1

    # A5 turn direction (control SPmT)
    turn_rows = []
    for path in sorted((RAW / "control").glob("*_SelfPace_matTURN.csv")):
        pid = path.name.removesuffix("_SelfPace_matTURN.csv").replace(" (control)", "").strip()
        turn_rows.append({"participant": pid, **turn_direction(path)})
    turns = pd.DataFrame(turn_rows).merge(ctrl[ctrl["task"].eq("SPmT")], on="participant")
    turns.to_csv(HERE / "control_spmt_turns.csv", index=False)
    rho, p = spearmanr(turns["net_turn"], turns["log_R_over_L"], nan_policy="omit")
    results["A5_turn_direction"] = {"n": int(turns["net_turn"].notna().sum()),
                                    "net_turn_counts": turns["net_turn"].value_counts().to_dict(),
                                    "spearman_rho": float(rho), "p": float(p)}

    # A2 task-matched, A3 SP with SPmT theta, A4 AUC
    a2, a4 = {}, {}
    for task in ("SPmT", "TUG"):
        d = pd_task(wide, task)
        c = ctrl[ctrl["task"].eq(task)]["log_R_over_L"]
        theta = float(c.median())
        a2[task] = {"theta": theta, "raw": agreement_with_offset(d, 0.0),
                    "adjusted": agreement_with_offset(d, theta), **boot_offset_diff(d, c)}
        a4[task] = auc_ci(d)
    sp = pd_task(wide, "SP")
    c_spmt = ctrl[ctrl["task"].eq("SPmT")]["log_R_over_L"]
    theta_sp = float(c_spmt.median())
    results["A2_task_matched"] = a2
    results["A3_SP_with_SPmT_theta"] = {"theta": theta_sp, "raw": agreement_with_offset(sp, 0.0),
                                        "adjusted": agreement_with_offset(sp, theta_sp), **boot_offset_diff(sp, c_spmt)}
    a4["SP"] = auc_ci(sp)
    results["A4_auc"] = a4

    # B coverage and magnitude logistic (SP primary cohort)
    curve = coverage_curve(sp, theta_sp)
    curve.to_csv(HERE / "coverage_curve_SP.csv", index=False)
    curve_raw = coverage_curve(sp, 0.0)
    curve_raw.to_csv(HERE / "coverage_curve_SP_unadjusted.csv", index=False)
    pick = {}
    for withhold in (0.0, 0.25, 0.5):
        row = curve.iloc[(curve["coverage"] - (1 - withhold)).abs().argmin()]
        pick[f"withhold_{int(withhold*100)}pct"] = row.to_dict()
    sp = sp.assign(
        agree_adj=(np.where(sp["sensor_log_R_over_L"] - theta_sp < 0, "R", "L") == sp["clinical_side"]).astype(int),
        abs_sensor_adj=(sp["sensor_log_R_over_L"] - theta_sp).abs() / 0.10,
        abs_clinical=(sp["upper_R"] - sp["upper_L"]).abs(),
    )
    results["B_coverage"] = pick
    results["B_logit_adjusted"] = logit(sp, "agree_adj", ["abs_sensor_adj", "abs_clinical"])
    sp_raw = sp.assign(agree_raw=sp["agreement"].astype(int), abs_sensor_raw=sp["sensor_log_R_over_L"].abs() / 0.10)
    results["B_logit_unadjusted"] = logit(sp_raw, "agree_raw", ["abs_sensor_raw", "abs_clinical"])
    rho, p = spearmanr(sp["abs_clinical"], sp["agree_adj"])
    results["B_clinical_magnitude_spearman"] = {"rho": float(rho), "p": float(p)}

    # C V2 direction flips vs baseline magnitude
    v2 = pd.read_csv(ANALYSIS / "v2_arm_swing_metrics_wide.csv")
    v2 = v2[v2["task"].eq("SP")].copy()
    v2["lr"] = np.log(v2["peak_R"] / v2["peak_L"])
    pv = v2.pivot(index="participant", columns="session", values="lr").dropna()
    pv.columns = ["s1", "s2"]
    c_out = {}
    for label, th in (("unadjusted", 0.0), ("adjusted", theta_sp)):
        s1, s2 = pv["s1"] - th, pv["s2"] - th
        kept = np.sign(s1) == np.sign(s2)
        small = s1.abs() < 0.10
        tab = [[int((kept & small).sum()), int((~kept & small).sum())],
               [int((kept & ~small).sum()), int((~kept & ~small).sum())]]
        c_out[label] = {"n": int(len(pv)), "kept": int(kept.sum()),
                        "small_baseline": prop(int((kept & small).sum()), int(small.sum())),
                        "large_baseline": prop(int((kept & ~small).sum()), int((~small).sum())),
                        "fisher_p": float(fisher_exact(tab)[1]),
                        "mwu_abs_s1_flip_vs_kept_p": float(mannwhitneyu(s1.abs()[~kept], s1.abs()[kept]).pvalue) if (~kept).any() else np.nan,
                        "median_abs_s1_flip": float(s1.abs()[~kept].median()) if (~kept).any() else np.nan,
                        "median_abs_s1_kept": float(s1.abs()[kept].median())}
    results["C_v2_flips"] = c_out

    # D DBS
    dd = sp[sp["DBS?"].isin(["Yes", "No"])].copy()
    dd["dbs"] = dd["DBS?"].eq("Yes").astype(int)
    dd["years"] = pd.to_numeric(dd["Years since PD diagnosis"], errors="coerce")
    dd["agree_raw"] = dd["agreement"].astype(int)
    d_out = {"n_recorded": int(len(dd)), "n_missing": int(len(sp) - len(dd))}
    for col in ("agree_raw", "agree_adj"):
        k_yes, n_yes = int(dd.loc[dd.dbs.eq(1), col].sum()), int(dd.dbs.eq(1).sum())
        k_no, n_no = int(dd.loc[dd.dbs.eq(0), col].sum()), int(dd.dbs.eq(0).sum())
        d_out[col] = {"dbs_yes": prop(k_yes, n_yes), "dbs_no": prop(k_no, n_no),
                      "fisher_p": float(fisher_exact([[k_no, n_no - k_no], [k_yes, n_yes - k_yes]])[1])}
        try:
            d_out[col]["logit"] = logit(dd, col, ["dbs", "abs_clinical", "years"])
        except Exception as exc:  # separation or singularity
            d_out[col]["logit"] = f"failed: {exc}"
    d_out["dbs_vs_no_characteristics"] = dd.groupby("dbs")[["years", "abs_clinical"]].median().to_dict()
    results["D_dbs"] = d_out

    json_dump(HERE / "conditional_validity_results.json", results)
    print(json.dumps(results, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
