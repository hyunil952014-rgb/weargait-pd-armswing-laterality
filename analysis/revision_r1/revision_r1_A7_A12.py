"""Revision analyses A7-A12 for sensors-4689297 (added at peer review, exploratory).

Uses cache/runs_cache.pkl (built and verified against the locked outputs by build_cache.py).

A7   Resultant angular velocity without PCA projection (patients and controls).
A8   One-at-a-time sensitivity of swing-detection and periodicity thresholds.
A9   Interval rejection, swing detection and exclusions by task and clinically worse side.
A10  Sensitivity of the control-referenced threshold to control age, sex and site composition.
A11  Cross-validated threshold estimated from patient data.
A12  Within-session direction stability by pass-level split (Supplementary Figure S1, right panel).
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import roc_auc_score
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r1_signal as rs  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]  # analysis/
OUT = HERE / "output_A7_A12"
SEED = 20261118
TASKS = ["SP", "HP", "SPmT", "TUG", "Tandem"]
CTRL_SITE = {"HC": "Control-only site", "NLS": "JHOC", "WHC": "VA-Seattle"}


def load_locked_module():
    spec = importlib.util.spec_from_file_location("locked_ro", PROJ / "05_analysis" / "run_locked_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # defines functions only; main() is guarded
    mod.RAW = rs.RAW
    mod.CLINICAL_FILE = rs.RAW / "PD - Demographic+Clinical - datasetV1.csv"
    return mod


def prop(k, n) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "pct": None, "ci95": [None, None]}
    lo, hi = proportion_confint(int(k), int(n), alpha=0.05, method="beta")
    return {"k": int(k), "n": int(n), "pct": round(100 * k / n, 1), "ci95": [round(100 * lo, 1), round(100 * hi, 1)]}


def side_summary(d: pd.DataFrame, ratio: str = "ratio", theta: float | np.ndarray = 0.0) -> dict:
    lab = np.where(d[ratio].to_numpy() - theta < 0, "R", "L")
    agree = lab == d["clinical_side"].to_numpy()
    out = {"overall": prop(agree.sum(), len(d))}
    for side, name in (("L", "left_worse"), ("R", "right_worse")):
        m = d["clinical_side"].to_numpy() == side
        out[name] = prop(agree[m].sum(), m.sum())
    if out["left_worse"]["n"] and out["right_worse"]["n"]:
        out["balanced_pct"] = round(100 * (out["left_worse"]["k"] / out["left_worse"]["n"] + out["right_worse"]["k"] / out["right_worse"]["n"]) / 2, 1)
    out["sensor_right_lower"] = prop((lab == "R").sum(), len(d))
    b = int(((d["clinical_side"] == "L").to_numpy() & (lab == "R")).sum())
    c = int(((d["clinical_side"] == "R").to_numpy() & (lab == "L")).sum())
    out["mcnemar_exact_p"] = round(float(binomtest(min(b, c), b + c).pvalue), 4) if b + c else 1.0
    return out


def auc_point(d: pd.DataFrame, ratio: str = "ratio"):
    y = d["clinical_side"].eq("R").astype(int).to_numpy()
    return round(float(roc_auc_score(y, -d[ratio].to_numpy())), 3) if 0 < y.sum() < len(y) else None


class Data:
    def __init__(self):
        with open(HERE / "cache" / "runs_cache.pkl", "rb") as f:
            self.cache = pickle.load(f)
        self.locked = load_locked_module()
        clin = self.locked.load_clinical()
        self.clin = clin
        self.universe = clin[clin["clinical_side"].isin(["R", "L"]) & clin["Days since Part III Clinical Evaluation"].eq(0)].copy()
        self._memo = {}

    def runs(self, group, pid, task, side):
        c = self.cache.get((group, pid, task))
        if c is None or "error" in c:
            return None
        return c[side]["runs"]

    def pd_metrics(self, task: str, p: rs.Params = rs.DEFAULT) -> pd.DataFrame:
        key = (task, p)
        if key in self._memo:
            return self._memo[key]
        rows = []
        for _, r in self.universe.iterrows():
            rec = {"participant": r["participant"], "clinical_side": r["clinical_side"], "has_file": True}
            for side in ("R", "L"):
                runs = self.runs("PD", r["participant"], task, side)
                if runs is None:
                    rec["has_file"] = False
                    rec[f"elig_{side}"] = False
                    continue
                m = rs.swing_metrics(runs, p)
                rec[f"elig_{side}"] = m["eligible"]
                rec[f"v_{side}"] = m.get("median_peak_velocity_deg_s", np.nan)
            rows.append(rec)
        d = pd.DataFrame(rows)
        d["bilateral"] = d["elig_R"] & d["elig_L"]
        d["ratio"] = np.log(d["v_R"] / d["v_L"])
        self._memo[key] = d
        return d

    def ctrl_metrics(self, task: str, p: rs.Params = rs.DEFAULT) -> pd.DataFrame:
        rows = []
        for (g, pid, t), c in self.cache.items():
            if g != "Control" or t != task or "error" in c:
                continue
            mr, ml = rs.swing_metrics(c["R"]["runs"], p), rs.swing_metrics(c["L"]["runs"], p)
            if mr["eligible"] and ml["eligible"]:
                rows.append({"participant": pid, "ratio": float(np.log(mr["median_peak_velocity_deg_s"] / ml["median_peak_velocity_deg_s"]))})
        return pd.DataFrame(rows)


# ------------------------------------------------------------------ A7
def a7_resultant(D: Data) -> dict:
    res = {"definition": ("median of local maxima of the resultant angular velocity ||omega|| (3 Hz low-pass, no projection); "
                          "find_peaks height>=10 deg/s, prominence>=5 deg/s, spacing>=0.25 s, 0.51 s edges trimmed; >=4 peaks per wrist. "
                          "Local maxima approximate one peak per forward or backward swing but are not segmented by half cycle."),
           "variants": {"same_intervals": "intervals restricted to those in which the primary PCA pipeline accepted swings (interval selection depends on PCA)",
                        "all_valid_intervals": "all intervals >=2 s with >=95% valid samples; no PCA or periodicity step (projection-free)"},
           "patients": {}, "controls": {}, "common_sample": {}}

    def resultant_ratio(group, pid, task, variant):
        vals = {}
        for side in ("R", "L"):
            runs = D.runs(group, pid, task, side)
            if not runs:
                return np.nan
            mask = None
            if variant == "same_intervals":
                m = rs.swing_metrics(runs, return_runs=True)
                mask = [len(v) > 0 for v in m["per_run_velocity"]]
            r = rs.resultant_metrics(runs, mask)
            if not r["eligible"]:
                return np.nan
            vals[side] = r["median_peak_resultant_deg_s"]
        return float(np.log(vals["R"] / vals["L"]))

    for task in TASKS:
        base = D.pd_metrics(task)
        base = base[base["bilateral"]]
        row = {"primary_pipeline": {**side_summary(base), "auc": auc_point(base), "n": len(base)}}
        ratios = {"primary_pipeline": base.set_index("participant")["ratio"]}
        if task == "Tandem":
            amp3 = D.pd_metrics(task, rs.Params(amplitude_deg=3.0))
            ratios["primary_amplitude_3deg"] = amp3[amp3["bilateral"]].set_index("participant")["ratio"]
        for variant in ("same_intervals", "all_valid_intervals"):
            d = D.universe[["participant", "clinical_side"]].copy()
            d["ratio"] = [resultant_ratio("PD", pid, task, variant) for pid in d["participant"]]
            d = d.dropna(subset=["ratio"])
            ratios[variant] = d.set_index("participant")["ratio"]
            merged = d.merge(base[["participant", "ratio"]].rename(columns={"ratio": "ratio_primary"}), on="participant")
            row[variant] = {**side_summary(d), "auc": auc_point(d), "n": len(d),
                            "direction_agreement_with_primary": prop((np.sign(merged["ratio"]) == np.sign(merged["ratio_primary"])).sum(), len(merged)),
                            "spearman_with_primary": round(float(spearmanr(merged["ratio"], merged["ratio_primary"])[0]), 3)}
        res["patients"][task] = row
        # same participants for every method, so that differences are not due to eligibility
        ids = sorted(set.intersection(*[set(s.index) for s in ratios.values()]))
        clin = D.universe.set_index("participant").loc[ids, "clinical_side"]
        common = {"n": len(ids)}
        for name, s in ratios.items():
            d = pd.DataFrame({"participant": ids, "clinical_side": clin.to_numpy(), "ratio": s.loc[ids].to_numpy()})
            common[name] = {**side_summary(d), "auc": auc_point(d)}
        res["common_sample"][task] = common
    for task in ("SPmT", "SP", "TUG", "HP"):
        pids = sorted({pid for (g, pid, t) in D.cache if g == "Control" and t == task})
        row = {}
        for variant in ("same_intervals", "all_valid_intervals"):
            r = np.array([resultant_ratio("Control", pid, task, variant) for pid in pids])
            r = r[np.isfinite(r)]
            k = int((r < 0).sum())
            row[variant] = {"right_lower": prop(k, len(r)), "binomial_p": round(float(binomtest(k, len(r)).pvalue), 4),
                            "median_log_ratio": round(float(np.median(r)), 3)}
        c = D.ctrl_metrics(task)
        k = int((c["ratio"] < 0).sum())
        row["primary_pipeline"] = {"right_lower": prop(k, len(c)), "median_log_ratio": round(float(c["ratio"].median()), 3)}
        res["controls"][task] = row
    return res


# ------------------------------------------------------------------ A8
def a8_grid(D: Data) -> list[dict]:
    grid = [("default", rs.DEFAULT)]
    for a in (3.0, 7.0):
        grid.append((f"amplitude {a:g} deg", rs.Params(amplitude_deg=a)))
    for v in (5.0, 15.0):
        grid.append((f"peak velocity {v:g} deg/s", rs.Params(velocity_deg_s=v)))
    for pr in (1.0, 3.0):
        grid.append((f"prominence {pr:g} deg", rs.Params(prominence_deg=pr)))
    for q in (0.65, 0.85):
        grid.append((f"periodic power {q:g}", rs.Params(periodic_min=q)))
    grid.append(("all relaxed (3 deg, 5 deg/s, 1 deg, 0.65)", rs.Params(amplitude_deg=3, velocity_deg_s=5, prominence_deg=1, periodic_min=0.65)))
    grid.append(("all stricter (7 deg, 15 deg/s, 3 deg, 0.85)", rs.Params(amplitude_deg=7, velocity_deg_s=15, prominence_deg=3, periodic_min=0.85)))
    rows = []
    for name, p in grid:
        for task in TASKS:
            d = D.pd_metrics(task, p)
            d = d[d["bilateral"]]
            s = side_summary(d)
            rows.append({"setting": name, "task": task, "n": len(d), "overall": s["overall"]["pct"],
                         "left_worse": f'{s["left_worse"]["k"]}/{s["left_worse"]["n"]} ({s["left_worse"]["pct"]})',
                         "right_worse": f'{s["right_worse"]["k"]}/{s["right_worse"]["n"]} ({s["right_worse"]["pct"]})',
                         "left_pct": s["left_worse"]["pct"], "right_pct": s["right_worse"]["pct"],
                         "auc": auc_point(d)})
    return rows


# ------------------------------------------------------------------ A9
def a9_counts(D: Data) -> tuple[list[dict], dict]:
    rows = []
    for task in TASKS:
        for side_worse in ("L", "R"):
            sub = D.universe[D.universe["clinical_side"].eq(side_worse)]
            agg = {"task": task, "clinically_worse": side_worse, "participants": len(sub), "with_file": 0,
                   "wrist_recordings": 0, "label_intervals": 0, "intervals_lt_2s": 0, "intervals_missing_data": 0,
                   "intervals_analysed": 0, "rejected_periodicity": 0, "rejected_short_after_edges": 0,
                   "no_accepted_swing": 0, "intervals_with_swings": 0, "candidate_half_cycles": 0,
                   "accepted_swings": 0, "wrists_lt4_swings": 0, "participants_not_bilateral": 0}
            swings_per_wrist = []
            for pid in sub["participant"]:
                c = D.cache.get(("PD", pid, task))
                if c is None or "error" in c:
                    continue
                agg["with_file"] += 1
                agg["label_intervals"] += c["n_label_runs"] * 2
                agg["intervals_lt_2s"] += c["n_runs_lt2s"] * 2
                elig = []
                for s in ("R", "L"):
                    agg["wrist_recordings"] += 1
                    agg["intervals_missing_data"] += c[s]["rejected_missing"]
                    m = rs.swing_metrics(c[s]["runs"])
                    agg["intervals_analysed"] += m["n_runs"]
                    if m["n_runs"]:
                        st = m["run_status"]
                        agg["rejected_periodicity"] += st["periodicity"]
                        agg["rejected_short_after_edges"] += st["too_short_after_edges"]
                        agg["no_accepted_swing"] += st["no_accepted_swing"]
                        agg["intervals_with_swings"] += st["accepted"]
                        agg["candidate_half_cycles"] += m["n_candidates"]
                        agg["accepted_swings"] += m["n_swings"]
                        swings_per_wrist.append(m["n_swings"])
                    else:
                        swings_per_wrist.append(0)
                    if not m["eligible"]:
                        agg["wrists_lt4_swings"] += 1
                    elig.append(m["eligible"])
                if not all(elig):
                    agg["participants_not_bilateral"] += 1
            agg["median_swings_per_wrist"] = float(np.median(swings_per_wrist)) if swings_per_wrist else None
            agg["candidate_acceptance_pct"] = round(100 * agg["accepted_swings"] / agg["candidate_half_cycles"], 1) if agg["candidate_half_cycles"] else None
            rows.append(agg)
    # exclusion by side: Fisher test of non-bilateral by clinically worse side, per task
    from scipy.stats import fisher_exact
    tests = {}
    for task in TASKS:
        a = next(r for r in rows if r["task"] == task and r["clinically_worse"] == "L")
        b = next(r for r in rows if r["task"] == task and r["clinically_worse"] == "R")
        t = [[a["participants_not_bilateral"], a["with_file"] - a["participants_not_bilateral"]],
             [b["participants_not_bilateral"], b["with_file"] - b["participants_not_bilateral"]]]
        tests[task] = {"excluded_left_worse": f'{t[0][0]}/{a["with_file"]}', "excluded_right_worse": f'{t[1][0]}/{b["with_file"]}',
                       "fisher_p": round(float(fisher_exact(t)[1]), 4)}
    return rows, tests


# ------------------------------------------------------------------ A10
def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(cw, 0.5)])


def a10_control_reference(D: Data) -> dict:
    demo = pd.read_csv(rs.RAW / "CONTROLS - Demographic+Clinical - datasetV1.csv", header=1)
    demo = demo.rename(columns={"Subject ID": "participant"})[["participant", "Age", "Sex"]]
    demo["participant"] = demo["participant"].astype(str).str.strip()
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")
    demo["male"] = demo["Sex"].astype(str).str.lower().eq("male").astype(int)
    demo["site"] = demo["participant"].str.extract(r"^([A-Za-z]+)")[0].map(CTRL_SITE)

    pts = D.pd_metrics("SP")
    pts = pts[pts["bilateral"]].merge(D.clin[["participant", "Age (years)", "Sex"]], on="participant")
    pts["age"] = pd.to_numeric(pts["Age (years)"], errors="coerce")
    pts["male"] = pts["Sex"].astype(str).str.lower().eq("male").astype(int)
    assert len(pts) == 73

    def band(a):
        return pd.cut(a, [0, 59.999, 69.999, 200], labels=["<60", "60-69", ">=70"])

    out = {"patients": {"n": len(pts), "age_mean": round(pts["age"].mean(), 1), "male_pct": round(100 * pts["male"].mean(), 1),
                        "age_p10_p90": [float(pts["age"].quantile(0.1)), float(pts["age"].quantile(0.9))]}}
    lo, hi = out["patients"]["age_p10_p90"]
    for ref_task in ("SPmT", "SP"):
        c = D.ctrl_metrics(ref_task).merge(demo, on="participant", how="left")
        assert c["Age"].notna().all(), "control demographics missing"
        block = {"controls": {"n": len(c), "age_mean": round(c["Age"].mean(), 1), "male_pct": round(100 * c["male"].mean(), 1),
                              "by_site_n": c["site"].value_counts().to_dict()}}
        thetas = {"all controls (reported)": float(c["ratio"].median())}
        # post-stratified weights (age band x sex) to patient composition
        c["stratum"] = band(c["Age"]).astype(str) + "_" + c["male"].astype(str)
        pts["stratum"] = band(pts["age"]).astype(str) + "_" + pts["male"].astype(str)
        pshare = pts["stratum"].value_counts(normalize=True)
        cshare = c["stratum"].value_counts(normalize=True)
        w = c["stratum"].map(lambda s: pshare.get(s, 0) / cshare[s]).to_numpy()
        thetas["reweighted to patient age band x sex"] = weighted_median(c["ratio"].to_numpy(), w)
        block["reweighting_effective_n"] = round(float(w.sum() ** 2 / (w ** 2).sum()), 1)
        thetas["male controls only"] = float(c.loc[c["male"].eq(1), "ratio"].median())
        thetas[f"controls aged {lo:g}-{hi:g} (patient 10th-90th pct)"] = float(c.loc[c["Age"].between(lo, hi), "ratio"].median())
        thetas["male controls aged within patient 10th-90th pct"] = float(c.loc[c["Age"].between(lo, hi) & c["male"].eq(1), "ratio"].median())
        for site in ("Control-only site", "VA-Seattle"):
            thetas[f"{site} controls only"] = float(c.loc[c["site"].eq(site), "ratio"].median())
        block["n_by_subset"] = {"male": int(c["male"].sum()), "age_range": int(c["Age"].between(lo, hi).sum()),
                                "male_age_range": int((c["Age"].between(lo, hi) & c["male"].eq(1)).sum())}
        # median regression on age and sex, predicted at each patient's covariates
        X = np.column_stack([np.ones(len(c)), c["Age"].to_numpy(), c["male"].to_numpy()])
        qr = QuantReg(c["ratio"].to_numpy(), X).fit(q=0.5)
        block["median_regression"] = {"coef_intercept_age_male": [round(float(v), 4) for v in qr.params],
                                      "p_age_male": [round(float(v), 4) for v in qr.pvalues[1:]]}
        theta_i = qr.params[0] + qr.params[1] * pts["age"].to_numpy() + qr.params[2] * pts["male"].to_numpy()
        block["median_regression"]["patient_threshold_median_range"] = [round(float(np.median(theta_i)), 3),
                                                                       round(float(theta_i.min()), 3), round(float(theta_i.max()), 3)]
        Xs = np.column_stack([X, pd.get_dummies(c["site"], drop_first=True).to_numpy(dtype=float)])
        qrs = QuantReg(c["ratio"].to_numpy(), Xs).fit(q=0.5)
        block["median_regression_with_site_p"] = [round(float(v), 4) for v in qrs.pvalues[1:]]
        results = {}
        for name, th in thetas.items():
            s = side_summary(pts, theta=th)
            results[name] = {"theta": round(th, 3), "overall": s["overall"]["pct"], "left_worse": s["left_worse"],
                             "right_worse": s["right_worse"], "balanced_pct": s["balanced_pct"]}
        s = side_summary(pts, theta=theta_i)
        results["median regression on age + sex (patient-specific)"] = {"theta": "patient-specific", "overall": s["overall"]["pct"],
                                                                         "left_worse": s["left_worse"], "right_worse": s["right_worse"],
                                                                         "balanced_pct": s["balanced_pct"]}
        s0 = side_summary(pts, theta=0.0)
        results["zero threshold (pre-specified)"] = {"theta": 0.0, "overall": s0["overall"]["pct"], "left_worse": s0["left_worse"],
                                                     "right_worse": s0["right_worse"], "balanced_pct": s0["balanced_pct"]}
        block["thresholds_applied_to_standard_walking"] = results
        out[f"reference_{ref_task}"] = block
    return out


# ------------------------------------------------------------------ A11
def best_threshold(x: np.ndarray, y: np.ndarray, criterion: str) -> float:
    xs = np.sort(np.unique(x))
    cands = np.r_[xs[0] - 1e-6, (xs[:-1] + xs[1:]) / 2, xs[-1] + 1e-6]
    best, val = [], -1
    for t in cands:
        pred_r = x < t
        if criterion == "balanced":
            sc = 0.5 * (pred_r[y == 1].mean() + (~pred_r)[y == 0].mean())
        else:
            sc = (pred_r == (y == 1)).mean()
        if sc > val + 1e-12:
            best, val = [t], sc
        elif abs(sc - val) <= 1e-12:
            best.append(t)
    best = np.array(best)
    return float(best[np.argmin(np.abs(best))])  # among ties, the threshold closest to zero


def a11_cv_threshold(D: Data) -> dict:
    rng = np.random.default_rng(SEED)
    out = {"rule": "label right-worse if log(R/L) < theta; theta chosen in training folds; ties resolved toward zero",
           "repeats": 200, "folds": 10}
    for task in ("SP", "HP", "SPmT", "TUG"):
        d = D.pd_metrics(task)
        d = d[d["bilateral"]].reset_index(drop=True)
        x, y = d["ratio"].to_numpy(), d["clinical_side"].eq("R").to_numpy().astype(int)
        row = {"n": len(d)}
        for crit in ("balanced", "accuracy"):
            apparent = best_threshold(x, y, crit)
            s_app = side_summary(d, theta=apparent)
            # LOO
            pred = np.empty(len(x), bool)
            th_loo = []
            for i in range(len(x)):
                m = np.ones(len(x), bool)
                m[i] = False
                t = best_threshold(x[m], y[m], crit)
                th_loo.append(t)
                pred[i] = x[i] < t
            agree = pred == (y == 1)
            loo = {"overall": prop(agree.sum(), len(x)), "left_worse": prop(agree[y == 0].sum(), (y == 0).sum()),
                   "right_worse": prop(agree[y == 1].sum(), (y == 1).sum()),
                   "theta_median_range": [round(float(np.median(th_loo)), 3), round(float(np.min(th_loo)), 3), round(float(np.max(th_loo)), 3)]}
            loo["balanced_pct"] = round(100 * (loo["left_worse"]["k"] / loo["left_worse"]["n"] + loo["right_worse"]["k"] / loo["right_worse"]["n"]) / 2, 1)
            # repeated stratified 10-fold
            stats = {"overall": [], "left": [], "right": [], "theta": []}
            idx_r, idx_l = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            for _ in range(out["repeats"]):
                fold = np.empty(len(x), int)
                fold[rng.permutation(idx_r)] = np.arange(len(idx_r)) % 10
                fold[rng.permutation(idx_l)] = np.arange(len(idx_l)) % 10
                p = np.empty(len(x), bool)
                for f in range(10):
                    te = fold == f
                    t = best_threshold(x[~te], y[~te], crit)
                    stats["theta"].append(t)
                    p[te] = x[te] < t
                a = p == (y == 1)
                stats["overall"].append(a.mean())
                stats["left"].append(a[y == 0].mean())
                stats["right"].append(a[y == 1].mean())
            cv = {k: [round(100 * float(np.mean(v)), 1), [round(100 * float(q), 1) for q in np.quantile(v, [0.025, 0.975])]]
                  for k, v in stats.items() if k != "theta"}
            cv["theta_median_iqr"] = [round(float(q), 3) for q in np.quantile(stats["theta"], [0.5, 0.25, 0.75])]
            row[crit] = {"apparent_theta": round(apparent, 3),
                         "apparent": {"overall": s_app["overall"]["pct"], "left": s_app["left_worse"]["pct"], "right": s_app["right_worse"]["pct"]},
                         "loo": loo, "repeated_10fold_mean_and_2.5_97.5": cv}
        s0 = side_summary(d)
        row["zero_threshold"] = {"overall": s0["overall"]["pct"], "left": s0["left_worse"]["pct"], "right": s0["right_worse"]["pct"]}
        out[task] = row
    return out


# ------------------------------------------------------------------ A12
def a12_split_stability(D: Data) -> dict:
    d = D.pd_metrics("SP")
    d = d[d["bilateral"]].copy()
    bands = [(0, 0.10, "<0.10"), (0.10, 0.25, "0.10-0.25"), (0.25, 0.50, "0.25-0.50"), (0.50, 99, ">0.50")]

    def half_label(pid, which, scheme):
        vals = {}
        for side in ("R", "L"):
            c = D.cache[("PD", pid, "SP")][side]
            runs, starts = c["runs"], c["starts"]
            order = np.argsort(starts)
            n = len(order)
            if scheme == "first_vs_second_half_of_passes":
                sel = order[: n // 2] if which == 0 else order[n // 2:]
            else:  # alternating passes
                sel = order[which::2]
            m = rs.swing_metrics([runs[i] for i in sel])
            if not m["eligible"]:
                return None
            vals[side] = m["median_peak_velocity_deg_s"]
        return "R" if vals["R"] < vals["L"] else "L"

    out = {"previous_reported_(sample_midpoint_of_concatenated_walk)": {"<0.10": "7/18", "0.10-0.25": "11/11", "0.25-0.50": "18/18", ">0.50": "25/25"}}
    d["n_passes_R"] = [len(D.cache[("PD", p, "SP")]["R"]["runs"]) for p in d["participant"]]
    out["passes_per_wrist_median_range"] = [float(d["n_passes_R"].median()), int(d["n_passes_R"].min()), int(d["n_passes_R"].max())]
    for scheme in ("first_vs_second_half_of_passes", "alternating_passes"):
        h1 = [half_label(p, 0, scheme) for p in d["participant"]]
        h2 = [half_label(p, 1, scheme) for p in d["participant"]]
        dd = d.assign(h1=h1, h2=h2)
        dd = dd[dd["h1"].notna() & dd["h2"].notna()]
        dd["stable"] = dd["h1"].eq(dd["h2"])
        dd["absr"] = dd["ratio"].abs()
        res = {"n_evaluable": len(dd), "overall": prop(dd["stable"].sum(), len(dd))}
        for lo, hi, name in bands:
            s = dd[(dd["absr"] >= lo) & (dd["absr"] < hi)]
            res[name] = prop(s["stable"].sum(), len(s))
        out[scheme] = res
    return out


def clean(obj):
    """Replace NaN with None so the JSON is standard (allow_nan=False)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and obj != obj:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    OUT.mkdir(exist_ok=True)
    D = Data()
    assert len(D.universe) == 76, len(D.universe)
    sp = D.pd_metrics("SP")
    sp = sp[sp["bilateral"]]
    assert len(sp) == 73 and side_summary(sp)["overall"]["k"] == 48
    results = {"seed": SEED}
    print("A7", flush=True); results["A7_resultant"] = a7_resultant(D)
    print("A8", flush=True); grid = a8_grid(D); pd.DataFrame(grid).to_csv(OUT / "A8_threshold_grid.csv", index=False); results["A8_threshold_grid"] = grid
    print("A9", flush=True); rows, tests = a9_counts(D); pd.DataFrame(rows).to_csv(OUT / "A9_detection_counts.csv", index=False)
    results["A9_detection_counts"] = rows; results["A9_exclusion_by_side"] = tests
    print("A10", flush=True); results["A10_control_reference"] = a10_control_reference(D)
    print("A11", flush=True); results["A11_cv_threshold"] = a11_cv_threshold(D)
    print("A12", flush=True); results["A12_split_stability"] = a12_split_stability(D)
    (OUT / "results_A7_A12.json").write_text(json.dumps(clean(results), indent=2, ensure_ascii=False, default=str, allow_nan=False))
    print("done")


if __name__ == "__main__":
    main()
