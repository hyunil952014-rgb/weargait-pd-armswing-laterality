#!/usr/bin/env python3
"""Post-result exploratory (2026-10-26): same-task control thresholds for standard (SP) and fast (HP) walking.

Control SelfPace/HurriedPace CSVs were missing from the initial local download and were retrieved from
Synapse (syn61370552). The locked signal pipeline is applied unchanged; PD metrics are the locked n=73 outputs.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
CTRL_DIR = Path(os.environ["WEARGAIT_RAW"]) / "control_SP_HP_syn61370552"
SEED = 20261026
N_BOOT = 5000
RNG = np.random.default_rng(SEED)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


locked = load_module(ANALYSIS / "run_locked_analysis.py", "locked_v8")
v6 = load_module(ANALYSIS / "exploratory_v6" / "run_conditional_validity_v6.py", "v6_for_v8")


def prop(k, n):
    lo, hi = proportion_confint(k, n, method="beta")
    return {"k": int(k), "n": int(n), "p": k / n, "ci95": [float(lo), float(hi)],
            "binom_p_vs_0.5": float(binomtest(int(k), int(n)).pvalue)}


def control_metrics() -> pd.DataFrame:
    cache = HERE / "control_SP_HP_metrics.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for suffix, task in (("_SelfPace.csv", "SP"), ("_HurriedPace.csv", "HP")):
        for path in sorted(CTRL_DIR.glob(f"*{suffix}")):
            pid = path.name.removesuffix(suffix).replace(" (control)", "").strip()
            rows.extend(locked.extract_file(path, pid, task, axes="XYZ"))
    m = pd.DataFrame(rows)
    m.to_csv(HERE / "control_SP_HP_extraction_log.csv", index=False)
    good = m[m["eligible"].eq(True)]
    w = good.pivot_table(index=["participant", "task"], columns="side",
                         values="median_peak_velocity_deg_s", aggfunc="first").dropna().reset_index()
    w["log_R_over_L"] = np.log(w["R"] / w["L"])
    w.to_csv(cache, index=False)
    return w


def boot_median(x: np.ndarray) -> list[float]:
    meds = [np.median(x[RNG.integers(0, len(x), len(x))]) for _ in range(N_BOOT)]
    return np.quantile(meds, [0.025, 0.975]).tolist()


def balanced(res: dict) -> float:
    return (res["clinical_L"]["p"] + res["clinical_R"]["p"]) / 2


def boot_balanced_change(d: pd.DataFrame, c: np.ndarray) -> list[float]:
    """Change in balanced accuracy (referenced - zero), resampling patients and controls, as in Table 3."""
    rng = np.random.default_rng(SEED + 1)
    x, side = d["sensor_log_R_over_L"].to_numpy(), d["clinical_side"].to_numpy()
    bal = lambda a, s: (a[s == "L"].mean() + a[s == "R"].mean()) / 2
    lab = lambda v, t: np.where(v - t < 0, "R", "L")
    diffs = []
    for _ in range(N_BOOT):
        i, j = rng.integers(0, len(x), len(x)), rng.integers(0, len(c), len(c))
        s = side[i]
        if len(set(s)) < 2:
            continue
        diffs.append(bal(lab(x[i], np.median(c[j])) == s, s) - bal(lab(x[i], 0.0) == s, s))
    return np.quantile(diffs, [0.025, 0.975]).tolist()


def main() -> None:
    ctrl = control_metrics()
    pd_wide = pd.read_csv(ANALYSIS / "v1_arm_swing_metrics_wide.csv")
    turn = pd.read_csv(ANALYSIS / "exploratory_v6" / "control_task_metrics.csv")
    theta_turn = float(turn.loc[turn.task.eq("SPmT"), "log_R_over_L"].median())
    out = {"control_files_dir": str(CTRL_DIR), "seed": SEED, "n_boot": N_BOOT,
           "control_files": {t: len(list(CTRL_DIR.glob(f"*_{s}.csv"))) for t, s in (("SP", "SelfPace"), ("HP", "HurriedPace"))},
           "turning_threshold_SPmT": theta_turn, "tasks": {}}
    for task in ("SP", "HP"):
        c = ctrl.loc[ctrl.task.eq(task), "log_R_over_L"].dropna().to_numpy()
        theta = float(np.median(c))
        d = v6.pd_task(pd_wide, task)
        zero = v6.agreement_with_offset(d, 0.0)
        same = v6.agreement_with_offset(d, theta)
        boot = v6.boot_offset_diff(d, pd.Series(c))
        task_out = {
            "control_n": int(len(c)),
            "control_right_lower": prop(int((c < 0).sum()), len(c)),
            "control_median": theta, "control_median_ci95": boot_median(c),
            "pd_n": int(len(d)),
            "zero": zero, "same_task_reference": same,
            "change_points": (same["overall"]["p"] - zero["overall"]["p"]) * 100,
            "change_ci95_points": [v * 100 for v in boot["agreement_change_ci95"]],
            "balanced_zero": balanced(zero), "balanced_same_task": balanced(same),
            "balanced_change_ci95_points": [v * 100 for v in boot_balanced_change(d, c)],
        }
        if task == "SP":
            turn_ref = v6.agreement_with_offset(d, theta_turn)
            task_out["turning_reference_previous"] = turn_ref
            task_out["balanced_turning"] = balanced(turn_ref)
            task_out["label_changes_vs_turning"] = int(((d.sensor_log_R_over_L - theta) < 0).ne((d.sensor_log_R_over_L - theta_turn) < 0).sum())
        out["tasks"][task] = task_out
    (HERE / "same_task_control_v8_results.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
