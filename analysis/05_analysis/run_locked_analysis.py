#!/usr/bin/env python3
"""Locked SAP analysis for the WearGait-PD arm-swing laterality study."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from scipy.stats import binomtest, shapiro, skew, ttest_rel, wilcoxon
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


PROJECT = Path(__file__).resolve().parents[1]
EXPLORE = PROJECT / "03_exploration"
OUT = PROJECT / "05_analysis"
RAW = Path("/Users/hyeonilkim/Desktop/투고 논문들/교수님과제/02_데이터인벤토리/raw")
CLINICAL_FILE = RAW / "PD - Demographic+Clinical - datasetV1.csv"
SAP_FILE = PROJECT / "04_sap/sap_v1.md"
FS = 100.0
RNG = np.random.default_rng(20260907)
TASK_ORDER = ["SP", "HP", "SPmT", "TUG", "Tandem"]
GYR_ALL = [f"{side}_Wrist_Gyr_{axis}" for side in ("R", "L") for axis in "XYZ"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_clinical() -> pd.DataFrame:
    d = pd.read_csv(CLINICAL_FILE, header=1, na_values=["-", "", " "])
    d = d.rename(columns={"Subject ID": "participant"})
    numeric = [
        "MDSUPDRS_3-3-RUE", "MDSUPDRS_3-3-LUE",
        "MDSUPDRS_3-4-R", "MDSUPDRS_3-4-L",
        "MDSUPDRS_3-5-R", "MDSUPDRS_3-5-L",
        "MDSUPDRS_3-6-R", "MDSUPDRS_3-6-L",
        "Days since Part III Clinical Evaluation", "Age (years)",
        "Years since PD diagnosis", "Modified Hoehn & Yahr Score",
    ]
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["rigidity_R"] = d["MDSUPDRS_3-3-RUE"]
    d["rigidity_L"] = d["MDSUPDRS_3-3-LUE"]
    d["brady_R"] = d[["MDSUPDRS_3-4-R", "MDSUPDRS_3-5-R", "MDSUPDRS_3-6-R"]].sum(axis=1, min_count=3)
    d["brady_L"] = d[["MDSUPDRS_3-4-L", "MDSUPDRS_3-5-L", "MDSUPDRS_3-6-L"]].sum(axis=1, min_count=3)
    d["upper_R"] = d["rigidity_R"] + d["brady_R"]
    d["upper_L"] = d["rigidity_L"] + d["brady_L"]
    d["upper_total"] = d["upper_R"] + d["upper_L"]

    def side(r: pd.Series, l: pd.Series) -> pd.Series:
        return pd.Series(np.select([r > l, r < l], ["R", "L"], default="TIE"), index=d.index)

    d["clinical_side"] = side(d["upper_R"], d["upper_L"])
    d["rigidity_side"] = side(d["rigidity_R"], d["rigidity_L"])
    d["brady_side"] = side(d["brady_R"], d["brady_L"])
    d["site"] = d["participant"].str.extract(r"^([A-Za-z]+)")[0]
    keep = [
        "participant", "site", "Age (years)", "Sex", "Years since PD diagnosis",
        "Modified Hoehn & Yahr Score", "Days since Part III Clinical Evaluation", "DBS?",
        "upper_R", "upper_L", "upper_total", "clinical_side",
        "rigidity_R", "rigidity_L", "rigidity_side", "brady_R", "brady_L", "brady_side",
    ]
    return d[keep].copy()


def contiguous_runs(mask: np.ndarray, minimum_samples: int = 200) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2) if b - a >= minimum_samples]


def lowpass(x: np.ndarray) -> np.ndarray:
    sos = butter(2, 3.0, btype="lowpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def moving_average(x: np.ndarray, width: int = 101) -> np.ndarray:
    return np.convolve(x, np.ones(width) / width, mode="same")


def run_swings(omega_rad_s: np.ndarray) -> dict:
    omega_deg_s = np.rad2deg(omega_rad_s)
    freq, power = welch(omega_deg_s, fs=FS, nperseg=min(512, len(omega_deg_s)))
    arm_band = (freq >= 0.3) & (freq <= 3.0)
    ref_band = (freq >= 0.05) & (freq <= 5.0)
    peak_frequency = float(freq[arm_band][np.argmax(power[arm_band])])
    periodic_power = float(power[arm_band].sum() / power[ref_band].sum())
    if periodic_power < 0.75:
        return {"periodic_power": periodic_power, "peak_frequency": peak_frequency, "amplitude": [], "velocity": []}

    angle = np.rad2deg(cumulative_trapezoid(omega_rad_s, dx=1 / FS, initial=0))
    angle = angle - moving_average(angle)
    edge = 51
    usable = angle[edge:-edge] if len(angle) > 2 * edge else np.array([])
    if len(usable) < 80:
        return {"periodic_power": periodic_power, "peak_frequency": peak_frequency, "amplitude": [], "velocity": []}

    min_distance = max(20, int(0.30 * FS / max(peak_frequency, 0.3)))
    maxima, _ = find_peaks(usable, prominence=2.0, distance=min_distance)
    minima, _ = find_peaks(-usable, prominence=2.0, distance=min_distance)
    extrema = sorted([(int(i), "max") for i in maxima] + [(int(i), "min") for i in minima])
    alternating: list[tuple[int, str]] = []
    for item in extrema:
        if not alternating or alternating[-1][1] != item[1]:
            alternating.append(item)
        else:
            old_i, kind = alternating[-1]
            if (kind == "max" and usable[item[0]] > usable[old_i]) or (
                kind == "min" and usable[item[0]] < usable[old_i]
            ):
                alternating[-1] = item

    amps: list[float] = []
    vels: list[float] = []
    for (i, _), (j, _) in zip(alternating[:-1], alternating[1:]):
        amp = abs(float(usable[j] - usable[i]))
        vel = float(np.max(np.abs(omega_deg_s[edge + i : edge + j + 1])))
        if amp >= 5.0 and vel >= 10.0:
            amps.append(amp)
            vels.append(vel)
    return {"periodic_power": periodic_power, "peak_frequency": peak_frequency, "amplitude": amps, "velocity": vels}


def extract_side(df: pd.DataFrame, label: str, side: str, axes: str = "XYZ") -> dict:
    runs = contiguous_runs(df["GeneralEvent"].eq(label).to_numpy())
    columns = [f"{side}_Wrist_Gyr_{axis}" for axis in axes]
    filtered_runs: list[np.ndarray] = []
    retained_runs: list[int] = []
    for start, stop in runs:
        data = df.iloc[start:stop][columns].apply(pd.to_numeric, errors="coerce")
        valid = data.notna().all(axis=1)
        if valid.mean() < 0.95 or valid.sum() < 200:
            continue
        values = data.loc[valid].to_numpy(dtype=float)
        filtered_runs.append(lowpass(values))
        retained_runs.append(len(values))
    if not filtered_runs:
        return {"eligible": False, "reason": "no_valid_walk_run"}

    pooled = np.vstack([x - np.mean(x, axis=0, keepdims=True) for x in filtered_runs])
    _, singular, vt = np.linalg.svd(pooled, full_matrices=False)
    loading = vt[0]
    pc1_share = float(singular[0] ** 2 / np.sum(singular**2))
    all_amp: list[float] = []
    all_vel: list[float] = []
    periodic: list[float] = []
    frequencies: list[float] = []
    accepted_runs = 0
    for x in filtered_runs:
        omega = (x - np.mean(x, axis=0, keepdims=True)) @ loading
        result = run_swings(omega)
        periodic.append(result["periodic_power"])
        frequencies.append(result["peak_frequency"])
        if result["amplitude"]:
            accepted_runs += 1
            all_amp.extend(result["amplitude"])
            all_vel.extend(result["velocity"])
    eligible = len(all_vel) >= 4
    payload = {
        "eligible": eligible,
        "reason": "ok" if eligible else "fewer_than_4_swings",
        "n_runs": len(filtered_runs),
        "accepted_runs": accepted_runs,
        "walk_seconds": float(sum(retained_runs) / FS),
        "n_swings": len(all_vel),
        "pc1_variance_share": pc1_share,
        "median_periodic_power": float(np.median(periodic)),
        "median_peak_frequency_hz": float(np.median(frequencies)),
        "median_amplitude_deg": float(np.median(all_amp)) if all_amp else np.nan,
        "median_peak_velocity_deg_s": float(np.median(all_vel)) if all_vel else np.nan,
    }
    for axis, value in zip(axes, loading):
        payload[f"loading_{axis}"] = float(value)
    return payload


def extract_file(path: Path, participant: str, task: str, session: int | None = None, axes: str = "XYZ") -> list[dict]:
    usecols = ["GeneralEvent"] + GYR_ALL
    try:
        df = pd.read_csv(path, usecols=lambda c: c in usecols)
    except Exception as exc:
        return [{"participant": participant, "task": task, "session": session, "side": side, "axes": axes, "eligible": False, "reason": f"read_error:{type(exc).__name__}"} for side in ("R", "L")]
    if any(c not in df.columns for c in GYR_ALL):
        return [{"participant": participant, "task": task, "session": session, "side": side, "axes": axes, "eligible": False, "reason": "missing_wrist_columns"} for side in ("R", "L")]
    label = "TandemWalk" if task == "Tandem" else "Walk"
    rows = []
    for side in ("R", "L"):
        rows.append({"participant": participant, "task": task, "session": session, "side": side, "axes": axes, "path": str(path), **extract_side(df, label, side, axes)})
    return rows


def make_wide(metrics: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    good = metrics[metrics["eligible"].eq(True)].copy()
    value_cols = ["median_peak_velocity_deg_s", "median_amplitude_deg", "pc1_variance_share", "median_periodic_power", "n_swings"]
    wide = good.pivot_table(index=["participant", "task"], columns="side", values=value_cols, aggfunc="first")
    wide.columns = [f"{name}_{side}" for name, side in wide.columns]
    wide = wide.reset_index().merge(clinical, on="participant", how="left")
    both = wide["median_peak_velocity_deg_s_R"].notna() & wide["median_peak_velocity_deg_s_L"].notna()
    wide = wide[both].copy()
    wide["sensor_lower_side"] = np.where(wide["median_peak_velocity_deg_s_R"] < wide["median_peak_velocity_deg_s_L"], "R", "L")
    wide["sensor_lower_amplitude_side"] = np.where(wide["median_amplitude_deg_R"] < wide["median_amplitude_deg_L"], "R", "L")
    wide["agreement"] = wide["sensor_lower_side"].eq(wide["clinical_side"])
    wide["sensor_log_R_over_L"] = np.log(wide["median_peak_velocity_deg_s_R"] / wide["median_peak_velocity_deg_s_L"])
    wide["aligned_log_peak"] = np.where(
        wide["clinical_side"].eq("R"),
        np.log(wide["median_peak_velocity_deg_s_L"] / wide["median_peak_velocity_deg_s_R"]),
        np.log(wide["median_peak_velocity_deg_s_R"] / wide["median_peak_velocity_deg_s_L"]),
    )
    wide["aligned_log_amplitude"] = np.where(
        wide["clinical_side"].eq("R"),
        np.log(wide["median_amplitude_deg_L"] / wide["median_amplitude_deg_R"]),
        np.log(wide["median_amplitude_deg_R"] / wide["median_amplitude_deg_L"]),
    )
    return wide


def kappa_binary(a: pd.Series, b: pd.Series) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    po = float(np.mean(a == b))
    pa_r = float(np.mean(a == "R"))
    pb_r = float(np.mean(b == "R"))
    pe = pa_r * pb_r + (1 - pa_r) * (1 - pb_r)
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def bootstrap_kappa(a: pd.Series, b: pd.Series, n_boot: int = 5000) -> tuple[float, float]:
    av = np.asarray(a)
    bv = np.asarray(b)
    values = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(av), len(av))
        value = kappa_binary(pd.Series(av[idx]), pd.Series(bv[idx]))
        if np.isfinite(value):
            values.append(value)
    return tuple(np.quantile(values, [0.025, 0.975]))


def agreement_result(df: pd.DataFrame, clinical_col: str = "clinical_side", sensor_col: str = "sensor_lower_side") -> dict:
    use = df[df[clinical_col].isin(["R", "L"])].copy()
    agree = use[sensor_col].eq(use[clinical_col])
    count = int(agree.sum())
    n = int(len(agree))
    ci = proportion_confint(count, n, method="beta") if n else (np.nan, np.nan)
    p = binomtest(count, n, 0.5, alternative="two-sided").pvalue if n else np.nan
    kappa = kappa_binary(use[clinical_col], use[sensor_col])
    kappa_ci = bootstrap_kappa(use[clinical_col], use[sensor_col]) if n else (np.nan, np.nan)
    return {
        "n": n,
        "agreements": count,
        "agreement_proportion": count / n if n else np.nan,
        "agreement_ci95": [float(ci[0]), float(ci[1])],
        "exact_binomial_p_two_sided": float(p),
        "cohen_kappa": float(kappa),
        "cohen_kappa_bootstrap_ci95": [float(kappa_ci[0]), float(kappa_ci[1])],
        "confusion": pd.crosstab(use[clinical_col], use[sensor_col]).reindex(index=["L", "R"], columns=["L", "R"], fill_value=0).to_dict(),
    }


def paired_change(df: pd.DataFrame, value: str = "aligned_log_peak") -> dict:
    p = df.pivot(index="participant", columns="task", values=value).dropna(subset=["SP", "HP"])
    diff = (p["HP"] - p["SP"]).to_numpy(dtype=float)
    z = (diff - np.mean(diff)) / np.std(diff, ddof=1)
    diagnostics = {
        "skew": float(skew(diff, bias=False)),
        "shapiro_p": float(shapiro(diff).pvalue),
        "max_abs_standardized": float(np.max(np.abs(z))),
    }
    use_wilcoxon = abs(diagnostics["skew"]) > 1.0 or diagnostics["max_abs_standardized"] > 4.0
    t = ttest_rel(p["HP"], p["SP"])
    w = wilcoxon(p["HP"], p["SP"], alternative="two-sided", zero_method="wilcox")
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    mean = float(np.mean(diff))
    return {
        "n": int(len(diff)),
        "SP_mean": float(p["SP"].mean()),
        "HP_mean": float(p["HP"].mean()),
        "mean_change_HP_minus_SP": mean,
        "mean_change_ci95_normal": [mean - 1.96 * se, mean + 1.96 * se],
        "median_change": float(np.median(diff)),
        "distribution_diagnostics": diagnostics,
        "prespecified_test_selected": "wilcoxon_signed_rank" if use_wilcoxon else "paired_t",
        "paired_t_statistic": float(t.statistic),
        "paired_t_p": float(t.pvalue),
        "wilcoxon_statistic": float(w.statistic),
        "wilcoxon_p": float(w.pvalue),
    }


def mixed_task_model(long: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    use = long[long["clinical_side"].isin(["R", "L"]) & long["Days since Part III Clinical Evaluation"].eq(0)].copy()
    use["task"] = pd.Categorical(use["task"], TASK_ORDER, ordered=True)
    model = mixedlm("aligned_log_peak ~ C(task, Treatment(reference='SP'))", use, groups=use["participant"])
    fit = model.fit(reml=True, method="lbfgs", maxiter=1000, disp=False)
    rows = []
    for task in TASK_ORDER[1:]:
        name = f"C(task, Treatment(reference='SP'))[T.{task}]"
        rows.append({
            "contrast": f"{task}-SP",
            "estimate": float(fit.params[name]),
            "se": float(fit.bse[name]),
            "ci_low": float(fit.conf_int().loc[name, 0]),
            "ci_high": float(fit.conf_int().loc[name, 1]),
            "p_raw": float(fit.pvalues[name]),
        })
    contrasts = pd.DataFrame(rows)
    contrasts["p_fdr_bh"] = multipletests(contrasts["p_raw"], method="fdr_bh")[1]
    payload = {
        "n_participants": int(use["participant"].nunique()),
        "n_observations": int(len(use)),
        "random_intercept_variance": float(fit.cov_re.iloc[0, 0]),
        "residual_variance": float(fit.scale),
        "converged": bool(fit.converged),
        "task_descriptives": use.groupby("task", observed=True)["aligned_log_peak"].agg(["count", "mean", "std", "median"]).reset_index().to_dict(orient="records"),
        "contrasts": contrasts.to_dict(orient="records"),
    }
    return payload, use


def icc_a1(values: pd.DataFrame) -> float:
    arr = values.to_numpy(dtype=float)
    n, k = arr.shape
    grand = arr.mean()
    row_mean = arr.mean(axis=1)
    col_mean = arr.mean(axis=0)
    msr = k * np.sum((row_mean - grand) ** 2) / (n - 1)
    msc = n * np.sum((col_mean - grand) ** 2) / (k - 1)
    residual = arr - row_mean[:, None] - col_mean[None, :] + grand
    mse = np.sum(residual**2) / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def bootstrap_icc(values: pd.DataFrame, n_boot: int = 3000) -> tuple[float, float]:
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(values), len(values))
        x = values.iloc[idx]
        try:
            val = icc_a1(x)
            if np.isfinite(val):
                vals.append(val)
        except Exception:
            pass
    return tuple(np.quantile(vals, [0.025, 0.975]))


def longitudinal_metrics(v2_wide: pd.DataFrame, clinical: pd.DataFrame, task: str) -> tuple[dict, pd.DataFrame]:
    use = v2_wide[v2_wide["task"].eq(task)].merge(clinical[["participant", "clinical_side"]], on="participant", how="left")
    use["log_ratio"] = np.log(use["peak_R"] / use["peak_L"])
    use["sensor_lower_side"] = np.where(use["peak_R"] < use["peak_L"], "R", "L")
    use["aligned_log"] = np.where(use["clinical_side"].eq("R"), -use["log_ratio"], use["log_ratio"])
    paired_count = int(use.pivot(index="participant", columns="session", values="log_ratio").dropna().shape[0])
    payload: dict = {"task": task, "n_participants": paired_count, "metrics": {}}
    for metric in ["peak_R", "peak_L", "log_ratio", "aligned_log"]:
        p = use.pivot(index="participant", columns="session", values=metric).dropna()
        if len(p) < 3:
            continue
        icc = icc_a1(p[[1, 2]])
        ci = bootstrap_icc(p[[1, 2]])
        change = p[2] - p[1]
        se = float(change.std(ddof=1) / math.sqrt(len(change)))
        payload["metrics"][metric] = {
            "n": int(len(p)), "icc_a1": icc, "icc_bootstrap_ci95": [float(ci[0]), float(ci[1])],
            "session1_mean": float(p[1].mean()), "session2_mean": float(p[2].mean()),
            "mean_change_s2_minus_s1": float(change.mean()),
            "change_ci95_normal": [float(change.mean() - 1.96 * se), float(change.mean() + 1.96 * se)],
        }
    sides = use.pivot(index="participant", columns="session", values="sensor_lower_side").dropna()
    payload["direction_stability"] = {
        "n": int(len(sides)),
        "agreement": float(np.mean(sides[1] == sides[2])),
        "kappa": kappa_binary(sides[1], sides[2]),
        "kappa_bootstrap_ci95": list(bootstrap_kappa(sides[1], sides[2])),
    }
    return payload, use


def plot_primary(primary_df: pd.DataFrame, result: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    matrix = pd.crosstab(primary_df["clinical_side"], primary_df["sensor_lower_side"]).reindex(index=["L", "R"], columns=["L", "R"], fill_value=0)
    im = axes[0].imshow(matrix.values, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, int(matrix.iloc[i, j]), ha="center", va="center", fontsize=14)
    axes[0].set_xticks([0, 1], ["L", "R"])
    axes[0].set_yticks([0, 1], ["L", "R"])
    axes[0].set_xlabel("Lower arm-swing velocity side")
    axes[0].set_ylabel("Clinical more-affected side")
    axes[0].set_title("Directional agreement")
    prop = result["agreement_proportion"]
    lo, hi = result["agreement_ci95"]
    axes[1].bar([0], [prop], color="#0072B2", width=0.5)
    axes[1].errorbar([0], [prop], yerr=[[prop - lo], [hi - prop]], fmt="none", color="black", capsize=5)
    axes[1].axhline(0.5, ls="--", color="gray")
    axes[1].set_xlim(-0.7, 0.7)
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks([0], ["SP"])
    axes[1].set_ylabel("Agreement proportion (95% CI)")
    axes[1].set_title(f"{result['agreements']}/{result['n']} agreements")
    fig.tight_layout()
    fig.savefig(OUT / "figure_primary_agreement.png", dpi=200)
    plt.close(fig)


def plot_tasks(long: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(20260907)
    data = [long.loc[long.task.eq(task), "aligned_log_peak"].dropna().to_numpy() for task in TASK_ORDER]
    ax.boxplot(data, labels=TASK_ORDER, showfliers=False, widths=0.55)
    for i, values in enumerate(data, start=1):
        ax.scatter(i + rng.normal(0, 0.05, len(values)), values, s=10, alpha=0.35, color="#0072B2")
    ax.axhline(0, color="gray", ls="--")
    ax.set_ylabel("Aligned log arm-swing velocity ratio\n(+ = clinically more-affected side lower)")
    ax.set_xlabel("Task")
    ax.set_title("Task-dependent clinically aligned arm-swing asymmetry")
    fig.tight_layout()
    fig.savefig(OUT / "figure_task_aligned_asymmetry.png", dpi=200)
    plt.close(fig)


def plot_longitudinal(use: pd.DataFrame, task: str = "SP") -> None:
    p = use.pivot(index="participant", columns="session", values="log_ratio").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(p[1], p[2], color="#0072B2", alpha=0.75)
    lim = [float(min(p.min()) - 0.05), float(max(p.max()) + 0.05)]
    axes[0].plot(lim, lim, ls="--", color="gray")
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("Session 1 log(R/L)")
    axes[0].set_ylabel("Session 2 log(R/L)")
    axes[0].set_title(f"{task} sensor asymmetry stability")
    mean = (p[1] + p[2]) / 2
    diff = p[2] - p[1]
    md = diff.mean(); sd = diff.std(ddof=1)
    axes[1].scatter(mean, diff, color="#D55E00", alpha=0.75)
    axes[1].axhline(md, color="black")
    axes[1].axhline(md + 1.96 * sd, color="gray", ls="--")
    axes[1].axhline(md - 1.96 * sd, color="gray", ls="--")
    axes[1].set_xlabel("Mean of sessions")
    axes[1].set_ylabel("Session 2 − Session 1")
    axes[1].set_title("Bland–Altman")
    fig.tight_layout()
    fig.savefig(OUT / f"figure_v2_{task.lower()}_stability.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clinical = load_clinical()
    inv1 = pd.read_csv(EXPLORE / "v1_file_inventory.csv")
    inv2 = pd.read_csv(EXPLORE / "v2_file_inventory.csv")

    rows = []
    for _, rec in inv1.iterrows():
        rows.extend(extract_file(Path(rec["path"]), rec["participant"], rec["task"], axes="XYZ"))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "v1_arm_swing_metrics_long.csv", index=False)
    wide = make_wide(metrics, clinical)
    wide.to_csv(OUT / "v1_arm_swing_metrics_wide.csv", index=False)

    primary_df = wide[
        wide["task"].eq("SP")
        & wide["clinical_side"].isin(["R", "L"])
        & wide["Days since Part III Clinical Evaluation"].eq(0)
    ].copy()
    primary = agreement_result(primary_df)
    primary["success_criterion_met"] = bool(primary["agreement_ci95"][0] > 0.5)
    json_dump(OUT / "primary_results.json", primary)

    paired_df = wide[
        wide["clinical_side"].isin(["R", "L"])
        & wide["Days since Part III Clinical Evaluation"].eq(0)
        & wide["task"].isin(["SP", "HP"])
    ].copy()
    secondary = paired_change(paired_df)
    json_dump(OUT / "secondary_sphp_results.json", secondary)

    task_model, task_long = mixed_task_model(wide)
    json_dump(OUT / "exploratory_task_model.json", task_model)
    pd.DataFrame(task_model["contrasts"]).to_csv(OUT / "table_task_contrasts.csv", index=False)

    sensitivities: dict = {}
    sp_all = wide[wide["task"].eq("SP")].copy()
    sensitivities["clinical_date_le30"] = agreement_result(sp_all[sp_all["Days since Part III Clinical Evaluation"].le(30)])
    sensitivities["clinical_date_any"] = agreement_result(sp_all)
    sensitivities["rigidity_only"] = agreement_result(primary_df, "rigidity_side")
    sensitivities["bradykinesia_only"] = agreement_result(primary_df, "brady_side")
    sensitivities["amplitude_outcome"] = agreement_result(primary_df, "clinical_side", "sensor_lower_amplitude_side")
    sensitivities["amplitude_SP_HP_change"] = paired_change(paired_df, "aligned_log_amplitude")
    sensitivities["exclude_DBS"] = agreement_result(primary_df[~primary_df["DBS?"].astype(str).str.upper().eq("YES")])
    sensitivities["quality_strict"] = agreement_result(primary_df[
        primary_df["pc1_variance_share_R"].ge(0.50)
        & primary_df["pc1_variance_share_L"].ge(0.50)
        & primary_df["median_periodic_power_R"].ge(0.75)
        & primary_df["median_periodic_power_L"].ge(0.75)
    ])
    sensitivities["by_site"] = {site: agreement_result(group) for site, group in primary_df.groupby("site")}
    sensitivities["leave_one_site_out"] = {site: agreement_result(primary_df[primary_df["site"].ne(site)]) for site in sorted(primary_df["site"].dropna().unique())}

    axis_sensitivity = {}
    sp_inventory = inv1[inv1["task"].eq("SP")]
    for axes in ("XY", "XZ", "YZ"):
        axis_rows = []
        for _, rec in sp_inventory.iterrows():
            axis_rows.extend(extract_file(Path(rec["path"]), rec["participant"], rec["task"], axes=axes))
        axis_metrics = pd.DataFrame(axis_rows)
        axis_wide = make_wide(axis_metrics, clinical)
        axis_primary = axis_wide[
            axis_wide["clinical_side"].isin(["R", "L"])
            & axis_wide["Days since Part III Clinical Evaluation"].eq(0)
        ]
        axis_sensitivity[axes] = agreement_result(axis_primary)
        axis_metrics.to_csv(OUT / f"sensitivity_sp_{axes}_metrics.csv", index=False)
    sensitivities["axis_pairs"] = axis_sensitivity
    json_dump(OUT / "sensitivity_results.json", sensitivities)

    v2_rows = []
    for _, rec in inv2.iterrows():
        v2_rows.extend(extract_file(Path(rec["path"]), rec["participant"], rec["task"], int(rec["session"]), axes="XYZ"))
    v2_metrics = pd.DataFrame(v2_rows)
    v2_metrics.to_csv(OUT / "v2_arm_swing_metrics_long.csv", index=False)
    good = v2_metrics[v2_metrics["eligible"].eq(True)].copy()
    v2_wide = good.pivot_table(
        index=["participant", "task", "session"], columns="side",
        values="median_peak_velocity_deg_s", aggfunc="first"
    ).rename(columns={"R": "peak_R", "L": "peak_L"}).dropna().reset_index()
    v2_wide.to_csv(OUT / "v2_arm_swing_metrics_wide.csv", index=False)
    v2_sp, v2_sp_use = longitudinal_metrics(v2_wide, clinical, "SP")
    v2_tandem, v2_tandem_use = longitudinal_metrics(v2_wide, clinical, "Tandem")
    longitudinal = {"SP": v2_sp, "Tandem_exploratory": v2_tandem}
    json_dump(OUT / "longitudinal_results.json", longitudinal)

    flow = {
        "v1_files": int(len(inv1)),
        "v1_files_bilateral_algorithm_eligible": int(wide[["participant", "task"]].drop_duplicates().shape[0]),
        "primary_expected_before_algorithm": 70,
        "primary_analyzed": int(len(primary_df)),
        "paired_SP_HP_analyzed": int(paired_df.pivot(index="participant", columns="task", values="aligned_log_peak").dropna().shape[0]),
        "v2_SP_two_session_analyzed": int(v2_sp["n_participants"]),
        "v2_Tandem_two_session_analyzed": int(v2_tandem["n_participants"]),
    }
    json_dump(OUT / "participant_flow.json", flow)

    plot_primary(primary_df, primary)
    plot_tasks(task_long)
    plot_longitudinal(v2_sp_use, "SP")

    manifest = {
        "analysis": "locked SAP v1",
        "sap_sha256": sha256(SAP_FILE),
        "script_sha256": sha256(Path(__file__)),
        "random_seed": 20260907,
        "software": {"python": "3", "pandas": pd.__version__, "numpy": np.__version__},
        "outputs": {},
        "deviations": [
            "The prespecified gait-speed-adjusted exploratory task model was not run because no validated cross-task gait-speed variable was available for all five tasks; this omission does not affect primary H1 or secondary H2.",
            "Official wrist longitudinal-axis documentation was not machine-readable; all three two-axis pairs were analyzed as prespecified orientation sensitivity variants instead of choosing one pair post hoc.",
        ],
    }
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "results_manifest.json":
            manifest["outputs"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    json_dump(OUT / "results_manifest.json", manifest)

    print(json.dumps({"primary": primary, "secondary": secondary, "flow": flow, "longitudinal": longitudinal}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
