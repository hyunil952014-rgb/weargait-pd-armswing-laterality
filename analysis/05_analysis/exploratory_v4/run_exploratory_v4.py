#!/usr/bin/env python3
"""Post-result exploratory analyses for manuscript revision v4.

The locked SAP v1 results are inputs and are never overwritten. These analyses
were specified after review of the locked results and the existing Claude
pipeline; every inferential output is therefore labelled exploratory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LOCKED = PROJECT / "05_analysis"
RAW = Path(os.environ["WEARGAIT_RAW"])  # local copy of the WearGait-PD data
PROVENANCE = HERE / "provenance"
RNG = np.random.default_rng(20260908)
FS = 100.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gait = load_module(HERE / "gait_extract_snapshot.py", "gait_extract_snapshot")
locked = load_module(LOCKED / "run_locked_analysis.py", "locked_analysis_v1")


def exact_proportion(count: int, n: int) -> dict:
    result = stats.binomtest(count, n, 0.5)
    ci = result.proportion_ci(confidence_level=0.95, method="exact")
    return {
        "agreements": int(count),
        "n": int(n),
        "proportion": float(count / n),
        "exact_ci95": [float(ci.low), float(ci.high)],
        "exact_p_vs_0_5": float(result.pvalue),
    }


def signed_side(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def extract_step_lengths(primary: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    paths = inventory[inventory["task"].eq("SP")].set_index("participant")["path"].to_dict()
    rows = []
    for participant in primary["participant"]:
        path = Path(paths[participant])
        data = gait.load_walkway(path)
        footfalls = gait.footfall_centroids(data, gait.annotate_phase(gait.build_footfalls(data), data))
        steps, _, _ = gait.spatial_metrics(footfalls, phases=("Walk",))
        grouped = steps.groupby("side")["step_length"].agg(["mean", "size"])
        if not {"L", "R"}.issubset(grouped.index) or grouped["size"].min() < 4:
            rows.append({"participant": participant, "eligible": False, "reason": "fewer_than_4_steps_per_side"})
            continue
        ordered = steps.sort_values("idx")
        midpoint = len(ordered) // 2
        halves = {}
        for label, part in (("h1", ordered.iloc[:midpoint]), ("h2", ordered.iloc[midpoint:])):
            g = part.groupby("side")["step_length"].mean()
            halves[label] = float(np.log(g["R"] / g["L"])) if {"L", "R"}.issubset(g.index) else np.nan
        rows.append({
            "participant": participant,
            "eligible": True,
            "path": str(path),
            "left_step_cm": float(grouped.loc["L", "mean"]),
            "right_step_cm": float(grouped.loc["R", "mean"]),
            "n_left_steps": int(grouped.loc["L", "size"]),
            "n_right_steps": int(grouped.loc["R", "size"]),
            "step_log_R_over_L": float(np.log(grouped.loc["R", "mean"] / grouped.loc["L", "mean"])),
            "step_h1_log_R_over_L": halves["h1"],
            "step_h2_log_R_over_L": halves["h2"],
        })
    result = pd.DataFrame(rows)
    result.to_csv(HERE / "step_length_selfpace_exact_sample.csv", index=False)
    return result


def half_arm_direction(path: Path) -> dict:
    columns = ["GeneralEvent"] + locked.GYR_ALL
    data = pd.read_csv(path, usecols=lambda c: c in columns)
    walk = data[data["GeneralEvent"].astype(str).eq("Walk")].reset_index(drop=True)
    midpoint = len(walk) // 2
    payload = {}
    for label, part in (("h1", walk.iloc[:midpoint].copy()), ("h2", walk.iloc[midpoint:].copy())):
        part.loc[:, "GeneralEvent"] = "Walk"
        sides = {side: locked.extract_side(part, "Walk", side, axes="XYZ") for side in ("L", "R")}
        if all(sides[side].get("eligible", False) for side in ("L", "R")):
            left = sides["L"]["median_peak_velocity_deg_s"]
            right = sides["R"]["median_peak_velocity_deg_s"]
            payload[f"arm_{label}_log_R_over_L"] = float(np.log(right / left))
            payload[f"arm_{label}_lower_side"] = "R" if right < left else "L"
        else:
            payload[f"arm_{label}_log_R_over_L"] = np.nan
            payload[f"arm_{label}_lower_side"] = None
    return payload


BP = signal.butter(4, [0.3 / (FS / 2), 3.0 / (FS / 2)], btype="band", output="sos")


def rms_pc1(values: np.ndarray) -> float:
    frame = pd.DataFrame(values).interpolate().bfill().ffill().to_numpy(dtype=float)
    if len(frame) < 500 or not np.isfinite(frame).all():
        return np.nan
    filtered = signal.sosfiltfilt(BP, frame, axis=0)
    filtered -= filtered.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(filtered, full_matrices=False)
    projection = filtered @ vt[0]
    return float(np.sqrt(np.mean(projection ** 2)))


def alternate_rms_ratio(path: Path) -> float:
    columns = ["GeneralEvent"] + locked.GYR_ALL
    data = pd.read_csv(path, usecols=lambda c: c in columns)
    walk = data[data["GeneralEvent"].astype(str).eq("Walk")]
    values = {}
    for side in ("L", "R"):
        cols = [f"{side}_Wrist_Gyr_{axis}" for axis in "XYZ"]
        values[side] = rms_pc1(walk[cols].apply(pd.to_numeric, errors="coerce").to_numpy())
    return float(np.log(values["R"] / values["L"])) if np.isfinite(list(values.values())).all() else np.nan


def fit_logistic(outcome: pd.Series, predictor: pd.Series, scale: float = 0.1) -> dict:
    frame = pd.DataFrame({"y": outcome.astype(float), "x": predictor.astype(float) / scale}).dropna()
    fit = sm.GLM(frame["y"], sm.add_constant(frame[["x"]]), family=sm.families.Binomial()).fit()
    low, high = fit.conf_int().loc["x"]
    return {
        "n": int(len(frame)),
        "predictor_scale": float(scale),
        "odds_ratio": float(np.exp(fit.params["x"])),
        "ci95": [float(np.exp(low)), float(np.exp(high))],
        "p_exploratory": float(fit.pvalues["x"]),
        "intercept": float(fit.params["const"]),
    }


def paired_bootstrap_difference(frame: pd.DataFrame, n_boot: int = 10000) -> dict:
    observed = float(frame["arm_ok"].mean() - frame["step_ok"].mean())
    values = []
    for _ in range(n_boot):
        sampled = frame.iloc[RNG.integers(0, len(frame), len(frame))]
        values.append(float(sampled["arm_ok"].mean() - sampled["step_ok"].mean()))
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "difference": observed,
        "bootstrap_ci95": [float(low), float(high)],
        "bootstrap_probability_difference_le_zero": float(np.mean(np.asarray(values) <= 0)),
        "resamples": int(n_boot),
    }


def bootstrap_spearman_difference(frame: pd.DataFrame, n_boot: int = 10000) -> dict:
    arm = stats.spearmanr(frame["arm_signed_laterality"], frame["clinical_difference"])
    step = stats.spearmanr(frame["step_signed_laterality"], frame["clinical_difference"])
    values = []
    for _ in range(n_boot):
        sampled = frame.iloc[RNG.integers(0, len(frame), len(frame))]
        ra = stats.spearmanr(sampled["arm_signed_laterality"], sampled["clinical_difference"]).statistic
        rs = stats.spearmanr(sampled["step_signed_laterality"], sampled["clinical_difference"]).statistic
        if np.isfinite(ra) and np.isfinite(rs):
            values.append(float(ra - rs))
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "arm_spearman_rho": float(arm.statistic),
        "arm_p_exploratory": float(arm.pvalue),
        "step_spearman_rho": float(step.statistic),
        "step_p_exploratory": float(step.pvalue),
        "difference_rho": float(arm.statistic - step.statistic),
        "paired_bootstrap_ci95": [float(low), float(high)],
        "resamples": int(n_boot),
    }


def control_construct_check() -> pd.DataFrame:
    data = pd.read_csv(PROVENANCE / "claude_armswing_SPmT.csv")
    pd_clin = pd.read_csv(RAW / "PD - Demographic+Clinical - datasetV1.csv", header=1).rename(
        columns={"Subject ID": "pid", "Age (years)": "age", "Height (in)": "height", "Sex": "sex"}
    )[["pid", "age", "height", "sex"]]
    ct_clin = pd.read_csv(RAW / "CONTROLS - Demographic+Clinical - datasetV1.csv", header=1).rename(
        columns={"Subject ID": "pid", "Age": "age", "Height (in)": "height", "Sex": "sex"}
    )[["pid", "age", "height", "sex"]]
    clinical = pd.concat([pd_clin, ct_clin], ignore_index=True)
    clinical["sex01"] = clinical["sex"].eq("Male").astype(int)
    clinical["age"] = pd.to_numeric(clinical["age"], errors="coerce")
    clinical["height"] = pd.to_numeric(clinical["height"], errors="coerce")
    data = data.merge(clinical, on="pid", how="left")
    data["minimum_amplitude"] = data[["L_p2p_deg", "R_p2p_deg"]].min(axis=1)
    data["mean_amplitude"] = data[["L_p2p_deg", "R_p2p_deg"]].mean(axis=1)

    def smd(a: pd.Series, b: pd.Series) -> float:
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        return float((a.mean() - b.mean()) / pooled)

    rows = []
    for column, label in (
        ("asa_p2p_deg", "Amplitude asymmetry index"),
        ("asa_rms", "Angular velocity asymmetry index"),
        ("minimum_amplitude", "Lower-swing arm amplitude"),
        ("mean_amplitude", "Bilateral mean amplitude"),
    ):
        use = data.dropna(subset=[column, "age", "sex01", "height"]).copy()
        pd_values = use.loc[use["grp"].eq("PD"), column]
        ct_values = use.loc[use["grp"].eq("Control"), column]
        design = sm.add_constant(use[["age", "sex01", "height"]].assign(PD=use["grp"].eq("PD").astype(int)))
        fit = sm.OLS(use[column], design).fit()
        rows.append({
            "metric": label,
            "pd_n": int(len(pd_values)),
            "control_n": int(len(ct_values)),
            "pd_mean": float(pd_values.mean()),
            "control_mean": float(ct_values.mean()),
            "smd": smd(pd_values, ct_values),
            "adjusted_pd_beta": float(fit.params["PD"]),
            "adjusted_p_exploratory": float(fit.pvalues["PD"]),
        })
    output = pd.DataFrame(rows)
    output.to_csv(HERE / "control_construct_check.csv", index=False)
    return output


def make_figure(frame: pd.DataFrame, bins: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    arm = frame["arm_ok"].mean()
    step = frame["step_ok"].mean()
    arm_ci = exact_proportion(int(frame["arm_ok"].sum()), len(frame))["exact_ci95"]
    step_ci = exact_proportion(int(frame["step_ok"].sum()), len(frame))["exact_ci95"]
    axes[0].bar([0, 1], [arm, step], color=["#1f77b4", "#9aa0a6"], width=0.62)
    axes[0].errorbar(
        [0, 1], [arm, step],
        yerr=[[arm - arm_ci[0], step - step_ci[0]], [arm_ci[1] - arm, step_ci[1] - step]],
        fmt="none", color="black", capsize=5,
    )
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xticks([0, 1], ["Arm swing", "Step length"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Agreement with clinical laterality")
    axes[0].set_title("Exact locked-sample comparison")

    x = np.arange(len(bins))
    axes[1].plot(x, bins["clinical_agreement"], marker="o", color="#1f77b4", linewidth=2, label="Clinical agreement")
    axes[1].plot(x, bins["half_direction_stability"], marker="s", color="#d55e00", linewidth=2, label="Within-bout stability")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, bins["magnitude_bin"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Absolute arm-swing log ratio")
    axes[1].set_ylabel("Proportion")
    axes[1].set_title("Magnitude-dependent interpretability")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "figure_head_to_head_and_calibration.png", dpi=300)
    plt.close(fig)


def main() -> None:
    wide = pd.read_csv(LOCKED / "v1_arm_swing_metrics_wide.csv")
    inventory = pd.read_csv(PROJECT / "03_exploration/v1_file_inventory.csv")
    primary = wide[
        wide["task"].eq("SP")
        & wide["clinical_side"].isin(["L", "R"])
        & wide["Days since Part III Clinical Evaluation"].eq(0)
    ].copy()
    # Rerun 2026-10-21 on the full V1 download: the cohort is expected to differ from the original 67.
    print(f"[rerun] primary cohort n={len(primary)} (original locked run: 67)")

    steps = extract_step_lengths(primary, inventory)
    if not steps["eligible"].all():
        raise RuntimeError("Not every locked primary participant has eligible bilateral step length")
    merged = primary.merge(steps.drop(columns=["eligible"]), on="participant", validate="one_to_one")
    merged["clinical_difference"] = merged["upper_R"] - merged["upper_L"]
    merged["clinical_side_numeric"] = merged["clinical_side"].map({"L": -1, "R": 1})
    merged["arm_side_numeric"] = merged["sensor_lower_side"].map({"L": -1, "R": 1})
    merged["step_side_numeric"] = -merged["step_log_R_over_L"].map(signed_side)
    merged["arm_ok"] = merged["arm_side_numeric"].eq(merged["clinical_side_numeric"])
    merged["step_ok"] = merged["step_side_numeric"].eq(merged["clinical_side_numeric"])
    merged["arm_signed_laterality"] = -merged["sensor_log_R_over_L"]
    merged["step_signed_laterality"] = -merged["step_log_R_over_L"]
    merged["absolute_arm_log_ratio"] = merged["sensor_log_R_over_L"].abs()

    paths = inventory[inventory["task"].eq("SP")].set_index("participant")["path"].to_dict()
    half_rows = []
    alternate_rows = []
    for participant in merged["participant"]:
        path = Path(paths[participant])
        half_rows.append({"participant": participant, **half_arm_direction(path)})
        alternate_rows.append({"participant": participant, "alternate_rms_log_R_over_L": alternate_rms_ratio(path)})
    merged = merged.merge(pd.DataFrame(half_rows), on="participant", validate="one_to_one")
    merged = merged.merge(pd.DataFrame(alternate_rows), on="participant", validate="one_to_one")
    merged["half_direction_stable"] = merged["arm_h1_lower_side"].eq(merged["arm_h2_lower_side"])
    merged.loc[merged[["arm_h1_lower_side", "arm_h2_lower_side"]].isna().any(axis=1), "half_direction_stable"] = np.nan
    merged["alternate_rms_lower_side"] = np.where(merged["alternate_rms_log_R_over_L"] < 0, "R", "L")
    merged["alternate_rms_ok"] = merged["alternate_rms_lower_side"].eq(merged["clinical_side"])
    merged["alternate_vs_locked_direction_match"] = merged["alternate_rms_lower_side"].eq(merged["sensor_lower_side"])
    merged.to_csv(HERE / "head_to_head_exact_sample.csv", index=False)

    arm_only = int((merged["arm_ok"] & ~merged["step_ok"]).sum())
    step_only = int((~merged["arm_ok"] & merged["step_ok"]).sum())
    head_to_head = {
        "analysis_status": "post-result exploratory",
        "sample": "locked same-day SP primary cohort with paired same-bout step length",
        "arm": exact_proportion(int(merged["arm_ok"].sum()), len(merged)),
        "step": exact_proportion(int(merged["step_ok"].sum()), len(merged)),
        "paired_difference": paired_bootstrap_difference(merged),
        "discordant_pairs": {"arm_only_correct": arm_only, "step_only_correct": step_only},
        "mcnemar_exact_p_exploratory": float(stats.binomtest(arm_only, arm_only + step_only, 0.5).pvalue),
        "continuous_associations": bootstrap_spearman_difference(merged),
        "claim_boundary": "Estimate and triangulation only; no confirmatory claim that arm swing is superior to step length.",
    }
    json_dump(HERE / "head_to_head_results.json", head_to_head)

    magnitude_bins = pd.cut(
        merged["absolute_arm_log_ratio"],
        bins=[-np.inf, 0.10, 0.25, 0.50, np.inf],
        labels=["<0.10", "0.10-0.25", "0.25-0.50", ">0.50"],
        right=False,
    )
    merged["magnitude_bin"] = magnitude_bins
    bins = merged.groupby("magnitude_bin", observed=True).agg(
        n=("participant", "size"),
        clinical_agreement=("arm_ok", "mean"),
        half_eligible=("half_direction_stable", "count"),
        half_direction_stability=("half_direction_stable", "mean"),
    ).reset_index()
    bins["magnitude_bin"] = bins["magnitude_bin"].astype(str)
    bins.to_csv(HERE / "discordance_bins.csv", index=False)
    discordance = {
        "analysis_status": "post-result exploratory",
        "clinical_agreement_logistic": fit_logistic(merged["arm_ok"], merged["absolute_arm_log_ratio"]),
        "within_bout_direction_stability_logistic": fit_logistic(
            merged["half_direction_stable"], merged["absolute_arm_log_ratio"]
        ),
        "half_direction_eligible_n": int(merged["half_direction_stable"].notna().sum()),
        "descriptive_bins_file": "discordance_bins.csv",
        "threshold_safeguard": "The fixed bins are descriptive only; no participant was excluded and 0.10 was not used to redefine the primary result.",
    }
    json_dump(HERE / "discordance_calibration.json", discordance)

    alternate = {
        "analysis_status": "post-result exploratory analytic triangulation on overlapping data",
        "n": int(merged["alternate_rms_log_R_over_L"].notna().sum()),
        "clinical_agreement": exact_proportion(int(merged["alternate_rms_ok"].sum()), len(merged)),
        "direction_match_with_locked_peak_velocity_pipeline": float(merged["alternate_vs_locked_direction_match"].mean()),
        "direction_matches_n": int(merged["alternate_vs_locked_direction_match"].sum()),
        "spearman_with_locked_log_ratio": float(stats.spearmanr(
            merged["alternate_rms_log_R_over_L"], merged["sensor_log_R_over_L"]
        ).statistic),
        "not_external_validation": True,
    }
    json_dump(HERE / "alternate_pipeline_results.json", alternate)

    controls = control_construct_check()
    original_step = pd.read_csv(PROVENANCE / "claude_step_asym_SelfPace.csv")
    step_check = steps.merge(original_step, left_on="participant", right_on="pid", how="left")
    maximum_step_difference = float((step_check["step_log_R_over_L"] - step_check["logratio_step_sp"]).abs().max())
    original_h2h = pd.read_csv(PROVENANCE / "claude_H2H_paired.csv")
    overlap = set(merged["participant"]) & set(original_h2h["pid"])
    triangulation = {
        "locked_primary_n": int(len(merged)),
        "claude_h2h_n": int(len(original_h2h)),
        "participant_overlap_n": int(len(overlap)),
        "all_locked_primary_participants_in_claude_h2h": bool(len(overlap) == len(merged)),
        "independent_cohort": False,
        "reproduced_step_logratio_maximum_absolute_difference": maximum_step_difference,
        "control_construct_file": "control_construct_check.csv",
        "control_construct_summary": controls.to_dict(orient="records"),
        "source_snapshot_hashes": {
            path.name: sha256(path) for path in sorted(PROVENANCE.iterdir()) if path.is_file()
        },
    }
    json_dump(HERE / "triangulation_audit.json", triangulation)
    make_figure(merged, bins)

    outputs = {}
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.name != "results_manifest_v4_exploratory.json":
            outputs[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "analysis": "post-result exploratory manuscript revision v4",
        "original_locked_results_manifest_sha256": sha256(LOCKED / "results_manifest.json"),
        "sap_amendment_sha256": sha256(PROJECT / "04_protocol/amendments/amendment-001.json"),
        "script_sha256": sha256(Path(__file__)),
        "random_seed": 20260908,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else None,
            "statsmodels": sm.__version__ if hasattr(sm, "__version__") else None,
        },
        "outputs": outputs,
        "claim_boundary": "All new analyses are post-result exploratory. The locked primary and secondary results remain unchanged.",
    }
    json_dump(HERE / "results_manifest_v4_exploratory.json", manifest)
    print(json.dumps({
        "head_to_head": head_to_head,
        "discordance": discordance,
        "alternate": alternate,
        "step_reproduction_max_abs_difference": maximum_step_difference,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
