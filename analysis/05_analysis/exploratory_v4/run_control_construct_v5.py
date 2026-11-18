#!/usr/bin/env python3
"""Post-result PD-control construct check using the locked primary pipeline."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RAW = Path(os.environ["WEARGAIT_RAW"])  # local copy of the WearGait-PD data
LOCKED = PROJECT / "05_analysis" / "run_locked_analysis.py"
INVENTORY = PROJECT / "03_exploration" / "v1_file_inventory.csv"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


locked = load_module(LOCKED, "locked_analysis_v5_control")


def cohort_clinical(path: Path, group: str) -> pd.DataFrame:
    frame = pd.read_csv(path, header=1)
    height = "Height (in)"
    sex = "Sex" if "Sex" in frame.columns else "Gender"
    output = frame.rename(columns={"Subject ID": "participant", "Age (years)": "age", "Age": "age", height: "height", sex: "sex"})
    output = output[["participant", "age", "height", "sex"]].copy()
    output["participant"] = output["participant"].astype(str).str.strip()
    output["group"] = group
    output["age"] = pd.to_numeric(output["age"], errors="coerce")
    output["height"] = pd.to_numeric(output["height"], errors="coerce")
    output["male"] = output["sex"].astype(str).str.strip().eq("Male").astype(int)
    return output


def control_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted((RAW / "control").glob("*_SelfPace_matTURN.csv")):
        participant = path.name.removesuffix("_SelfPace_matTURN.csv").replace(" (control)", "")
        rows.append({"participant": participant, "path": str(path), "task": "SPmT"})
    return pd.DataFrame(rows)


def extract_metrics(inventory: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for item in inventory.itertuples(index=False):
        rows.extend(locked.extract_file(Path(item.path), item.participant, item.task, axes="XYZ"))
    metrics = pd.DataFrame(rows)
    good = metrics[metrics["eligible"].eq(True)].copy()
    wide = good.pivot_table(
        index="participant",
        columns="side",
        values=["median_peak_velocity_deg_s", "median_amplitude_deg"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{side}" for metric, side in wide.columns]
    wide = wide.reset_index()
    required = [
        "median_peak_velocity_deg_s_R", "median_peak_velocity_deg_s_L",
        "median_amplitude_deg_R", "median_amplitude_deg_L",
    ]
    wide = wide.dropna(subset=required).copy()
    wide["group"] = group
    wide["absolute_log_peak_velocity_ratio"] = np.abs(np.log(
        wide["median_peak_velocity_deg_s_R"] / wide["median_peak_velocity_deg_s_L"]
    ))
    wide["bilateral_mean_amplitude_deg"] = wide[[
        "median_amplitude_deg_R", "median_amplitude_deg_L"
    ]].mean(axis=1)
    return wide


def smd(pd_values: pd.Series, control_values: pd.Series) -> float:
    pooled = np.sqrt(((len(pd_values) - 1) * pd_values.var(ddof=1) + (len(control_values) - 1) * control_values.var(ddof=1)) / (len(pd_values) + len(control_values) - 2))
    return float((pd_values.mean() - control_values.mean()) / pooled)


def summarize(frame: pd.DataFrame, metric: str, label: str) -> dict:
    use = frame.dropna(subset=[metric, "age", "height", "male"]).copy()
    pd_values = use.loc[use["group"].eq("PD"), metric]
    control_values = use.loc[use["group"].eq("Control"), metric]
    design = sm.add_constant(use[["age", "height", "male"]].assign(PD=use["group"].eq("PD").astype(int)))
    fit = sm.OLS(use[metric], design).fit()
    return {
        "metric": label,
        "pd_n": int(len(pd_values)),
        "control_n": int(len(control_values)),
        "pd_mean": float(pd_values.mean()),
        "control_mean": float(control_values.mean()),
        "smd_pd_minus_control": smd(pd_values, control_values),
        "adjusted_pd_beta": float(fit.params["PD"]),
        "adjusted_p_exploratory": float(fit.pvalues["PD"]),
    }


def main() -> None:
    pd_inventory = pd.read_csv(INVENTORY).query("task == 'SPmT'")[["participant", "path", "task"]]
    control_paths = control_inventory()
    pd_metrics = extract_metrics(pd_inventory, "PD")
    control_metrics = extract_metrics(control_paths, "Control")
    clinical = pd.concat([
        cohort_clinical(RAW / "PD - Demographic+Clinical - datasetV1.csv", "PD"),
        cohort_clinical(RAW / "CONTROLS - Demographic+Clinical - datasetV1.csv", "Control"),
    ], ignore_index=True)
    metrics = pd.concat([pd_metrics, control_metrics], ignore_index=True).merge(
        clinical, on=["participant", "group"], how="left", validate="one_to_one"
    )
    details_path = HERE / "control_construct_v5_participant_metrics.csv"
    metrics.to_csv(details_path, index=False)
    summary = [
        summarize(metrics, "absolute_log_peak_velocity_ratio", "Absolute peak angular-velocity asymmetry |log(R/L)|"),
        summarize(metrics, "bilateral_mean_amplitude_deg", "Bilateral mean arm-swing amplitude (degrees)"),
    ]
    output = {
        "analysis_id": "A-EXP-004",
        "evidence_status": "post_lock_exploratory",
        "shared_task": "SelfPace_matTURN",
        "pipeline": "Locked three-axis peak-angular-velocity extraction",
        "covariates": ["age", "sex", "height"],
        "results": summary,
        "claim_boundary": "Supportive construct evidence only; not external validation and not validation of individual clinical-side classification.",
    }
    (HERE / "control_construct_v5_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
