#!/usr/bin/env python3
"""Outcome-blind feasibility checks for the selected WearGait-PD question.

This script deliberately does not test clinical associations or p-values. It only
maps clinical laterality, inventories files, and evaluates wrist-signal usability.
"""

from __future__ import annotations

import json
import re
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch


RAW = Path(os.environ["WEARGAIT_RAW"])  # local copy of the WearGait-PD data
OUT = Path(__file__).resolve().parent
CLINICAL = RAW / "PD - Demographic+Clinical - datasetV1.csv"
FULL = RAW / "full"
TANDEM = RAW / "tandem"
V2 = RAW / "v2"
FS = 100.0

WRIST_GYR = [f"{side}_Wrist_Gyr_{axis}" for side in ("R", "L") for axis in "XYZ"]
UPPER_COLUMNS = {
    "R": ["MDSUPDRS_3-3-RUE", "MDSUPDRS_3-4-R", "MDSUPDRS_3-5-R", "MDSUPDRS_3-6-R"],
    "L": ["MDSUPDRS_3-3-LUE", "MDSUPDRS_3-4-L", "MDSUPDRS_3-5-L", "MDSUPDRS_3-6-L"],
}


def load_clinical() -> pd.DataFrame:
    df = pd.read_csv(CLINICAL, header=1, na_values=["-", "", " "])
    df = df.rename(columns={"Subject ID": "participant"})
    for cols in UPPER_COLUMNS.values():
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["upper_R"] = df[UPPER_COLUMNS["R"]].sum(axis=1, min_count=len(UPPER_COLUMNS["R"]))
    df["upper_L"] = df[UPPER_COLUMNS["L"]].sum(axis=1, min_count=len(UPPER_COLUMNS["L"]))
    df["upper_total"] = df["upper_R"] + df["upper_L"]
    df["clinical_R_minus_L"] = df["upper_R"] - df["upper_L"]
    df["clinical_more_affected"] = np.select(
        [df["clinical_R_minus_L"] > 0, df["clinical_R_minus_L"] < 0],
        ["R", "L"],
        default="TIE",
    )
    keep = [
        "participant",
        "Age (years)",
        "Sex",
        "Years since PD diagnosis",
        "Days since Part III Clinical Evaluation",
        "Modified Hoehn & Yahr Score",
        "upper_R",
        "upper_L",
        "upper_total",
        "clinical_R_minus_L",
        "clinical_more_affected",
    ] + UPPER_COLUMNS["R"] + UPPER_COLUMNS["L"]
    return df[keep].copy()


def parse_v1(path: Path) -> tuple[str, str] | None:
    name = path.stem
    mapping = {
        "_SelfPace": "SP",
        "_HurriedPace": "HP",
        "_SelfPace_matTURN": "SPmT",
        "_TUG": "TUG",
        "_TandemGait": "Tandem",
    }
    for suffix, task in sorted(mapping.items(), key=lambda x: -len(x[0])):
        if name.endswith(suffix):
            return name[: -len(suffix)], task
    return None


def inventory_v1() -> pd.DataFrame:
    rows = []
    for directory in (FULL, TANDEM):
        for path in sorted(directory.glob("*.csv")):
            parsed = parse_v1(path)
            if not parsed:
                continue
            participant, task = parsed
            try:
                head = pd.read_csv(path, nrows=2)
                wrist_complete = all(c in head.columns for c in WRIST_GYR)
            except Exception:
                wrist_complete = False
            rows.append(
                {
                    "participant": participant,
                    "task": task,
                    "path": str(path),
                    "wrist_columns_present": wrist_complete,
                }
            )
    return pd.DataFrame(rows)


def inventory_v2() -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"^(?P<participant>.+)s(?P<session>[12])_(?P<task>SelfPace|TandemGait)$")
    for path in sorted(V2.glob("*.csv")):
        match = pattern.match(path.stem)
        if not match:
            continue
        task = {"SelfPace": "SP", "TandemGait": "Tandem"}[match.group("task")]
        try:
            head = pd.read_csv(path, nrows=2)
            wrist_complete = all(c in head.columns for c in WRIST_GYR)
        except Exception:
            wrist_complete = False
        rows.append(
            {
                "participant": match.group("participant"),
                "session": int(match.group("session")),
                "task": task,
                "path": str(path),
                "wrist_columns_present": wrist_complete,
            }
        )
    return pd.DataFrame(rows)


def select_qc_participants(clinical: pd.DataFrame, inv: pd.DataFrame, n_per_side: int = 5) -> list[str]:
    task_counts = inv[inv["wrist_columns_present"]].groupby("participant")["task"].nunique()
    complete = set(task_counts[task_counts == 5].index)
    eligible = clinical[
        clinical["participant"].isin(complete)
        & clinical["clinical_more_affected"].isin(["R", "L"])
        & clinical["upper_total"].notna()
    ].copy()
    selected: list[str] = []
    for side in ("R", "L"):
        side_df = eligible[eligible["clinical_more_affected"] == side].sort_values(
            ["upper_total", "participant"]
        )
        if len(side_df) <= n_per_side:
            selected.extend(side_df["participant"].tolist())
            continue
        positions = np.linspace(0, len(side_df) - 1, n_per_side).round().astype(int)
        selected.extend(side_df.iloc[positions]["participant"].tolist())
    return selected


def label_for_task(task: str) -> str:
    return "TandemWalk" if task == "Tandem" else "Walk"


def bandpass(signal: np.ndarray) -> np.ndarray:
    sos = butter(4, [0.3, 3.0], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, signal)


def signal_metrics(path: Path, participant: str, task: str) -> list[dict]:
    usecols = ["GeneralEvent"] + WRIST_GYR
    df = pd.read_csv(path, usecols=usecols)
    segment = df[df["GeneralEvent"].eq(label_for_task(task))].copy()
    rows = []
    for side in ("R", "L"):
        for axis in "XYZ":
            col = f"{side}_Wrist_Gyr_{axis}"
            x = pd.to_numeric(segment[col], errors="coerce").to_numpy(dtype=float)
            missing_fraction = float(np.mean(~np.isfinite(x))) if len(x) else 1.0
            x = x[np.isfinite(x)]
            row = {
                "participant": participant,
                "task": task,
                "side": side,
                "axis": axis,
                "n_walk_samples": int(len(x)),
                "walk_seconds": float(len(x) / FS),
                "missing_fraction": missing_fraction,
                "std": np.nan,
                "robust_range_q95_q05": np.nan,
                "band_rms": np.nan,
                "peak_frequency_hz": np.nan,
            }
            if len(x) >= 500:
                filtered = bandpass(x)
                freq, power = welch(filtered, fs=FS, nperseg=min(1024, len(filtered)))
                band = (freq >= 0.3) & (freq <= 3.0)
                row.update(
                    {
                        "std": float(np.std(x, ddof=1)),
                        "robust_range_q95_q05": float(np.quantile(x, 0.95) - np.quantile(x, 0.05)),
                        "band_rms": float(np.sqrt(np.mean(filtered**2))),
                        "peak_frequency_hz": float(freq[band][np.argmax(power[band])]),
                    }
                )
            rows.append(row)
    return rows


def make_trace_figure(inv: pd.DataFrame, selected: list[str], metrics: pd.DataFrame) -> None:
    # Plot two participants selected independently of wrist outcomes: one per clinical side.
    clinical = load_clinical().set_index("participant")
    examples = []
    for side in ("R", "L"):
        candidates = [p for p in selected if clinical.loc[p, "clinical_more_affected"] == side]
        if candidates:
            examples.append(candidates[len(candidates) // 2])
    tasks = ["SP", "HP", "SPmT", "TUG", "Tandem"]
    fig, axes = plt.subplots(len(examples), len(tasks), figsize=(18, 6), sharey=True)
    axes = np.atleast_2d(axes)
    for i, participant in enumerate(examples):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            path = Path(inv[(inv.participant == participant) & (inv.task == task)].iloc[0].path)
            d = pd.read_csv(path, usecols=["GeneralEvent"] + WRIST_GYR)
            d = d[d["GeneralEvent"].eq(label_for_task(task))]
            # Dominant axis is determined at group-task level, not per participant outcome.
            subset = metrics[(metrics.task == task)]
            axis = subset.groupby("axis")["band_rms"].median().idxmax()
            n = min(len(d), 1000)
            t = np.arange(n) / FS
            for side, color in (("R", "#D55E00"), ("L", "#0072B2")):
                x = pd.to_numeric(d[f"{side}_Wrist_Gyr_{axis}"], errors="coerce").iloc[:n].interpolate().bfill().to_numpy()
                if len(x) >= 20:
                    ax.plot(t, bandpass(x), lw=0.8, color=color, label=side)
            ax.set_title(f"{participant} | {task} | Gyr-{axis}")
            ax.set_xlabel("Walk time (s)")
            if j == 0:
                ax.set_ylabel("Band-passed angular velocity")
            if i == 0 and j == 0:
                ax.legend(frameon=False)
    fig.suptitle("Outcome-blind wrist gyroscope feasibility traces (0.3–3 Hz)")
    fig.tight_layout()
    fig.savefig(OUT / "qc_wrist_traces.png", dpi=180)
    plt.close(fig)


def main() -> None:
    clinical = load_clinical()
    inv1 = inventory_v1()
    inv2 = inventory_v2()

    clinical.to_csv(OUT / "clinical_laterality_map.csv", index=False)
    inv1.to_csv(OUT / "v1_file_inventory.csv", index=False)
    inv2.to_csv(OUT / "v2_file_inventory.csv", index=False)

    selected = select_qc_participants(clinical, inv1)
    selected_df = clinical[clinical["participant"].isin(selected)].copy()
    selected_df["selection_basis"] = "clinical side and upper-motor burden only; wrist outcomes unseen"
    selected_df.to_csv(OUT / "qc_participant_selection.csv", index=False)

    metric_rows = []
    for participant in selected:
        for _, row in inv1[inv1["participant"].eq(participant)].iterrows():
            metric_rows.extend(signal_metrics(Path(row["path"]), participant, row["task"]))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "qc_axis_metrics.csv", index=False)
    make_trace_figure(inv1, selected, metrics)

    clinical_counts = clinical["clinical_more_affected"].value_counts(dropna=False).to_dict()
    complete_v1 = (
        inv1[inv1["wrist_columns_present"]]
        .groupby("participant")["task"]
        .nunique()
        .eq(5)
    )
    v2_pair = (
        inv2[inv2["wrist_columns_present"]]
        .groupby(["participant", "task"])["session"]
        .nunique()
        .eq(2)
    )
    axis_summary = (
        metrics.groupby(["task", "side", "axis"], as_index=False)["band_rms"]
        .median()
        .sort_values(["task", "side", "band_rms"], ascending=[True, True, False])
    )
    dominant = axis_summary.groupby(["task", "side"]).first().reset_index()
    summary = {
        "guardrail": "No clinical-association test, p-value, or outcome-based participant selection was performed.",
        "sampling_rate_hz": FS,
        "clinical_definition": {
            "right": UPPER_COLUMNS["R"],
            "left": UPPER_COLUMNS["L"],
            "rule": "higher four-item upper-limb rigidity+bradykinesia sum; ties retained but excluded from primary directional agreement",
            "counts": {str(k): int(v) for k, v in clinical_counts.items()},
        },
        "v1": {
            "participants_any_file": int(inv1["participant"].nunique()),
            "participants_all_five_tasks_with_wrist": int(complete_v1.sum()),
            "task_counts": {str(k): int(v) for k, v in inv1.groupby("task")["participant"].nunique().to_dict().items()},
            "complete_five_tasks_with_directional_laterality": int(
                clinical[clinical["participant"].isin(complete_v1[complete_v1].index)]["clinical_more_affected"]
                .isin(["R", "L"])
                .sum()
            ),
            "complete_five_tasks_directional_and_same_day_clinical": int(
                (
                    clinical[clinical["participant"].isin(complete_v1[complete_v1].index)]["clinical_more_affected"]
                    .isin(["R", "L"])
                    & pd.to_numeric(
                        clinical[clinical["participant"].isin(complete_v1[complete_v1].index)][
                            "Days since Part III Clinical Evaluation"
                        ],
                        errors="coerce",
                    ).eq(0)
                ).sum()
            ),
        },
        "v2": {
            "participants": int(inv2["participant"].nunique()),
            "two_session_pairs_by_task": {str(k): int(v) for k, v in v2_pair.groupby("task").sum().to_dict().items()},
            "complete_two_task_two_session_participants": int(
                v2_pair[v2_pair].groupby("participant").size().eq(2).sum()
            ),
        },
        "qc_sample": selected,
        "qc_files": int(metrics[["participant", "task"]].drop_duplicates().shape[0]),
        "dominant_axis_by_task_side": dominant.to_dict(orient="records"),
        "minimum_walk_seconds": float(metrics.groupby(["participant", "task"])["walk_seconds"].first().min()),
        "maximum_missing_fraction": float(metrics["missing_fraction"].max()),
        "unit_status": "Angular velocity unit confirmed as rad/s in the source documentation; exact wrist-axis convention will be treated through orientation-insensitive PCA and documented sensitivity analyses.",
    }
    (OUT / "feasibility_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
