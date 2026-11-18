#!/usr/bin/env python3
"""Pilot a literature-derived arm-swing extraction method on the locked QC sample.

No clinical association, task hypothesis test, or p-value is calculated here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, find_peaks, sosfiltfilt, welch


OUT = Path(__file__).resolve().parent
FS = 100.0
GYR = [f"{side}_Wrist_Gyr_{axis}" for side in ("R", "L") for axis in "XYZ"]


def contiguous_runs(mask: np.ndarray, minimum_samples: int = 250) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2) if b - a >= minimum_samples]


def lowpass(data: np.ndarray) -> np.ndarray:
    sos = butter(2, 3.0, btype="lowpass", fs=FS, output="sos")
    return sosfiltfilt(sos, data, axis=0)


def pca_first(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centered = data - np.mean(data, axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    component = centered @ vt[0]
    variance_share = float(singular[0] ** 2 / np.sum(singular**2))
    return component, vt[0], variance_share


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


def extract_run(data_rad_s: np.ndarray) -> dict:
    filtered = lowpass(data_rad_s)
    omega, loading, variance_share = pca_first(filtered)
    omega_deg = np.rad2deg(omega)

    freq, power = welch(omega_deg, fs=FS, nperseg=min(512, len(omega_deg)))
    arm_band = (freq >= 0.3) & (freq <= 3.0)
    reference_band = (freq >= 0.05) & (freq <= 5.0)
    peak_freq = float(freq[arm_band][np.argmax(power[arm_band])])
    periodic_power = float(power[arm_band].sum() / power[reference_band].sum())

    angle = np.rad2deg(cumulative_trapezoid(omega, dx=1 / FS, initial=0))
    angle = angle - moving_average(angle, 101)
    edge = 51
    usable = angle[edge:-edge] if len(angle) > 2 * edge else np.array([])
    if len(usable) < 150:
        return {
            "peak_frequency_hz": peak_freq,
            "periodic_power_fraction": periodic_power,
            "pca_variance_share": variance_share,
            "loading_X": float(loading[0]),
            "loading_Y": float(loading[1]),
            "loading_Z": float(loading[2]),
            "n_swings": 0,
            "median_amplitude_deg": np.nan,
            "median_peak_velocity_deg_s": np.nan,
        }

    minimum_distance = max(20, int(0.30 * FS / max(peak_freq, 0.3)))
    maxima, _ = find_peaks(usable, prominence=2.0, distance=minimum_distance)
    minima, _ = find_peaks(-usable, prominence=2.0, distance=minimum_distance)
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

    amplitudes = []
    velocities = []
    for first, second in zip(alternating[:-1], alternating[1:]):
        i, _ = first
        j, _ = second
        amp = abs(float(usable[j] - usable[i]))
        segment_velocity = abs(omega_deg[edge + i : edge + j + 1])
        peak_velocity = float(np.max(segment_velocity)) if len(segment_velocity) else np.nan
        if amp >= 5.0 and peak_velocity >= 10.0:
            amplitudes.append(amp)
            velocities.append(peak_velocity)

    return {
        "peak_frequency_hz": peak_freq,
        "periodic_power_fraction": periodic_power,
        "pca_variance_share": variance_share,
        "loading_X": float(loading[0]),
        "loading_Y": float(loading[1]),
        "loading_Z": float(loading[2]),
        "n_swings": len(amplitudes),
        "median_amplitude_deg": float(np.median(amplitudes)) if amplitudes else np.nan,
        "median_peak_velocity_deg_s": float(np.median(velocities)) if velocities else np.nan,
    }


def main() -> None:
    selection = pd.read_csv(OUT / "qc_participant_selection.csv")
    inventory = pd.read_csv(OUT / "v1_file_inventory.csv")
    selected = set(selection["participant"])
    inventory = inventory[inventory["participant"].isin(selected)].copy()

    rows = []
    for _, rec in inventory.iterrows():
        path = Path(rec["path"])
        df = pd.read_csv(path, usecols=["GeneralEvent"] + GYR)
        label = "TandemWalk" if rec["task"] == "Tandem" else "Walk"
        runs = contiguous_runs(df["GeneralEvent"].eq(label).to_numpy())
        for run_number, (start, stop) in enumerate(runs, start=1):
            for side in ("R", "L"):
                columns = [f"{side}_Wrist_Gyr_{axis}" for axis in "XYZ"]
                data = df.iloc[start:stop][columns].apply(pd.to_numeric, errors="coerce")
                valid = data.notna().all(axis=1).to_numpy()
                if valid.mean() < 0.95 or valid.sum() < 250:
                    continue
                result = extract_run(data.loc[valid].to_numpy(dtype=float))
                rows.append(
                    {
                        "participant": rec["participant"],
                        "task": rec["task"],
                        "run": run_number,
                        "side": side,
                        "duration_seconds": float(valid.sum() / FS),
                        **result,
                    }
                )

    results = pd.DataFrame(rows)
    results["accepted"] = (
        results["periodic_power_fraction"].ge(0.75)
        & results["pca_variance_share"].ge(0.50)
        & results["n_swings"].ge(4)
    )
    results.to_csv(OUT / "pilot_arm_swing_run_metrics.csv", index=False)

    summary_by_task = (
        results.groupby(["task", "side"], as_index=False)
        .agg(
            runs=("run", "size"),
            acceptance_rate=("accepted", "mean"),
            median_pca_variance=("pca_variance_share", "median"),
            median_periodic_power=("periodic_power_fraction", "median"),
            median_detected_swings=("n_swings", "median"),
            median_run_seconds=("duration_seconds", "median"),
        )
    )
    summary_by_task.to_csv(OUT / "pilot_arm_swing_qc_by_task.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    task_order = ["SP", "HP", "SPmT", "TUG", "Tandem"]
    plot_data = summary_by_task.copy()
    plot_data["task"] = pd.Categorical(plot_data["task"], task_order, ordered=True)
    for side, color in (("R", "#D55E00"), ("L", "#0072B2")):
        part = plot_data[plot_data["side"].eq(side)].sort_values("task")
        axes[0].plot(part["task"].astype(str), part["acceptance_rate"], "o-", color=color, label=side)
        axes[1].plot(part["task"].astype(str), part["median_pca_variance"], "o-", color=color)
        axes[2].plot(part["task"].astype(str), part["median_detected_swings"], "o-", color=color)
    axes[0].set_ylabel("Accepted run proportion")
    axes[1].set_ylabel("Median PC1 variance share")
    axes[2].set_ylabel("Median detected half-swings/run")
    for ax in axes:
        ax.set_xlabel("Task")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Pilot extraction QC only — no clinical association tests")
    fig.tight_layout()
    fig.savefig(OUT / "pilot_arm_swing_qc.png", dpi=180)
    plt.close(fig)

    summary = {
        "guardrail": "QC sample selected without wrist outcomes; no clinical association or task hypothesis test performed.",
        "method_basis": "Warmerdam et al. 2020: 3-Hz low-pass, PCA swing direction, numerical integration, 1-s drift removal, 0.3–3-Hz periodicity, minimum 5-degree amplitude and 10-degree/s peak velocity.",
        "adaptation": "Three-axis PCA is used in the pilot because WearGait straight-walk annotations exclude turns and the local sensor-axis contribution changes in tandem gait. The SAP will predefine primary and two-axis sensitivity variants.",
        "participants": int(results["participant"].nunique()),
        "runs": int(len(results)),
        "overall_acceptance_rate": float(results["accepted"].mean()),
        "by_task_side": summary_by_task.to_dict(orient="records"),
    }
    (OUT / "pilot_algorithm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

