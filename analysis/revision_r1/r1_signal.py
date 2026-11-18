"""Shared signal-processing helpers for the R1 revision analyses (A7-A12).

`extract_runs` reproduces the interval selection and filtering of the locked pipeline
(run_locked_analysis.py: contiguous_runs, validity filter, 3 Hz low-pass) and keeps the
filtered three-axis runs so that alternative metrics can be computed without re-reading CSVs.
`swing_metrics` reproduces extract_side()/run_swings() with adjustable thresholds; with the
default parameters it must return exactly the locked values (checked in build_cache.py).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

FS = 100.0
RAW = Path(os.environ["WEARGAIT_RAW"])
GYR_ALL = [f"{side}_Wrist_Gyr_{axis}" for side in ("R", "L") for axis in "XYZ"]
_SOS = butter(2, 3.0, btype="lowpass", fs=FS, output="sos")


@dataclass(frozen=True)
class Params:
    min_run_samples: int = 200       # 2 s
    min_valid_fraction: float = 0.95
    periodic_min: float = 0.75       # share of 0.05-5 Hz power within 0.3-3 Hz
    prominence_deg: float = 2.0
    amplitude_deg: float = 5.0
    velocity_deg_s: float = 10.0
    min_swings: int = 4


DEFAULT = Params()


def label_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def extract_runs(path: Path, task: str) -> dict:
    """Read one CSV and return filtered runs per side plus interval bookkeeping."""
    df = pd.read_csv(path, usecols=lambda c: c in ["GeneralEvent"] + GYR_ALL)
    if any(c not in df.columns for c in GYR_ALL):
        return {"error": "missing_wrist_columns"}
    label = "TandemWalk" if task == "Tandem" else "Walk"
    runs_all = label_runs(df["GeneralEvent"].eq(label).to_numpy())
    out = {"n_label_runs": len(runs_all), "n_runs_lt2s": sum(1 for a, b in runs_all if b - a < DEFAULT.min_run_samples)}
    for side in ("R", "L"):
        cols = [f"{side}_Wrist_Gyr_{axis}" for axis in "XYZ"]
        kept, rejected_missing, starts = [], 0, []
        for a, b in runs_all:
            if b - a < DEFAULT.min_run_samples:
                continue
            data = df.iloc[a:b][cols].apply(pd.to_numeric, errors="coerce")
            valid = data.notna().all(axis=1)
            if valid.mean() < DEFAULT.min_valid_fraction or valid.sum() < DEFAULT.min_run_samples:
                rejected_missing += 1
                continue
            kept.append(sosfiltfilt(_SOS, data.loc[valid].to_numpy(dtype=float), axis=0))
            starts.append(a)
        out[side] = {"runs": kept, "starts": starts, "rejected_missing": rejected_missing}
    return out


def moving_average(x: np.ndarray, width: int = 101) -> np.ndarray:
    return np.convolve(x, np.ones(width) / width, mode="same")


def periodicity(omega_deg_s: np.ndarray) -> tuple[float, float]:
    freq, power = welch(omega_deg_s, fs=FS, nperseg=min(512, len(omega_deg_s)))
    arm = (freq >= 0.3) & (freq <= 3.0)
    ref = (freq >= 0.05) & (freq <= 5.0)
    return float(power[arm].sum() / power[ref].sum()), float(freq[arm][np.argmax(power[arm])])


def run_swings(omega_rad_s: np.ndarray, p: Params) -> dict:
    omega_deg_s = np.rad2deg(omega_rad_s)
    periodic, peak_f = periodicity(omega_deg_s)
    res = {"periodic_power": periodic, "peak_frequency": peak_f, "amplitude": [], "velocity": [],
           "status": "ok", "candidates": 0}
    if periodic < p.periodic_min:
        res["status"] = "periodicity"
        return res
    angle = np.rad2deg(cumulative_trapezoid(omega_rad_s, dx=1 / FS, initial=0))
    angle = angle - moving_average(angle)
    edge = 51
    usable = angle[edge:-edge] if len(angle) > 2 * edge else np.array([])
    if len(usable) < 80:
        res["status"] = "too_short_after_edges"
        return res
    min_distance = max(20, int(0.30 * FS / max(peak_f, 0.3)))
    maxima, _ = find_peaks(usable, prominence=p.prominence_deg, distance=min_distance)
    minima, _ = find_peaks(-usable, prominence=p.prominence_deg, distance=min_distance)
    extrema = sorted([(int(i), "max") for i in maxima] + [(int(i), "min") for i in minima])
    alt: list[tuple[int, str]] = []
    for item in extrema:
        if not alt or alt[-1][1] != item[1]:
            alt.append(item)
        else:
            old, kind = alt[-1]
            if (kind == "max" and usable[item[0]] > usable[old]) or (kind == "min" and usable[item[0]] < usable[old]):
                alt[-1] = item
    res["candidates"] = max(0, len(alt) - 1)
    for (i, _), (j, _) in zip(alt[:-1], alt[1:]):
        amp = abs(float(usable[j] - usable[i]))
        vel = float(np.max(np.abs(omega_deg_s[edge + i: edge + j + 1])))
        if amp >= p.amplitude_deg and vel >= p.velocity_deg_s:
            res["amplitude"].append(amp)
            res["velocity"].append(vel)
    return res


def swing_metrics(runs: list[np.ndarray], p: Params = DEFAULT, return_runs: bool = False) -> dict:
    """Locked extract_side() on pre-filtered runs, with adjustable thresholds."""
    if not runs:
        return {"eligible": False, "reason": "no_valid_walk_run", "n_runs": 0}
    pooled = np.vstack([x - x.mean(axis=0, keepdims=True) for x in runs])
    _, singular, vt = np.linalg.svd(pooled, full_matrices=False)
    loading = vt[0]
    vel, amp, periodic, per_run = [], [], [], []
    counts = {"periodicity": 0, "too_short_after_edges": 0, "no_accepted_swing": 0, "accepted": 0}
    candidates = 0
    for x in runs:
        omega = (x - x.mean(axis=0, keepdims=True)) @ loading
        r = run_swings(omega, p)
        periodic.append(r["periodic_power"])
        candidates += r["candidates"]
        if r["status"] != "ok":
            counts[r["status"]] += 1
        elif r["velocity"]:
            counts["accepted"] += 1
        else:
            counts["no_accepted_swing"] += 1
        vel.extend(r["velocity"])
        amp.extend(r["amplitude"])
        per_run.append(r["velocity"])
    out = {
        "eligible": len(vel) >= p.min_swings,
        "n_runs": len(runs),
        "n_swings": len(vel),
        "n_candidates": candidates,
        "run_status": counts,
        "pc1_variance_share": float(singular[0] ** 2 / np.sum(singular ** 2)),
        "median_periodic_power": float(np.median(periodic)),
        "median_peak_velocity_deg_s": float(np.median(vel)) if vel else np.nan,
        "median_amplitude_deg": float(np.median(amp)) if amp else np.nan,
    }
    if return_runs:
        out["per_run_velocity"] = per_run
    return out


def resultant_metrics(runs: list[np.ndarray], accepted_mask: list[bool] | None = None,
                      height: float = 10.0, prominence: float = 5.0, min_peaks: int = 4) -> dict:
    """Peak resultant angular velocity ||omega|| without projection.

    Each half swing (forward or backward) produces one local maximum of ||omega||; peaks are
    detected with a minimum spacing of 0.25 s, height >= 10 deg/s and prominence >= 5 deg/s,
    after trimming the same 0.51 s edges as the primary pipeline.
    """
    peaks_all = []
    for k, x in enumerate(runs):
        if accepted_mask is not None and not accepted_mask[k]:
            continue
        r = np.rad2deg(np.linalg.norm(x, axis=1))
        edge = 51
        if len(r) <= 2 * edge + 80:
            continue
        seg = r[edge:-edge]
        idx, _ = find_peaks(seg, height=height, prominence=prominence, distance=25)
        peaks_all.extend(seg[idx].tolist())
    return {"eligible": len(peaks_all) >= min_peaks, "n_peaks": len(peaks_all),
            "median_peak_resultant_deg_s": float(np.median(peaks_all)) if peaks_all else np.nan}
