"""Read every wrist file once and cache filtered runs (PD V1 tasks + control tasks).

Then verify that the parameterised pipeline reproduces the locked per-wrist outputs exactly.
Cache: cache/runs_cache.pkl (derived from WearGait-PD raw data; local use only, not for sharing).
"""
from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r1_signal as rs  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "runs_cache.pkl"
PROJ = Path(__file__).resolve().parents[1]  # analysis/
INV = PROJ / "03_exploration" / "v1_file_inventory.csv"
LOCKED_LONG = PROJ / "05_analysis" / "v1_arm_swing_metrics_long.csv"
CTRL_V6 = PROJ / "05_analysis" / "exploratory_v6" / "control_task_metrics.csv"
CTRL_V8 = PROJ / "05_analysis" / "exploratory_v8" / "control_SP_HP_metrics.csv"


def file_list() -> list[tuple[str, str, str, str]]:
    items = []
    inv = pd.read_csv(INV)
    for _, r in inv.iterrows():
        items.append(("PD", r["participant"], r["task"], r["path"]))
    for suffix, task in (("_SelfPace_matTURN.csv", "SPmT"), ("_TUG.csv", "TUG")):
        for p in sorted((rs.RAW / "control").glob(f"*{suffix}")):
            items.append(("Control", p.name.removesuffix(suffix).replace(" (control)", "").strip(), task, str(p)))
    for suffix, task in (("_SelfPace.csv", "SP"), ("_HurriedPace.csv", "HP")):
        for p in sorted((rs.RAW / "control_SP_HP_syn61370552").glob(f"*{suffix}")):
            items.append(("Control", p.name.removesuffix(suffix).replace(" (control)", "").strip(), task, str(p)))
    return items


def work(item):
    group, pid, task, path = item
    try:
        return item, rs.extract_runs(Path(path), task)
    except Exception as exc:  # noqa: BLE001
        return item, {"error": f"{type(exc).__name__}: {exc}"}


def build() -> dict:
    items = file_list()
    print(f"files: {len(items)}", flush=True)
    cache = {}
    with ProcessPoolExecutor(max_workers=7) as ex:
        for k, (item, res) in enumerate(ex.map(work, items, chunksize=2)):
            group, pid, task, path = item
            cache[(group, pid, task)] = {"path": path, **res}
            if k % 50 == 0:
                print(f"  {k}/{len(items)}", flush=True)
    CACHE.parent.mkdir(exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def verify(cache: dict) -> None:
    long = pd.read_csv(LOCKED_LONG)
    bad = 0
    for _, r in long.iterrows():
        c = cache[("PD", r["participant"], r["task"])]
        if "error" in c:
            if r["eligible"]:
                bad += 1
            continue
        m = rs.swing_metrics(c[r["side"]]["runs"])
        if bool(m["eligible"]) != bool(r["eligible"]):
            bad += 1
            continue
        if r["eligible"] and not (np.isclose(m["median_peak_velocity_deg_s"], r["median_peak_velocity_deg_s"], atol=1e-9)
                                  and int(m["n_swings"]) == int(r["n_swings"])):
            bad += 1
    print(f"PD locked rows: {len(long)}, mismatches: {bad}")
    for path in (CTRL_V6, CTRL_V8):
        ctrl = pd.read_csv(path)
        badc = 0
        for _, r in ctrl.iterrows():
            c = cache[("Control", r["participant"], r["task"])]
            mr = rs.swing_metrics(c["R"]["runs"])
            ml = rs.swing_metrics(c["L"]["runs"])
            if not (np.isclose(mr["median_peak_velocity_deg_s"], r["R"], atol=1e-9) and np.isclose(ml["median_peak_velocity_deg_s"], r["L"], atol=1e-9)):
                badc += 1
        print(f"Control rows {path.name}: {len(ctrl)}, mismatches: {badc}")


if __name__ == "__main__":
    if CACHE.exists() and "--rebuild" not in sys.argv:
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = build()
    verify(cache)
