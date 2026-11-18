#!/usr/bin/env python3
"""Independent consistency checks for the locked WearGait-PD analysis outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint


HERE = Path(__file__).resolve().parent


def load_json(name: str):
    with (HERE / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(float(a) - float(b)) <= tol


checks: list[dict] = []


def check(name: str, condition: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(condition), "detail": detail})


primary = load_json("primary_results.json")
flow = load_json("participant_flow.json")
longitudinal = load_json("longitudinal_results.json")
manifest = load_json("results_manifest.json")
wide = pd.read_csv(HERE / "v1_arm_swing_metrics_wide.csv")
v2wide = pd.read_csv(HERE / "v2_arm_swing_metrics_wide.csv")

# Reconstruct the primary cohort from participant-level outputs only.
p = wide.loc[
    (wide["task"] == "SP")
    & (wide["clinical_side"].isin(["L", "R"]))
    & (wide["Days since Part III Clinical Evaluation"] <= 7)
    & (wide["sensor_lower_side"].isin(["L", "R"]))
].copy()
p["agreement_recomputed"] = p["clinical_side"] == p["sensor_lower_side"]
n = len(p)
agreements = int(p["agreement_recomputed"].sum())
agreement = agreements / n
ci = proportion_confint(agreements, n, alpha=0.05, method="beta")
p_exact = binomtest(agreements, n, p=0.5, alternative="two-sided").pvalue
kappa = cohen_kappa_score(p["clinical_side"], p["sensor_lower_side"], labels=["L", "R"])

check("primary unique participants", p["participant"].nunique() == n, f"n={n}")
check("primary n", n == primary["n"] == flow["primary_analyzed"], f"recomputed={n}")
check("primary agreements", agreements == primary["agreements"], f"recomputed={agreements}")
check("primary proportion", close(agreement, primary["agreement_proportion"]), f"recomputed={agreement:.12f}")
check("primary exact CI", close(ci[0], primary["agreement_ci95"][0]) and close(ci[1], primary["agreement_ci95"][1]), f"recomputed={ci}")
check("primary exact p", close(p_exact, primary["exact_binomial_p_two_sided"]), f"recomputed={p_exact:.12f}")
check("primary kappa", close(kappa, primary["cohen_kappa"]), f"recomputed={kappa:.12f}")

conf = pd.crosstab(p["clinical_side"], p["sensor_lower_side"]).reindex(index=["L", "R"], columns=["L", "R"], fill_value=0)
check("confusion total", int(conf.to_numpy().sum()) == n, f"matrix={conf.to_dict()}")

# Reconstruct the V2 paired samples from the wide output.
for task, expected in [("SP", 42), ("Tandem", 37)]:
    task_df = v2wide.loc[v2wide["task"] == task]
    paired = task_df.pivot(index="participant", columns="session", values=["peak_L", "peak_R"]).dropna()
    reported = longitudinal["SP" if task == "SP" else "Tandem_exploratory"]["n_participants"]
    check(f"V2 {task} paired n", len(paired) == expected == reported, f"recomputed={len(paired)}")
    check(f"V2 {task} unique participants", paired.index.nunique() == len(paired), f"n={len(paired)}")

# Verify every output hash in the results manifest, excluding the manifest itself.
for filename, entry in manifest.get("outputs", {}).items():
    path = HERE / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"
    check(f"manifest hash: {filename}", digest == entry["sha256"], digest)

report = {
    "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
    "n_checks": len(checks),
    "n_passed": sum(item["passed"] for item in checks),
    "checks": checks,
}
(HERE / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(0 if report["status"] == "PASS" else 1)
