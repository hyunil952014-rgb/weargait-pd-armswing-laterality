#!/usr/bin/env python3
"""Leave-one-participant-out influence check for the primary agreement (run at revision, 2026-10-26).

Planned in SAP section 9 but no archived output was found; this run is labelled as performed at revision.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
w = pd.read_csv(HERE.parent / "v1_arm_swing_metrics_wide.csv")
d = w[w.task.eq("SP") & w.clinical_side.isin(["L", "R"]) & w["Days since Part III Clinical Evaluation"].eq(0)
      & w.sensor_log_R_over_L.notna()].reset_index(drop=True)
assert len(d) == 73
lab = np.where(d.sensor_log_R_over_L < 0, "R", "L")
y = d.clinical_side.to_numpy()
rows = []
for i in range(len(d)):
    m = np.arange(len(d)) != i
    ok = lab[m] == y[m]
    lo, hi = proportion_confint(ok.sum(), ok.size, method="beta")
    rows.append({"left_out": d.participant[i], "agree": ok.mean(), "ci_low": lo, "kappa": cohen_kappa_score(y[m], lab[m])})
r = pd.DataFrame(rows)
r.to_csv(HERE / "lopo_influence.csv", index=False)
out = {"n_runs": len(r), "agreement_range": [r.agree.min(), r.agree.max()], "kappa_range": [r.kappa.min(), r.kappa.max()],
       "ci_low_range": [r.ci_low.min(), r.ci_low.max()], "success_criterion_met_in": int((r.ci_low > 0.5).sum())}
(HERE / "lopo_influence_results.json").write_text(json.dumps(out, indent=2, default=float))
print(out)
