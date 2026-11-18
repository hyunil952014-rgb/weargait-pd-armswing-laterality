# Analyses added at peer review (first revision, November 2026)

These analyses were added in response to the reviewers and were not part of the plan or its amendments. All are exploratory; p values are descriptive and not adjusted for multiple comparisons. Code: `analysis/revision_r1/`.

| ID | Analysis | Script | Reported in |
|---|---|---|---|
| A1 | Exact McNemar test of the sensor label against the clinical label, each task | `revision_r1_A1_A6.py` | Section 3.1; Supplementary Table S11 |
| A2 | Rigidity vs. bradykinesia laterality (concordant, opposite, tied) | `revision_r1_A1_A6.py` | Section 2.3; Supplementary Table S15 |
| A3 | Agreement and AUC by composite asymmetry (1 point vs. ≥2 points) | `revision_r1_A1_A6.py` | Table 5; Supplementary Table S15 |
| A4 | Agreement by DBS status (confirmed absence, reported DBS, unknown) | `revision_r1_A1_A6.py` | Table 5; Supplementary Table S5 |
| A5 | Site-adjusted side-specific difference (Mantel–Haenszel; logistic) | `revision_r1_A1_A6.py` | Table 5 |
| A5b | Cross-site references by clinically worse side | `revision_r1_A5b_cross_site.py` | Section 3.2; Table 4; Supplementary Table S7 |
| A6 | Paired task comparisons in common participants | `revision_r1_A1_A6.py` | Section 3.3; Supplementary Table S15 |
| A7 | Resultant angular velocity without projection; same-participant comparison | `revision_r1_A7_A12.py` | Section 3.4; Table 5; Supplementary Table S18 |
| A8 | One-at-a-time swing-detection and periodicity thresholds | `revision_r1_A7_A12.py` | Section 3.3; Supplementary Table S19 |
| A9 | Interval rejection, swing detection and exclusions by task and side | `revision_r1_A7_A12.py` | Supplementary Table S10 (B) |
| A10 | Control-reference sensitivity to age, sex and site | `revision_r1_A7_A12.py` | Section 3.2; Supplementary Table S16 |
| A11 | Patient-derived threshold with leave-one-out and repeated 10-fold cross-validation | `revision_r1_A7_A12.py` | Section 3.2; Table 5; Supplementary Table S17 |
| A12 | Within-session direction stability by pass-level split | `revision_r1_A7_A12.py` | Supplementary Figure S1 |

The signal-processing helper `r1_signal.py` reproduces the locked pipeline with adjustable thresholds. `build_cache.py` reads each raw file once, stores the filtered walking intervals, and checks that the default settings reproduce every locked per-wrist output exactly before any revision analysis is run.
