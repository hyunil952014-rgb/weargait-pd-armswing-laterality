# Chronology of data retrieval, planning, code locking and outcome inspection

All times are local (Korea Standard Time) and come from the authors' local file records (file creation or modification times, run logs and download records). They are not third-party timestamps, and the plan was not registered before the analysis.

| Date (2026) | Event | Record |
|---|---|---|
| 3–4 September | Initial download from Synapse: V1 PD walking files, V1 control files, tandem files and V2 files | Creation times of the local raw-data folders and files |
| 7 September, 16:11–16:16 | Feasibility check before the plan: signal quality was piloted in 10 participants with PD selected on clinical information only (clinical side and upper-limb burden); no clinical–sensor agreement, task difference or p value was computed | `analysis/03_exploration/run_feasibility.py`; pilot participant selection recorded as "wrist outcomes unseen" |
| 7 September, 16:16 | Statistical analysis plan v1.0 | `docs/SAP_v1_2026-09-07.md` (translation) |
| 7 September, 19:59 | Locked primary analysis run; the run manifest records the SHA-256 of the analysis script (`4bf6c823…`) and of the original plan file | `analysis/05_analysis/run_locked_analysis.py` (byte-identical to the locked script) |
| 8 September | Post-result exploratory plan (amendment 001): step-length comparison, magnitude analysis, alternative pipeline | `docs/amendments/A001_2026-09-08_post_result_plan.json` |
| 9 September | PD–control construct check plan | `docs/amendments/A001b_2026-09-09_control_construct_plan.json` |
| 7 October | Amendment 002 (control reference, AUC, magnitude, six-month direction, DBS) and amendment 003 (cross-site reference), written before running; sections recording what had already been seen are part of the amendment | `docs/amendments/A002_2026-10-07_conditional_validity.md`; `analysis/05_analysis/exploratory_v6/` |
| 21 October | Data-coverage correction (amendment 004): missing V1 files downloaded from Synapse (download record with Synapse IDs and checksums); locked script rerun unchanged (n = 73); exploratory analyses rerun; balanced accuracy added after results were seen | `docs/amendments/A004_2026-10-21_data_coverage.md` |
| 26 October, 18:50–18:56 | Straight self-paced and fast-walking control files downloaded (Synapse syn61370552) | Download folder |
| 26 October | Same-task control thresholds (v8) and leave-one-participant-out influence check (v9) | `analysis/05_analysis/exploratory_v8/`, `exploratory_v9/` (dated script headers) |
| 27 October | Manuscript submitted | — |
| 17–18 November | Analyses added at peer review (R1) | `analysis/revision_r1/`, `docs/peer_review_R1_analyses.md` |
