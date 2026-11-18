# Amendment 004 — Reanalysis after correction of data coverage (21 October 2026)

Written: 21 October 2026. Nature: correction of **input data coverage**, not a change in analysis methods.

> Translated from the Korean amendment; local folder references were replaced by their role. The original file is available from the authors on request.

## What was wrong

- The plan's primary population is participants with PD with bilateral standard-walking data, complete same-day MDS-UPDRS Part III upper-limb items, non-tied laterality and at least four eligible swings per wrist.
- 76 participants are clinically eligible under these criteria. The local copy of the V1 walking files, however, had been downloaded for an earlier project with a narrower selection (complete Part III and four tasks, 88 participants) and contained files for only 70 of the 76.
- The locked analysis (n = 67) was therefore run on narrower data than the plan specified. The cause was the download scope, not a selection based on results.
- How it was found: rechecked directly after an external manuscript appraisal on 21 October 2026.

## What was done

1. V1 files for the 12 missing participants with PD (55 files) and one missing tandem file (WPD029) were downloaded from Synapse (syn61370558) and added to the local raw folders, with a download record (Synapse IDs and checksums).
2. The project was copied, and only `03_exploration/v1_file_inventory.csv` was regenerated from the raw folders (438 → 494 rows; all earlier rows retained).
3. `run_locked_analysis.py` was run byte-identical to the original (SHA-256 4bf6c823e26dfb0f722fb42f00c5be60f67aa81508ebc5a4474f78e61e4a6228).
4. The exploratory analyses (v4, v5, v6, v7) were rerun with the same scripts. The only code change: in v4, the check "stop if the primary cohort is not 67" was replaced by printing n.

## Results

- Primary analysis n = 73 (all six added participants passed the signal criteria). Agreement 48/73 = 65.8% (95% CI 53.7–76.5), kappa 0.343; the pre-specified success criterion was met. Essentially the same as the original 44/67 = 65.7%.
- Sensor values of the original 67 participants were identical in both runs (maximum difference 0).
- Control results and V2 six-month results were unchanged by design.

## Reporting principles

- The primary result in the manuscript is replaced by this reanalysis (n = 73), which matches the plan's criteria; it would have been replaced even if the result had become worse.
- The original analysis (n = 67) is preserved. The manuscript states the data-coverage correction and its (essentially null) impact (Supplementary Table S9).
