# Wrist arm-swing laterality in Parkinson disease (WearGait-PD secondary analysis)

Analysis code, statistical analysis plan, amendments and chronology for:

> Kim H, Lim C. Does the Lower-Swing Wrist Mark the More Affected Side in Parkinson Disease? Agreement and Exploratory Side-Specific Accuracy. *Sensors* (under review).

The study compares the wrist with lower peak angular velocity during walking with the clinically more affected upper limb (MDS-UPDRS rigidity and bradykinesia items) in the open WearGait-PD dataset.

## Contents

| Folder | What it holds |
|---|---|
| `docs/` | Statistical analysis plan (`SAP_v1_2026-09-07.md`), amendments (`amendments/`), chronology of data retrieval, code locking and outcome inspection (`chronology.md`), and the analyses added at peer review (`peer_review_R1_analyses.md`) |
| `analysis/03_exploration/` | File inventories (paths relative to the data folder), inventory preparation, and the feasibility scripts run before the plan |
| `analysis/05_analysis/run_locked_analysis.py` | Locked primary analysis, **byte-identical** to the script run on 7 September 2026 (SHA-256 `4bf6c823e26dfb0f722fb42f00c5be60f67aa81508ebc5a4474f78e61e4a6228`) |
| `analysis/05_analysis/run_locked.py` | Wrapper that runs the locked script with the data path set by `WEARGAIT_RAW`; the locked script still contains the authors' original local data path, which the wrapper overrides |
| `analysis/05_analysis/exploratory_v4` … `v9` | Post-result exploratory analyses defined in the amendments |
| `analysis/07_reporting/` | Statistics computed at the reporting stage (Table 3, Figure 1, paired AUC, participant flow) |
| `analysis/revision_r1/` | Analyses added at peer review (see `docs/peer_review_R1_analyses.md`) |

Apart from the locked script, the scripts are the authors' scripts with hard-coded local data paths replaced by the `WEARGAIT_RAW` environment variable; the analysis logic is unchanged.

## Data

The data are not redistributed here. WearGait-PD is available on Synapse under CC BY 4.0 after registration (parent project syn52540892; Version 1 syn55052683; Version 2 syn74686431; straight and fast-walking control files syn61370552). Dataset descriptor: Anderson et al., *Sci. Data* 2026, 13, 440, doi:10.1038/s41597-026-06806-2.

Place the files in one folder and set `WEARGAIT_RAW` to it, with this layout:

```
WEARGAIT_RAW/
  PD - Demographic+Clinical - datasetV1.csv
  CONTROLS - Demographic+Clinical - datasetV1.csv
  full/                         Version 1 PD walking files (e.g. NLS002_SelfPace.csv)
  tandem/                       Version 1 PD tandem files
  v2/                           Version 2 files (e.g. NLS005s1_SelfPace.csv)
  control/                      Version 1 control files (*_SelfPace_matTURN.csv, *_TUG.csv)
  control_SP_HP_syn61370552/    control straight and fast-walking files
```

The exact files used are listed in `analysis/03_exploration/v1_file_inventory_relative.csv` and `v2_file_inventory_relative.csv`.

## Running

```bash
pip install -r requirements.txt
WEARGAIT_RAW=/path/to/weargait_data bash run_all.sh
```

`run_all.sh` runs, in order: inventories → locked primary analysis → validation → exploratory analyses (v4–v9) → reporting statistics → peer-review analyses. Outputs are written next to each script. The peer-review step first checks that its signal-processing helper reproduces every locked per-wrist output exactly. The whole run takes a few minutes on a laptop.

The exploratory v4 step (same-bout step length and an alternative processing pipeline, Supplementary Tables S2 and S3) compares against participant-level outputs of an earlier parallel pipeline (`analysis/05_analysis/exploratory_v4/provenance/*.csv`). These files, and the scripts of that earlier pipeline, are not redistributed and are available from the authors on request; place the files in `analysis/05_analysis/exploratory_v4/provenance/` to run this step. `run_all.sh` skips this step when they are absent.

## Verification

Before release, the repository was copied to an empty folder and `run_all.sh` was run against the WearGait-PD files (about 7.5 minutes). Every output of the locked, exploratory and peer-review analyses matched the outputs used in the article value by value (JSON and CSV files; local paths and file hashes excluded), the validation step passed all 32 checks, and the reporting statistics reproduced Table 3 and the paired AUC difference.

## Where results appear in the article

| Result | Script |
|---|---|
| Primary agreement, kappa, fast-walking hypothesis, task model, six-month stability (Table 2), sensitivity analyses (Supplementary Tables S1, S11, S13) | `analysis/05_analysis/run_locked_analysis.py` |
| Step length, alternative pipeline (Supplementary Tables S2, S3) | `exploratory_v4/run_exploratory_v4.py` |
| PD–control asymmetry (Supplementary Table S4) | `exploratory_v4/run_control_construct_v5.py` |
| Control reference, AUC, DBS, magnitude, six-month direction (Table 3; Supplementary Tables S5, S6, S8; Supplementary Figure S3) | `exploratory_v6/run_conditional_validity_v6.py` |
| Cross-site reference (Supplementary Table S7) | `exploratory_v6/run_cross_site_v7.py` |
| Same-task control thresholds (Supplementary Table S14) | `exploratory_v8/run_same_task_control_v8.py` |
| Leave-one-participant-out check (Supplementary Table S12) | `exploratory_v9/run_lopo_influence_v9.py` |
| Balanced accuracy and change CIs, paired standard-vs-tandem AUC, participant flow, signal characteristics (Tables 1 and 3, Figure 1, Supplementary Table S10 (A)) | `analysis/07_reporting/compute_reported_statistics.py` |
| Analyses added at peer review (Sections 3.2–3.4, Tables 4 and 5, Supplementary Tables S10 (B), S15–S19, Supplementary Figure S1) | `analysis/revision_r1/` |

## Planning and transparency

The analysis plan was dated 7 September 2026, before the primary analysis was run; it was not registered in advance. `docs/chronology.md` lists when the data were downloaded, when the plan and amendments were written, and what had been seen before each step, based on the authors' local file records. Post-result and peer-review analyses are exploratory. Claude (Anthropic) and ChatGPT/Codex (OpenAI) assisted with code development; the authors reviewed the code and take responsibility for it.

## Licence

Code: MIT (see `LICENSE`). Documents in `docs/`: CC BY 4.0.
