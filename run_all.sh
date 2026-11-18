#!/usr/bin/env bash
# Reproduce all analyses from the WearGait-PD files.
# Usage: WEARGAIT_RAW=/path/to/weargait_data bash run_all.sh
set -euo pipefail
: "${WEARGAIT_RAW:?Set WEARGAIT_RAW to the folder holding the WearGait-PD files (see README)}"
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

echo "== 1. File inventories";            $PY analysis/03_exploration/prepare_inventory.py
echo "== 2. Locked primary analysis";     $PY analysis/05_analysis/run_locked.py > logs_locked.txt
echo "== 3. Validation of locked outputs"; $PY analysis/05_analysis/validate_analysis.py
if [ -f analysis/05_analysis/exploratory_v4/provenance/claude_armswing_SPmT.csv ]; then
  echo "== 4a. Exploratory v4 (step length, alternative pipeline)"; $PY analysis/05_analysis/exploratory_v4/run_exploratory_v4.py
else
  echo "== 4a. Exploratory v4 skipped: provenance inputs not present (see README)"
fi
echo "== 4b. PD-control construct check"; $PY analysis/05_analysis/exploratory_v4/run_control_construct_v5.py
echo "== 5. Control reference, AUC, magnitude (amendment 002)"; $PY analysis/05_analysis/exploratory_v6/run_conditional_validity_v6.py
echo "== 6. Cross-site reference (amendment 003)"; $PY analysis/05_analysis/exploratory_v6/run_cross_site_v7.py
echo "== 7. Same-task control thresholds"; $PY analysis/05_analysis/exploratory_v8/run_same_task_control_v8.py
echo "== 8. Leave-one-participant-out";    $PY analysis/05_analysis/exploratory_v9/run_lopo_influence_v9.py
echo "== 9. Reporting statistics";         $PY analysis/07_reporting/compute_reported_statistics.py
echo "== 10. Peer-review analyses";       $PY analysis/revision_r1/build_cache.py
$PY analysis/revision_r1/revision_r1_A1_A6.py > logs_revision_A1_A6.txt
$PY analysis/revision_r1/revision_r1_A7_A12.py
$PY analysis/revision_r1/revision_r1_A5b_cross_site.py > logs_revision_A5b.txt
echo "done"
