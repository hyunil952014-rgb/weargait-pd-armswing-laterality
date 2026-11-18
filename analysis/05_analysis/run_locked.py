"""Run the locked primary analysis without editing it.

run_locked_analysis.py is kept byte-identical to the script used on 7 September 2026 (SHA-256 4bf6c823...).
It contains the authors' local data path, so this wrapper loads it and points the data paths to WEARGAIT_RAW.
The run manifest records the SHA-256 of the plan file it is given; here that is the English translation
in docs/, so this hash differs from the original Korean plan recorded in the authors' 7 September manifest.
"""
import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = Path(os.environ["WEARGAIT_RAW"])

spec = importlib.util.spec_from_file_location("run_locked_analysis", HERE / "run_locked_analysis.py")
locked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(locked)
locked.RAW = RAW
locked.CLINICAL_FILE = RAW / "PD - Demographic+Clinical - datasetV1.csv"
locked.SAP_FILE = HERE.parents[1] / "docs" / "SAP_v1_2026-09-07.md"
locked.main()
