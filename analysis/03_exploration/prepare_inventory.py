"""Write the file inventories used by the analysis scripts, with paths under WEARGAIT_RAW.

The repository stores the inventories with paths relative to the data folder
(full/, tandem/, v2/), as organised by the authors after download from Synapse.
"""
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = Path(os.environ["WEARGAIT_RAW"])
for name in ("v1_file_inventory", "v2_file_inventory"):
    inv = pd.read_csv(HERE / f"{name}_relative.csv")
    inv["path"] = [str(RAW / p) for p in inv["path"]]
    missing = [p for p in inv["path"] if not Path(p).exists()]
    inv.to_csv(HERE / f"{name}.csv", index=False)
    print(f"{name}: {len(inv)} files, {len(missing)} missing under WEARGAIT_RAW")
