"""Create a tiny cache by streaming until one true prepared vector per category is found.

This is a one-time offline preparation command, never called by FastAPI. It reads at
most 50,000 rows into memory at a time and stops once the listed categories are found.
The source CSVs are project-prepared/scaled files.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "data" / "demo_samples.json"
CATEGORIES = ["BenignTraffic", "Backdoor_Malware", "BrowserHijacking", "CommandInjection", "Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain", "SqlInjection", "Uploading_Attack", "XSS"]
FEATURE_FILE = ROOT / "model_training" / "outputs" / "zero_shield_feature_columns.json"
FEATURES = json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
found = {}

def collect(path: Path, category_source: str):
    wanted = [c for c in CATEGORIES if c not in found]
    if not wanted or not path.exists():
        return
    for chunk in pd.read_csv(path, usecols=FEATURES + [category_source], chunksize=50_000, low_memory=True):
        labels = chunk[category_source].astype(str).str.strip()
        for category in wanted:
            matches = chunk.loc[labels.eq(category)]
            if not matches.empty:
                row = matches.iloc[0][FEATURES].replace([float("inf"), float("-inf")], float("nan")).fillna(0)
                found[category] = {feature: float(row[feature]) for feature in FEATURES}
        if len(found) == len(CATEGORIES):
            break

collect(ROOT / "validation_set.csv", "label")
collect(ROOT / "zero_day_test_scaled_v2.csv", "label")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(found, indent=2), encoding="utf-8")
print(f"Cached {len(found)}/{len(CATEGORIES)} true labeled prepared samples at {OUT}")
if len(found) < len(CATEGORIES):
    print("Missing categories:", ", ".join(c for c in CATEGORIES if c not in found))
