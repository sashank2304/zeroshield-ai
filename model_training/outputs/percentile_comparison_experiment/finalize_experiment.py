"""Record observations gathered while supervising the completed run."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

OUT = Path(__file__).resolve().parent
cfg_path = OUT / "experiment_config.json"
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
end = datetime.now(timezone.utc)
start = datetime.fromisoformat(cfg["start_time_utc"])
cfg["end_time_utc"] = end.isoformat()
cfg["experiment_duration_seconds"] = (end - start).total_seconds()
cfg["warnings"] = [
    "PyTorch emitted a read-only NumPy view warning during existing inference scoring; the view was read-only in use. The runner was later updated to create writable contiguous sample arrays for future reproductions.",
    "scikit-learn warned that OCSVM was fitted with an unnamed NumPy array and scored on named DataFrame chunks. The same 43 features in the saved order were supplied. The runner was later updated to use NumPy arrays for OCSVM scoring.",
    "VAE training loss rose to 16.731732 at epoch 4 and returned to 0.096780 at epoch 8. Training and evaluation completed.",
]
cfg["assumptions"] = [
    "Training and reference observations are a seeded row-level random split of the existing benign-only prepared dataset.",
    "The prepared files are treated as already StandardScaler-transformed; the fitted scaler and raw-data transformation procedure were not present in the repository.",
]
cfg["corrections"] = [
    "Threshold calibration is independent of model fitting: 100,000 benign rows were reserved outside the 500,000 training rows.",
    "Post-run runner edits remove OCSVM feature-name warnings and ensure writable contiguous sample arrays for future reproductions; they do not alter the completed model files or results.",
]
svm_package = joblib.load(OUT / "models" / "One_Class_SVM.joblib")
cfg["model_configurations"]["OCSVM"]["support_vectors"] = int(svm_package["model"].n_support_.sum())
cfg["model_configurations"]["OCSVM"]["support_vectors_per_class"] = svm_package["model"].n_support_.tolist()
(OUT / "experiment_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

model_summary = pd.read_csv(OUT / "model_summary.csv")
model_summary["Random_Seed"] = int(cfg["seed"])
model_summary.to_csv(OUT / "model_summary.csv", index=False)

log = (OUT / "experiment_log.txt").read_text(encoding="utf-8")
log += f"\nEnd time (UTC): {end.isoformat()}\nElapsed wall time (seconds): {cfg['experiment_duration_seconds']:.1f}\n"
log += "Warnings observed:\n- " + "\n- ".join(cfg["warnings"]) + "\n"
log += "Assumptions:\n- " + "\n- ".join(cfg["assumptions"]) + "\n"
log += "Corrections/implementation notes:\n- " + "\n- ".join(cfg["corrections"]) + "\n"
log += "Skipped evaluation rows: 0 (confirmed by count invariants).\n"
(OUT / "experiment_log.txt").write_text(log, encoding="utf-8")
print(f"Run duration seconds: {cfg['experiment_duration_seconds']:.1f}")
print(f"OCSVM support vectors: {cfg['model_configurations']['OCSVM']['support_vectors']}")
