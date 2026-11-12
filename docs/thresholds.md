# Thresholds in this repository

Several threshold values exist. They belong to different models or experiments and must not be mixed.

| Value | Where it comes from | Used by |
|---|---|---|
| **0.06141921002417763** | Stored inside `model_training/outputs/Deep_SVDD_Tuned.pt` (`threshold_percentile` 95) | The backend (`model_service.py` reads it from the checkpoint) and `Deep_SVDD_Tuned_evaluation_results.json` |
| 0.04083193272352222 (low, 92nd percentile) and 0.12980574011802667 (high, 98th) | `Deep_SVDD_hierarchical_config.json`; rule: highest validation recall with FPR at most 10% (low) or 2% (high) | Research outputs for hierarchical routing; not used by the API |
| 0.33 to 1.18 (90th to 98th percentile) | `Deep_SVDD_selected_thresholds.csv`, from a balanced 100,000 benign / 100,000 attack tuning sample, on a different score scale | An earlier tuning experiment; not used by the API |
| 0.06141921 (hard-coded) | Copied from the tuned checkpoint into `LLM/xai_explain.py`, `evaluate_zero_day.py` and `score_validation.py`, which load the legacy `Deep_SVDD.pt` | **Legacy**: pairs the old network (no stored threshold) with another network's threshold. See `LLM/README.md` |
| 0.04865155 | A separate autoencoder (not in this repository): 90th percentile of benign validation errors | Different model and score scale; never compare with the values above |

## What the active threshold gives (from `Deep_SVDD_Tuned_evaluation_results.json`)

- Validation, 31,159,073 rows: recall 0.9877, false-positive rate 0.0501, precision 0.9997.
- Zero-day set, 2,658,953 rows: detection rate 0.9938 overall, but per family it varies widely (for example `Backdoor_Malware` 0.432). Report zero-day results per family, not as one blended number.

## Check before comparing models

The SVDD evaluation used 31,159,073 validation rows; the autoencoder evaluation used 43,149,298. Confirm both used the same validation split before putting their numbers side by side.
