# ZeroShield AI - Teammate Handoff

> **LEGACY NOTICE - read first.** `xai_explain.py`, `evaluate_zero_day.py`, `score_validation.py` and
> `diagnose_svdd.py` load `Deep_SVDD.pt`, the older network (43 -> 64 -> 32 -> 16) saved by `models.py`.
> It stores **no threshold**. These scripts score it against 0.06141921, which is the stored threshold of a
> different network, `model_training/outputs/Deep_SVDD_Tuned.pt` (hidden layers [128, 64], 95th-percentile threshold).
> Everything derived from them (`xai_results/`, `soc_reports/`, `sample_*_alert.json`, and the copies in
> `ZeroShield_Verification_Handoff/soc_reports/`) therefore pairs one model with another model's threshold.
> In those files, `predicted_anomaly: false` for Mirai flows is an artifact of that pairing, not a result about
> the detector the application uses. Use them as **report-format examples only**.
> The application (`backend/`) uses the tuned checkpoint consistently. See `docs/thresholds.md`.

## Project pipeline
Deep SVDD anomaly detection
-> XAI feature explanations
-> MITRE ATT&CK retrieval
-> Local Ollama LLM SOC report
-> Post-LLM verification (next implementation stage)

## Requirements
- Python 3.11 or 3.12 recommended
- PyTorch
- NumPy
- pandas
- scikit-learn
- Ollama, if running the local LLM assistant
- Ollama model: llama3.2:3b

## Setup
Create a Python virtual environment and install compatible packages.
Check each script's imports before running it.

Install Ollama separately and run:
ollama pull llama3.2:3b

Start Ollama and check:
http://localhost:11434

## Important limitations
- The existing threshold 0.06141921 is provisional, not validated.
- MITRE TF-IDF similarity scores are not probabilities.
- XAI feature sensitivities are not proof of causation.
- Existing reports are example outputs, not proof of detector accuracy.
- Large datasets are intentionally excluded from this handoff.
- Scripts that read files under Dataset/ need the required dataset files.
- Run scripts from the project root so relative paths resolve correctly.

## Next task
Implement post-LLM verification:
1. Parse the LLM report into a defined schema.
2. Check claims against detector scores, XAI evidence and retrieved MITRE data.
3. Flag unsupported claims and invalid technique IDs.
4. Produce a verified report with pass/fail results and reasons.
5. Keep the original LLM output for comparison and auditing.
