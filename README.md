# ZeroShield AI

ZeroShield AI is a research prototype for behavior-based network-flow anomaly detection and evidence-grounded SOC triage. It combines the existing benign-only Deep SVDD detector with feature perturbation explanations, MITRE ATT&CK context retrieval, optional LLM reasoning, rule-based verification, and a simulated contextual response policy. It is not a production security control.

## Current implementation

The repository contains the original training/evaluation and XAI code, a FastAPI/React demo, MITRE retrieval assets, verification code, and contextual response experimentation. Stages 1–7 are present. The dashboard and simulated action/feedback workflow also contain parts of Stages 8–10; their limits are listed below. See [docs/project-overview.md](docs/project-overview.md), [docs/methodology.md](docs/methodology.md), and [README_FOR_STAGES_8_10.md](README_FOR_STAGES_8_10.md).

| Stage | Current implementation |
|---|---|
| 1. Ingestion and features | Training/evaluation scripts consume prepared CSVs; runtime app accepts one prepared 43-feature row. |
| 2. Detection | `backend/services/model_service.py` loads `model_training/outputs/Deep_SVDD_Tuned.pt`; no training at startup. |
| 3. XAI | Per-feature zero-baseline perturbation in prepared/scaled feature space. |
| 4. MITRE context | `LLM/rag_store/mitre_tfidf.pkl` TF-IDF retrieval plus project category mapping. |
| 5. LLM reasoning | Optional OpenAI API; labeled evidence-based fallback when no key is configured or a call fails. |
| 6. Verification | Structured checks in `backend/services/verification_service.py`; separate report verifier in `ZeroShield_Verification_Handoff/`. |
| 7. Response policy | LinUCB and four simulated actions in `backend/services/bandit_service.py`. |
| 8–10 | Simulated execution, feedback/policy updates, API and dashboard views exist in part; they are not real containment or a production SOC platform. |

## Runtime artifacts and data

The app requires the small checkpoint `model_training/outputs/Deep_SVDD_Tuned.pt`, its feature list `model_training/outputs/zero_shield_feature_columns.json`, mapping file `model_training/outputs/Deep_SVDD_MITRE_mapping_for_LLM.json`, retrieval source `LLM/data/mitre_enterprise_techniques.json`, and the active TF-IDF index `LLM/rag_store/mitre_tfidf.pkl`. The checked-in `backend/data/demo_samples.json` is a small prepared-space demo cache. The SQLite ledger is created locally and ignored by Git.

The detector checkpoint has 43 features, hidden layers `[128, 64]`, latent dimension 16, and stores its frozen threshold (currently 0.06141921002417763). The app has no scaler artifact or raw-to-scaled transform. Inputs must already be in the project's StandardScaler-prepared feature space. The three root CSV datasets are large and excluded; the app does not need them to start or analyze the small demo cache. See the inventory in the handoff document for artifact details.

## Windows setup and launch

Prerequisites: Git, Python 3.11 (64-bit recommended), and Node.js/npm. Commands below are PowerShell commands run from the clone root. Replace the clone URL with the URL provided by the repository owner.

```powershell
git clone <repository-url> zeroshield-ai
Set-Location .\zeroshield-ai

# Backend
py -3.11 -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Set-Location .\backend
python -m uvicorn main:app --reload
```

Open a second PowerShell window at the clone root:

```powershell
Set-Location .\zeroshield-ai\frontend
npm ci
npm run dev
```

Open `http://localhost:5173`; API docs are at `http://localhost:8000/docs`. Check `http://localhost:8000/api/health` for `{"status":"ok"...}`. If PowerShell blocks venv activation, use `backend\.venv\Scripts\python.exe -m pip ...` and `backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload` without activating.

No API key is needed for detector, XAI, retrieval, verification, or fallback reasoning. For optional OpenAI reasoning, set the key only in the current PowerShell session (never commit it):

```powershell
$env:OPENAI_API_KEY = "paste-your-own-key-here"
$env:OPENAI_MODEL = "gpt-4o-mini"
```

To rebuild the MITRE index locally instead of using the checked-in 5.9 MB index:

```powershell
Set-Location .\zeroshield-ai\LLM
..\backend\.venv\Scripts\python.exe -m pip install scikit-learn==1.7.2
..\backend\.venv\Scripts\python.exe .\build_mitre_rag.py
```

The source JSON is included. Run index generation from `LLM` because the existing builder uses paths relative to its working directory. `pickle` indexes must only be loaded from this trusted repository or rebuilt locally.

## Demo and lightweight checks

Use the UI's demo samples or send one JSON object containing all 43 named features to `POST /api/analyze`. Values must already be prepared/scaled. Do not upload or load the full datasets for a smoke test.

```powershell
Set-Location .\backend
python -m unittest discover -s tests -v
Set-Location ..\ZeroShield_Verification_Handoff
python .\test_verify_soc_report.py
Set-Location ..\frontend
npm ci
npm run build
```

The standalone verifier expects the `ZeroShield_Verification_Handoff` fixture-relative working directory as shown above.

## Research data and training

`train_benign_only.csv`, `validation_set.csv`, and `zero_day_test_scaled_v2.csv` are local large datasets and are intentionally ignored. Obtain them through the project's authorized dataset source and prepare them according to the research scripts before running training/evaluation. No download URL or scaler artifact is present in this repository. The app does not require dataset access. Do not retrain or present illustrative sample output as measured performance.

## Limitations and security

- The existing fitted scaler itself was not found. Only prepared/scaled inputs can be analyzed faithfully; there is no supported raw feature ingestion transform.
- LLM output is optional and untrusted; verification checks evidence consistency, not whether an alert is truly malicious.
- MITRE candidates and TF-IDF similarities are contextual suggestions, not confirmed techniques.
- Actions are simulated; no host/network operation occurs. Labels enter only through simulation feedback.
- Dataset provenance/access instructions are not included. Arrange authorized access separately.
- On 2026-10-21, `npm audit --omit=dev` reported two high-severity advisories affecting `source-map-js` and Vite in the current frontend lockfile. Review/update the frontend lockfile before production use.
- Keep `.env`, API keys, local data, virtual environments, generated output, and the mutable database out of source control.

## Repository status

This working directory did not contain a `.git` directory when readiness was assessed. Therefore there is no verified branch, remote, index, or commit to inspect, and an actual fresh `git clone` test cannot be claimed here. Run the steps above after the repository is initialized and pushed to its approved remote; check every ignored runtime artifact is represented in the published repository before declaring clean-clone readiness.
