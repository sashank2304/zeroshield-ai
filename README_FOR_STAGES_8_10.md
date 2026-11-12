# ZeroShield AI — technical handoff for continuing development

## A. Project overview

ZeroShield AI is a research prototype for detecting unusual network-flow behavior with a benign-only Deep SVDD model, explaining score sensitivity, retrieving candidate ATT&CK context, producing optional evidence-bound reasoning, checking that reasoning, and selecting a simulated response. A detector alert is not proof of an attack; the feature data alone do not establish endpoint or application behavior.

## B. Current pipeline

1. **Stage 1 — ingestion and feature engineering:** training/evaluation scripts work with prepared, scaled CSV data. The running API accepts one prepared 43-feature record; it does not perform raw-to-scaled conversion.
2. **Stage 2 — detector:** the saved Deep SVDD checkpoint is loaded by `backend/services/model_service.py`.
3. **Stage 3 — XAI:** the API computes zero-baseline feature perturbation sensitivities for the current record.
4. **Stage 4 — MITRE retrieval:** the API turns the clearly abnormal top XAI features into a plain-language behaviour query and retrieves network-observable techniques from the local TF-IDF index, abstaining when there is no clear behaviour. A separate project category mapping is also retained.
5. **Stage 5 — LLM:** optional OpenAI JSON reasoning; a labeled fallback is returned without a key or on provider failure.
6. **Stage 6 — verification:** the backend checks schema, score/status, numeric grounding, feature support/direction, retrieved technique IDs, and uncertainty. The standalone verifier checks saved Markdown report fixtures.
7. **Stage 7 — contextual policy:** LinUCB chooses among `AUTO_REMEDIATE`, `CONFIRM_TO_ACT`, `HUMAN_REVIEW`, and `MONITOR_ONLY`; selected action and evidence are recorded in SQLite.
8. **Stage 8 — simulated action:** currently represented by simulated status and approval handling; no host/network operation occurs.
9. **Stage 9 — feedback/reward:** feedback endpoints apply the designed reward table and update policy state once. Dataset labels are used only as offline simulated outcomes.
10. **Stage 10 — dashboard/reporting:** the React UI and API expose analysis, alerts, approvals, replay, and experiment results. This is a prototype, not a complete or production SOC dashboard.

Stages 8–10 are the next development scope unless inspection at implementation time shows the particular required behaviors are complete. Reuse existing code and close gaps; do not start another application.

## C. Actual architecture

- `backend/main.py`: FastAPI app and stage orchestration.
- `backend/services/model_service.py`: detector load, inference, XAI.
- `backend/services/mitre_service.py`: project mapping and behavior retrieval.
- `backend/services/llm_service.py`: optional API and fallback reasoning.
- `backend/services/verification_service.py`: structured response checks.
- `backend/services/bandit_service.py`: LinUCB, action simulation, approval, feedback, SQLite ledger.
- `backend/schemas/analysis.py`, `backend/schemas/response.py`: request schemas (strict boolean for feedback, approval choice).
- `backend/security.py`: localhost-or-token access control for state-changing endpoints.
- `backend/data/demo_samples.json`: small prepared-space cache. `zeroshield_ledger.sqlite3` is generated locally and ignored.
- `backend/tests/`: backend policy and adapter tests.
- `backend/run_response_experiment.py`, `backend/response_experiment_config.json`: replay/experiment code and configuration.
- `frontend/src/`, `frontend/package.json`, `frontend/package-lock.json`: React/Vite dashboard.
- `model_training/outputs/Deep_SVDD_Tuned.pt`: active detector checkpoint.
- `model_training/outputs/zero_shield_feature_columns.json`: active feature order.
- `model_training/outputs/Deep_SVDD_MITRE_mapping_for_LLM.json`: category map used by API.
- `LLM/data/mitre_enterprise_techniques.json`: technique source for rebuilding retrieval index.
- `LLM/rag_store/mitre_tfidf.pkl`: existing runtime retrieval index.
- `Deep_SVDD_XAI.py`, `svdd.py`, `svdd_eval.py`, `models.py`, `Iso_forr.py`, `eval_for_iso_forr.py`, `ZeroShield_MITRE_mapping_final.py`: existing research implementation/scripts.
- `ZeroShield_Verification_Handoff/`: standalone saved-report verifier and fixtures.
- `Documentation/`: research and presentation material.

## D. Existing ML implementation

The runtime model is Deep SVDD, loaded with `torch.load(..., map_location="cpu")` from `model_training/outputs/Deep_SVDD_Tuned.pt`. The checkpoint contains `features`, `hidden`, `latent_dim`, `center`, `threshold`, and `state_dict`. It is a 43-input network with hidden dimensions `[128, 64]` and latent dimension 16. Feature order is taken from the checkpoint and cross-checked at 43; the companion definition is `model_training/outputs/zero_shield_feature_columns.json`.

Score is squared Euclidean distance from the network embedding to the stored center. `score >= checkpoint threshold` is anomalous; the current checkpoint threshold is `0.06141921002417763`. No fitted scaler artifact or raw preprocessing transformer was found. Inputs must already be in the prepared StandardScaler feature space. XAI replaces each feature with zero in that space and reports original score minus perturbed score; this is local sensitivity, not causation. The 60 KB active checkpoint is present locally and should be tracked. No retraining is required.

## E. Stage 4 — MITRE ATT&CK retrieval

`backend/services/mitre_service.py` loads `LLM/rag_store/mitre_tfidf.pkl` containing documents, fitted `TfidfVectorizer`, and sparse matrix. It uses cosine-equivalent dot products from the normalized TF-IDF representation, returning up to five records with IDs, names, descriptions, URLs, tactics, platforms, and similarity. The query is plain-language behaviour built from the top eight XAI features that push the score up and sit clearly outside the benign range (hand-written feature-to-behaviour map `BEHAVIOR_ONTOLOGY`), never from attack labels or raw numbers. Only techniques with a network-observable tactic and a similarity of at least 0.15 are returned; with no clear behaviour the stage abstains. Held-out evaluation (`python backend/evaluate_retrieval.py`, details in `docs/mitre-retrieval-evaluation.md`): a relevant technique for 19 of 53 flagged samples, i.e. 19 of 30 Mirai floods and 0 of 23 web attacks. `model_training/outputs/Deep_SVDD_MITRE_mapping_for_LLM.json` separately contains project category candidates.

The 5.9 MB index is small enough to include and is required for behavior retrieval. It can be rebuilt from the included `LLM/data/mitre_enterprise_techniques.json` by running from `LLM`:

```powershell
..\backend\.venv\Scripts\python.exe -m pip install scikit-learn==1.7.2
..\backend\.venv\Scripts\python.exe .\build_mitre_rag.py
```

The script writes `LLM/rag_store/mitre_tfidf.pkl`; execute it only against trusted source data because Python pickle loading can execute code. Similarity is a ranking score, not probability or confirmation.

## F. Stage 5 — LLM reasoning

`backend/services/llm_service.py` sends a structured JSON object containing detector status/score/threshold, top feature evidence, optional category context, and retrieved candidates to OpenAI Chat Completions. `OPENAI_API_KEY` is required only for provider mode. `OPENAI_MODEL` is optional and defaults to `gpt-4o-mini`. There is no `.env` required and no key belongs in Git. Without the key, or if the call fails, a clearly named `Demo Fallback Reasoning` record is returned. The local `LLM/local_llm_soc_assistant.py` is separate and is not used by the backend path; its requirements/configuration must be inspected before choosing it.

## G. Stage 6 — verification

The live entry point is `backend/services/verification_service.py:verify_record`. Inputs are detector analysis, MITRE result, and reasoning record. Outputs include verdict (`VERIFIED`, `UNCERTAIN`, `REJECTED`), equal-weight check pass fraction, per-check details, and failed check names. Critical failures reject; any failed check makes the record uncertain. It runs 11 checks: required report fields, the cited score, negation-aware agreement with the detector decision ("not anomalous" for a flagged flow fails), numeric claims grounded in the evidence (coarse roundings must stay within 10% of the real value), XAI feature names and direction, retrieved technique IDs only, no invented IPs/MACs/dates/ports/URLs, no unhedged certainty claims ("this confirms..."), no claim that a response was already carried out, and a limitations statement. When the reasoning is the built-in template (no API key) the result carries `reasoning_mode` and a `limitation` saying the check only confirms internal consistency, because code wrote the report from the same evidence. Tests: `backend/tests/test_verification_service.py` (damaged reports must fail, honest hedged reports must pass) and `test_pipeline_adapters.py`.

Standalone `ZeroShield_Verification_Handoff/verify_soc_report.py` verifies saved Markdown report content against its evidence record; run its tests from that directory. Its fixtures include explicit caveats that the sample threshold is not verified for that checkpoint. Do not conflate this separate report verifier with the live structured adapter.

## H. Stage 7 — contextual bandit

`backend/services/bandit_service.py` implements LinUCB (alpha 0.25; deterministic initial seed) using seven context fields: intercept, normalized score, detected flag, verification confidence, XAI evidence fraction, leading XAI magnitude, and MITRE context presence. The four actions and reward table are:

| Outcome | AUTO_REMEDIATE | CONFIRM_TO_ACT | HUMAN_REVIEW | MONITOR_ONLY |
|---|---:|---:|---:|---:|
| attack | 5 | 3 | 2 | -6 |
| benign | -12 | -3 | -2 | 1 |

This is reward table `designed-simulation-v2`. Under v1, `CONFIRM_TO_ACT` paid -2 on benign flows, so it was at least as good as `HUMAN_REVIEW` for both outcomes and review could never be the best action. v2 makes approving a containment of a benign flow cost more than plain triage (analyst time plus the risk of a wrongful block being approved), which gives every action a region: monitor below an attack probability of 0.27, human review 0.27-0.50, confirm 0.50-0.82, auto-remediate above 0.82 (`decision_regions()` in `bandit_service.py`; v1 is kept in `REWARD_TABLES` so older results stay reproducible, and every rewarded record stores its `reward_version`). These values are designed trade-offs, not measured costs; change the table deliberately and bump the version. Rejected approval has separate reward values (attack -6, benign -2). Feedback is idempotent and only updates after feedback; action context excludes labels/outcomes. The SQLite database stores alert ledger and policy state. Baselines include fixed rule, seeded random, and always human review. Replay and results are in `backend/run_response_experiment.py` and `backend/experiments/contextual_bandit_replay_v1/`. These are offline simulated rewards, not real operational utility. Automatic remediation is only a simulation and is safety-masked based on detection and verification confidence.

## I. Stages 8–10 implementation contract

Before extending, inspect the current response endpoints in `backend/main.py`, `bandit_service.py`, frontend components, and replay. Keep all actions simulated. Record action, simulation status, timestamps, approval requirement, and approval decision. Require approval for `CONFIRM_TO_ACT`; never implement destructive host/network operations.

Feedback must append observed outcome, reward/table version, and policy update version to the same alert ledger record; prevent duplicate outcome processing and updates. Keep replay reproducible, report cumulative reward and policy versions, and make simulation labels unavailable to pre-action policy context.

Use the existing React/FastAPI application for any missing dashboard functions. Required views include alert detail, score/threshold, XAI, MITRE context, LLM explanation, verification, chosen action, approval queue, ledger, reward analytics, learning curve, and baseline comparisons. Do not reconstruct the trace from unrelated UI state.

## J. Single alert record contract

The existing record is assembled in `backend/main.py:analyze` and persisted by the bandit service. Existing names include `alert_id`, `sample_id`, `timestamp`, `anomaly_score`, `threshold`, `detected`, `xai_evidence`, `retrieval_query`, `retrieved_techniques`, `llm_explanation`, `verification_result`, `verification_checks`, `verification_confidence`, `context_features`, `policy_version`, `action_scores`, `selected_action`, `approval_status`, `simulated_execution_status`, `observed_outcome`, `reward`, and `policy_update_status`. Preserve them. A compatible extension may add explicit schema/policy/reward version fields and cumulative replay metadata, but each stage must append to this logical alert record.

## K. Development rules

Inspect the clone before editing; reuse existing Stages 1–7 and implement only real gaps. Preserve detector, architecture, feature order, preprocessing assumptions, checkpoint threshold, and stage order. Do not retrain; fabricate no scores, explanations, techniques, or results; do not load the giant datasets into RAM. Use small fixtures and reproducible alert replay. Keep actions simulated and credentials out of Git. Run existing integration tests after changes. Update docs and report changed files, tests, limitations, and launch commands.

## L. Teammate startup

Prerequisites are Python 3.11, Git, and Node.js/npm. From the clone root in PowerShell:

```powershell
py -3.11 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install --upgrade pip
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Set-Location .\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload
```

In another window:

```powershell
Set-Location <clone-path>\frontend
npm ci
npm run dev
```

Open `http://localhost:5173` and check `http://localhost:8000/api/health`. Optional API reasoning uses session-only `$env:OPENAI_API_KEY` and `$env:OPENAI_MODEL`; no local LLM is required for the app. The active checkpoint, feature list, MITRE map, technique source/index, and demo cache must be present as identified in Sections C–E. Large CSV datasets are excluded and unnecessary for app startup. Their authorized source and scaler artifact are not documented/found; coordinate access and prepared-input transformation separately. Do not invent a download URL.

### Clean clone inventory

| Class | Items and handling |
|---|---|
| Required source | Backend, frontend, detector/XAI, MITRE builder, verifier, scripts; commit these. |
| Required small config/data | Feature list, mapping JSON, MITRE technique JSON, demo cache, package manifests/lockfile, backend requirements; commit these. |
| Runtime model/index | 60 KB active Deep SVDD checkpoint and 5.9 MB TF-IDF pickle index; include both or rebuild index as above. |
| Optional large datasets | Three root CSV datasets (about 0.6 GB, 22 GB, and 2 GB locally); excluded, not needed to launch. |
| Regenerable/local state | SQLite ledger, Python venv, `node_modules`, frontend `dist`, logs and caches; exclude. |
| Secrets | `OPENAI_API_KEY`; local session/environment only, optional. |
| Research outputs | Keep concise reports and figures where suitable; exclude large intermediate arrays and alternative model checkpoints. |

## M. Changes made after the repository audit

Ledger and policy (`bandit_service.py`, schema `alert-record-v2`):
- Outcome, reward and policy update commit in one SQLite transaction. A failure rolls everything back and the same feedback can be retried. Records stranded by the old two-step write (reward saved, policy never updated) are repaired by the next feedback call.
- Every record keeps an append-only `history` (`ACTION_SELECTED`, `APPROVAL_DECIDED`, `OUTCOME_OBSERVED`, `POLICY_UPDATED`) with timestamps, plus `schema_version`, `reward_version`, `feedback_at`, `policy_updated_at`, `policy_state_at_selection` and `updated_policy_version`.
- Approval and feedback take the write lock first, so two simultaneous clicks cannot both succeed.
- A rejected approval on a benign flow costs -2 (analyst time only); an approved one costs -3.

Replay (`run_response_experiment.py`): only detector-flagged flows are routed; k-fold evaluation (each alert evaluated once by a policy that did not train on it); reward version taken from the code; `--stream`, `--folds`, `--seed`; logic is importable without PyTorch and tested in `tests/test_replay_experiment.py`. Previous run folders were removed: they scored flows the detector never flagged. Regenerate with `python run_response_experiment.py`.

API and dependencies: strict request schemas, operator access control, checkpoint loaded with `weights_only=True`, SHA-256 check before the TF-IDF index is unpickled, `/api/health` no longer invents a threshold, frontend dependencies updated to 0 audit vulnerabilities with React packages pinned.

## N. Open work for Stages 8-10

- **Replay stream.** The ten-row demo cache routes five flows and all five are attacks, so it contains no false positive and cannot separate the policies (on it the bandit and the fixed rule both average +5.0). Build a larger prepared stream from the local datasets that includes benign flows the detector flags, run it with `--stream` over several seeds, and report the mean and spread. Test how sensitive the policy comparison is to the reward values.
- **Dashboard.** Add an alert-detail view that loads the stored record (`BanditService.get_alert` exists; there is no endpoint yet), a timeline from `history`, policy and reward versions, a live cumulative-reward chart from the ledger, and a paginated ledger (it shows the 15 newest plus a raw JSON drawer).
- **Verification on a real LLM.** In the app, verification has only run on template reasoning. Run it on OpenAI or local-LLM output and report the verified/uncertain/rejected shares.
- **Legacy `LLM/` outputs** pair the old checkpoint with the tuned threshold (see `LLM/README.md`). Regenerating them needs the datasets.
- **No scaler artifact**, so the app and any live-capture demo accept only already-scaled rows.
