# Stage 7 integration plan and experiment contract

## Audit and integration plan

- Preserve the existing FastAPI + React/Vite application and its single-sample flow.
- Reuse `LLM/rag_store/mitre_tfidf.pkl` and `LLM/data/mitre_enterprise_techniques.json` through a per-alert behavior query; keep the original category mapping implementation intact for existing reports.
- Reuse the Stage 5 evidence-bound structured response service and its explicitly labeled fallback.
- Adapt the Stage 6 handoff's critical-failure/evidence-check contract to the API's structured JSON output. Keep the handoff verifier and its tests unchanged.
- Add a small LinUCB policy, SQLite ledger, simulator feedback route, and dashboard response/ledger panel.
- Keep the detector checkpoint, architecture, 43 ordered inputs, feature perturbation XAI, threshold, and pipeline order unchanged.

## Context schema

Version `context-v1` uses, in order: intercept; anomaly score divided by the existing threshold and clipped to `[0,5]`; detector status; Stage 6 confidence; number of XAI features divided by ten and clipped to `[0,1]`; maximum absolute supplied XAI contribution clipped to `[0,5]`; whether ATT&CK context was retrieved. No asset criticality or analyst history is present or synthesized. Labels, rewards, outcomes, and post-action data are excluded.

## Algorithm

LinUCB uses one regularized linear reward estimate per action, identity initialization, and `alpha=0.25` for controlled exploration. The maximum upper-confidence score selects the action, with fixed action-order tie breaking. The update is `A[a] += x xᵀ`, `b[a] += reward*x`; state is persisted in SQLite. Greedy selection is available for held-out evaluation. Seed: `20261020`.

## Designed simulation reward table

| Outcome | AUTO_REMEDIATE | CONFIRM_TO_ACT | HUMAN_REVIEW | MONITOR_ONLY |
|---|---:|---:|---:|---:|
| Attack | +5 | +3 | +2 | -6 |
| Benign | -12 | -2 | -2 | +1 |

Version `designed-simulation-v1`. The values encode a deliberate high penalty for benign remediation and a substantial penalty for missing an attack. They are not measured operational costs and do not demonstrate real-world optimality. Sensitivity to alternate values has not yet been evaluated.

If a reviewer rejects a `CONFIRM_TO_ACT`, the simulated attack reward is -6 because it remains uncontained; a rejected benign confirmation retains -2 because analyst approval workload was incurred.

## Baselines and evaluation

Required baselines are a deterministic fixed rule (verified + detected + confidence at least 0.9 => AUTO_REMEDIATE; detected => HUMAN_REVIEW; otherwise MONITOR_ONLY), always HUMAN_REVIEW, and seeded random choice. The same held-out alert sequence and simulator/reward table are used for every policy. Training and evaluation are separated; evaluation labels score actions only and do not update the policy. The replay artifacts are in `backend/experiments/contextual_bandit_replay_v1/run_20261020_181512/`.

The run trained on five attack-category cache rows and evaluated on the benign cache row plus four attack-category rows:

| Policy | Mean reward | Cumulative reward | Action distribution (auto, confirm, human, monitor) |
|---|---:|---:|---|
| LinUCB, greedy held-out | 2.4 | 12 | 1, 4, 0, 0 |
| Fixed rules | -2.4 | -12 | 1, 0, 0, 4 |
| Always HUMAN_REVIEW | 1.2 | 6 | 0, 0, 5, 0 |
| Seeded random | 1.0 | 5 | 1, 1, 1, 2 |

LinUCB received five feedback updates during training and no updates during the five-alert evaluation. These are descriptive results from one tiny stream and one seed, with one benign example; they do not support a general performance or statistical claim. Offline replay assumes confirmation requests are approved so the reward can be observed. Live dashboard approvals remain pending until a reviewer records a decision.

## Current limitations

- The checked-in cache has only one prepared feature row for each category. It is a ten-row demo source, not a representative performance dataset.
- Dataset family labels are used only as explicitly selected offline simulator outcomes after selection; they are not policy context. User-input alerts need separately supplied outcome feedback.
- Approval state is local demo state; all actions are simulated and no network or host controls are invoked.
- Stage 6's teammate verifier is Markdown-schema-specific. The dashboard uses a structured-record adapter with documented equal-weight checks; this requires broader adversarial fixtures before being treated as a complete replacement for the original verifier.
- This build does not establish detection improvement. Deep SVDD is unchanged.
