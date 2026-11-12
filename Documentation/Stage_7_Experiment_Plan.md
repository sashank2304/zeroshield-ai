# Stage 7 integration plan and experiment contract

## Audit and integration plan

- Preserve the existing FastAPI + React/Vite application and its single-sample flow.
- Reuse `LLM/rag_store/mitre_tfidf.pkl` and `LLM/data/mitre_enterprise_techniques.json` through a per-alert behavior query; keep the original category mapping implementation intact for existing reports.
- Reuse the Stage 5 evidence-bound structured response service and its explicitly labeled fallback.
- Adapt the Stage 6 handoff's critical-failure/evidence-check contract to the API's structured JSON output. Keep the handoff verifier and its tests unchanged.
- Add a small LinUCB policy, SQLite ledger, simulator feedback route, and dashboard response/ledger panel.
- Keep the detector checkpoint, architecture, 43 ordered inputs, feature perturbation XAI, threshold, and pipeline order unchanged.

## Context schema

Version `context-v1` uses, in order: intercept; anomaly score divided by the existing threshold and clipped to `[0,5]`; detector status; Stage 6 confidence; number of XAI features divided by ten and clipped to `[0,1]`; maximum absolute supplied XAI contribution clipped to `[0,5]`; whether ATT&CK context was retrieved. No asset criticality or analyst history is present or synthesized. Labels, rewards, outcomes, and post-action data are excluded (tested: the context is identical when label-like fields are added).

## Algorithm

LinUCB uses one regularized linear reward estimate per action, identity initialization, and `alpha=0.25` for controlled exploration. The maximum upper-confidence score selects the action, with fixed action-order tie breaking. The update is `A[a] += x xT`, `b[a] += reward*x`; it is committed in the same SQLite transaction as the outcome, so a failure leaves nothing half-updated and the feedback can be retried. Greedy selection is available for held-out evaluation. Seed: `20261020`. `AUTO_REMEDIATE` is masked unless the flow was detected and verification confidence is at least 0.85.

## Designed simulation reward table

Version `designed-simulation-v2` (default):

| Outcome | AUTO_REMEDIATE | CONFIRM_TO_ACT | HUMAN_REVIEW | MONITOR_ONLY |
|---|---:|---:|---:|---:|
| Attack | +5 | +3 | +2 | -6 |
| Benign | -12 | -3 | -2 | +1 |

Expected reward is `p*attack + (1-p)*benign`, where `p` is the probability the flow is an attack. Solving it gives the best action by `p`: monitor below 0.27, human review 0.27-0.50, confirm 0.50-0.82, auto-remediate above 0.82. Each action is best somewhere, so the policy has four meaningful choices.

Why v2 exists: under v1 (benign confirm = -2) `CONFIRM_TO_ACT` was at least as good as `HUMAN_REVIEW` for both outcomes, so review was never the best action and the four-action design collapsed to three. v2 charges -3 for approving a containment of a benign flow (analyst time, as in plain triage, plus the risk of a wrongful block being approved). v1 is kept in `REWARD_TABLES` for reproducing earlier results, and every rewarded ledger record stores its `reward_version`.

If a reviewer rejects a `CONFIRM_TO_ACT`, no containment occurred: an attack stays uncontained (-6) and a benign flow costs only the approval time (-2).

These values are designed trade-offs, not measured operational costs. They do not demonstrate real-world optimality, and sensitivity to alternate values has not yet been evaluated.

## Baselines and evaluation

Baselines: a deterministic fixed rule (verified + detected + confidence at least 0.9 => AUTO_REMEDIATE; detected => HUMAN_REVIEW; otherwise MONITOR_ONLY), always HUMAN_REVIEW, and seeded random choice.

Design (`backend/run_response_experiment.py`):
- Only flows the detector flagged are routed to the policy. A flow below the threshold never becomes an alert, so no policy acts on it; unflagged flows are counted separately.
- Routed alerts are split into k folds. For each fold a fresh policy trains with exploration and feedback on the other folds, then is evaluated greedily on the held-out fold. Every routed alert is evaluated exactly once, by a policy that did not train on it. All policies see the same held-out alerts and the same reward table.
- The simulator label is read after an action is chosen, only to compute reward, and is never in the policy's context. A `CONFIRM_TO_ACT` is assumed approved offline so a reward can be observed; live dashboard approvals remain pending until a reviewer decides.
- Runs are reproducible for a given seed and stream (tested).

## Current results and limitations

- **No comparison is claimed yet.** On the ten cached samples the tuned checkpoint flags five (Mirai x3, XSS, Backdoor_Malware), all attacks. The benign row and four other Web-Based attacks fall below the threshold. With only attacks routed, auto-remediate is always right, so the bandit and the fixed rule both average +5.0 per alert (always-review +2.0, random -0.4). There is no false positive to handle, so this cache cannot separate the policies.
- An earlier run (removed) scored flows the detector never flagged and reported the bandit well ahead of the fixed rule. That gap came from those unrouted flows and is not evidence.
- A meaningful comparison needs a larger prepared stream that includes benign flows the detector flags; run it with `--stream`.
- The checked-in cache has one prepared row per category and only offers Mirai, Web-Based and benign samples.
- Dataset family labels are used only as offline simulator outcomes after selection; user-input alerts need separately supplied outcome feedback.
- Approval state is local demo state; all actions are simulated and no network or host controls are invoked.
- In the demo and replay, reasoning is the built-in template, so Stage 6 only confirms internal consistency. The dashboard shows this.
- This build does not establish detection improvement. Deep SVDD is unchanged.
