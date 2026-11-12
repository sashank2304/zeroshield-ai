# Repository audit: problems found and what was done

Each row names the problem, the fix, and the test that now guards it. Run the backend suite from `backend/` with `python -m unittest discover -s tests` (84 tests) or `python -m pytest tests`.

| # | Problem | Fix | Test |
|---|---|---|---|
| 1 | A failure after the reward was saved left the alert stuck in `WAITING_FOR_UPDATE`. Retries did nothing, so the policy never learned from it. | Outcome, reward and policy update commit in one transaction. A failure rolls back and can be retried. Stranded records are repaired on the next feedback call. | `test_ledger_integrity`: failed update retried; stranded record repaired |
| 2 | Ledger records had no reward version, feedback time or audit trail. | Schema `alert-record-v2`: append-only `history`, `reward_version`, `schema_version`, `feedback_at`, `policy_updated_at`, policy state at selection. | `test_history_versions_and_timestamps_are_recorded` |
| 3 | Two quick approvals or feedback calls could both pass the status check. | Write lock taken before reading (`BEGIN IMMEDIATE`). | `test_concurrent_approvals_only_one_wins`, `test_concurrent_feedback_updates_policy_once` |
| 4 | `HUMAN_REVIEW` could never be the best action: `CONFIRM_TO_ACT` was at least as good for both outcomes. | Reward table `designed-simulation-v2` (approving a containment of a benign flow costs -3). v1 kept for reproducibility. `decision_regions()` shows where each action wins. | `test_v1_table_never_prefers_human_review`, `test_v2_table_gives_every_action_a_region` |
| 5 | The replay fed the policy flows the detector never flagged, used one 5-and-5 split, and hard-coded the reward version. | Only flagged flows are routed. K-fold, each alert evaluated once. Version read from code. `--stream/--folds/--seed`. Stale runs removed. | `test_replay_experiment` (9 tests) |
| 6 | The dashboard's verifier passed "not anomalous" for a flagged flow, "this confirms an attack", "the host was isolated", and made-up values like 0.1. | Four added checks (negation-aware status, certainty claims, completed actions, invented specifics), a tighter number match, and a `limitation` note when reasoning is the built-in template. | `test_verification_service`: 23 tests, 14 of which fail on the old verifier |
| 7 | Feedback used `bool(payload[...])`, so the string `"false"` became an attack label. Payloads were unvalidated. | Pydantic schemas with a strict boolean and an approval enum. | `test_response_api` |
| 8 | Approve, feedback, policy reset and replay start were open to anyone who could reach the port. | `backend/security.py`: localhost only, or `X-Admin-Token` when `ZEROSHIELD_ADMIN_TOKEN` is set. | `test_remote_clients_cannot_change_state`, `test_admin_token_is_required_and_enables_remote_use` |
| 9 | The checkpoint was loaded with `weights_only=False` (can run code). | `weights_only=True`. The checkpoint only references classes that mode allows. | Not exercised here (needs PyTorch): run the app once to confirm it loads |
| 10 | The pickled TF-IDF index was loaded with no integrity check. | SHA-256 sidecar checked before unpickling. `build_mitre_rag.py` writes it. | `IndexIntegrityTests` |
| 11 | `/api/health` reported a made-up threshold when the model failed to load. | Reports `null`. | `test_degraded_health_does_not_invent_a_threshold` |
| 12 | The README's fallback launch command failed (`No module named 'services'`). | Corrected to `uvicorn main:app --app-dir backend`. | Reproduced the failure before the fix |
| 13 | `LLM/` scripts pair the old checkpoint with the tuned model's threshold, so their outputs (including "Mirai missed") say nothing about the app's detector. | Warning banners, a notice in `LLM/README.md`, corrected handoff README. Outputs are not regenerated (needs the datasets). | n/a |
| 14 | Several conflicting threshold files. | `docs/thresholds.md` says which is used where. | n/a |
| 15 | Frontend: 2 high `npm audit` advisories; React packages set to `latest`. | Dependencies updated (0 vulnerabilities), React packages pinned, replay caption corrected, template-reasoning note shown. | Clean `npm ci` + `npm run build` |
| 16 | **MITRE retrieval returned the same irrelevant techniques for every alert** (0 relevant in 180 samples; 159 got the same persistence technique). | Behaviour-phrase query, network-observable tactics only, similarity floor, abstention, reliability note. Held-out: 19 of 53 flagged samples (19 of 30 floods, 0 of 23 web attacks). | `test_mitre_retrieval` (9 tests), `evaluate_retrieval.py`, `docs/mitre-retrieval-evaluation.md` |
| 17 | Duplicate and stale files. | Removed 9 stale replay runs, 4 root copies of the MITRE mapping, 2 duplicate Word documents. Architecture diagram moved to `Documentation/`. | n/a |

## Not fixed (needs data or a decision)

- **Retrieval for web attacks.** Flow statistics do not contain the evidence, so retrieval is correct for floods and not for application-layer attacks (precision 51% when it answers). Better evidence is needed, not better wording.
- **Replay evidence.** The ten-row cache gives five flagged flows, all attacks, so no policy comparison is possible yet. A larger prepared stream with benign false positives is needed (see `README_FOR_STAGES_8_10.md`, section N).
- **Reward values** are designed, not measured. v2 is a proposal; change it deliberately and bump the version.
- **Verification on real LLM output** has not been measured; in the app it has only run on template reasoning.
- **Detector input** still has no scaler artifact, so only already-scaled rows can be analysed.
- **Same-split check** between the SVDD (31.2M validation rows) and autoencoder (43.1M) evaluations (`docs/thresholds.md`).
