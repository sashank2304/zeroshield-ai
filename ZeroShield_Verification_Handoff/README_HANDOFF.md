# ZeroShield AI - Verification Stage Handoff

## Where this fits

    Detector -> XAI evidence -> MITRE retrieval -> LLM report -> VERIFICATION (done) -> Response agent (next)

This package contains the finished verification stage. It reads the LLM's Markdown
SOC report and checks every claim it can against the evidence the LLM was given.

## Files

| File | Purpose |
|---|---|
| `verify_soc_report.py` | The verifier (standard library only, Python 3.10+) |
| `test_verify_soc_report.py` | Tests: the real sample must pass, 21 deliberately damaged copies must be caught |
| `soc_reports/sample_0_soc_report.json` | Evidence given to the LLM (detector score, XAI features, MITRE candidates) |
| `soc_reports/sample_0_llm_soc_report.json` | The LLM's report (Markdown inside `llm_report_markdown`) |

## Run

From this folder:

    python test_verify_soc_report.py     # expect: ALL TESTS PASSED
    python verify_soc_report.py          # writes soc_reports/sample_0_verified_report.json and .md

## Output you can read (soc_reports/<name>_verified_report.json)

| Field | Meaning |
|---|---|
| `verdict` | `VERIFIED`, `UNCERTAIN` or `REJECTED` |
| `verification_confidence` | 0 to 1, weighted share of checks passed (computed by the verifier, not the LLM) |
| `alert_status` | `ALERT` or `NO_ALERT` - whether the detector actually flagged the sample |
| `failed_checks` | names of checks that did not fully pass |
| `checks` | every check with level (critical / major / minor), score and a plain-English detail |
| `original_llm_record` | the LLM output, unchanged (for audit) |
| `original_report_sha256` | hash of the original report text |

Verdict rules: any failed critical check -> REJECTED. Any failed major check -> UNCERTAIN
(REJECTED if overall confidence is below 0.5). Otherwise VERIFIED at confidence >= 0.85.
A report the verifier cannot parse fails safe to UNCERTAIN.

Detector data also available in `original_llm_record`: `anomaly_score`, `threshold`,
`predicted_anomaly`, `dataset_label`, `binary_label`, `mitre_candidates_supplied`.

## What VERIFIED does and does not mean

- It means the report is faithful to its evidence. Numbers, technique IDs, URLs and
  contributions match, nothing is invented, and the required caveats are present.
- It does NOT mean the flow is really an attack. A benign flow wrongly flagged by the
  detector can still receive a faithful report and pass.
- Never let the response agent act automatically on a report that is not VERIFIED.

## Open issues (not solved yet)

1. **Which detector is final?** The teammate handoff uses Deep SVDD with threshold 0.06141921
   (provisional, not validated). The validated Model A is an autoencoder with threshold
   0.04865155 (98.17% recall on validation). One must be chosen before any evaluation.
2. **MITRE candidates are the same for every sample.** They come from one fixed query
   (`results.behavior_focused.top_5`), not from the sample's own XAI evidence. Any flood-like
   claim therefore passes the technique check. Retrieval should be built per sample.
3. **Only one real report exists (`sample_0`).** It is a Mirai flow the detector missed
   (score 0.0212 < threshold 0.0614). A batch of reports on flagged samples is needed
   before anything can be trained or evaluated.
4. `sample_mirai_alert.json` / `sample_weak_alert.json` in the teammate handoff are demo
   placeholders with a different schema. Do not use them for evaluation.

## Suggested next task: Stage 7, contextual bandit response agent (edit if yours differs)

- Actions: auto-remediate, confirm to act, human review, monitor.
- State you can build from real data now: anomaly score relative to threshold, verification
  confidence, verdict, MITRE technique (and tactic).
- State that must be SIMULATED (the dataset has none): asset criticality, analyst feedback
  history. Say so plainly in the report.
- Reward: design a written reward table (for example, blocking benign traffic is penalised,
  missing an attack is penalised more, escalating a true attack is rewarded). Use the dataset
  labels only to compute reward in simulation, never as an input feature.
- Hard safety rule: auto-remediate is allowed only when the verdict is VERIFIED.
- Baselines to compare against: fixed if/else rules, always human review, random.
- Report what the policy learned relative to the designed reward, not real-world optimality.

## Limitations of the verifier

- Tested on one real report plus 21 damaged copies of it. The LLM is `llama3.2:3b`, so
  formatting may drift on other samples; such reports are marked UNCERTAIN, not silently passed.
- Cut-offs (0.85 verified, 0.50 uncertain) are defaults and should be reviewed on a larger batch.
