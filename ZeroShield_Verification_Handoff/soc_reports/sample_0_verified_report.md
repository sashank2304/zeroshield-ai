# ZeroShield AI - Verified SOC Report

- **Verdict:** VERIFIED
- **Verification confidence:** 1.0
- **Reason:** every checkable claim matches the evidence
- **Alert status:** NO_ALERT (detector did not flag this sample)
- **Verified at:** 2026-10-17T09:18:25.123765+00:00
- **Original report SHA-256:** `db486610af2690a2f1ee824e176f17d31d39b9dffb504b484f1415011be95643`

## Check results

| Check | Level | Result | Detail |
|---|---|---|---|
| inputs_consistent | major | PASS | evidence file matches the LLM record |
| required_sections | major | PASS | all 6 sections present |
| status_fields_present | major | PASS | score, threshold and prediction stated |
| status_values_correct | critical | PASS | 9/9 correct |
| threshold_comparison_correct | critical | PASS | 1/1 correct |
| detection_claims_consistent | critical | PASS | no claim contradicts the detector prediction |
| technique_ids_valid | critical | PASS | 5 technique ID(s), all supplied |
| technique_details_correct | major | PASS | 5/5 correct |
| urls_grounded | critical | PASS | no invented URLs |
| feature_contributions_correct | major | PASS | 10/10 correct |
| no_unsupported_feature_mentions | major | PASS | every named feature is in the XAI evidence |
| direction_claims_consistent | major | PASS | direction statements match the contribution signs |
| no_ungrounded_numbers | major | PASS | 22 number(s), all in the evidence |
| no_invented_specifics | critical | PASS | no invented IPs, MACs, dates or ports |
| no_certainty_overclaims | major | PASS | no unhedged confirmation or proof claims |
| no_completed_action_claims | major | PASS | no claim that a response was carried out |
| required_caveats_present | major | PASS | all required caveats present |

---

## Original LLM report (unmodified)

# ZeroShield AI - LLM-Assisted SOC Report
## 1. Executive Summary
A sample from the "Mirai-greeth_flood" dataset was analyzed using ZeroShield AI's XAI report. The analysis revealed a low anomaly score of 0.021153250709176, which is below the threshold of 0.06141921. The detector prediction is false.

## 2. Sample and Detector Status
* Sample Index: 0
* Dataset Label: Mirai-greeth_flood
* Binary Label: 1
* Anomaly Score: 0.021153250709176
* Threshold: 0.06141921
* Detector Prediction: False

## 3. Explainable AI Evidence
The XAI report provides evidence of local perturbation sensitivity by replacing one standardized feature with zero. The contributions for each feature are:
* Protocol Type: 0.0193799929693341
* ack_flag_number: 0.0087769366800785
* TCP: 0.00729466881603
* Header_Length: 0.0042119622230529
* psh_flag_number: 0.0024477243423461
* SYNRatio: 0.0019859988242387
* HTTP: 0.0014407467097043
* UDP: 0.0010012052953243
* urg_count: 0.0006544813513755
* RSTRatio: 0.0005441978573799

These contributions indicate that replacing these features with zero reduces the anomaly score, but does not prove causation.

## 4. MITRE ATT&CK Candidate Mapping
The MITRE ATT&CK retrieval method used is TF-IDF similarity. The top 5 MITRE candidates are:
* T1498.001: Direct Network Flood (similarity: 0.365956)
* T1499: Endpoint Denial of Service (similarity: 0.163066)
* T1498.002: Reflection Amplification (similarity: 0.136123)
* T1110.001: Password Guessing (similarity: 0.132707)
* T1584.005: Botnet (similarity: 0.124071)

These candidates are retrieval suggestions, not confirmed attack classifications.

## 5. Recommended Analyst Validation
Further validation is required to confirm the attack classification and determine the likelihood of a successful attack.

## 6. Limitations
This report is based on the provided XAI report and MITRE ATT&CK retrieval candidates. ZeroShield AI has not inspected live traffic, endpoints, or external systems. The detector threshold is unverified for this checkpoint.
