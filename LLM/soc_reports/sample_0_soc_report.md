# ZeroShield AI — SOC Evidence Report

## 1. Sample overview

- **Sample index:** 0
- **Dataset label:** Mirai-greeth_flood
- **Binary label:** 1
- **Anomaly score:** 0.021153
- **Current threshold:** 0.061419
- **Predicted anomaly:** False
- **Threshold verified for checkpoint:** False

### Interpretation

The current detector did **not** flag this sample as anomalous. The dataset label is shown separately from the model prediction.

## 2. Explainable AI evidence

Feature values are standardized values, not raw packet counts or raw network measurements.

| Rank | Feature | Standardized value | Perturbed score | Contribution |
|---:|---|---:|---:|---:|
| 1 | Protocol Type | 17.485764 | 0.001773 | 0.019380 |
| 2 | ack_flag_number | -2.360826 | 0.012376 | 0.008777 |
| 3 | TCP | -2.478800 | 0.013859 | 0.007295 |
| 4 | Header_Length | -0.761199 | 0.016941 | 0.004212 |
| 5 | psh_flag_number | -0.134757 | 0.018706 | 0.002448 |
| 6 | SYNRatio | -0.881435 | 0.019167 | 0.001986 |
| 7 | HTTP | -0.200159 | 0.019713 | 0.001441 |
| 8 | UDP | -0.270352 | 0.020152 | 0.001001 |
| 9 | urg_count | -0.704494 | 0.020499 | 0.000654 |
| 10 | RSTRatio | -0.746813 | 0.020609 | 0.000544 |

A positive contribution means replacing the feature with zero reduced the anomaly score for this sample. This is a local perturbation explanation, not proof of causation.

## 3. MITRE ATT&CK retrieval candidates

Candidates below were retrieved using behavior-focused TF-IDF search. Similarity scores are not probabilities and do not confirm that a technique occurred.

| Rank | Technique ID | Technique | TF-IDF similarity | Reference |
|---:|---|---|---:|---|
| 1 | T1498.001 | Direct Network Flood | 0.365956 | [Official ATT&CK page](https://attack.mitre.org/techniques/T1498/001) |
| 2 | T1499 | Endpoint Denial of Service | 0.163066 | [Official ATT&CK page](https://attack.mitre.org/techniques/T1499) |
| 3 | T1498.002 | Reflection Amplification | 0.136123 | [Official ATT&CK page](https://attack.mitre.org/techniques/T1498/002) |
| 4 | T1110.001 | Password Guessing | 0.132707 | [Official ATT&CK page](https://attack.mitre.org/techniques/T1110/001) |
| 5 | T1584.005 | Botnet | 0.124071 | [Official ATT&CK page](https://attack.mitre.org/techniques/T1584/005) |

## 4. Analyst notes and limitations

- MITRE matches are candidates, not confirmed classifications.
- TF-IDF similarity is a text-retrieval score, not an attack probability.
- Feature contributions indicate local model sensitivity, not causation.
- The checkpoint threshold has not been verified unless explicitly confirmed by the XAI summary.
- Review other telemetry and independent evidence before making an incident decision.

## 5. Source files

- `xai_results/xai_top_features.csv`
- `xai_results/xai_summary.json`
- `data/mitre_retrieval_comparison_v2.json`
