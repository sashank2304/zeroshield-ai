# ZeroShield AI - Local LLM SOC Assistant

- **Generated:** 2026-10-17T06:34:07.143520+05:30
- **Local model:** `llama3.2:3b`
- **Execution:** Local Ollama inference
- **API key used:** No
- **Dataset label:** `Mirai-greeth_flood`
- **Detector prediction:** `False`
- **MITRE candidates supplied:** 5

---

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

---

## Evidence and Provenance Notes

- This report was generated from the existing XAI report and MITRE retrieval results.
- MITRE mappings are candidates, not confirmation of an attack technique.
- TF-IDF similarity is not an attack probability.
- XAI perturbation contributions are not causal proof.
- The detector threshold verification status is preserved from the input report.
- Verify interpretations against the underlying evidence before using them operationally.
