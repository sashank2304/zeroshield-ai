# Stage 4 (MITRE ATT&CK retrieval): what was wrong, what changed, what was measured

## The problem

The query sent to the TF-IDF index was feature names and numbers ("syn_count value -1.0099, perturbation contribution 2.326; ..."). ATT&CK descriptions are plain language ("high volume of packets", "SYN flood", "scanning"), so the two share almost no words. Measured on the 180 labelled samples of the tuned model's XAI evidence, the old query returned a relevant technique for **0 of 180** (0 of 100 flagged) samples, and **159 of 180** alerts got the same top result, T1547.014 Active Setup (a persistence technique that cannot be seen in network flow data).

## What changed (`backend/services/mitre_service.py`)

- **Behaviour query.** A hand-written map (`BEHAVIOR_ONTOLOGY`) turns each feature that is clearly outside the benign range (scaled value at or beyond +/-1) and pushes the score up into the behaviour it signals, for example a very high `PacketDensity` into "very high packet density in a very short time high volume flood denial of service". It is written from networking knowledge, not fitted to labels, and the query never contains a dataset label.
- **Network-observable tactics only.** Flow statistics cannot support persistence, privilege-escalation, defense-evasion or execution techniques, so only techniques with a network-observable tactic can be returned.
- **Similarity floor of 0.15** and **abstention.** If no feature is clearly outside the benign range, or nothing clears the floor, the result says so instead of returning irrelevant techniques.
- A protocol that is absent adds nothing (absence of HTTPS is not attack behaviour).
- Every result carries the evidence features behind the query and a reliability note.

## How it was measured (`python backend/evaluate_retrieval.py`)

Data: `model_training/outputs/Deep_SVDD_XAI_per_sample.csv` (180 samples of nine labelled attack families, top-10 XAI features each). The label is used only to score an answer. A technique counts as relevant if it is in the expected family (Mirai floods: T1498, T1499, T1584; web attacks: T1190, T1059, T1505, T1185, T1189). That is generous but still label-based.

The samples were split by position within each family. The similarity floor was chosen on the tuning half; the other half is the reported result. The ontology was revised once after looking at the results (dropping absence phrases for protocols), so the held-out half was inspected twice. The change left the tuning-half relevant count unchanged (18).

## Result (held-out half, detector-flagged samples, n = 53)

| | Before | After |
|---|---:|---:|
| Relevant technique in the top 5 | 0 of 100 (both halves) | **19 of 53 (36%)** |
| Answers given | all | 37 (the rest abstain) |
| Precision when it answers | 0% | 51% |

| Family (flagged, held-out half) | Flagged | Answered | Relevant |
|---|---:|---:|---:|
| Mirai-greeth_flood | 10 | 10 | 9 |
| Mirai-greip_flood | 10 | 10 | 9 |
| Mirai-udpplain | 10 | 4 | 1 |
| **All Mirai floods** | **30** | **24** | **19** |
| Web attacks (SQL injection, XSS, command injection, backdoor, uploading, browser hijacking) | 23 | 13 | **0** |

## How to read this honestly

- Flow statistics can point to **network-level techniques** such as floods (19 of 30). They **cannot identify application-layer attacks**: for web attacks the evidence does not contain the information, and all 13 answers given were wrong. Treat candidates for such flows as low confidence. The dashboard and the LLM context carry this reliability note.
- The five candidates score close together. Roughly half of the flagged alerts get Credential Stuffing (T1110.004) as the first candidate, even for floods whose relevant technique is lower in the list. Do not read the order as a ranking.
- Some "wrong" answers are faithful to the evidence: an XSS flow whose statistics look like a flood is given flood techniques and scored wrong, because the label says XSS.
- Better retrieval for web attacks needs better evidence (application-layer features), not better wording.
