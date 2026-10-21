# ZeroShield AI — MITRE ATT&CK Mapping Report

## Purpose
This report is the contextualization layer after Deep SVDD anomaly detection and feature-level XAI. ATT&CK mapping is not a second detector. The mapping is driven by the documented CIC-IoT-2023 attack semantics and the observed model evidence.

XAI summary input: `./model_training\outputs\Deep_SVDD_XAI_category_summary.csv`

## Mapping table

| Attack class | Status | Technique | Tactic | Confidence | Top XAI features |
|---|---|---|---|---|---|
| BenignTraffic | no_mapping | None | — | — | IAT, Number, Weight, TCP, ack_flag_number, UDP, HTTP, ack_count, psh_flag_number, Radius |
| Backdoor_Malware | conditional | T1505.003 — Server Software Component: Web Shell | Persistence | medium | IAT, Number, Weight, TCP, ack_flag_number, SYNRatio, HTTPS, RSTRatio, UDP, Std |
| BrowserHijacking | candidate_requires_host_evidence | T1185 — Browser Session Hijacking | Collection | low | Weight, IAT, Number, syn_count, TCP, HTTPS, UDP, ack_flag_number, psh_flag_number, urg_count |
| BrowserHijacking | candidate_requires_host_evidence | T1189 — Drive-by Compromise | Initial Access | low | Weight, IAT, Number, syn_count, TCP, HTTPS, UDP, ack_flag_number, psh_flag_number, urg_count |
| CommandInjection | contextual | T1190 — Exploit Public-Facing Application | Initial Access | high | IAT, TCP, HTTPS, flow_duration, ack_flag_number, RSTRatio, UDP, Radius, HTTP, AVG |
| CommandInjection | candidate_requires_execution_evidence | T1059 — Command and Scripting Interpreter | Execution | medium | IAT, TCP, HTTPS, flow_duration, ack_flag_number, RSTRatio, UDP, Radius, HTTP, AVG |
| Mirai-greeth_flood | contextual | T1498.001 — Network Denial of Service: Direct Network Flood | Impact | high | Protocol Type, PacketDensity, HTTPS, ack_flag_number, RSTRatio, SYNRatio, syn_count, psh_flag_number, Std, ACKRatio |
| Mirai-greip_flood | contextual | T1498.001 — Network Denial of Service: Direct Network Flood | Impact | high | Protocol Type, HTTPS, ack_flag_number, PacketDensity, RSTRatio, SYNRatio, syn_count, psh_flag_number, Std, ACKRatio |
| Mirai-udpplain | contextual | T1498.001 — Network Denial of Service: Direct Network Flood | Impact | high | PacketDensity, Radius, Duration, Header_Length, ACKRatio, urg_count, SYN_ACK_Ratio, rst_count, flow_duration, Rate |
| SqlInjection | contextual | T1190 — Exploit Public-Facing Application | Initial Access | high | Weight, Number, flow_duration, HTTPS, TCP, IAT, UDP, RSTRatio, Std, AVG |
| Uploading_Attack | contextual | T1190 — Exploit Public-Facing Application | Initial Access | high | HTTP, ack_flag_number, rst_count, Std, TCP, syn_count, AVG, flow_duration, UDP, ack_count |
| Uploading_Attack | candidate | T1105 — Ingress Tool Transfer | Command and Control | medium | HTTP, ack_flag_number, rst_count, Std, TCP, syn_count, AVG, flow_duration, UDP, ack_count |
| XSS | contextual | T1189 — Drive-by Compromise | Initial Access | medium | AVG, flow_duration, rst_count, TCP, Radius, Header_Length, Std, ack_count, SYNRatio, syn_count |

## Important qualifications

* `contextual` means the known dataset attack semantics support the ATT&CK context, while the flow model still does the anomaly detection.
* `conditional` / `candidate` mappings require the listed host, browser, web-server, or application evidence before they should be reported as confirmed techniques.
* Do not turn a feature such as `IAT`, `PacketDensity`, `SYNRatio`, `HTTP`, or `HTTPS` into an ATT&CK technique by itself. Those features support the anomaly explanation; the attack label/behavior provides the contextual link.

## Official sources

CIC-IoT-2023: https://www.unb.ca/cic/datasets/iotdataset-2023.html
MITRE ATT&CK Enterprise techniques: https://attack.mitre.org/techniques/

## Implementation rule for the next LLM stage

The LLM receives the observed attack label (when known), anomaly score/threshold, top XAI features, and this mapping table. It may explain or summarize only these supplied candidates. It must not invent a new ATT&CK technique if the supplied evidence is insufficient.