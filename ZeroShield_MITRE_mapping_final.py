"""
ZeroShield AI
MITRE ATT&CK Contextualization Layer

Purpose
-------
Map the known CIC-IoT-2023 attack labels used by ZeroShield AI to
MITRE ATT&CK Enterprise techniques in an evidence-driven way.

This script DOES NOT:
- train or modify Deep SVDD
- infer new attack labels
- let an LLM invent MITRE technique IDs
- treat XAI features as proof of a technique

Instead it:
1. Reads the existing Deep SVDD XAI category summary when available.
2. Applies a fixed, human-reviewed mapping table based on:
   - the documented CIC-IoT-2023 attack definition/source
   - current MITRE ATT&CK technique definitions
   - the observed XAI feature evidence
3. Marks uncertain mappings as candidate/conditional rather than confirmed.
4. Writes CSV + JSON + a report for the next LLM stage.

Expected XAI input:
    model_training/outputs/Deep_SVDD_XAI_category_summary.csv

Outputs:
    model_training/outputs/Deep_SVDD_MITRE_mapping.csv
    model_training/outputs/Deep_SVDD_MITRE_mapping.json
    model_training/outputs/Deep_SVDD_MITRE_mapping_for_LLM.json
    model_training/outputs/Deep_SVDD_MITRE_mapping_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# 1. DEFAULT PATHS
# ============================================================

DEFAULT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 2. FIXED, HUMAN-REVIEWED ATT&CK MAPPING RULES
# ============================================================
#
# Important:
# - "confirmed" below means the attack label semantics directly support
#   the ATT&CK context.
# - It does NOT mean the flow-only detector independently proved the
#   complete ATT&CK technique.
# - "candidate" / "conditional" mappings require additional endpoint,
#   browser, web-server, or application evidence.
#

MAPPING_RULES: dict[str, list[dict[str, Any]]] = {

    "Backdoor_Malware": [
        {
            "mapping_status": "conditional",
            "technique_id": "T1505.003",
            "technique_name": "Server Software Component: Web Shell",
            "tactic": "Persistence",
            "confidence": "medium",
            "reason": (
                "CIC-IoT-2023 defines Backdoor Malware as installing malware that "
                "creates a hidden entry point for later unauthorized access. "
                "Because this is a web-based attack and the dataset uses DVWA/Remot3d, "
                "T1505.003 is a reasonable contextual candidate when the observed backdoor "
                "is implemented as a web shell or web-accessible server-side script."
            ),
            "qualification": (
                "Do not mark T1505.003 as confirmed from flow features alone. "
                "Confirm with web-root/file-creation/process evidence or a documented "
                "web-shell implementation."
            ),
            "mitre_url": "https://attack.mitre.org/techniques/T1505/003/",
        }
    ],

    "BrowserHijacking": [
        {
            "mapping_status": "candidate_requires_host_evidence",
            "technique_id": "T1185",
            "technique_name": "Browser Session Hijacking",
            "tactic": "Collection",
            "confidence": "low",
            "reason": (
                "The dataset's Browser Hijacking attack modifies browser settings such as "
                "the home page, search engine, or bookmarks and redirects the user. "
                "T1185 covers browser/session hijacking, but its ATT&CK definition is "
                "more specific to session takeover, cookies, certificates, or browser pivoting."
            ),
            "qualification": (
                "Keep this as a candidate only. Do not report T1185 as confirmed unless "
                "host/browser telemetry demonstrates session hijacking behavior."
            ),
            "mitre_url": "https://attack.mitre.org/techniques/T1185/",
        },
        {
            "mapping_status": "candidate_requires_host_evidence",
            "technique_id": "T1189",
            "technique_name": "Drive-by Compromise",
            "tactic": "Initial Access",
            "confidence": "low",
            "reason": (
                "T1189 includes malicious script injection into web pages, but the dataset's "
                "Browser Hijacking label does not by itself prove a drive-by compromise."
            ),
            "qualification": (
                "Use only when browser/network evidence shows compromise through a user's "
                "visit to attacker-controlled or compromised web content."
            ),
            "mitre_url": "https://attack.mitre.org/techniques/T1189/",
        },
    ],

    "CommandInjection": [
        {
            "mapping_status": "contextual",
            "technique_id": "T1190",
            "technique_name": "Exploit Public-Facing Application",
            "tactic": "Initial Access",
            "confidence": "high",
            "reason": (
                "CIC-IoT-2023 describes Command Injection as targeting a web application "
                "by injecting commands into an input field. MITRE T1190 explicitly covers "
                "exploitation of Internet-facing web applications/services."
            ),
            "qualification": (
                "The mapping identifies the exploitation context. Flow-only data does not "
                "prove which command interpreter was executed after exploitation."
            ),
            "mitre_url": "https://attack.mitre.org/techniques/T1190/",
        },
        {
            "mapping_status": "candidate_requires_execution_evidence",
            "technique_id": "T1059",
            "technique_name": "Command and Scripting Interpreter",
            "tactic": "Execution",
            "confidence": "medium",
            "reason": (
                "The dataset states that successful Command Injection can result in "
                "arbitrary command execution. T1059 covers adversary abuse of command/script "
                "interpreters."
            ),
            "qualification": (
                "Only promote to a reported technique when endpoint/process telemetry or "
                "application logs show the interpreter used. Do not guess a sub-technique "
                "such as PowerShell, Windows Command Shell, Unix Shell, or JavaScript."
            ),
            "mitre_url": "https://attack.mitre.org/techniques/T1059/",
        },
    ],

    "Mirai-greeth_flood": [
        {
            "mapping_status": "contextual",
            "technique_id": "T1498.001",
            "technique_name": "Network Denial of Service: Direct Network Flood",
            "tactic": "Impact",
            "confidence": "high",
            "reason": (
                "CIC-IoT-2023 documents GREETH as a Mirai flood attack using packet "
                "encapsulation based on the Ethernet header. MITRE T1498.001 covers "
                "direct high-volume network floods, including attacks using many protocols."
            ),
            "qualification": (
                "The ZeroShield flow evidence supports anomaly detection of the flood "
                "behavior; ATT&CK attribution is contextual, not a replacement for the detector."
            ),
            "mitre_url": "https://attack.mitre.org/techniques/T1498/001/",
        }
    ],

    "Mirai-greip_flood": [
        {
            "mapping_status": "contextual",
            "technique_id": "T1498.001",
            "technique_name": "Network Denial of Service: Direct Network Flood",
            "tactic": "Impact",
            "confidence": "high",
            "reason": (
                "CIC-IoT-2023 documents GREIP as a Mirai flood in which GRE packets "
                "carry encapsulated traffic. MITRE T1498.001 covers direct high-volume "
                "network flooding against a target."
            ),
            "qualification": (
                "The ATT&CK mapping is made from the known attack label/behavior plus "
                "network-flow evidence; it is not inferred from one feature alone."
            ),
            "mitre_url": "https://attack.mitre.org/techniques/T1498/001/",
        }
    ],

    "Mirai-udpplain": [
        {
            "mapping_status": "contextual",
            "technique_id": "T1498.001",
            "technique_name": "Network Denial of Service: Direct Network Flood",
            "tactic": "Impact",
            "confidence": "high",
            "reason": (
                "CIC-IoT-2023 documents UDPPlain as a Mirai attack that floods the target "
                "with UDP packets. MITRE T1498.001 explicitly includes UDP as a common "
                "protocol for direct flood attacks."
            ),
            "qualification": (
                "Treat this as ATT&CK context for a known dataset attack class, not as an "
                "independent classification result from Deep SVDD."
            ),
            "mitre_url": "https://attack.mitre.org/techniques/T1498/001/",
        }
    ],

    "SqlInjection": [
        {
            "mapping_status": "contextual",
            "technique_id": "T1190",
            "technique_name": "Exploit Public-Facing Application",
            "tactic": "Initial Access",
            "confidence": "high",
            "reason": (
                "CIC-IoT-2023 defines SQL Injection as malicious SQL code injected into "
                "web-application input. MITRE T1190 explicitly discusses exploiting websites "
                "and databases and includes SQL injection in its procedure examples."
            ),
            "qualification": (
                "Flow-level XAI explains the anomalous network behavior; it does not prove "
                "the exact SQL payload or post-exploitation effect."
            ),
            "mitre_url": "https://attack.mitre.org/techniques/T1190/",
        }
    ],

    "Uploading_Attack": [
        {
            "mapping_status": "contextual",
            "technique_id": "T1190",
            "technique_name": "Exploit Public-Facing Application",
            "tactic": "Initial Access",
            "confidence": "high",
            "reason": (
                "CIC-IoT-2023 describes Uploading Attack as exploiting a web application's "
                "file-upload functionality to upload malicious files and potentially execute "
                "arbitrary code. That is consistent with exploitation of a web-facing application."
            ),
            "qualification": (
                "Report T1190 as the exploitation context. Exact application vulnerability "
                "and post-upload execution require application/host evidence."
            ),
            "mitre_url": "https://attack.mitre.org/techniques/T1190/",
        },
        {
            "mapping_status": "candidate",
            "technique_id": "T1105",
            "technique_name": "Ingress Tool Transfer",
            "tactic": "Command and Control",
            "confidence": "medium",
            "reason": (
                "The dataset describes malicious-file upload into the target environment. "
                "MITRE T1105 covers transfer of tools or files from an external system into "
                "a compromised environment."
            ),
            "qualification": (
                "Keep this as secondary context unless the observed workflow clearly shows "
                "a malicious file being transferred into an already-compromised environment."
            ),
            "mitre_url": "https://attack.mitre.org/techniques/T1105/",
        },
    ],

    "XSS": [
        {
            "mapping_status": "contextual",
            "technique_id": "T1189",
            "technique_name": "Drive-by Compromise",
            "tactic": "Initial Access",
            "confidence": "medium",
            "reason": (
                "CIC-IoT-2023 defines XSS as injecting malicious script into a web page for "
                "execution by a browser. MITRE T1189 explicitly lists cross-site scripting "
                "as a way malicious scripts or iFrames may be inserted into a compromised website."
            ),
            "qualification": (
                "Because the ZeroShield inputs are flow-level, do not claim browser execution "
                "unless browser/client telemetry confirms the follow-on behavior."
            ),
            "mitre_url": "https://attack.mitre.org/techniques/T1189/",
        }
    ],
}


DATASET_CLASS = {
    "BenignTraffic": "Benign",
    "Backdoor_Malware": "Web-based",
    "BrowserHijacking": "Web-based",
    "CommandInjection": "Web-based",
    "SqlInjection": "Web-based",
    "Uploading_Attack": "Web-based",
    "XSS": "Web-based",
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
}

DATASET_ATTACK_DEFINITIONS = {
    "Backdoor_Malware": (
        "Installing malware that creates a hidden entry point for later unauthorized access."
    ),
    "BrowserHijacking": (
        "Modifying browser settings such as the home page, default search engine, or bookmarks "
        "to redirect the user or display unwanted content."
    ),
    "CommandInjection": (
        "Injecting malicious commands into a web-application input field to gain access, steal "
        "information, or execute commands on the target."
    ),
    "Mirai-greeth_flood": (
        "Mirai GREETH flood using packet encapsulation based on the Ethernet header."
    ),
    "Mirai-greip_flood": (
        "Mirai GREIP flood using GRE packet encapsulation with internal traffic."
    ),
    "Mirai-udpplain": (
        "Mirai UDPPlain flood using UDP packets against the target."
    ),
    "SqlInjection": (
        "Injecting malicious SQL code into web-application input fields."
    ),
    "Uploading_Attack": (
        "Exploiting web-application file-upload functionality to upload malicious files."
    ),
    "XSS": (
        "Injecting malicious script into a web page for execution by a user's browser."
    ),
}


# ============================================================
# 3. HELPERS
# ============================================================

def normalize_label(value: Any) -> str:
    label = str(value).strip()
    aliases = {
        "Mirai-greeth_flood": "Mirai-greeth_flood",
        "Mirai-greip_flood": "Mirai-greip_flood",
        "Mirai-udpplain": "Mirai-udpplain",
        "SqlInjection": "SqlInjection",
        "SQLInjection": "SqlInjection",
        "SQL Injection": "SqlInjection",
    }
    return aliases.get(label, label)


def read_xai_summary(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    required = {"attack_category", "feature"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"XAI summary is missing required columns: {sorted(missing)}"
        )

    return df


def top_xai_features(
    xai_df: pd.DataFrame | None,
    category: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if xai_df is None:
        return []

    subset = xai_df[
        xai_df["attack_category"].astype(str).map(normalize_label) == category
    ].copy()

    if subset.empty:
        return []

    # Prefer the exact ranking used by the existing Deep SVDD XAI aggregation.
    sort_cols = []
    ascending = []
    for col in [
        "mean_positive_contribution",
        "mean_absolute_contribution",
        "top10_frequency",
    ]:
        if col in subset.columns:
            sort_cols.append(col)
            ascending.append(False)

    if sort_cols:
        subset = subset.sort_values(sort_cols, ascending=ascending)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for _, row in subset.iterrows():
        feature = str(row["feature"])
        if feature in seen:
            continue
        seen.add(feature)

        item: dict[str, Any] = {"feature": feature}
        for col in [
            "mean_positive_contribution",
            "mean_absolute_contribution",
            "top5_frequency",
            "top10_frequency",
            "sample_count",
            "detected_count",
        ]:
            if col in row.index and pd.notna(row[col]):
                value = row[col]
                if isinstance(value, (int, float)):
                    item[col] = float(value)
                else:
                    item[col] = value

        rows.append(item)
        if len(rows) >= top_k:
            break

    return rows


def build_records(
    categories: list[str],
    xai_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for category in categories:
        attack_class = DATASET_CLASS.get(category, "Unknown")
        xai_features = top_xai_features(xai_df, category, top_k=10)
        xai_feature_names = [item["feature"] for item in xai_features]

        rules = MAPPING_RULES.get(category, [])

        if not rules:
            records.append(
                {
                    "observed_traffic_attack_class": category,
                    "dataset_group": attack_class,
                    "mapping_status": "no_mapping",
                    "technique_id": None,
                    "technique_name": None,
                    "tactic": None,
                    "confidence": None,
                    "dataset_attack_definition": None,
                    "xai_evidence_available": bool(xai_features),
                    "top_xai_features": xai_feature_names,
                    "reason_for_mapping": "Benign traffic is not mapped to an adversary ATT&CK technique.",
                    "qualification": "Use as the baseline / no-attack case.",
                    "source_dataset": "CIC-IoT-2023",
                    "source_mitre": None,
                }
            )
            continue

        for rule in rules:
            records.append(
                {
                    "observed_traffic_attack_class": category,
                    "dataset_group": attack_class,
                    "mapping_status": rule["mapping_status"],
                    "technique_id": rule["technique_id"],
                    "technique_name": rule["technique_name"],
                    "tactic": rule["tactic"],
                    "confidence": rule["confidence"],
                    "dataset_attack_definition": DATASET_ATTACK_DEFINITIONS.get(category),
                    "xai_evidence_available": bool(xai_features),
                    "top_xai_features": xai_feature_names,
                    "reason_for_mapping": rule["reason"],
                    "qualification": rule["qualification"],
                    "source_dataset": "https://www.unb.ca/cic/datasets/iotdataset-2023.html",
                    "source_mitre": rule["mitre_url"],
                }
            )

    return records


def build_llm_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a constrained lookup table for the future LLM layer."""

    by_attack: dict[str, dict[str, Any]] = {}

    for record in records:
        attack = record["observed_traffic_attack_class"]
        entry = by_attack.setdefault(
            attack,
            {
                "attack_class": attack,
                "allowed_mitre_candidates": [],
                "top_xai_features": record["top_xai_features"],
                "important_rule": (
                    "Use only the candidates below. Do not create a new ATT&CK ID. "
                    "A conditional/candidate mapping requires the listed qualification evidence."
                ),
            },
        )

        technique = record["technique_id"]
        if technique is None:
            continue

        candidate = {
            "technique_id": technique,
            "technique_name": record["technique_name"],
            "tactic": record["tactic"],
            "mapping_status": record["mapping_status"],
            "confidence": record["confidence"],
            "reason": record["reason_for_mapping"],
            "qualification": record["qualification"],
            "source": record["source_mitre"],
        }
        entry["allowed_mitre_candidates"].append(candidate)

    return {
        "project": "ZeroShield AI",
        "purpose": "Evidence-constrained MITRE ATT&CK contextualization",
        "rules": [
            "The detector/XAI evidence comes before ATT&CK contextualization.",
            "The LLM may select only from allowed candidates supplied for the observed class.",
            "The LLM must not invent a technique ID, attack class, feature, or evidence.",
            "Candidate/conditional techniques must retain their qualification text.",
            "Flow-level XAI is supporting evidence, not proof of host/browser execution or persistence.",
        ],
        "mappings": list(by_attack.values()),
    }


def write_report(
    output_path: Path,
    records: list[dict[str, Any]],
    xai_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# ZeroShield AI — MITRE ATT&CK Mapping Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append(
        "This report is the contextualization layer after Deep SVDD anomaly detection and feature-level XAI. "
        "ATT&CK mapping is not a second detector. The mapping is driven by the documented CIC-IoT-2023 attack "
        "semantics and the observed model evidence."
    )
    lines.append("")
    lines.append(f"XAI summary input: `{xai_path}`")
    lines.append("")
    lines.append("## Mapping table")
    lines.append("")
    lines.append(
        "| Attack class | Status | Technique | Tactic | Confidence | Top XAI features |"
    )
    lines.append("|---|---|---|---|---|---|")

    # One concise line per record.
    for r in records:
        technique = (
            f"{r['technique_id']} — {r['technique_name']}"
            if r["technique_id"]
            else "None"
        )
        features = ", ".join(r["top_xai_features"][:10])
        lines.append(
            f"| {r['observed_traffic_attack_class']} | {r['mapping_status']} | {technique} | "
            f"{r['tactic'] or '—'} | {r['confidence'] or '—'} | {features or 'XAI summary not found'} |"
        )

    lines.append("")
    lines.append("## Important qualifications")
    lines.append("")
    lines.append(
        "* `contextual` means the known dataset attack semantics support the ATT&CK context, while the flow model still "
        "does the anomaly detection."
    )
    lines.append(
        "* `conditional` / `candidate` mappings require the listed host, browser, web-server, or application evidence "
        "before they should be reported as confirmed techniques."
    )
    lines.append(
        "* Do not turn a feature such as `IAT`, `PacketDensity`, `SYNRatio`, `HTTP`, or `HTTPS` into an ATT&CK technique "
        "by itself. Those features support the anomaly explanation; the attack label/behavior provides the contextual link."
    )
    lines.append("")
    lines.append("## Official sources")
    lines.append("")
    lines.append(
        "CIC-IoT-2023: https://www.unb.ca/cic/datasets/iotdataset-2023.html"
    )
    lines.append("MITRE ATT&CK Enterprise techniques: https://attack.mitre.org/techniques/")
    lines.append("")
    lines.append("## Implementation rule for the next LLM stage")
    lines.append("")
    lines.append(
        "The LLM receives the observed attack label (when known), anomaly score/threshold, top XAI features, and this "
        "mapping table. It may explain or summarize only these supplied candidates. It must not invent a new ATT&CK "
        "technique if the supplied evidence is insufficient."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 4. MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Build ZeroShield AI MITRE ATT&CK contextualization outputs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="ZeroShield project root directory.",
    )
    parser.add_argument(
        "--xai",
        type=Path,
        default=None,
        help="Optional path to Deep_SVDD_XAI_category_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <root>/model_training/outputs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.root / "model_training" / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    xai_path = args.xai or (output_dir / "Deep_SVDD_XAI_category_summary.csv")

    xai_df = read_xai_summary(xai_path)

    categories = ["BenignTraffic"] + sorted(MAPPING_RULES.keys())
    records = build_records(categories, xai_df)

    mapping_csv = output_dir / "Deep_SVDD_MITRE_mapping.csv"
    mapping_json = output_dir / "Deep_SVDD_MITRE_mapping.json"
    llm_json = output_dir / "Deep_SVDD_MITRE_mapping_for_LLM.json"
    report_md = output_dir / "Deep_SVDD_MITRE_mapping_report.md"

    # Make a flat CSV-friendly view.
    csv_rows = []
    for r in records:
        row = dict(r)
        row["top_xai_features"] = ", ".join(r["top_xai_features"])
        csv_rows.append(row)

    pd.DataFrame(csv_rows).to_csv(mapping_csv, index=False)

    mapping_json.write_text(
        json.dumps(
            {
                "project": "ZeroShield AI",
                "layer": "MITRE ATT&CK contextualization",
                "xai_input": str(xai_path),
                "xai_input_found": xai_df is not None,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    llm_payload = build_llm_payload(records)
    llm_payload["xai_input"] = str(xai_path)
    llm_payload["xai_input_found"] = xai_df is not None
    llm_json.write_text(
        json.dumps(llm_payload, indent=2),
        encoding="utf-8",
    )

    write_report(report_md, records, xai_path)

    print("\n" + "=" * 78)
    print("ZEROSHIELD AI — MITRE ATT&CK MAPPING COMPLETE")
    print("=" * 78)
    print(f"XAI summary found : {xai_df is not None}")
    print(f"Mappings written  : {len(records)}")
    print(f"CSV               : {mapping_csv}")
    print(f"JSON              : {mapping_json}")
    print(f"LLM lookup        : {llm_json}")
    print(f"Report            : {report_md}")


if __name__ == "__main__":
    main()
