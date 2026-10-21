"""
Tests for verify_soc_report.py, using the real teammate sample.
Run from the project root:   python test_verify_soc_report.py
"""

import copy
import json
import re
from pathlib import Path

from verify_soc_report import SocReportVerifier

EVIDENCE = json.loads(Path("soc_reports/sample_0_soc_report.json").read_text(encoding="utf-8"))
RECORD = json.loads(Path("soc_reports/sample_0_llm_soc_report.json").read_text(encoding="utf-8"))
ORIGINAL = RECORD["llm_report_markdown"]


def run(text, evidence=EVIDENCE):
    rec = copy.deepcopy(RECORD)
    rec["llm_report_markdown"] = text
    return SocReportVerifier().verify(evidence, rec)


def drop_sentences(text, pattern):
    keep = [l for l in text.split("\n") if not re.search(pattern, l, re.I)]
    return "\n".join(keep)


def sub_once(text, old, new):
    assert old in text, f"test setup problem: '{old}' not found in the report"
    return text.replace(old, new, 1)


# (name, damaged report text, expected to be caught)
CASES = [
    ("wrong anomaly score",
     sub_once(ORIGINAL, "* Anomaly Score: 0.021153250709176", "* Anomaly Score: 0.5")),
    ("wrong threshold",
     sub_once(ORIGINAL, "* Threshold: 0.06141921", "* Threshold: 0.2")),
    ("flipped detector prediction",
     sub_once(ORIGINAL, "* Detector Prediction: False", "* Detector Prediction: True")),
    ("wrong dataset label",
     sub_once(ORIGINAL, "* Dataset Label: Mirai-greeth_flood", "* Dataset Label: DDoS-SYN_Flood")),
    ("invented technique ID",
     ORIGINAL + "\n* T1059: Command and Scripting Interpreter (similarity: 0.4)\n"),
    ("wrong technique name",
     sub_once(ORIGINAL, "T1498.001: Direct Network Flood", "T1498.001: SYN Flood Attack")),
    ("wrong similarity",
     sub_once(ORIGINAL, "similarity: 0.365956", "similarity: 0.91")),
    ("invented URL",
     ORIGINAL + "\nSee https://attack.mitre.org/techniques/T9999 for details.\n"),
    ("invented IP address",
     ORIGINAL + "\nTraffic originated from 192.168.1.45.\n"),
    ("invented date",
     ORIGINAL + "\nThe flow was captured on 2026-03-14.\n"),
    ("percentage probability",
     ORIGINAL + "\nThere is an 87% likelihood of a flood attack.\n"),
    ("wrong feature contribution",
     sub_once(ORIGINAL, "* Protocol Type: 0.0193799929693341", "* Protocol Type: 0.5")),
    ("feature not in the evidence",
     ORIGINAL + "\nPacketDensity was also strongly elevated.\n"),
    ("certainty overclaim",
     ORIGINAL + "\nThis confirms a Mirai botnet attack.\n"),
    ("completed action claim",
     ORIGINAL + "\nThe source host was isolated from the network.\n"),
    ("threshold comparison reversed",
     sub_once(ORIGINAL, "which is below the threshold", "which exceeds the threshold")),
    ("direction reversed",
     sub_once(ORIGINAL, "reduces the anomaly score", "increases the anomaly score")),
    ("claims a detection that did not happen",
     ORIGINAL + "\nThe sample was flagged as an attack by the detector.\n"),
    ("caveats removed",
     drop_sentences(ORIGINAL, r"unverified|causation|not confirmed|suggestions")),
    ("format drift: contributions as prose",
     re.sub(r"^\* (Protocol Type|ack_flag_number|TCP|Header_Length|psh_flag_number|SYNRatio|HTTP|UDP|urg_count|RSTRatio): ",
            r"Feature \1 had value ", ORIGINAL, flags=re.M)),
    ("missing section",
     drop_sentences(ORIGINAL, r"^## 5\.|^Further validation")),
]


def test_real_sample_verifies():
    result = run(ORIGINAL)
    assert result["verdict"] == "VERIFIED", result["failed_checks"]
    assert result["original_llm_record"]["llm_report_markdown"] == ORIGINAL
    print(f"real sample -> {result['verdict']} ({result['verification_confidence']})")


def test_every_damaged_copy_is_caught():
    missed = []
    for name, text in CASES:
        result = run(text)
        caught = result["verdict"] != "VERIFIED"
        print(f"  {'caught' if caught else 'MISSED'}  {name:42s} -> {result['verdict']:9s} "
              f"{result['failed_checks'][:2]}")
        if not caught:
            missed.append(name)
    assert not missed, f"verifier missed: {missed}"


def test_empty_report_rejected():
    assert run("")["verdict"] == "REJECTED"


def test_original_output_is_not_modified():
    before = ORIGINAL
    result = run(before)
    assert result["original_llm_record"]["llm_report_markdown"] == before


if __name__ == "__main__":
    test_real_sample_verifies()
    test_every_damaged_copy_is_caught()
    test_empty_report_rejected()
    test_original_output_is_not_modified()
    print("ALL TESTS PASSED")
