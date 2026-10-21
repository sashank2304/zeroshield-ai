"""
ZeroShield AI - Stage 6: post-LLM verification of the SOC report.

Input 1 (evidence):  soc_reports/sample_0_soc_report.json
    detector score/threshold/prediction, XAI feature_evidence, MITRE candidates.
Input 2 (LLM output): soc_reports/sample_0_llm_soc_report.json
    the free-form Markdown report in "llm_report_markdown".

The report is parsed and every checkable claim is compared with the evidence:
numbers, detector status, technique IDs/names/similarities, URLs, feature
contributions, causal or confirmation overclaims, invented specifics (IPs,
dates, ports), completed-action claims and the caveats the system prompt
requires. Each check is a plain rule, so every verdict has a stated reason.

The ORIGINAL LLM output is kept unchanged inside the verified report, with a
SHA-256 hash, so the two can be compared and audited.

What this does NOT do: it checks that the report is faithful to the evidence.
It does not prove the flow is really an attack, and it cannot judge how good
the detector or the MITRE retrieval was.

Run (from the project root):
    python verify_soc_report.py
    python verify_soc_report.py --evidence X.json --llm Y.json --out soc_reports
"""

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_SECTIONS = [
    "Executive Summary",
    "Sample and Detector Status",
    "Explainable AI Evidence",
    "MITRE ATT&CK Candidate Mapping",
    "Recommended Analyst Validation",
    "Limitations",
]

FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
    "fin_count", "urg_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP",
    "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "AVG", "Std", "IAT",
    "Number", "Radius", "Weight", "SYNRatio", "ACKRatio", "RSTRatio", "FINRatio",
    "SYN_ACK_Ratio", "PacketDensity",
]
# Names that are also ordinary words or protocol names are skipped when scanning
# free text, so "UDP traffic" or "the rate" is not read as a feature claim.
AMBIGUOUS = {"Rate", "Number", "Weight", "Radius", "Duration", "Std", "AVG", "IAT",
             "TCP", "UDP", "DNS", "ARP", "ICMP", "HTTP", "HTTPS", "SSH", "IRC",
             "SMTP", "DHCP", "Telnet", "IPv"}
SCANNED_FEATURES = [f for f in FEATURES if f not in AMBIGUOUS]

NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
TECH_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
URL_RE = re.compile(r"https?://[^\s)>\]\"']+")
NEG = re.compile(r"\b(not|no|never|cannot|without|neither|nor)\b|n't", re.I)
HEDGE = re.compile(r"\b(not|no|never|cannot|without|require[sd]?|further|unconfirmed|"
                   r"unverified|candidates?|suggest\w*|possible|possibly|may|might|could|"
                   r"if|whether|to confirm|to determine|should)\b|n't", re.I)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
@dataclass
class Check:
    name: str
    score: float          # 0.0 to 1.0
    weight: float
    level: str            # "critical", "major" or "minor"
    detail: str


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def split_sections(text):
    """Map section title -> body, using '## N. Title' headers."""
    parts = re.split(r"^##\s*\d+\.\s*(.+?)\s*$", text, flags=re.M)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def number_matches(token, value):
    """True if the quoted number equals the evidence value or its correct rounding."""
    try:
        t = float(token)
    except ValueError:
        return False
    if math.isclose(t, value, rel_tol=1e-6, abs_tol=1e-12):
        return True
    if "e" in token.lower() or "." not in token:
        return abs(t - value) < 1e-9          # integers must match exactly
    decimals = len(token.split(".")[1])
    return abs(round(value, decimals) - t) < 1e-12


def evidence_numbers(evidence, candidates):
    vals = [evidence.get("anomaly_score"), evidence.get("threshold"),
            evidence.get("sample_index"), evidence.get("binary_label"),
            len(candidates), len(evidence.get("feature_evidence", [])), 0, 1]
    for f in evidence.get("feature_evidence", []):
        vals += [f.get("original_value"), f.get("perturbed_score"), f.get("contribution")]
    for c in candidates:
        vals += [c.get("rank"), c.get("similarity")]
    return [float(v) for v in vals
            if isinstance(v, (int, float)) and not isinstance(v, bool)]


def fraction_check(name, found, ok, weight, level, none_detail, bad):
    if found == 0:
        return Check(name, 1.0, weight, level, none_detail)
    return Check(name, ok / found, weight, level,
                 "; ".join(bad) if bad else f"{ok}/{found} correct")


# ----------------------------------------------------------------------
# the verifier
# ----------------------------------------------------------------------
class SocReportVerifier:
    def __init__(self, verified_min=0.85, uncertain_min=0.50):
        self.verified_min = verified_min
        self.uncertain_min = uncertain_min

    def verify(self, evidence, llm_record):
        text = llm_record.get("llm_report_markdown", "")
        candidates = (llm_record.get("mitre_candidates_supplied")
                      or evidence.get("mitre_candidates", []))
        ev_json = json.dumps(evidence) + json.dumps(llm_record.get("mitre_candidates_supplied", []))

        if not text.strip():
            check = Check("report_present", 0.0, 1.0, "critical", "LLM report is empty")
            return self._result(evidence, llm_record, [check])

        sections = split_sections(text)
        checks = [
            self._metadata_consistent(evidence, llm_record),
            self._structure(sections),
            self._status_fields_present(text),
            self._status_numbers(text, evidence),
            self._threshold_comparison(text, evidence),
            self._detection_claims(text, evidence),
            self._technique_ids(text, candidates),
            self._technique_details(text, candidates),
            self._urls(text, candidates, ev_json),
            self._feature_contributions(sections, text, evidence),
            self._unlisted_features(text, evidence),
            self._direction_claims(text, evidence),
            self._ungrounded_numbers(text, evidence, candidates),
            self._invented_specifics(text, ev_json),
            self._overclaims(text),
            self._completed_actions(text),
            self._caveats(text, evidence),
        ]
        return self._result(evidence, llm_record, checks)

    # ---- structure and metadata --------------------------------------
    def _metadata_consistent(self, evidence, rec):
        bad = []
        for key in ("anomaly_score", "threshold", "predicted_anomaly", "dataset_label"):
            if key in rec and key in evidence and rec[key] != evidence[key]:
                bad.append(f"{key}: evidence={evidence[key]} vs llm_record={rec[key]}")
        return Check("inputs_consistent", 0.0 if bad else 1.0, 1.0, "major",
                     "; ".join(bad) if bad else "evidence file matches the LLM record")

    def _structure(self, sections):
        titles = [t.lower() for t in sections]
        missing = [s for s in REQUIRED_SECTIONS if not any(s.lower() in t for t in titles)]
        return Check("required_sections", 1 - len(missing) / len(REQUIRED_SECTIONS), 1.0,
                     "major", f"missing sections: {missing}" if missing else "all 6 sections present")

    def _status_fields_present(self, text):
        need = {"anomaly score": r"anomaly\s+score", "threshold": r"threshold",
                "detector prediction": r"detector\s+prediction"}
        missing = [k for k, p in need.items() if not re.search(p, text, re.I)]
        return Check("status_fields_present", 1 - len(missing) / len(need), 1.0, "major",
                     f"not mentioned: {missing}" if missing else "score, threshold and prediction stated")

    # ---- detector status ----------------------------------------------
    def _status_numbers(self, text, ev):
        found = ok = 0
        bad = []

        def check_num(label, pattern, value):
            nonlocal found, ok
            for m in re.finditer(pattern, text, re.I):
                found += 1
                if number_matches(m.group(1), value):
                    ok += 1
                else:
                    bad.append(f"{label}: report says {m.group(1)}, evidence says {value}")

        check_num("anomaly score", rf"anomaly\s+score[^\d\n.;]{{0,40}}({NUM})", ev["anomaly_score"])
        check_num("threshold", rf"threshold[^\d\n.;]{{0,40}}({NUM})", ev["threshold"])
        if "binary_label" in ev:
            check_num("binary label", rf"binary\s+label\s*[:=]\s*[`*]*({NUM})", ev["binary_label"])
        if "sample_index" in ev:
            check_num("sample index", rf"sample\s+index\s*[:=]\s*[`*]*({NUM})", ev["sample_index"])

        for m in re.finditer(r"dataset\s+label\s*[:=]\s*[`*\"']*([A-Za-z0-9_\-.]+)", text, re.I):
            found += 1
            if m.group(1).rstrip(".") == str(ev.get("dataset_label")):
                ok += 1
            else:
                bad.append(f"dataset label: report says {m.group(1)}, evidence says {ev.get('dataset_label')}")

        expected = str(bool(ev["predicted_anomaly"]))
        for m in re.finditer(r"detector\s+prediction\s*(?:is|was|:)?\s*[`*]*\s*(true|false)\b", text, re.I):
            found += 1
            if m.group(1).capitalize() == expected:
                ok += 1
            else:
                bad.append(f"detector prediction: report says {m.group(1)}, evidence says {expected}")
        if found == 0:
            return Check("status_values_correct", 0.0, 3.0, "major",
                         "could not parse any status value (format changed or values omitted)")
        return fraction_check("status_values_correct", found, ok, 3.0, "critical",
                              "", bad)

    def _threshold_comparison(self, text, ev):
        score, thr = ev["anomaly_score"], ev["threshold"]
        found = ok = 0
        bad = []
        patterns = [
            (r"\b(below|under|less than|lower than|beneath)\b[^.\n]{0,30}\bthreshold", "below"),
            (r"\b(above|over|exceeds?|exceeded|greater than|higher than)\b[^.\n]{0,30}\bthreshold", "above"),
        ]
        for pattern, kind in patterns:
            for m in re.finditer(pattern, text, re.I):
                found += 1
                negated = bool(NEG.search(text[max(0, m.start() - 25):m.start()]))
                claim = (score < thr) if kind == "below" else (score > thr)
                claim = (not claim) if negated else claim
                if claim:
                    ok += 1
                else:
                    bad.append(f"report says score {'not ' if negated else ''}{kind} threshold; "
                               f"score={score:.6f}, threshold={thr:.6f}")
        return fraction_check("threshold_comparison_correct", found, ok, 3.0, "critical",
                              "no score-versus-threshold statement", bad)

    def _detection_claims(self, text, ev):
        predicted = bool(ev["predicted_anomaly"])
        positive = re.compile(r"\b(flagged|detected|identified|classified|predicted)\b[^.\n]{0,50}"
                              r"\b(anomal\w*|attack\w*|malicious|threat)\b", re.I)
        negative = re.compile(r"\b(not|never|no|did not|was not|wasn't)\b[^.\n]{0,30}"
                              r"\b(flagged|detected|identified)\b", re.I)
        bad = []
        for s in sentences(text):
            if not predicted and positive.search(s) and not NEG.search(s):
                bad.append(f"claims a detection, but the detector did not flag it: '{s[:90]}'")
            if predicted and negative.search(s):
                bad.append(f"says it was not flagged, but the detector flagged it: '{s[:90]}'")
        return Check("detection_claims_consistent", 0.0 if bad else 1.0, 3.0, "critical",
                     "; ".join(bad) if bad else "no claim contradicts the detector prediction")

    # ---- MITRE ----------------------------------------------------------
    def _technique_ids(self, text, candidates):
        allowed = {c["id"] for c in candidates}
        allowed |= {i.split(".")[0] for i in allowed}
        no_urls = URL_RE.sub("", text)
        found = TECH_RE.findall(no_urls)
        invalid = sorted(set(found) - allowed)
        return Check("technique_ids_valid", 0.0 if invalid else 1.0, 3.0, "critical",
                     f"technique IDs not in the supplied candidates: {invalid}" if invalid
                     else f"{len(set(found))} technique ID(s), all supplied")

    def _technique_details(self, text, candidates):
        by_id = {c["id"]: c for c in candidates}
        pat = re.compile(rf"(T\d{{4}}(?:\.\d{{3}})?)\s*:\s*([^(\n]+?)\s*\(\s*similarity\s*:\s*({NUM})\s*\)", re.I)
        found = ok = 0
        bad = []
        for m in pat.finditer(text):
            tid, name, sim = m.group(1), m.group(2).strip(), m.group(3)
            if tid not in by_id:
                continue                         # reported by technique_ids_valid
            found += 1
            name_ok = name.lower() == by_id[tid]["name"].lower()
            sim_ok = number_matches(sim, by_id[tid]["similarity"])
            if name_ok and sim_ok:
                ok += 1
            else:
                bad.append(f"{tid}: report '{name}' / {sim}, supplied '{by_id[tid]['name']}' / {by_id[tid]['similarity']}")
        if found == 0:
            if TECH_RE.search(URL_RE.sub("", text)):
                return Check("technique_details_correct", 0.0, 2.0, "major",
                             "technique IDs appear but no 'ID: name (similarity: x)' line could be parsed")
            return Check("technique_details_correct", 1.0, 2.0, "major", "no technique mentioned")
        return fraction_check("technique_details_correct", found, ok, 2.0, "major",
                              "", bad)

    def _urls(self, text, candidates, ev_json):
        allowed = {c.get("url", "").rstrip("/") for c in candidates}
        bad = []
        for url in URL_RE.findall(text):
            u = url.rstrip(".,;:/")
            if u not in allowed and u not in ev_json:
                bad.append(u)
        return Check("urls_grounded", 0.0 if bad else 1.0, 2.0, "critical",
                     f"URLs not supplied as evidence: {bad}" if bad else "no invented URLs")

    # ---- XAI evidence -----------------------------------------------------
    def _feature_contributions(self, sections, text, ev):
        by_name = {f["feature"]: f for f in ev.get("feature_evidence", [])}
        body = next((b for t, b in sections.items() if "explainable" in t.lower()), text)
        pat = re.compile(rf"^\s*[*\-]\s*([A-Za-z_][\w ]*?)\s*:\s*({NUM})\s*$", re.M)
        found = ok = 0
        bad = []
        for m in pat.finditer(body):
            name, val = m.group(1).strip(), m.group(2)
            found += 1
            if name not in by_name:
                bad.append(f"'{name}' is not in the supplied XAI evidence")
            elif number_matches(val, by_name[name]["contribution"]):
                ok += 1
            else:
                bad.append(f"{name}: report says {val}, evidence says {by_name[name]['contribution']}")
        if found == 0 and by_name:
            return Check("feature_contributions_correct", 0.0, 3.0, "major",
                         "XAI evidence exists but no 'feature: value' lines could be parsed")
        return fraction_check("feature_contributions_correct", found, ok, 3.0, "major",
                              "no XAI evidence supplied", bad)

    def _unlisted_features(self, text, ev):
        known = {f["feature"] for f in ev.get("feature_evidence", [])}
        mentioned = {f for f in SCANNED_FEATURES if re.search(rf"\b{re.escape(f)}\b", text)}
        extra = sorted(mentioned - known)
        return Check("no_unsupported_feature_mentions", 0.0 if extra else 1.0, 2.0, "major",
                     f"features mentioned without XAI evidence: {extra}" if extra
                     else "every named feature is in the XAI evidence")

    def _direction_claims(self, text, ev):
        contribs = [f["contribution"] for f in ev.get("feature_evidence", [])]
        if not contribs:
            return Check("direction_claims_consistent", 1.0, 1.0, "major", "no XAI evidence to compare")
        bad = []
        for s in sentences(text):
            if not re.search(r"replac", s, re.I):
                continue
            if re.search(r"\breduc\w*|\bdecreas\w*|\blower\w*", s, re.I) and not all(c > 0 for c in contribs):
                bad.append(f"says replacement reduces the score, but not every contribution is positive: '{s[:90]}'")
            if re.search(r"\bincreas\w*|\braise\w*|\bhigher\b", s, re.I) and not all(c < 0 for c in contribs):
                bad.append(f"says replacement increases the score, but not every contribution is negative: '{s[:90]}'")
        return Check("direction_claims_consistent", 0.0 if bad else 1.0, 2.0, "major",
                     "; ".join(bad) if bad else "direction statements match the contribution signs")

    # ---- generic hallucination checks --------------------------------------
    def _ungrounded_numbers(self, text, ev, candidates):
        body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
        body = URL_RE.sub(" ", TECH_RE.sub(" ", body))
        percents = re.findall(rf"{NUM}\s*%", body)
        tokens = re.findall(rf"(?<![\w.]){NUM}(?![\w])", body)
        values = evidence_numbers(ev, candidates)
        bad = [t for t in tokens if not any(number_matches(t, v) for v in values)]
        details = []
        if bad:
            details.append(f"numbers not found in the evidence: {sorted(set(bad))}")
        if percents:
            details.append(f"percentages (not in the evidence; similarity is not a probability): {percents}")
        total = len(tokens) or 1
        score = max(0.0, 1 - (len(bad) + len(percents)) / total)
        return Check("no_ungrounded_numbers", score, 2.0, "major",
                     "; ".join(details) if details else f"{len(tokens)} number(s), all in the evidence")

    def _invented_specifics(self, text, ev_json):
        patterns = {
            "IP address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "MAC address": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
            "date/time": r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b",
            "port number": r"\bport\s+\d+\b",
        }
        bad = []
        for kind, pat in patterns.items():
            for m in re.finditer(pat, text, re.I):
                if m.group(0) not in ev_json:
                    bad.append(f"{kind} '{m.group(0)}'")
        return Check("no_invented_specifics", 0.0 if bad else 1.0, 3.0, "critical",
                     f"specifics that are not in the evidence: {bad}" if bad
                     else "no invented IPs, MACs, dates or ports")

    def _overclaims(self, text):
        words = re.compile(r"\b(confirm(?:s|ed)?|prove[sd]?|proven|definitive(?:ly)?|"
                           r"certain(?:ly)?|undoubtedly|conclusive(?:ly)?|clearly an? attack)\b", re.I)
        bad = [s[:90] for s in sentences(text) if words.search(s) and not HEDGE.search(s)]
        return Check("no_certainty_overclaims", 0.0 if bad else 1.0, 2.0, "major",
                     f"unhedged certainty claims: {bad}" if bad else "no unhedged confirmation or proof claims")

    def _completed_actions(self, text):
        verbs = "blocked|isolated|quarantined|remediated|contained|investigated|mitigated|patched|terminated|disabled"
        pat = re.compile(rf"\b(?:was|were|has been|have been|had been)\s+(?:{verbs})\b|"
                         rf"\bwe\s+(?:{verbs}|verified|confirmed)\b", re.I)
        bad = [s[:90] for s in sentences(text) if pat.search(s) and not NEG.search(s)]
        return Check("no_completed_action_claims", 0.0 if bad else 1.0, 2.0, "major",
                     f"claims an action was done: {bad}" if bad else "no claim that a response was carried out")

    def _caveats(self, text, ev):
        required = {}
        if not ev.get("threshold_verified_for_checkpoint", False):
            required["threshold unverified"] = (
                r"threshold[^.]{0,80}(unverified|not\s+(?:been\s+)?(?:verified|validated|calibrated)|"
                r"provisional|uncalibrated)|(unverified|provisional)[^.]{0,80}threshold")
        if not ev.get("mitre_confirmation", False):
            required["MITRE not confirmed"] = (
                r"not\s+(?:a\s+)?confirm|candidates?[^.]{0,80}(suggestion|require|further validation)|"
                r"retrieval suggestion")
        required["XAI not causal"] = (
            r"does not prove|not\s+(?:a\s+)?(?:proof|causal|causation)|no\s+causal|not\s+prove")
        missing = [k for k, p in required.items() if not re.search(p, text, re.I)]
        return Check("required_caveats_present", 1 - len(missing) / len(required), 2.0, "major",
                     f"caveats missing: {missing}" if missing else "all required caveats present")

    # ---- verdict -----------------------------------------------------------
    def _result(self, evidence, llm_record, checks):
        total_w = sum(c.weight for c in checks)
        confidence = sum(c.score * c.weight for c in checks) / total_w
        critical = [c.name for c in checks if c.level == "critical" and c.score < 1.0]
        major = [c.name for c in checks if c.level == "major" and c.score < 1.0]

        if critical:
            verdict, reason = "REJECTED", f"critical check failed: {critical}"
        elif major:
            if confidence >= self.uncertain_min:
                verdict, reason = "UNCERTAIN", f"unsupported claims found: {major}"
            else:
                verdict, reason = "REJECTED", f"many unsupported claims: {major}"
        elif confidence >= self.verified_min:
            verdict, reason = "VERIFIED", "every checkable claim matches the evidence"
        else:
            verdict, reason = "UNCERTAIN", "no single failure, but overall support is weak"

        original = llm_record.get("llm_report_markdown", "")
        alert = bool(evidence.get("predicted_anomaly"))
        return {
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "sample_index": evidence.get("sample_index"),
            "alert_status": "ALERT" if alert else "NO_ALERT (detector did not flag this sample)",
            "verdict": verdict,
            "verification_confidence": round(confidence, 3),
            "reason": reason,
            "failed_checks": [c.name for c in checks if c.score < 1.0],
            "checks": [asdict(c) for c in checks],
            "original_report_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
            "original_llm_record": llm_record,        # unchanged, kept for audit
        }


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------
def render_markdown(result):
    rows = []
    for c in result["checks"]:
        status = "PASS" if c["score"] >= 1.0 else ("PARTIAL" if c["score"] > 0 else "FAIL")
        rows.append(f"| {c['name']} | {c['level']} | {status} | {c['detail']} |")
    original = result["original_llm_record"].get("llm_report_markdown", "")
    return "\n".join([
        "# ZeroShield AI - Verified SOC Report",
        "",
        f"- **Verdict:** {result['verdict']}",
        f"- **Verification confidence:** {result['verification_confidence']}",
        f"- **Reason:** {result['reason']}",
        f"- **Alert status:** {result['alert_status']}",
        f"- **Verified at:** {result['verified_at']}",
        f"- **Original report SHA-256:** `{result['original_report_sha256']}`",
        "",
        "## Check results",
        "",
        "| Check | Level | Result | Detail |",
        "|---|---|---|---|",
        *rows,
        "",
        "---",
        "",
        "## Original LLM report (unmodified)",
        "",
        original,
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description="Verify an LLM SOC report against its evidence.")
    parser.add_argument("--evidence", default="soc_reports/sample_0_soc_report.json")
    parser.add_argument("--llm", default="soc_reports/sample_0_llm_soc_report.json")
    parser.add_argument("--out", default="soc_reports")
    args = parser.parse_args()

    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    record = json.loads(Path(args.llm).read_text(encoding="utf-8"))
    result = SocReportVerifier().verify(evidence, record)

    stem = Path(args.llm).stem.replace("_llm_soc_report", "")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}_verified_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{stem}_verified_report.md").write_text(render_markdown(result), encoding="utf-8")

    print(f"Verdict: {result['verdict']}  (confidence {result['verification_confidence']})")
    print(f"Reason:  {result['reason']}")
    print(f"Alert:   {result['alert_status']}")
    for c in result["checks"]:
        if c["score"] < 1.0:
            print(f"  - {c['name']}: {c['detail']}")


if __name__ == "__main__":
    main()
