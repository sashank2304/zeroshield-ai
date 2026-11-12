"""Live-record adapter for the Stage 6 evidence verification contract.

The handoff verifier consumes fixed-section Markdown reports. The dashboard emits
structured JSON, so this adapter applies its evidence-bound checks to that schema.

Every check is a plain rule. A VERIFIED verdict means the report is faithful to its
evidence; it does not mean the flow is really an attack.

When the reasoning comes from the built-in template (no API key) the report is generated
by code from the same evidence, so verification only confirms internal consistency; the
result says so in ``limitation``.
"""
import json
import re

NUMERIC = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z0-9])")
TECHNIQUE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.I)
NEG = re.compile(r"\b(not|no|never|cannot|without|neither|nor)\b|n't", re.I)
HEDGE = re.compile(r"\b(not|no|never|cannot|without|require[sd]?|further|unconfirmed|unverified|candidates?|"
                   r"suggest\w*|possible|possibly|may|might|could|if|whether|to confirm|to determine|should)\b|n't", re.I)
BELOW = re.compile(r"\b(below|under|beneath|less than|lower than)\b[^.\n]{0,30}\bthreshold", re.I)
ABOVE = re.compile(r"\b(above|over|exceed(?:s|ed)?|greater than|higher than)\b[^.\n]{0,30}\bthreshold", re.I)
OVERCLAIM = re.compile(r"\b(confirm(?:s|ed)?|prove[sd]?|proven|definitive(?:ly)?|certain(?:ly)?|undoubtedly|"
                       r"conclusive(?:ly)?|clearly an? attack)\b", re.I)
ACTION_VERBS = "blocked|isolated|quarantined|remediated|contained|investigated|mitigated|patched|terminated|disabled|stopped"
DONE = re.compile(rf"\b(?:was|were|has been|have been|had been|is now|are now)\s+(?:{ACTION_VERBS})\b|"
                  rf"\bwe\s+(?:{ACTION_VERBS}|verified|confirmed)\b", re.I)
SPECIFICS = {
    "IP address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "MAC address": re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    "date/time": re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b"),
    "port number": re.compile(r"\bport\s+\d+\b", re.I),
    "URL": re.compile(r"https?://[^\s)>\]\"']+"),
}
FALLBACK_MODE = "Demo Fallback Reasoning"


def _flatten(value) -> str:
    if isinstance(value, (list, tuple)):
        return "\n".join(_flatten(v) for v in value)
    return str(value)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _matches_supported_number(token: str, supported: list[float]) -> bool:
    value = float(token)
    if "e" in token.lower() or "." not in token:
        return any(abs(value - candidate) < 1e-9 for candidate in supported)
    digits = len(token.split(".", 1)[1])
    for candidate in supported:
        if abs(round(candidate, digits) - value) >= 1e-12:
            continue
        # A coarse rounding (1-3 decimals) must still be close to the real number, otherwise
        # a made-up "0.1" would pass as a rounded 0.0614.
        if digits >= 4 or abs(value - candidate) <= 0.1 * abs(candidate):
            return True
    return False


def _status_contradictions(text: str, detected: bool) -> tuple[list[str], int]:
    """Statements that disagree with the detector decision, and how many agree with it."""
    contradictions, agreeing = [], 0
    for pattern, kind in ((ABOVE, "above"), (BELOW, "below")):
        for m in pattern.finditer(text):
            negated = bool(NEG.search(text[max(0, m.start() - 25):m.start()]))
            says_detected = (kind == "above") != negated
            if says_detected == detected:
                agreeing += 1
            else:
                contradictions.append(text[max(0, m.start() - 20):m.end()].strip().replace("\n", " "))
    for sentence in _sentences(text):
        if detected and (re.search(r"\bnot\s+(?:an?\s+)?(?:anomal\w+|abnormal|flagged|detected)", sentence, re.I)
                         or re.search(r"\bnon-anomalous\b", sentence, re.I)
                         or re.search(r"\b(?:is|was|appears(?: to be)?|classified as|labeled as)\s+(?:normal|benign)\b", sentence, re.I)):
            contradictions.append(sentence[:90])
        if not detected and not NEG.search(sentence) and re.search(
                r"\b(?:is|was|appears(?: to be)?|flagged as|classified as|detected as|identified as)\s+(?:an?\s+)?"
                r"(?:anomalous|anomaly|abnormal|malicious|attack)\b", sentence, re.I):
            contradictions.append(sentence[:90])
    return contradictions, agreeing


def verify_record(analysis: dict, mitre: dict, reasoning: dict) -> dict:
    report = reasoning.get("report", {})
    evidence_text = _flatten(report.get("evidence", []))
    all_text = "\n".join(_flatten(v) for v in report.values())
    technique_ids = {t.get("technique_id") for t in mitre.get("techniques", [])}
    checks = []

    def check(name, passed, detail, critical=False):
        checks.append({"name": name, "passed": bool(passed), "detail": detail, "level": "critical" if critical else "major"})

    check("structured_schema", all(report.get(k) for k in ("incident_summary", "evidence", "attack_context", "interpretation", "recommended_action", "limitations")), "Required structured report fields are present.", True)
    check("detector_score", str(analysis["anomaly_score"]) in evidence_text or f"{analysis['anomaly_score']:.6f}" in evidence_text, "Explanation cites the actual anomaly score.", True)

    detected = bool(analysis["detected"])
    contradictions, agreeing = _status_contradictions(all_text, detected)
    affirmed_by_word = bool(re.search(r"\banomalous\b", all_text, re.I)) if detected else bool(re.search(r"\bnormal\b", all_text, re.I))
    check("threshold_and_status", "threshold" in evidence_text.lower() and not contradictions and (agreeing > 0 or affirmed_by_word),
          "Explanation agrees with the frozen threshold decision." if not contradictions
          else "Statements contradict the detector decision: " + "; ".join(contradictions), True)

    numeric_text = TECHNIQUE.sub("", all_text)
    supported_numbers = [float(analysis["anomaly_score"]), float(analysis["threshold"]), 0.0, 1.0,
                         float(len(analysis.get("feature_names", []))), float(len(technique_ids))]
    for feature in analysis.get("top_features", []):
        supported_numbers.extend([float(feature["value"]), float(feature["contribution"])])
    unsupported_numbers = [m.group(0) for m in NUMERIC.finditer(numeric_text)
                           if not _matches_supported_number(m.group(0), supported_numbers)]
    check("numeric_claims_grounded", not unsupported_numbers,
          "Unsupported numeric claims: " + (", ".join(unsupported_numbers) if unsupported_numbers else "none"), True)

    supported_names = set(analysis["feature_names"])
    project_names = set(analysis.get("all_feature_names", supported_names))
    feature_mentions = [name for name in project_names if re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", evidence_text, re.I)]
    unsupported = [name for name in feature_mentions if name not in supported_names]
    check("feature_claims_supported", not unsupported, "Unsupported XAI feature claims: " + (", ".join(unsupported) if unsupported else "none"))

    direction_errors = []
    directional = re.compile(r"\b(replac\w*|perturb\w*|mask\w*|remov\w*|increas\w*|rais\w*|boost\w*|decreas\w*|reduc\w*|lower\w*|positiv\w*|negativ\w*)\b", re.I)
    for feature in analysis.get("top_features", []):
        name, contribution = feature["feature"], float(feature["contribution"])
        for sentence in re.split(r"[.!?\n]+", all_text):
            if not re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", sentence, re.I): continue
            words = directional.findall(sentence.lower())
            if not words: continue
            reducing = any(w.startswith(("decreas", "reduc", "lower")) for w in words)
            increasing = any(w.startswith(("increas", "rais", "boost")) for w in words)
            removal_context = any(w.startswith(("replac", "perturb", "mask", "remov")) for w in words)
            says_positive = any(w.startswith("positiv") for w in words)
            says_negative = any(w.startswith("negativ") for w in words)
            if removal_context and ((reducing and contribution <= 0) or (increasing and contribution >= 0)):
                direction_errors.append(name)
            if (says_positive and contribution <= 0) or (says_negative and contribution >= 0):
                direction_errors.append(name)
    check("feature_directions_consistent", not direction_errors,
          "Contradictory XAI direction claims: " + (", ".join(sorted(set(direction_errors))) if direction_errors else "none"))

    mentioned = set(re.findall(r"\bT\d{4}(?:\.\d{3})?\b", all_text))
    check("technique_ids_retrieved", mentioned.issubset(technique_ids), "Every technique ID in the explanation was retrieved.", True)

    haystack = json.dumps({"mitre": mitre, "top_features": analysis.get("top_features", [])}, default=str)
    invented = [f"{kind} '{m.group(0)}'" for kind, pattern in SPECIFICS.items()
                for m in pattern.finditer(all_text) if m.group(0).rstrip(".,;:/") not in haystack]
    check("no_invented_specifics", not invented,
          "Specifics not present in the evidence: " + (", ".join(invented) if invented else "none"), True)

    overclaims = [s[:90] for s in _sentences(all_text) if OVERCLAIM.search(s) and not HEDGE.search(s)]
    check("certainty_claims_hedged", not overclaims,
          "Unhedged certainty claims: " + ("; ".join(overclaims) if overclaims else "none"))

    done = [s[:90] for s in _sentences(all_text) if DONE.search(s) and not NEG.search(s)]
    check("no_completed_action_claims", not done,
          "Claims that a response was already carried out: " + ("; ".join(done) if done else "none"))

    check("uncertainty_present", bool(report.get("limitations")), "Limitations and uncertainty are explicitly represented.")

    confidence = sum(c["passed"] for c in checks) / len(checks)
    failed_critical = any(not c["passed"] and c["level"] == "critical" for c in checks)
    failed_major = any(not c["passed"] for c in checks)
    verdict = "REJECTED" if failed_critical else "UNCERTAIN" if failed_major or confidence < 0.85 else "VERIFIED"
    mode = reasoning.get("mode")
    result = {"verdict": verdict, "verification_confidence": confidence, "checks": checks,
              "failed_checks": [c["name"] for c in checks if not c["passed"]],
              "reasoning_mode": mode,
              "method": "Stage 6 rule-based structured-record adapter; confidence is equal-weight pass fraction; any critical failure rejects and any failed check is uncertain"}
    if mode == FALLBACK_MODE:
        result["limitation"] = ("The reasoning is a built-in template generated from the same evidence, so this verdict "
                                "only confirms internal consistency. It does not test an LLM's faithfulness.")
    return result
