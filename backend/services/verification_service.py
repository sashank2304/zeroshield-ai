"""Live-record adapter for the Stage 6 evidence verification contract.

The handoff verifier consumes fixed-section Markdown reports. The dashboard emits
structured JSON, so this adapter applies its evidence-bound checks to that schema.
"""
import re

NUMERIC = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z0-9])")
TECHNIQUE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.I)


def _matches_supported_number(token: str, supported: list[float]) -> bool:
    value = float(token)
    if "e" in token.lower() or "." not in token:
        return any(abs(value - candidate) < 1e-9 for candidate in supported)
    digits = len(token.split(".", 1)[1])
    return any(abs(round(candidate, digits) - value) < 1e-12 for candidate in supported)


def verify_record(analysis: dict, mitre: dict, reasoning: dict) -> dict:
    report = reasoning.get("report", {})
    evidence_text = " ".join(str(v) for v in report.get("evidence", []))
    all_text = " ".join(str(v) for v in report.values())
    technique_ids = {t.get("technique_id") for t in mitre.get("techniques", [])}
    checks = []

    def check(name, passed, detail, critical=False):
        checks.append({"name": name, "passed": bool(passed), "detail": detail, "level": "critical" if critical else "major"})

    check("structured_schema", all(report.get(k) for k in ("incident_summary", "evidence", "attack_context", "interpretation", "recommended_action", "limitations")), "Required structured report fields are present.", True)
    check("detector_score", str(analysis["anomaly_score"]) in evidence_text or f"{analysis['anomaly_score']:.6f}" in evidence_text, "Explanation cites the actual anomaly score.", True)
    status_words = ["exceeded", "anomalous"] if analysis["detected"] else ["did not exceed", "normal"]
    check("threshold_and_status", "threshold" in evidence_text.lower() and any(x in all_text.lower() for x in status_words), "Explanation agrees with the frozen threshold decision.", True)
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
            phrase = sentence.lower()
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
    check("uncertainty_present", bool(report.get("limitations")), "Limitations and uncertainty are explicitly represented.")
    confidence = sum(c["passed"] for c in checks) / len(checks)
    failed_critical = any(not c["passed"] and c["level"] == "critical" for c in checks)
    failed_major = any(not c["passed"] for c in checks)
    verdict = "REJECTED" if failed_critical else "UNCERTAIN" if failed_major or confidence < 0.85 else "VERIFIED"
    return {"verdict": verdict, "verification_confidence": confidence, "checks": checks,
            "failed_checks": [c["name"] for c in checks if not c["passed"]],
            "method": "Stage 6 rule-based structured-record adapter; confidence is equal-weight pass fraction; any critical failure rejects and any failed check is uncertain"}
