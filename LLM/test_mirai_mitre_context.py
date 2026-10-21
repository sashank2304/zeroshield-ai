
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

TECHNIQUE_ID = "T1498.001"
CACHE_PATH = Path("data/mitre_enterprise_techniques.json")


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def assess_alert(alert, cache):
    technique = cache.get(TECHNIQUE_ID)

    if not technique:
        raise KeyError(
            f"{TECHNIQUE_ID} not found in {CACHE_PATH}. "
            "Run python mitre_api.py first."
        )

    # These values must come from the actual alert, not be inferred.
    detector_status = str(alert.get("detector_status", "unknown")).lower()
    anomaly_score = alert.get("anomaly_score")
    threshold = alert.get("threshold")

    features = alert.get("top_features", [])
    indicators = alert.get("traffic_indicators", {})

    # Explicit traffic indicators and comparisons against a baseline
    # are more useful than an anomaly score by itself.
    volume_keys = (
        "packets_per_second",
        "bytes_per_second",
        "total_packets",
        "total_bytes",
    )

    observed_volume = []
    for key in volume_keys:
        value = indicators.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            observed_volume.append({"metric": key, "value": value})

    baseline_comparisons = indicators.get("baseline_comparisons", {})
    elevated_metrics = []

    if isinstance(baseline_comparisons, dict):
        for metric, comparison in baseline_comparisons.items():
            if not isinstance(comparison, dict):
                continue

            ratio = comparison.get("ratio_to_baseline")
            if isinstance(ratio, (int, float)) and ratio > 1:
                elevated_metrics.append({
                    "metric": metric,
                    "ratio_to_baseline": ratio,
                })

    # A supplied score is only considered anomalous when it can be
    # compared to a supplied threshold using the project's score rule.
    score_exceeds_threshold = (
        isinstance(anomaly_score, (int, float))
        and isinstance(threshold, (int, float))
        and anomaly_score > threshold
    )

    status_indicates_alert = detector_status in {
        "anomaly", "alert", "malicious", "outlier"
    }

    anomaly_evidence = score_exceeds_threshold or status_indicates_alert
    volume_evidence = bool(observed_volume or elevated_metrics)

    if anomaly_evidence and volume_evidence:
        assessment = "candidate_mapping_supported"
        explanation = (
            "The supplied alert indicates anomalous behavior and includes "
            "traffic-volume or baseline-comparison evidence consistent with "
            "a possible network flood. This is a candidate mapping only."
        )
    elif anomaly_evidence:
        assessment = "insufficient_flood_evidence"
        explanation = (
            "An anomaly is indicated, but the supplied evidence does not "
            "include explicit traffic-volume or baseline-comparison "
            "indicators to support a network-flood interpretation."
        )
    elif volume_evidence:
        assessment = "insufficient_anomaly_evidence"
        explanation = (
            "Traffic-volume evidence was supplied, but the alert does not "
            "establish an anomaly using the supplied status or score and "
            "threshold."
        )
    else:
        assessment = "insufficient_evidence"
        explanation = (
            "The supplied evidence does not establish both anomalous "
            "behavior and indicators consistent with network flooding."
        )

    return {
        "test_name": "ZeroShield Mirai-style MITRE contextualization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "alert_id": alert.get("alert_id", "unspecified"),
        "detector": {
            "name": "Deep SVDD",
            "status_supplied": detector_status,
            "anomaly_score_supplied": anomaly_score,
            "threshold_supplied": threshold,
            "score_exceeds_threshold": score_exceeds_threshold,
            "note": (
                "The script does not recalculate the detector score. "
                "It assumes higher scores indicate greater anomalousness."
            ),
        },
        "observed_evidence": {
            "top_features_supplied": features,
            "traffic_volume_metrics": observed_volume,
            "elevated_baseline_comparisons": elevated_metrics,
        },
        "candidate_mapping": {
            "technique_id": TECHNIQUE_ID,
            "name": technique.get("name", ""),
            "tactics": technique.get("tactics", []),
            "description": technique.get("description", ""),
            "official_url": technique.get("url", ""),
            "assessment": assessment,
            "explanation": explanation,
        },
        "limitations": [
            "A candidate mapping is not confirmation of a DoS or DDoS attack.",
            "This test does not identify Mirai malware or attribute activity "
            "to a specific actor.",
            "Traffic-volume evidence must be interpreted in context; "
            "legitimate bursts can also produce high traffic.",
            "No source-IP distribution, target impact, packet capture, "
            "or host evidence is inferred when it was not supplied.",
            "ATT&CK metadata provides context; it does not validate "
            "the detector or independently prove the mapping.",
        ],
        "confirmation": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Contextualize a supplied ZeroShield alert with T1498.001."
    )
    parser.add_argument(
        "--evidence",
        required=True,
        help="Path to a JSON file containing one alert's evidence.",
    )
    parser.add_argument(
        "--output",
        default="data/mirai_mitre_context_result.json",
        help="Path for the JSON result.",
    )
    args = parser.parse_args()

    evidence_path = Path(args.evidence)
    if not evidence_path.exists():
        raise FileNotFoundError(f"Evidence file not found: {evidence_path}")
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"MITRE cache not found: {CACHE_PATH}. "
            "Run python mitre_api.py first."
        )

    alert = load_json(evidence_path)
    cache = load_json(CACHE_PATH)

    result = assess_alert(alert, cache)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 60)
    print("ZEROSHIELD AI - MITRE CONTEXTUALIZATION")
    print("=" * 60)
    print("Alert:", result["alert_id"])
    print("Technique:", TECHNIQUE_ID, "-", result["candidate_mapping"]["name"])
    print("Assessment:", result["candidate_mapping"]["assessment"])
    print("Explanation:", result["candidate_mapping"]["explanation"])
    print("Attack confirmed:", result["confirmation"])
    print("Result saved to:", output_path.resolve())


if __name__ == "__main__":
    main()
