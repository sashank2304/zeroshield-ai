
import json
from pathlib import Path

import pandas as pd


# ============================================================
# ZeroShield AI - SOC Evidence Report Generator
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

XAI_SUMMARY_PATH = BASE_DIR / "xai_results" / "xai_summary.json"
XAI_FEATURES_PATH = BASE_DIR / "xai_results" / "xai_top_features.csv"
MITRE_PATH = BASE_DIR / "data" / "mitre_retrieval_comparison_v2.json"

OUTPUT_DIR = BASE_DIR / "soc_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_INDEX = 0
TOP_FEATURE_COUNT = 10
TOP_MITRE_COUNT = 5


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_number(value, digits=6):
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def get_mitre_candidates(mitre_data):
    """
    Read the actual structure:
    mitre_data['results']['behavior_focused']['top_5']
    """
    results = mitre_data.get("results", {})
    behavior = results.get("behavior_focused", {})
    top_results = behavior.get("top_5", [])

    candidates = []

    for item in top_results[:TOP_MITRE_COUNT]:
        candidates.append({
            "rank": item.get("rank"),
            "id": item.get("id", "Unknown"),
            "name": item.get("name", "Name unavailable"),
            "similarity": item.get("similarity"),
            "url": item.get("url", ""),
        })

    return candidates


def main():
    summary = load_json(XAI_SUMMARY_PATH)
    mitre_data = load_json(MITRE_PATH)

    if not XAI_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"XAI features file not found: {XAI_FEATURES_PATH}"
        )

    df = pd.read_csv(XAI_FEATURES_PATH)

    required_columns = {
        "sample_index",
        "feature",
        "original_value",
        "contribution",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Missing columns in xai_top_features.csv: "
            + ", ".join(sorted(missing))
        )

    df["sample_index"] = pd.to_numeric(
        df["sample_index"], errors="coerce"
    )

    sample_df = df[df["sample_index"] == SAMPLE_INDEX].copy()

    if sample_df.empty:
        raise ValueError(
            f"No XAI evidence found for sample {SAMPLE_INDEX}"
        )

    if "feature_rank" in sample_df.columns:
        sample_df = sample_df.sort_values("feature_rank")
    else:
        sample_df = sample_df.sort_values(
            "contribution", ascending=False
        )

    sample_df = sample_df.head(TOP_FEATURE_COUNT)

    first = sample_df.iloc[0]

    label = first.get("label", "Unknown")
    binary_label = first.get("binary_label", "Unknown")
    anomaly_score = first.get("anomaly_score")
    threshold = first.get("threshold")
    predicted_anomaly = first.get("predicted_anomaly")

    # Preserve the verification status from the XAI summary.
    threshold_verified = summary.get(
        "threshold_verified_for_checkpoint", False
    )

    feature_evidence = []

    for _, row in sample_df.iterrows():
        original_value = row.get("original_value")
        contribution = row.get("contribution")
        perturbed_score = row.get("perturbed_score")

        feature_evidence.append({
            "feature": str(row["feature"]),
            "original_value": (
                None if pd.isna(original_value)
                else float(original_value)
            ),
            "perturbed_score": (
                None if pd.isna(perturbed_score)
                else float(perturbed_score)
            ),
            "contribution": (
                None if pd.isna(contribution)
                else float(contribution)
            ),
        })

    # FIXED: extract MITRE candidates from results.behavior_focused.top_5.
    mitre_candidates = get_mitre_candidates(mitre_data)

    report = {
        "project": "ZeroShield AI",
        "sample_index": SAMPLE_INDEX,
        "dataset_label": str(label),
        "binary_label": (
            None if pd.isna(binary_label) else binary_label
        ),
        "anomaly_score": (
            None if pd.isna(anomaly_score) else float(anomaly_score)
        ),
        "threshold": (
            None if pd.isna(threshold) else float(threshold)
        ),
        "predicted_anomaly": (
            None if pd.isna(predicted_anomaly)
            else bool(predicted_anomaly)
        ),
        "threshold_verified_for_checkpoint": bool(threshold_verified),
        "xai_method_note": (
            "Contribution is the original anomaly score minus the score "
            "after replacing one standardized feature with zero. A positive "
            "contribution means that replacement reduced the score. "
            "This is local perturbation evidence, not proof of causation."
        ),
        "feature_evidence": feature_evidence,
        "mitre_retrieval_method": "TF-IDF similarity",
        "mitre_candidates": mitre_candidates,
        "mitre_confirmation": False,
        "analyst_caveat": (
            "MITRE candidates are retrieval suggestions, not confirmed "
            "attack classifications. The dataset label and detector "
            "prediction must be reported separately."
        ),
    }

    json_path = OUTPUT_DIR / f"sample_{SAMPLE_INDEX}_soc_report.json"
    md_path = OUTPUT_DIR / f"sample_{SAMPLE_INDEX}_soc_report.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
    		report,
    		file,
    		indent=2,
    		ensure_ascii=False,
    		default=lambda obj: obj.item() if hasattr(obj, "item") else str(obj)
		)

    lines = [
        "# ZeroShield AI — SOC Evidence Report",
        "",
        "## 1. Sample overview",
        "",
        f"- **Sample index:** {SAMPLE_INDEX}",
        f"- **Dataset label:** {label}",
        f"- **Binary label:** {binary_label}",
        f"- **Anomaly score:** {format_number(anomaly_score)}",
        f"- **Current threshold:** {format_number(threshold)}",
        f"- **Predicted anomaly:** {predicted_anomaly}",
        f"- **Threshold verified for checkpoint:** {threshold_verified}",
        "",
        "### Interpretation",
        "",
    ]

    if str(predicted_anomaly).lower() in ("false", "0"):
        lines.append(
            "The current detector did **not** flag this sample as anomalous. "
            "The dataset label is shown separately from the model prediction."
        )
    elif str(predicted_anomaly).lower() in ("true", "1"):
        lines.append(
            "The current detector flagged this sample as anomalous. "
            "This does not independently confirm a particular attack type."
        )
    else:
        lines.append("The detector prediction is unavailable.")

    lines.extend([
        "",
        "## 2. Explainable AI evidence",
        "",
        "Feature values are standardized values, not raw packet counts "
        "or raw network measurements.",
        "",
        "| Rank | Feature | Standardized value | Perturbed score | Contribution |",
        "|---:|---|---:|---:|---:|",
    ])

    for rank, item in enumerate(feature_evidence, start=1):
        lines.append(
            f"| {rank} | {item['feature']} "
            f"| {format_number(item['original_value'])} "
            f"| {format_number(item['perturbed_score'])} "
            f"| {format_number(item['contribution'])} |"
        )

    lines.extend([
        "",
        "A positive contribution means replacing the feature with zero "
        "reduced the anomaly score for this sample. This is a local "
        "perturbation explanation, not proof of causation.",
        "",
        "## 3. MITRE ATT&CK retrieval candidates",
        "",
        "Candidates below were retrieved using behavior-focused TF-IDF "
        "search. Similarity scores are not probabilities and do not "
        "confirm that a technique occurred.",
        "",
    ])

    if not mitre_candidates:
        lines.append("No MITRE candidates were extracted from the JSON file.")
    else:
        lines.extend([
            "| Rank | Technique ID | Technique | TF-IDF similarity | Reference |",
            "|---:|---|---|---:|---|",
        ])

        for item in mitre_candidates:
            url = item["url"]
            reference = f"[Official ATT&CK page]({url})" if url else "N/A"

            lines.append(
                f"| {item['rank']} | {item['id']} | {item['name']} "
                f"| {format_number(item['similarity'])} | {reference} |"
            )

    lines.extend([
        "",
        "## 4. Analyst notes and limitations",
        "",
        "- MITRE matches are candidates, not confirmed classifications.",
        "- TF-IDF similarity is a text-retrieval score, not an attack probability.",
        "- Feature contributions indicate local model sensitivity, not causation.",
        "- The checkpoint threshold has not been verified unless explicitly "
        "confirmed by the XAI summary.",
        "- Review other telemetry and independent evidence before making "
        "an incident decision.",
        "",
        "## 5. Source files",
        "",
        "- `xai_results/xai_top_features.csv`",
        "- `xai_results/xai_summary.json`",
        "- `data/mitre_retrieval_comparison_v2.json`",
        "",
    ])

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("SOC report generated successfully.")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Sample label: {label}")
    print(f"Predicted anomaly: {predicted_anomaly}")
    print(f"Feature rows included: {len(feature_evidence)}")
    print(f"MITRE candidates included: {len(mitre_candidates)}")

    if mitre_candidates:
        print("\nRetrieved MITRE candidates:")
        for item in mitre_candidates:
            print(
                f"  {item['id']} - {item['name']} "
                f"(similarity={item['similarity']})"
            )


if __name__ == "__main__":
    main()
