
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


XAI_PATH = Path("xai_results/xai_top_features.csv")
INDEX_PATH = Path("rag_store/mitre_tfidf.pkl")
OUTPUT_PATH = Path("data/xai_mitre_rag_result.json")


def main():
    sample_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    if not XAI_PATH.exists() or not INDEX_PATH.exists():
        raise FileNotFoundError(
            "Missing XAI CSV or RAG index. Build the MITRE index first."
        )

    # Load XAI explanations for the selected sample.
    df = pd.read_csv(XAI_PATH)
    sample = df[df["sample_index"] == sample_index].copy()

    if sample.empty:
        raise ValueError(
            f"Sample index {sample_index} not found in XAI CSV."
        )

    first = sample.iloc[0]

    top = (
        sample[
            sample["in_top_k"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        ]
        .sort_values("feature_rank")
        .head(10)
    )

    features = [
        {
            "feature": str(row["feature"]),
            "rank": int(row["feature_rank"]),
            "contribution": float(row["contribution"]),
            "original_value": float(row["original_value"]),
        }
        for _, row in top.iterrows()
    ]

    label = str(first["label"])
    score = float(first["anomaly_score"])
    threshold = float(first["threshold"])

    # Parse the prediction robustly, including CSV strings.
    prediction_value = first["predicted_anomaly"]
    predicted_anomaly = (
        str(prediction_value).strip().lower() in ("true", "1", "yes")
    )

    # Build retrieval context from XAI feature evidence.
    # Feature presence does not prove a specific attack.
    feature_names = {item["feature"] for item in features}

    evidence_terms = [
        "network protocol characteristics",
        "packet header features",
        "network traffic behavior",
    ]

    if {"syn_flag_number", "syn_count", "SYNRatio"} & feature_names:
        evidence_terms.append(
            "TCP SYN flag and connection characteristics"
        )

    if {"ack_flag_number", "ack_count", "ACKRatio"} & feature_names:
        evidence_terms.append(
            "TCP ACK flag characteristics"
        )

    if {"Rate", "Drate", "PacketDensity", "Number"} & feature_names:
        evidence_terms.append(
            "traffic rate and packet density features"
        )

    if {"TCP", "UDP", "Protocol Type"} & feature_names:
        evidence_terms.append(
            "TCP UDP and network protocol features"
        )

    if predicted_anomaly:
        evidence_terms.append(
            "anomalous network activity requiring investigation"
        )

    # Do not use the dataset label to steer retrieval.
    query = (
        "MITRE ATT&CK techniques related to "
        + ", ".join(evidence_terms)
    )

    # Load the existing MITRE TF-IDF index.
    with INDEX_PATH.open("rb") as f:
        store = pickle.load(f)

    vectorizer = store["vectorizer"]
    matrix = store["matrix"]
    documents = store["documents"]

    # Retrieve the most textually similar MITRE techniques.
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked = scores.argsort()[::-1]

    candidates = []

    for idx in ranked:
        if scores[idx] <= 0:
            continue

        doc = documents[idx]

        candidates.append({
            "technique_id": doc["id"],
            "name": doc["name"],
            "relevance_score": round(float(scores[idx]), 4),
            "tactics": doc["tactics"],
            "platforms": doc["platforms"],
            "description": (
                doc["text"].split("Description: ", 1)[-1][:1200]
            ),
            "official_url": doc["url"],
        })

        if len(candidates) == 5:
            break

    # Construct the result for downstream analysis or LLM use.
    binary_label = first["binary_label"]
    if pd.isna(binary_label):
        binary_label = None
    else:
        binary_label = int(binary_label)

    result = {
        "pipeline": "ZeroShield XAI-to-MITRE RAG candidate retrieval",
        "sample_index": sample_index,
        "dataset_label": label,
        "binary_label": binary_label,
        "detector": {
            "name": "Deep SVDD",
            "anomaly_score": score,
            "threshold": threshold,
            "predicted_anomaly": predicted_anomaly,
            "score_exceeds_threshold": score > threshold,
            "threshold_verified": False,
        },
        "xai_method_note": (
            "Feature contributions are local perturbation sensitivities. "
            "They are not causal proof or direct evidence of an "
            "MITRE ATT&CK technique."
        ),
        "feature_value_note": (
            "Feature values are standardized model inputs. Do not interpret "
            "them as raw packet counts or physical traffic rates without "
            "verified preprocessing information."
        ),
        "top_features": features,
        "retrieval_query": query,
        "retrieval_method": "TF-IDF cosine similarity",
        "candidate_techniques": candidates,
        "confirmation": False,
        "caveat": (
            "Retrieved techniques are candidates ranked by text similarity, "
            "not confirmed attacks. The dataset label is retained for "
            "evaluation but is not used to construct the retrieval query. "
            "Feature presence alone does not establish attack behavior."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Display results.
    print("Sample:", sample_index, "| Label:", label)
    print("Anomaly score:", f"{score:.6f}")
    print("Threshold:", f"{threshold:.6f}")
    print("Predicted anomaly:", predicted_anomaly)

    print("\nRetrieval query:")
    print(query)

    print("\nTop XAI features:")
    for item in features[:5]:
        print(
            f"  {item['rank']}. {item['feature']} "
            f"(contribution={item['contribution']:.6f})"
        )

    print("\nRetrieved MITRE candidates:")
    if not candidates:
        print("  No candidates with a positive similarity score.")

    for item in candidates:
        print(
            f"  {item['technique_id']} - {item['name']} "
            f"(similarity={item['relevance_score']:.4f})"
        )
        print("   ", item["official_url"])

    print("\nConfirmation:", result["confirmation"])
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
