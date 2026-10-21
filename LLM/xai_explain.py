
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# --------------------------------------------------
# Configuration
# --------------------------------------------------
CHECKPOINT_PATH = Path("Deep_SVDD.pt")
DATASET_PATH = Path("Dataset/zero_day_test_scaled_v2.csv")
OUTPUT_DIR = Path("xai_results")

# Start small. Increase after validating the results.
SAMPLES_TO_EXPLAIN = 20
TOP_K = 10
BATCH_SIZE = 256

# This must match the threshold validated for this checkpoint.
# Treat this as provisional until confirmed.
ANOMALY_THRESHOLD = 0.0614192100


# --------------------------------------------------
# Model architecture from the actual checkpoint
# --------------------------------------------------
class DeepSVDD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(43, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    feature_names = checkpoint["features"]
    input_dim = int(checkpoint["input_dim"])
    latent_dim = int(checkpoint["latent_dim"])

    if input_dim != 43 or len(feature_names) != input_dim:
        raise ValueError(
            f"Unexpected input configuration: input_dim={input_dim}, "
            f"feature_count={len(feature_names)}"
        )

    model = DeepSVDD()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    center = torch.as_tensor(
        checkpoint["center"], dtype=torch.float32
    ).reshape(1, -1)

    if center.shape[1] != latent_dim:
        raise ValueError("Checkpoint center dimension mismatch.")

    print("Checkpoint loaded successfully.")
    print("Architecture: 43 -> 64 -> 32 -> 16")
    print("Feature count:", len(feature_names))

    # Read only a small number of rows for the first test.
    df = pd.read_csv(DATASET_PATH, nrows=SAMPLES_TO_EXPLAIN)

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = df[feature_names].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError(
            "Input contains NaN or infinite values. "
            "Handle these using the same preprocessing as training."
        )

    X_tensor = torch.from_numpy(X)

    # --------------------------------------------------
    # Original anomaly scores
    # score(x) = squared distance from benign center
    # --------------------------------------------------
    with torch.no_grad():
        embeddings = model(X_tensor)
        original_scores = (
            (embeddings - center) ** 2
        ).sum(dim=1).numpy()

    rows = []

    # --------------------------------------------------
    # Feature-perturbation explanations
    # Replace one standardized feature with 0 at a time.
    # Contribution = original score - perturbed score.
    # --------------------------------------------------
    for sample_idx in range(len(X)):
        original_score = float(original_scores[sample_idx])
        contributions = []

        for feature_idx, feature_name in enumerate(feature_names):
            perturbed = X_tensor[sample_idx:sample_idx + 1].clone()
            perturbed[0, feature_idx] = 0.0

            with torch.no_grad():
                embedding = model(perturbed)
                perturbed_score = float(
                    ((embedding - center) ** 2).sum().item()
                )

            contribution = original_score - perturbed_score

            contributions.append({
                "feature": feature_name,
                "original_value": float(X[sample_idx, feature_idx]),
                "perturbed_score": perturbed_score,
                "contribution": contribution,
            })

        # Rank features by contribution, largest first.
        contributions.sort(
            key=lambda item: item["contribution"],
            reverse=True,
        )

        label = (
            str(df.iloc[sample_idx]["label"])
            if "label" in df.columns else "unknown"
        )

        binary_label = (
            df.iloc[sample_idx]["binary_label"]
            if "binary_label" in df.columns else None
        )

        predicted_anomaly = original_score > ANOMALY_THRESHOLD

        for rank, item in enumerate(contributions, start=1):
            rows.append({
                "sample_index": sample_idx,
                "label": label,
                "binary_label": (
                    None if pd.isna(binary_label)
                    else str(binary_label)
                ),
                "anomaly_score": original_score,
                "threshold": ANOMALY_THRESHOLD,
                "predicted_anomaly": predicted_anomaly,
                "feature_rank": rank,
                "feature": item["feature"],
                "original_value": item["original_value"],
                "perturbed_score": item["perturbed_score"],
                "contribution": item["contribution"],
                "in_top_k": rank <= TOP_K,
            })

    results = pd.DataFrame(rows)

    all_path = OUTPUT_DIR / "xai_per_feature.csv"
    top_path = OUTPUT_DIR / "xai_top_features.csv"
    summary_path = OUTPUT_DIR / "xai_summary.json"

    results.to_csv(all_path, index=False)
    results[results["in_top_k"]].to_csv(top_path, index=False)

    summary = {
        "model": "Deep SVDD",
        "checkpoint": str(CHECKPOINT_PATH),
        "architecture": [43, 64, 32, 16],
        "latent_dimension": latent_dim,
        "score_definition": "squared Euclidean distance from center",
        "perturbation": "replace one standardized feature with zero",
        "contribution_definition": (
            "original anomaly score minus perturbed anomaly score"
        ),
        "threshold": ANOMALY_THRESHOLD,
        "threshold_verified_for_checkpoint": False,
        "samples_explained": len(X),
        "features_per_sample": len(feature_names),
        "top_k": TOP_K,
        "files": [
            str(all_path),
            str(top_path),
        ],
        "caveat": (
            "Local perturbation sensitivity, not causal proof. "
            "Threshold must be verified for this exact checkpoint."
        ),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nXAI test completed.")
    print("Samples explained:", len(X))
    print("Feature explanations:", len(results))
    print("Anomalies above provisional threshold:",
          int((original_scores > ANOMALY_THRESHOLD).sum()))
    print("All feature explanations:", all_path)
    print("Top features:", top_path)
    print("Summary:", summary_path)

    print("\nFirst sample's top features:")
    print(
        results[
            (results["sample_index"] == 0)
            & (results["in_top_k"])
        ][
            ["feature_rank", "feature", "original_value", "contribution"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
