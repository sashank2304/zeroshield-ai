
"""
ZeroShield AI
Evaluation-only script for the already trained / tuned Deep SVDD model.

It does NOT retrain the model.

It evaluates:
1) validation_set.csv       -> benign + regular attacks
2) zero_day_test_scaled_v2.csv -> held-out attack categories

The very large CSVs are processed chunk-by-chunk to avoid RAM crashes.

Expected saved model:
    model_training/outputs/Deep_SVDD_Tuned.pt
"""

from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

VALID_FILE = ROOT / "validation_set.csv"
ZERO_FILE = ROOT / "zero_day_test_scaled_v2.csv"

MODEL_FILE = (
    ROOT
    / "model_training"
    / "outputs"
    / "Deep_SVDD_Tuned.pt"
)

OUT_DIR = (
    ROOT
    / "model_training"
    / "outputs"
)

RESULTS_FILE = (
    OUT_DIR
    / "Deep_SVDD_Tuned_evaluation_results.json"
)


# ============================================================
# 2. SETTINGS
# ============================================================

CHUNK_SIZE = 50_000
BATCH_SIZE = 4096

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def clean_features(chunk, features):
    """
    Keep a DataFrame while reading.
    Convert to float32 only after cleaning.
    """

    return (
        chunk[features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .astype(np.float32)
    )


def clean_labels(chunk):
    """
    Expected:
        0 = Benign
        1 = Attack

    If binary_label is missing, reconstruct from the original
    label column:
        BenignTraffic -> 0
        all other labels -> 1
    """

    y = pd.to_numeric(
        chunk["binary_label"],
        errors="coerce"
    )

    missing = y.isna()

    if missing.any():

        if "label" not in chunk.columns:

            raise ValueError(
                "binary_label contains missing values, but "
                "label was not loaded for fallback recovery."
            )

        recovered = (
            chunk.loc[
                missing,
                "label"
            ]
            .astype(str)
            .str.strip()
            .ne("BenignTraffic")
            .astype(np.int8)
        )

        y.loc[missing] = recovered.to_numpy()

    if y.isna().any():

        raise ValueError(
            "Some labels remain missing after fallback recovery."
        )

    y = y.astype(
        np.int8
    ).to_numpy()

    invalid = np.setdiff1d(
        np.unique(y),
        np.array([0, 1], dtype=np.int8)
    )

    if len(invalid):

        raise ValueError(
            f"Unexpected binary labels: {invalid.tolist()}"
        )

    return y


def metrics_from_counts(tp, tn, fp, fn):

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    total = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else None
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else None
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "specificity": specificity,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


# ============================================================
# 4. MODEL CLASS
# ============================================================

class DeepSVDDNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden,
        latent_dim
    ):
        super().__init__()

        layers = []
        previous = input_dim

        for width in hidden:

            layers.append(
                nn.Linear(
                    previous,
                    width,
                    bias=False
                )
            )

            layers.append(
                nn.ReLU()
            )

            previous = width

        layers.append(
            nn.Linear(
                previous,
                latent_dim,
                bias=False
            )
        )

        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 5. LOAD SAVED MODEL
# ============================================================

section("LOADING SAVED DEEP SVDD MODEL")

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Saved model not found:\n{MODEL_FILE}"
    )

if not VALID_FILE.exists():

    raise FileNotFoundError(
        f"Validation file not found:\n{VALID_FILE}"
    )

if not ZERO_FILE.exists():

    raise FileNotFoundError(
        f"Zero-day file not found:\n{ZERO_FILE}"
    )


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

features = checkpoint["features"]
hidden = checkpoint["hidden"]
latent_dim = checkpoint["latent_dim"]
threshold = float(
    checkpoint["threshold"]
)

center = checkpoint["center"].to(
    DEVICE
)

model = DeepSVDDNetwork(
    input_dim=len(features),
    hidden=hidden,
    latent_dim=latent_dim
).to(
    DEVICE
)

model.load_state_dict(
    checkpoint["state_dict"]
)

model.eval()


print(
    f"Device              : {DEVICE}"
)

print(
    f"Features            : {len(features)}"
)

print(
    f"Hidden layers       : {hidden}"
)

print(
    f"Latent dimension    : {latent_dim}"
)

print(
    f"Frozen threshold    : {threshold:.10f}"
)

print(
    "\nNO RETRAINING WILL BE PERFORMED."
)


# ============================================================
# 6. SCORING
# ============================================================

@torch.no_grad()
def score_numpy(
    x,
    batch_size=BATCH_SIZE
):

    scores = []

    for start in range(
        0,
        len(x),
        batch_size
    ):

        stop = min(
            start + batch_size,
            len(x)
        )

        tensor = torch.from_numpy(
            x[start:stop]
        ).to(
            DEVICE,
            non_blocking=True
        )

        z = model(
            tensor
        )

        distance = (
            (z - center) ** 2
        ).sum(
            dim=1
        )

        scores.append(
            distance.cpu().numpy()
        )

    return np.concatenate(
        scores
    )


# ============================================================
# 7. VALIDATION EVALUATION
# ============================================================

def evaluate_validation():

    section(
        "REGULAR VALIDATION - CHUNKED EVALUATION"
    )

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    processed = 0

    # IMPORTANT:
    # label is included because some binary_label values
    # were found to be missing in the earlier validation run.

    usecols = (
        features
        + [
            "binary_label",
            "label"
        ]
    )

    reader = pd.read_csv(
        VALID_FILE,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=True
    )

    for chunk_no, chunk in enumerate(
        reader,
        start=1
    ):

        X_df = clean_features(
            chunk,
            features
        )

        X = X_df.to_numpy(
            dtype=np.float32,
            copy=False
        )

        y = clean_labels(
            chunk
        )

        scores = score_numpy(
            X
        )

        predictions = (
            scores >= threshold
        ).astype(
            np.int8
        )

        tp += int(
            (
                (predictions == 1)
                & (y == 1)
            ).sum()
        )

        tn += int(
            (
                (predictions == 0)
                & (y == 0)
            ).sum()
        )

        fp += int(
            (
                (predictions == 1)
                & (y == 0)
            ).sum()
        )

        fn += int(
            (
                (predictions == 0)
                & (y == 1)
            ).sum()
        )

        processed += len(
            chunk
        )

        if (
            chunk_no == 1
            or chunk_no % 20 == 0
        ):

            print(
                f"Processed validation rows: "
                f"{processed:,}"
            )

        del X_df
        del X
        del y
        del scores
        del predictions
        del chunk

    metrics = metrics_from_counts(
        tp,
        tn,
        fp,
        fn
    )

    print("\n" + "-" * 70)
    print("FINAL VALIDATION RESULTS")
    print("-" * 70)

    print(
        f"Rows        : {processed:,}"
    )

    print(
        f"Accuracy    : {metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision   : {metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall      : {metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1-score    : {metrics['f1'] * 100:.2f}%"
    )

    print(
        f"FPR         : {metrics['fpr'] * 100:.2f}%"
    )

    print(
        f"Specificity : {metrics['specificity'] * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        f"TP = {tp:,}"
    )

    print(
        f"TN = {tn:,}"
    )

    print(
        f"FP = {fp:,}"
    )

    print(
        f"FN = {fn:,}"
    )

    return {
        "rows": processed,
        "metrics": metrics
    }


# ============================================================
# 8. ZERO-DAY EVALUATION
# ============================================================

def evaluate_zero_day():

    section(
        "ZERO-DAY - FINAL HELD-OUT EVALUATION"
    )

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    processed = 0

    category_counts = {}

    usecols = (
        features
        + [
            "binary_label",
            "label"
        ]
    )

    reader = pd.read_csv(
        ZERO_FILE,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=True
    )

    for chunk_no, chunk in enumerate(
        reader,
        start=1
    ):

        X_df = clean_features(
            chunk,
            features
        )

        X = X_df.to_numpy(
            dtype=np.float32,
            copy=False
        )

        y = clean_labels(
            chunk
        )

        labels = (
            chunk["label"]
            .astype(str)
            .to_numpy()
        )

        scores = score_numpy(
            X
        )

        predictions = (
            scores >= threshold
        ).astype(
            np.int8
        )

        tp += int(
            (
                (predictions == 1)
                & (y == 1)
            ).sum()
        )

        tn += int(
            (
                (predictions == 0)
                & (y == 0)
            ).sum()
        )

        fp += int(
            (
                (predictions == 1)
                & (y == 0)
            ).sum()
        )

        fn += int(
            (
                (predictions == 0)
                & (y == 1)
            ).sum()
        )


        # Per-category counters
        for category in np.unique(
            labels
        ):

            mask = (
                labels == category
            )

            cy = y[mask]
            cp = predictions[mask]

            ctp = int(
                (
                    (cp == 1)
                    & (cy == 1)
                ).sum()
            )

            ctn = int(
                (
                    (cp == 0)
                    & (cy == 0)
                ).sum()
            )

            cfp = int(
                (
                    (cp == 1)
                    & (cy == 0)
                ).sum()
            )

            cfn = int(
                (
                    (cp == 0)
                    & (cy == 1)
                ).sum()
            )

            if category not in category_counts:

                category_counts[
                    category
                ] = {
                    "tp": 0,
                    "tn": 0,
                    "fp": 0,
                    "fn": 0
                }

            category_counts[
                category
            ]["tp"] += ctp

            category_counts[
                category
            ]["tn"] += ctn

            category_counts[
                category
            ]["fp"] += cfp

            category_counts[
                category
            ]["fn"] += cfn


        processed += len(
            chunk
        )

        if (
            chunk_no == 1
            or chunk_no % 20 == 0
        ):

            print(
                f"Processed zero-day rows: "
                f"{processed:,}"
            )

        del X_df
        del X
        del y
        del labels
        del scores
        del predictions
        del chunk


    metrics = metrics_from_counts(
        tp,
        tn,
        fp,
        fn
    )

    print("\n" + "-" * 70)
    print("FINAL ZERO-DAY RESULTS")
    print("-" * 70)

    print(
        f"Total attacks : "
        f"{tp + fn:,}"
    )

    print(
        f"Detected      : "
        f"{tp:,}"
    )

    print(
        f"Missed        : "
        f"{fn:,}"
    )

    detection_rate = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    print(
        f"Detection rate: "
        f"{detection_rate * 100:.2f}%"
    )

    print(
        "FPR/Specificity: N/A "
        "(zero-day file contains attacks only)"
    )


    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    print(
        "\nZero-day category detection:"
    )

    category_results = {}

    for category in sorted(
        category_counts
    ):

        c = category_counts[
            category
        ]

        total = (
            c["tp"]
            + c["fn"]
        )

        rate = (
            c["tp"] / total
            if total > 0
            else 0.0
        )

        category_results[
            category
        ] = metrics_from_counts(
            c["tp"],
            c["tn"],
            c["fp"],
            c["fn"]
        )

        print(
            f"  {category:25s}: "
            f"{rate * 100:7.2f}% "
            f"({c['tp']:,}/{total:,})"
        )


    return {
        "rows": processed,
        "metrics": metrics,
        "detection_rate": detection_rate,
        "categories": category_results
    }


# ============================================================
# 9. MAIN
# ============================================================

def main():

    validation_result = (
        evaluate_validation()
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    zero_day_result = (
        evaluate_zero_day()
    )

    results = {
        "model": "Deep SVDD",
        "model_file": str(
            MODEL_FILE
        ),
        "threshold": threshold,
        "threshold_source":
            "Loaded from saved tuned model",
        "validation": validation_result,
        "zero_day": zero_day_result
    }

    RESULTS_FILE.write_text(
        json.dumps(
            results,
            indent=4
        ),
        encoding="utf-8"
    )

    section(
        "EVALUATION COMPLETE"
    )

    print(
        f"Results saved to:\n"
        f"{RESULTS_FILE}"
    )

    print(
        "\nThe saved Deep SVDD model was NOT retrained."
    )


if __name__ == "__main__":
    main()
