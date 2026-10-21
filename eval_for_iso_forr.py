from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# ZERO SHIELD AI
# ISOLATION FOREST - VALIDATION EVALUATION ONLY
#
# IMPORTANT:
# The Isolation Forest has ALREADY been trained.
# This script loads the saved model and evaluates the
# 30+ GB validation CSV in chunks.
#
# No retraining is performed.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

VALID_FILE = PROJECT_ROOT / "validation_set.csv"

MODEL_FILE = (
    PROJECT_ROOT
    / "model_training"
    / "outputs"
    / "isolation_forest_model.joblib"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "model_training"
    / "outputs"
    / "isolation_forest_validation_results.json"
)


# ============================================================
# 2. SETTINGS
# ============================================================

CHUNK_SIZE = 50_000


# ============================================================
# 3. CHECK FILES
# ============================================================

print("=" * 70)
print("ZERO SHIELD AI - ISOLATION FOREST VALIDATION EVALUATION")
print("=" * 70)

if not VALID_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found:\n{VALID_FILE}"
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Saved Isolation Forest not found:\n{MODEL_FILE}"
    )


# ============================================================
# 4. LOAD SAVED MODEL
# ============================================================

print("\nLoading saved Isolation Forest...")

model_package = joblib.load(MODEL_FILE)

model = model_package["model"]
feature_columns = model_package["features"]
threshold = float(model_package["threshold"])

print("Saved model loaded.")

print(
    f"Number of features: "
    f"{len(feature_columns)}"
)

print(
    f"Stored threshold: "
    f"{threshold:.6f}"
)


# ============================================================
# 5. READ VALIDATION HEADER
# ============================================================

print("\nReading validation header...")

header = pd.read_csv(
    VALID_FILE,
    nrows=0
)

validation_columns = set(header.columns)


# Make sure all features exist.

missing_features = [
    feature
    for feature in feature_columns
    if feature not in validation_columns
]

if missing_features:

    raise ValueError(
        "The following model features are missing "
        f"from validation_set.csv:\n{missing_features}"
    )


if "binary_label" not in validation_columns:

    raise ValueError(
        "binary_label column is missing from validation_set.csv"
    )


# We ALSO load label because it provides a fallback if
# binary_label contains missing values.

if "label" not in validation_columns:

    raise ValueError(
        "label column is missing from validation_set.csv. "
        "We need it to safely reconstruct missing binary labels."
    )


# ============================================================
# 6. CONFUSION-MATRIX COUNTERS
# ============================================================

tp = 0
tn = 0
fp = 0
fn = 0

processed_rows = 0

missing_binary_labels = 0
reconstructed_binary_labels = 0


# ============================================================
# 7. CHUNKED VALIDATION READER
# ============================================================

usecols = (
    feature_columns
    + ["binary_label", "label"]
)

reader = pd.read_csv(
    VALID_FILE,
    usecols=usecols,
    chunksize=CHUNK_SIZE,
    low_memory=True
)


# ============================================================
# 8. PROCESS VALIDATION CHUNKS
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING VALIDATION DATA")
print("=" * 70)


for chunk_number, chunk in enumerate(
    reader,
    start=1
):

    # --------------------------------------------------------
    # 8.1 CLEAN FEATURES
    # --------------------------------------------------------
    #
    # Keep the DataFrame structure here.
    #
    # This avoids the sklearn warning about losing feature
    # names when decision_function() is called.
    #

    X_chunk = (
        chunk[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .astype(np.float32)
    )


    # --------------------------------------------------------
    # 8.2 PREPARE BINARY LABEL
    # --------------------------------------------------------
    #
    # Expected:
    #
    # 0 = Benign
    # 1 = Attack
    #
    # We first try the binary_label column.
    #
    # If a missing value exists, we reconstruct it from
    # the original label column:
    #
    # BenignTraffic -> 0
    # anything else  -> 1
    #
    # This follows the documented binary-label definition.
    #

    binary_raw = pd.to_numeric(
        chunk["binary_label"],
        errors="coerce"
    )

    missing_mask = binary_raw.isna()

    missing_count = int(
        missing_mask.sum()
    )

    if missing_count > 0:

        missing_binary_labels += missing_count

        reconstructed = (
            chunk.loc[
                missing_mask,
                "label"
            ]
            .astype(str)
            .str.strip()
            .ne("BenignTraffic")
            .astype(np.int8)
        )

        binary_raw.loc[
            missing_mask
        ] = reconstructed.to_numpy()

        reconstructed_binary_labels += (
            missing_count
        )


    # --------------------------------------------------------
    # 8.3 FINAL LABEL VALIDATION
    # --------------------------------------------------------

    # Check whether anything is STILL missing.

    if binary_raw.isna().any():

        bad_rows = int(
            binary_raw.isna().sum()
        )

        raise ValueError(
            f"\nUnable to reconstruct "
            f"{bad_rows} binary labels.\n"
            "The corresponding label values are also missing."
        )


    # Check that labels are actually 0/1.

    y_chunk = binary_raw.astype(
        np.int8
    ).to_numpy()


    unique_values = np.unique(
        y_chunk
    )

    invalid_values = [
        value
        for value in unique_values
        if value not in (0, 1)
    ]

    if invalid_values:

        raise ValueError(
            "Unexpected binary_label values found: "
            f"{invalid_values}"
        )


    # --------------------------------------------------------
    # 8.4 GENERATE ANOMALY SCORES
    # --------------------------------------------------------
    #
    # sklearn Isolation Forest:
    #
    # higher decision_function = more normal
    #
    # We negate it:
    #
    # higher anomaly score = more anomalous
    #

    anomaly_scores = (
        -model.decision_function(
            X_chunk
        )
    )


    # --------------------------------------------------------
    # 8.5 MAKE PREDICTIONS
    # --------------------------------------------------------
    #
    # 1 = Attack
    # 0 = Benign
    #

    predictions = (
        anomaly_scores >= threshold
    ).astype(np.int8)


    # --------------------------------------------------------
    # 8.6 UPDATE CONFUSION MATRIX
    # --------------------------------------------------------

    tp += int(
        (
            (predictions == 1) &
            (y_chunk == 1)
        ).sum()
    )

    tn += int(
        (
            (predictions == 0) &
            (y_chunk == 0)
        ).sum()
    )

    fp += int(
        (
            (predictions == 1) &
            (y_chunk == 0)
        ).sum()
    )

    fn += int(
        (
            (predictions == 0) &
            (y_chunk == 1)
        ).sum()
    )


    processed_rows += len(
        chunk
    )


    # --------------------------------------------------------
    # 8.7 PROGRESS
    # --------------------------------------------------------

    if (
        chunk_number == 1
        or chunk_number % 20 == 0
    ):

        print(
            f"Processed validation rows: "
            f"{processed_rows:,}"
        )


# ============================================================
# 9. CALCULATE METRICS
# ============================================================

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
    2 * precision * recall /
    (precision + recall)
    if precision + recall > 0
    else 0.0
)

total = (
    tp +
    tn +
    fp +
    fn
)

accuracy = (
    (tp + tn) / total
    if total > 0
    else 0.0
)

fpr = (
    fp / (fp + tn)
    if fp + tn > 0
    else 0.0
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ISOLATION FOREST - VALIDATION RESULTS")
print("=" * 70)

print(
    f"\nValidation rows processed: "
    f"{processed_rows:,}"
)

print(
    f"\nPrecision : "
    f"{precision:.4f}"
)

print(
    f"Recall    : "
    f"{recall:.4f}"
)

print(
    f"F1 Score  : "
    f"{f1:.4f}"
)

print(
    f"Accuracy  : "
    f"{accuracy:.4f}"
)

print(
    f"FPR       : "
    f"{fpr:.4f}"
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

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


# ============================================================
# 12. DATA-QUALITY INFORMATION
# ============================================================

print("\nValidation label handling:")

print(
    f"Missing binary_label values found: "
    f"{missing_binary_labels:,}"
)

print(
    f"Binary labels reconstructed from label: "
    f"{reconstructed_binary_labels:,}"
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results = {

    "model": "Isolation Forest",

    "threshold": threshold,

    "threshold_rule":
        "99th percentile of benign training scores",

    "validation_rows":
        int(processed_rows),

    "validation_metrics": {

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "accuracy": accuracy,

        "fpr": fpr,

        "tp": tp,

        "tn": tn,

        "fp": fp,

        "fn": fn
    },

    "label_handling": {

        "missing_binary_labels":
            missing_binary_labels,

        "reconstructed_from_original_label":
            reconstructed_binary_labels
    }
}


RESULTS_FILE.write_text(
    json.dumps(
        results,
        indent=4
    ),
    encoding="utf-8"
)


# ============================================================
# 14. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    RESULTS_FILE
)