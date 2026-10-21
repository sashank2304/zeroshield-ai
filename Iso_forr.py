from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


# ============================================================
# ZERO SHIELD AI
# ISOLATION FOREST - FINAL TRAINING + EVALUATION SCRIPT
# ============================================================
#
# TRAINING DATA:
#   train_benign_only.csv
#   -> BENIGN ONLY
#
# REGULAR EVALUATION:
#   validation_set.csv
#   -> held-out benign + regular attacks
#
# ZERO-DAY EVALUATION:
#   zero_day_test_scaled_v2.csv
#   -> Mirai + Web-Based attack categories
#
# IMPORTANT:
#   Validation and zero-day files are processed in chunks
#   to avoid loading huge CSV files into RAM.
#
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_FILE = PROJECT_ROOT / "train_benign_only.csv"
VALID_FILE = PROJECT_ROOT / "validation_set.csv"
ZERO_DAY_FILE = PROJECT_ROOT / "zero_day_test_scaled_v2.csv"

OUTPUT_DIR = PROJECT_ROOT / "model_training" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "isolation_forest_model.joblib"
RESULTS_FILE = OUTPUT_DIR / "isolation_forest_final_results.json"


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 200

# Used only for defining the anomaly threshold.
# Threshold = 99th percentile of benign training scores.
THRESHOLD_PERCENTILE = 99

# Memory-safe evaluation
VALIDATION_CHUNK_SIZE = 50_000
ZERO_DAY_CHUNK_SIZE = 50_000


# ============================================================
# 3. CHECK FILES
# ============================================================

print("=" * 70)
print("ZERO SHIELD AI - ISOLATION FOREST")
print("=" * 70)

required_files = {
    "Training": TRAIN_FILE,
    "Validation": VALID_FILE,
    "Zero-Day": ZERO_DAY_FILE
}

for name, path in required_files.items():

    if not path.exists():

        raise FileNotFoundError(
            f"\n{name} file not found:\n{path}"
        )

    print(f"{name} file found:")
    print(f"  {path}")


# ============================================================
# 4. READ TRAINING HEADER
# ============================================================
#
# We read only the header first so that we know the exact
# feature columns.
#
# label and binary_label are NOT model features.
#
# ============================================================

print("\nReading training file header...")

train_header = pd.read_csv(
    TRAIN_FILE,
    nrows=0
)

excluded_columns = {
    "label",
    "binary_label"
}

feature_columns = [
    column
    for column in train_header.columns
    if column not in excluded_columns
]

print(f"\nNumber of features: {len(feature_columns)}")

print("\nFeature columns:")

for i, feature in enumerate(feature_columns, start=1):
    print(f"{i:02d}. {feature}")


# ============================================================
# 5. LOAD BENIGN TRAINING DATA
# ============================================================
#
# IMPORTANT:
#
# train_benign_only.csv contains benign rows only.
#
# This is the ONLY dataset used for model.fit().
#
# ============================================================

print("\n" + "=" * 70)
print("LOADING BENIGN TRAINING DATA")
print("=" * 70)

train_df = pd.read_csv(
    TRAIN_FILE,
    usecols=feature_columns,
    low_memory=True
)

print(
    f"\nTraining matrix: "
    f"{train_df.shape[0]:,} rows × "
    f"{train_df.shape[1]} features"
)


# ============================================================
# 6. CLEAN TRAINING FEATURES
# ============================================================

X_train = (
    train_df
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
    .astype(np.float32)
)

del train_df

print(
    f"Training matrix after cleaning: "
    f"{X_train.shape[0]:,} × {X_train.shape[1]}"
)


# ============================================================
# 7. TRAIN ISOLATION FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ISOLATION FOREST")
print("=" * 70)

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    max_samples="auto",
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train)

print("\nIsolation Forest training completed.")


# ============================================================
# 8. CALCULATE BENIGN TRAINING ANOMALY SCORES
# ============================================================
#
# sklearn's decision_function:
#
#   higher value -> more normal
#
# We negate it:
#
#   higher value -> more anomalous
#
# This keeps the scoring convention identical to our
# Autoencoder / VAE / Deep SVDD experiments.
#
# ============================================================

print("\nCalculating benign training anomaly scores...")

train_scores = -model.decision_function(X_train)


# ============================================================
# 9. DETERMINE THRESHOLD
# ============================================================
#
# IMPORTANT:
#
# Threshold is derived ONLY from benign training data.
#
# The validation set and zero-day test set are NOT used.
#
# 99th percentile means that approximately the most extreme
# 1% of benign training scores lie above the threshold.
#
# ============================================================

threshold = float(
    np.percentile(
        train_scores,
        THRESHOLD_PERCENTILE
    )
)

print(
    f"\nAnomaly threshold: "
    f"{threshold:.6f}"
)

print(
    f"Threshold rule: "
    f"{THRESHOLD_PERCENTILE}th percentile "
    f"of benign training scores"
)


# ============================================================
# 10. SAVE TRAINED MODEL
# ============================================================

model_package = {
    "model": model,
    "features": feature_columns,
    "threshold": threshold,
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "n_estimators": N_ESTIMATORS
}

joblib.dump(
    model_package,
    MODEL_FILE
)

print("\nModel saved to:")
print(MODEL_FILE)


# ============================================================
# 11. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    tp,
    tn,
    fp,
    fn
):
    """
    Calculate classification metrics from confusion-matrix counts.
    """

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
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

    # FPR is undefined if there are no benign observations.
    if (fp + tn) == 0:
        fpr = None
    else:
        fpr = fp / (fp + tn)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


# ============================================================
# 12. REGULAR VALIDATION EVALUATION
# ============================================================
#
# validation_set.csv is very large (~30 GB), therefore:
#
#     read chunk
#         ↓
#     predict
#         ↓
#     update TP/TN/FP/FN
#         ↓
#     discard chunk
#         ↓
#     next chunk
#
# No full validation DataFrame is created.
#
# ============================================================

print("\n" + "=" * 70)
print("REGULAR VALIDATION - CHUNKED EVALUATION")
print("=" * 70)

validation_tp = 0
validation_tn = 0
validation_fp = 0
validation_fn = 0

processed_validation_rows = 0

validation_usecols = (
    feature_columns +
    ["binary_label"]
)


validation_reader = pd.read_csv(
    VALID_FILE,
    usecols=validation_usecols,
    chunksize=VALIDATION_CHUNK_SIZE,
    low_memory=True
)


for chunk_number, chunk in enumerate(
    validation_reader,
    start=1
):

    X_chunk = (
    chunk[feature_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
    .astype(np.float32)

    )

    y_chunk = (
        chunk["binary_label"]
        .astype(np.int8)
        .to_numpy()
    )


    # Generate anomaly scores.
    #
    # Higher score = more anomalous.

    scores = -model.decision_function(
        X_chunk
    )


    # Convert score to prediction.
    #
    # 1 = Attack
    # 0 = Benign

    predictions = (
        scores >= threshold
    ).astype(np.int8)


    # --------------------------------------------------------
    # Update confusion matrix counts
    # --------------------------------------------------------

    validation_tp += int(
        ((predictions == 1) & (y_chunk == 1)).sum()
    )

    validation_tn += int(
        ((predictions == 0) & (y_chunk == 0)).sum()
    )

    validation_fp += int(
        ((predictions == 1) & (y_chunk == 0)).sum()
    )

    validation_fn += int(
        ((predictions == 0) & (y_chunk == 1)).sum()
    )


    processed_validation_rows += len(chunk)


    # Progress display

    if (
        chunk_number == 1
        or chunk_number % 20 == 0
    ):

        print(
            f"Processed validation rows: "
            f"{processed_validation_rows:,}"
        )


# ============================================================
# 13. VALIDATION METRICS
# ============================================================

validation_metrics = calculate_metrics(
    validation_tp,
    validation_tn,
    validation_fp,
    validation_fn
)


print("\n" + "=" * 70)
print("ISOLATION FOREST - REGULAR VALIDATION RESULTS")
print("=" * 70)

print(
    f"Precision : "
    f"{validation_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{validation_metrics['recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{validation_metrics['f1']:.4f}"
)

print(
    f"Accuracy  : "
    f"{validation_metrics['accuracy']:.4f}"
)

print(
    f"FPR       : "
    f"{validation_metrics['fpr']:.4f}"
)

print("\nConfusion Matrix:")

print(
    f"TP = {validation_metrics['tp']:,}"
)

print(
    f"TN = {validation_metrics['tn']:,}"
)

print(
    f"FP = {validation_metrics['fp']:,}"
)

print(
    f"FN = {validation_metrics['fn']:,}"
)


# ============================================================
# 14. ZERO-DAY EVALUATION
# ============================================================
#
# zero_day_test_scaled_v2.csv is evaluated separately.
#
# It contains the held-out attack categories.
#
# It is NEVER used for:
#   - training
#   - threshold selection
#   - hyperparameter tuning
#
# ============================================================

print("\n" + "=" * 70)
print("ZERO-DAY EVALUATION - CHUNKED")
print("=" * 70)

zero_tp = 0
zero_tn = 0
zero_fp = 0
zero_fn = 0

processed_zero_rows = 0

# Fine-grained category counters
category_counts = {}


zero_usecols = (
    feature_columns +
    ["label", "binary_label"]
)


zero_reader = pd.read_csv(
    ZERO_DAY_FILE,
    usecols=zero_usecols,
    chunksize=ZERO_DAY_CHUNK_SIZE,
    low_memory=True
)


for chunk_number, chunk in enumerate(
    zero_reader,
    start=1
):

    X_chunk = (
    chunk[feature_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
    .astype(np.float32)

    )

    y_chunk = (
        chunk["binary_label"]
        .astype(np.int8)
        .to_numpy()
    )

    labels_chunk = (
        chunk["label"]
        .astype(str)
        .to_numpy()
    )


    # --------------------------------------------------------
    # Generate anomaly scores
    # --------------------------------------------------------

    scores = -model.decision_function(X_chunk)

    predictions = (
        scores >= threshold
    ).astype(np.int8)


    # --------------------------------------------------------
    # Overall zero-day counts
    # --------------------------------------------------------

    zero_tp += int(
        ((predictions == 1) & (y_chunk == 1)).sum()
    )

    zero_tn += int(
        ((predictions == 0) & (y_chunk == 0)).sum()
    )

    zero_fp += int(
        ((predictions == 1) & (y_chunk == 0)).sum()
    )

    zero_fn += int(
        ((predictions == 0) & (y_chunk == 1)).sum()
    )


    # --------------------------------------------------------
    # Individual attack category counts
    # --------------------------------------------------------

    for category in np.unique(
        labels_chunk
    ):

        mask = (
            labels_chunk == category
        )

        y_category = y_chunk[mask]
        pred_category = predictions[mask]

        tp = int(
            (
                (pred_category == 1) &
                (y_category == 1)
            ).sum()
        )

        tn = int(
            (
                (pred_category == 0) &
                (y_category == 0)
            ).sum()
        )

        fp = int(
            (
                (pred_category == 1) &
                (y_category == 0)
            ).sum()
        )

        fn = int(
            (
                (pred_category == 0) &
                (y_category == 1)
            ).sum()
        )


        if category not in category_counts:

            category_counts[category] = {
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0
            }


        category_counts[category]["tp"] += tp
        category_counts[category]["tn"] += tn
        category_counts[category]["fp"] += fp
        category_counts[category]["fn"] += fn


    processed_zero_rows += len(chunk)


    if (
        chunk_number == 1
        or chunk_number % 20 == 0
    ):

        print(
            f"Processed zero-day rows: "
            f"{processed_zero_rows:,}"
        )


# ============================================================
# 15. ZERO-DAY OVERALL METRICS
# ============================================================

zero_day_metrics = calculate_metrics(
    zero_tp,
    zero_tn,
    zero_fp,
    zero_fn
)


print("\n" + "=" * 70)
print("ISOLATION FOREST - ZERO-DAY OVERALL")
print("=" * 70)

print(
    f"Precision : "
    f"{zero_day_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{zero_day_metrics['recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{zero_day_metrics['f1']:.4f}"
)

print(
    f"Accuracy  : "
    f"{zero_day_metrics['accuracy']:.4f}"
)

if zero_day_metrics["fpr"] is None:

    print(
        "FPR       : N/A "
        "(zero-day dataset contains no benign samples)"
    )

else:

    print(
        f"FPR       : "
        f"{zero_day_metrics['fpr']:.4f}"
    )


print("\nConfusion Matrix:")

print(
    f"TP = {zero_day_metrics['tp']:,}"
)

print(
    f"TN = {zero_day_metrics['tn']:,}"
)

print(
    f"FP = {zero_day_metrics['fp']:,}"
)

print(
    f"FN = {zero_day_metrics['fn']:,}"
)


# ============================================================
# 16. ZERO-DAY RESULTS BY INDIVIDUAL CATEGORY
# ============================================================

print("\n" + "=" * 70)
print("ZERO-DAY RESULTS BY ATTACK CATEGORY")
print("=" * 70)

category_results = {}


for category in sorted(
    category_counts.keys()
):

    counts = category_counts[
        category
    ]

    metrics = calculate_metrics(
        counts["tp"],
        counts["tn"],
        counts["fp"],
        counts["fn"]
    )

    category_results[category] = metrics


    print(
        f"\n{category}"
    )

    print(
        f"  Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"  Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"  F1 Score  : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"  TP = {metrics['tp']:,} "
        f"| FN = {metrics['fn']:,}"
    )


# ============================================================
# 17. MIRAI VS WEB-BASED GROUPING
# ============================================================
#
# This gives us the higher-level comparison requested in
# the project notes:
#
#   Mirai
#   Web-Based
#
# while retaining the individual categories above.
#
# ============================================================

mirai_categories = {
    "Mirai-greeth_flood",
    "Mirai-greip_flood",
    "Mirai-udpplain"
}

web_based_categories = {
    "Backdoor_Malware",
    "SqlInjection",
    "CommandInjection",
    "Uploading_Attack",
    "XSS",
    "BrowserHijacking"
}


group_counts = {
    "Mirai": {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0
    },

    "Web-Based": {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0
    }
}


for category, counts in category_counts.items():

    if category in mirai_categories:

        group = "Mirai"

    elif category in web_based_categories:

        group = "Web-Based"

    else:

        continue


    for key in [
        "tp",
        "tn",
        "fp",
        "fn"
    ]:

        group_counts[group][key] += (
            counts[key]
        )


group_results = {}


for group, counts in group_counts.items():

    metrics = calculate_metrics(
        counts["tp"],
        counts["tn"],
        counts["fp"],
        counts["fn"]
    )

    group_results[group] = metrics


print("\n" + "=" * 70)
print("ZERO-DAY RESULTS: MIRAI VS WEB-BASED")
print("=" * 70)


for group in [
    "Mirai",
    "Web-Based"
]:

    metrics = group_results[group]

    print(
        f"\n{group}"
    )

    print(
        f"  Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"  Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"  F1 Score  : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"  TP = {metrics['tp']:,} "
        f"| FN = {metrics['fn']:,}"
    )


# ============================================================
# 18. SAVE ALL RESULTS
# ============================================================

final_results = {

    "model": "Isolation Forest",

    "training": {
        "file": str(TRAIN_FILE),
        "rows": int(len(X_train)),
        "features": int(len(feature_columns)),
        "feature_names": feature_columns,
        "n_estimators": N_ESTIMATORS
    },

    "threshold": {
        "value": threshold,
        "rule": (
            "99th percentile of benign "
            "training anomaly scores"
        )
    },

    "validation": validation_metrics,

    "zero_day_overall": zero_day_metrics,

    "zero_day_mirai_vs_web_based": group_results,

    "zero_day_by_category": category_results
}


with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_results,
        file,
        indent=4
    )


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ISOLATION FOREST EXPERIMENT COMPLETE")
print("=" * 70)

print("\nModel:")
print(MODEL_FILE)

print("\nResults:")
print(RESULTS_FILE)

print("\nTraining:")
print(
    f"{len(X_train):,} benign rows "
    f"× {len(feature_columns)} features"
)

print(
    f"\nThreshold: {threshold:.6f}"
)

print("\nValidation results:")

print(
    f"Precision = "
    f"{validation_metrics['precision']:.4f}"
)

print(
    f"Recall    = "
    f"{validation_metrics['recall']:.4f}"
)

print(
    f"F1        = "
    f"{validation_metrics['f1']:.4f}"
)

print(
    f"Accuracy  = "
    f"{validation_metrics['accuracy']:.4f}"
)

print(
    f"FPR       = "
    f"{validation_metrics['fpr']:.4f}"
    if validation_metrics["fpr"] is not None
    else "FPR       = N/A"
)

print("\nZero-day results:")

print(
    f"Recall    = "
    f"{zero_day_metrics['recall']:.4f}"
)

print(
    f"F1        = "
    f"{zero_day_metrics['f1']:.4f}"
)

print("\nDone.")