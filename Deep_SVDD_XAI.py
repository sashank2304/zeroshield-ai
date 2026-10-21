
"""
ZeroShield AI
Deep SVDD -> Feature-Level XAI

Purpose
-------
Explain WHY a trained Deep SVDD model considers a flow anomalous.

This script DOES NOT retrain or modify Deep SVDD.

XAI method
----------
Feature occlusion / perturbation:

1. Calculate the original Deep SVDD anomaly score.
2. Temporarily replace one standardized feature with 0.
   (Because the prepared data is StandardScaler output, 0 represents
    the training-data mean for that feature.)
3. Recalculate the anomaly score.
4. Contribution_i = original_score - masked_score

Interpretation:
    Positive contribution:
        removing the feature reduces the anomaly score.
        Therefore the feature contributed to the anomaly.

    Negative contribution:
        removing the feature increases the anomaly score.
        Therefore the feature was making the flow look more normal.

This is a direct feature-level explanation of the Deep SVDD
anomaly score. It avoids forcing SHAP onto a one-class anomaly
score before verifying that a particular SHAP formulation is
appropriate.

Input:
    validation_set.csv
    zero_day_test_scaled_v2.csv

The large CSV files are streamed in chunks.

Outputs:
    outputs/Deep_SVDD_XAI_per_sample.csv
    outputs/Deep_SVDD_XAI_category_summary.csv
    outputs/Deep_SVDD_XAI_report.json

The script selects representative samples:
    - benign validation samples
    - every zero-day attack category
"""

from pathlib import Path
import gc
import json
import random

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

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PER_SAMPLE_FILE = (
    OUT_DIR
    / "Deep_SVDD_XAI_per_sample.csv"
)

CATEGORY_SUMMARY_FILE = (
    OUT_DIR
    / "Deep_SVDD_XAI_category_summary.csv"
)

REPORT_FILE = (
    OUT_DIR
    / "Deep_SVDD_XAI_report.json"
)


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CSV_CHUNK_SIZE = 50_000

# Representative XAI sample counts.
N_BENIGN = 20
N_PER_ZERO_DAY_CATEGORY = 20

# Number of features shown in each explanation.
TOP_K = 10

# Batch size for ordinary model scoring.
SCORE_BATCH_SIZE = 4096


# ============================================================
# 3. MODEL
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
# 4. HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def clean_features(
    chunk,
    features
):
    """
    The prepared files are already scaled using the same
    fitted StandardScaler. Do not refit scaling here.
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


def clean_label(
    value
):
    value_str = str(
        value
    ).strip()

    if value_str == "BenignTraffic":
        return 0

    return 1


def score_tensor_batch(
    model,
    center,
    x
):
    """
    Deep SVDD anomaly score:
        squared Euclidean distance from the learned benign center.
    """

    z = model(x)

    return (
        (z - center) ** 2
    ).sum(
        dim=1
    )


@torch.no_grad()
def score_numpy(
    model,
    center,
    x
):

    scores = []

    for start in range(
        0,
        len(x),
        SCORE_BATCH_SIZE
    ):

        stop = min(
            start + SCORE_BATCH_SIZE,
            len(x)
        )

        tensor = torch.from_numpy(
            x[start:stop]
        ).to(
            DEVICE
        )

        score = score_tensor_batch(
            model,
            center,
            tensor
        )

        scores.append(
            score.cpu().numpy()
        )

    return np.concatenate(
        scores
    )


def get_identifier(
    row,
    source_file,
    row_number
):

    possible_ids = [
        "flow_id",
        "Flow ID",
        "flow_id_number",
        "id",
        "ID",
        "Index",
        "index"
    ]

    for column in possible_ids:

        if column in row.index:

            value = row[column]

            if (
                pd.notna(value)
                and str(value).strip()
                not in {"", "nan", "None"}
            ):

                return str(value)

    return (
        f"{source_file}:row_{row_number}"
    )


# ============================================================
# 5. LOAD SAVED MODEL
# ============================================================

section(
    "LOADING SAVED DEEP SVDD MODEL"
)

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Saved Deep SVDD model not found:\n{MODEL_FILE}"
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

features = list(
    checkpoint["features"]
)

hidden = list(
    checkpoint["hidden"]
)

latent_dim = int(
    checkpoint["latent_dim"]
)

threshold = float(
    checkpoint["threshold"]
)

center = (
    checkpoint["center"]
    .to(DEVICE)
    .float()
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
    f"Device           : {DEVICE}"
)

print(
    f"Feature count     : {len(features)}"
)

print(
    f"Hidden layers     : {hidden}"
)

print(
    f"Latent dimension  : {latent_dim}"
)

print(
    f"Frozen threshold  : {threshold:.10f}"
)

print(
    "\nThe trained Deep SVDD model will NOT be modified."
)


# ============================================================
# 6. REPRESENTATIVE SAMPLE COLLECTION
# ============================================================

section(
    "COLLECTING REPRESENTATIVE SAMPLES"
)

# We use reservoir sampling so we do not load giant files.

benign_samples = []

# category -> list of records
zero_day_samples = {}


def reservoir_add(
    reservoir,
    record,
    limit,
    seen_count,
    rng
):

    if seen_count <= limit:

        reservoir.append(
            record
        )

    else:

        j = rng.integers(
            0,
            seen_count
        )

        if j < limit:

            reservoir[j] = record

    return reservoir


rng = np.random.default_rng(
    SEED
)


# ------------------------------------------------------------
# Validation: benign samples
# ------------------------------------------------------------

print(
    "\nScanning validation_set.csv for representative benign samples..."
)

usecols_validation = (
    features
    + [
        "binary_label",
        "label"
    ]
)

validation_reader = pd.read_csv(
    VALID_FILE,
    usecols=usecols_validation,
    chunksize=CSV_CHUNK_SIZE,
    low_memory=True
)

benign_seen = 0
validation_rows = 0

for chunk_number, chunk in enumerate(
    validation_reader,
    start=1
):

    benign_mask = (
        pd.to_numeric(
            chunk["binary_label"],
            errors="coerce"
        )
        .fillna(
            chunk["label"].map(
                clean_label
            )
        )
        .astype(
            np.int8
        )
        .to_numpy()
        == 0
    )

    benign_chunk = chunk.loc[
        benign_mask
    ]

    for row_index, row in benign_chunk.iterrows():

        benign_seen += 1

        record = {
            "source_file": "validation_set.csv",
            "row_number": int(
                row_index + 2
            ),
            "identifier": get_identifier(
                row,
                "validation_set.csv",
                row_index + 2
            ),
            "attack_label": "BenignTraffic",
            "binary_label": 0,
            "features": row[
                features
            ].to_numpy(
                dtype=np.float32
            )
        }

        benign_samples = reservoir_add(
            benign_samples,
            record,
            N_BENIGN,
            benign_seen,
            rng
        )

    validation_rows += len(
        chunk
    )

    if (
        chunk_number == 1
        or chunk_number % 20 == 0
    ):

        print(
            f"  scanned {validation_rows:,} validation rows..."
        )


print(
    f"Representative benign samples collected: "
    f"{len(benign_samples)}"
)


# ------------------------------------------------------------
# Zero-day: representative samples for every category
# ------------------------------------------------------------

print(
    "\nScanning zero_day_test_scaled_v2.csv for representative "
    "samples from every attack category..."
)

zero_reader = pd.read_csv(
    ZERO_FILE,
    usecols=features + [
        "label",
        "binary_label"
    ],
    chunksize=CSV_CHUNK_SIZE,
    low_memory=True
)

category_seen = {}
zero_rows = 0

for chunk_number, chunk in enumerate(
    zero_reader,
    start=1
):

    for row_index, row in chunk.iterrows():

        category = str(
            row["label"]
        ).strip()

        if category not in zero_day_samples:

            zero_day_samples[
                category
            ] = []

            category_seen[
                category
            ] = 0

        category_seen[
            category
        ] += 1

        record = {
            "source_file":
                "zero_day_test_scaled_v2.csv",

            "row_number":
                int(row_index + 2),

            "identifier":
                get_identifier(
                    row,
                    "zero_day_test_scaled_v2.csv",
                    row_index + 2
                ),

            "attack_label":
                category,

            "binary_label":
                1,

            "features":
                row[
                    features
                ].to_numpy(
                    dtype=np.float32
                )
        }

        zero_day_samples[
            category
        ] = reservoir_add(
            zero_day_samples[
                category
            ],
            record,
            N_PER_ZERO_DAY_CATEGORY,
            category_seen[
                category
            ],
            rng
        )

    zero_rows += len(
        chunk
    )

    if (
        chunk_number == 1
        or chunk_number % 20 == 0
    ):

        print(
            f"  scanned {zero_rows:,} zero-day rows..."
        )


print(
    "\nZero-day categories found:"
)

for category in sorted(
    zero_day_samples
):

    print(
        f"  {category}: "
        f"{len(zero_day_samples[category])} samples"
    )


# ============================================================
# 7. FEATURE-LEVEL OCCLUSION XAI
# ============================================================

section(
    "RUNNING FEATURE-LEVEL XAI"
)


def explain_sample(
    record
):

    x = record[
        "features"
    ].copy()

    original_score = float(
        score_numpy(
            model,
            center,
            x.reshape(
                1,
                -1
            )
        )[0]
    )

    contributions = []

    # Because the input is StandardScaler output, 0 is the
    # training-data mean. We use this as the neutral baseline.

    for feature_index, feature in enumerate(
        features
    ):

        masked = x.copy()

        masked[
            feature_index
        ] = 0.0

        masked_score = float(
            score_numpy(
                model,
                center,
                masked.reshape(
                    1,
                    -1
                )
            )[0]
        )

        contribution = (
            original_score
            - masked_score
        )

        contributions.append({

            "feature":
                feature,

            "feature_index":
                feature_index,

            "feature_value_scaled":
                float(
                    x[feature_index]
                ),

            "masked_score":
                masked_score,

            "contribution":
                float(
                    contribution
                ),

            "absolute_contribution":
                float(
                    abs(contribution)
                )
        })


    # Strongest anomaly contributors first.
    ranked = sorted(
        contributions,
        key=lambda item:
            item["contribution"],
        reverse=True
    )

    top_positive = ranked[
        :TOP_K
    ]


    # Also retain strongest negative contributors for
    # diagnostic purposes.
    negative_ranked = sorted(
        contributions,
        key=lambda item:
            item["contribution"]
    )

    top_negative = negative_ranked[
        :TOP_K
    ]


    detected = (
        original_score
        >= threshold
    )

    return {

        "identifier":
            record["identifier"],

        "source_file":
            record["source_file"],

        "row_number":
            record["row_number"],

        "attack_label":
            record["attack_label"],

        "anomaly_score":
            original_score,

        "threshold":
            threshold,

        "detected":
            bool(detected),

        "top_positive_features":
            top_positive,

        "top_negative_features":
            top_negative,

        "all_feature_contributions":
            contributions
    }


# ============================================================
# 8. RUN XAI ON ALL REPRESENTATIVE SAMPLES
# ============================================================

records_to_explain = (
    benign_samples.copy()
)

for category in sorted(
    zero_day_samples
):

    records_to_explain.extend(
        zero_day_samples[
            category
        ]
    )


print(
    f"\nTotal representative samples for XAI: "
    f"{len(records_to_explain)}"
)

results = []

for index, record in enumerate(
    records_to_explain,
    start=1
):

    explanation = explain_sample(
        record
    )

    results.append(
        explanation
    )

    if (
        index == 1
        or index % 10 == 0
    ):

        print(
            f"  Explained "
            f"{index}/{len(records_to_explain)} samples..."
        )


# ============================================================
# 9. SAVE PER-SAMPLE XAI TABLE
# ============================================================

section(
    "SAVING PER-SAMPLE XAI RESULTS"
)

sample_rows = []

for result in results:

    for rank, item in enumerate(
        result[
            "top_positive_features"
        ],
        start=1
    ):

        sample_rows.append({

            "identifier":
                result["identifier"],

            "source_file":
                result["source_file"],

            "row_number":
                result["row_number"],

            "attack_label":
                result["attack_label"],

            "anomaly_score":
                result["anomaly_score"],

            "threshold":
                result["threshold"],

            "detected":
                result["detected"],

            "rank":
                rank,

            "feature":
                item["feature"],

            "feature_value_scaled":
                item["feature_value_scaled"],

            "masked_score":
                item["masked_score"],

            "contribution":
                item["contribution"],

            "absolute_contribution":
                item["absolute_contribution"]
        })


sample_df = pd.DataFrame(
    sample_rows
)

sample_df.to_csv(
    PER_SAMPLE_FILE,
    index=False
)

print(
    f"Per-sample XAI saved to:\n"
    f"{PER_SAMPLE_FILE}"
)


# ============================================================
# 10. CATEGORY-LEVEL XAI AGGREGATION
# ============================================================
#
# For each attack category:
#
#   mean positive contribution
#   mean absolute contribution
#   how often the feature appears in the top 5
#   how often the feature appears in the top 10
#
# This gives the "per-attack top contributing features" requested
# by the project handoff.
# ============================================================

section(
    "BUILDING CATEGORY-LEVEL XAI AGGREGATION"
)

category_accumulator = {}


for result in results:

    category = (
        result[
            "attack_label"
        ]
    )

    if category not in category_accumulator:

        category_accumulator[
            category
        ] = {

            "sample_count":
                0,

            "detected_count":
                0,

            "feature_contributions":
                {},

            "top5_counts":
                {},

            "top10_counts":
                {}
        }


    accumulator = category_accumulator[
        category
    ]

    accumulator[
        "sample_count"
    ] += 1

    if result[
        "detected"
    ]:

        accumulator[
            "detected_count"
        ] += 1


    # All features
    for item in result[
        "all_feature_contributions"
    ]:

        feature = item[
            "feature"
        ]

        if feature not in accumulator[
            "feature_contributions"
        ]:

            accumulator[
                "feature_contributions"
            ][feature] = []

        accumulator[
            "feature_contributions"
        ][feature].append(
            item["contribution"]
        )


    # Top 5
    for item in result[
        "top_positive_features"
    ][:5]:

        feature = item[
            "feature"
        ]

        accumulator[
            "top5_counts"
        ][feature] = (
            accumulator[
                "top5_counts"
            ].get(
                feature,
                0
            ) + 1
        )


    # Top 10
    for item in result[
        "top_positive_features"
    ][:10]:

        feature = item[
            "feature"
        ]

        accumulator[
            "top10_counts"
        ][feature] = (
            accumulator[
                "top10_counts"
            ].get(
                feature,
                0
            ) + 1
        )


category_rows = []

for category in sorted(
    category_accumulator
):

    accumulator = category_accumulator[
        category
    ]

    n = accumulator[
        "sample_count"
    ]

    for feature, values in (
        accumulator[
            "feature_contributions"
        ].items()
    ):

        values_array = np.asarray(
            values,
            dtype=np.float64
        )

        positive_values = values_array[
            values_array > 0
        ]

        mean_contribution = (
            float(
                values_array.mean()
            )
            if len(values_array) > 0
            else 0.0
        )

        mean_positive = (
            float(
                positive_values.mean()
            )
            if len(positive_values) > 0
            else 0.0
        )

        mean_absolute = float(
            np.abs(
                values_array
            ).mean()
        )

        top5_count = accumulator[
            "top5_counts"
        ].get(
            feature,
            0
        )

        top10_count = accumulator[
            "top10_counts"
        ].get(
            feature,
            0
        )

        category_rows.append({

            "attack_category":
                category,

            "sample_count":
                n,

            "detected_count":
                accumulator[
                    "detected_count"
                ],

            "feature":
                feature,

            "mean_contribution":
                mean_contribution,

            "mean_positive_contribution":
                mean_positive,

            "mean_absolute_contribution":
                mean_absolute,

            "top5_frequency":
                top5_count / n,

            "top10_frequency":
                top10_count / n
        })


category_df = pd.DataFrame(
    category_rows
)


# Sort with anomaly-positive contribution first.

category_df = category_df.sort_values(
    [
        "attack_category",
        "mean_positive_contribution",
        "mean_absolute_contribution",
        "top10_frequency"
    ],
    ascending=[
        True,
        False,
        False,
        False
    ]
)

category_df.to_csv(
    CATEGORY_SUMMARY_FILE,
    index=False
)

print(
    f"Category XAI summary saved to:\n"
    f"{CATEGORY_SUMMARY_FILE}"
)


# ============================================================
# 11. CONSOLE SUMMARY
# ============================================================

section(
    "XAI SUMMARY"
)

for category in sorted(
    category_accumulator
):

    subset = category_df[
        category_df[
            "attack_category"
        ] == category
    ]

    top = subset.head(
        TOP_K
    )

    print(
        f"\n{category}"
    )

    for _, row in top.iterrows():

        print(
            f"  {row['feature']:24s}"
            f"  mean_positive="
            f"{row['mean_positive_contribution']:.8f}"
            f"  top10="
            f"{row['top10_frequency'] * 100:.1f}%"
        )


# ============================================================
# 12. JSON REPORT
# ============================================================

report = {

    "model":
        "Deep SVDD",

    "model_file":
        str(MODEL_FILE),

    "xai_method":
        "Feature occlusion / perturbation",

    "baseline_value":
        0.0,

    "baseline_interpretation":
        "Because the prepared data is StandardScaler output, "
        "0 corresponds to the training-data mean of each feature.",

    "contribution_definition":
        "original anomaly score - anomaly score after replacing "
        "one feature with the neutral baseline",

    "positive_contribution_meaning":
        "Removing the feature lowers the anomaly score; the "
        "feature contributed to the anomaly.",

    "negative_contribution_meaning":
        "Removing the feature raises the anomaly score; the "
        "feature made the sample look relatively more normal.",

    "threshold":
        threshold,

    "feature_count":
        len(features),

    "top_k":
        TOP_K,

    "representative_samples": {

        "benign":
            len(benign_samples),

        "per_zero_day_category":
            N_PER_ZERO_DAY_CATEGORY,

        "total":
            len(results)
    },

    "category_summary":
        category_rows
}

REPORT_FILE.write_text(
    json.dumps(
        report,
        indent=4
    ),
    encoding="utf-8"
)


# ============================================================
# 13. COMPLETE
# ============================================================

section(
    "DEEP SVDD XAI COMPLETE"
)

print(
    f"Per-sample explanations:\n"
    f"{PER_SAMPLE_FILE}"
)

print(
    f"\nCategory aggregation:\n"
    f"{CATEGORY_SUMMARY_FILE}"
)

print(
    f"\nJSON report:\n"
    f"{REPORT_FILE}"
)

print(
    "\nThe Deep SVDD model itself was not changed."
)

print(
    "No SHAP model or classifier was added."
)

del model
del center
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
