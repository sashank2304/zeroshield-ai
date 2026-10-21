
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

CHECKPOINT = Path("Deep_SVDD.pt")
BENIGN_CSV = Path("Dataset/train_benign_only.csv")
ATTACK_CSV = Path("Dataset/zero_day_test_scaled_v2.csv")

N_ROWS = 2000


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


def score_file(model, center, features, path, n_rows):
    df = pd.read_csv(path, nrows=n_rows)

    X = df[features].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError(f"NaN or infinity found in {path}")

    with torch.no_grad():
        z = model(torch.from_numpy(X))
        scores = ((z - center) ** 2).sum(dim=1).numpy()

    result = pd.DataFrame({"score": scores})

    if "label" in df.columns:
        result["label"] = df["label"].astype(str)

    return result


def main():
    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )

    features = checkpoint["features"]

    model = DeepSVDD()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    center = torch.as_tensor(
        checkpoint["center"], dtype=torch.float32
    ).reshape(1, -1)

    benign = score_file(
        model, center, features, BENIGN_CSV, N_ROWS
    )
    attack = score_file(
        model, center, features, ATTACK_CSV, N_ROWS
    )

    print("\nBENIGN TRAINING FILE — score distribution")
    print(benign["score"].describe(
        percentiles=[0.50, 0.90, 0.95, 0.99]
    ).to_string())

    print("\nZERO-DAY FILE — score distribution by label")
    print(
        attack.groupby("label")["score"]
        .agg(["count", "min", "median", "max"])
        .sort_values("median", ascending=False)
        .to_string()
    )

    print("\nProvisional threshold: 0.06141921")
    threshold = 0.06141921

    print(
        "Benign rows above threshold:",
        f"{(benign['score'] > threshold).mean():.2%}"
    )
    print(
        "Zero-day rows above threshold:",
        f"{(attack['score'] > threshold).mean():.2%}"
    )

    print(
        "\nCaution: this compares the checkpoint against raw CSV feature "
        "values. It is valid only if these values use the model's "
        "original training preprocessing."
    )


if __name__ == "__main__":
    main()
