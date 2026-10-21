
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DATA_PATH = "Dataset/zero_day_test_scaled_v2.csv"
MODEL_PATH = "Deep_SVDD.pt"
THRESHOLD = 0.06141921  # Keep fixed for this evaluation
CHUNK_SIZE = 100_000

FEATURE_NAMES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration",
    "Rate", "Drate", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
    "fin_count", "urg_count", "rst_count", "HTTP", "HTTPS", "DNS",
    "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP",
    "ICMP", "IPv", "AVG", "Std", "IAT", "Number", "Radius",
    "Weight", "SYNRatio", "ACKRatio", "RSTRatio", "FINRatio",
    "SYN_ACK_Ratio", "PacketDensity"
]

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

checkpoint = torch.load(
    MODEL_PATH, map_location="cpu", weights_only=False
)
model = DeepSVDD()
model.load_state_dict(checkpoint["state_dict"])
model.eval()

center = torch.as_tensor(
    checkpoint["center"], dtype=torch.float32
).reshape(1, -1)

totals = {}

print("Evaluating zero-day data in chunks...")

for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
    missing = [c for c in FEATURE_NAMES if c not in chunk.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = chunk[FEATURE_NAMES].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float32)

    with torch.no_grad():
        z = model(torch.from_numpy(X))
        scores = ((z - center) ** 2).sum(dim=1).numpy()

    labels = chunk["label"].astype(str).to_numpy()
    flagged = scores > THRESHOLD

    for label in np.unique(labels):
        mask = labels == label
        if label not in totals:
            totals[label] = [0, 0, 0, 0.0]

        totals[label][0] += int(mask.sum())
        totals[label][1] += int(flagged[mask].sum())
        totals[label][2] += int((~flagged[mask]).sum())
        totals[label][3] = max(
            totals[label][3], float(scores[mask].max())
        )

print(f"\nThreshold: {THRESHOLD}")
print("\nDetection results by attack category:")
print(f"{'Label':<25} {'Total':>10} {'Detected':>10} {'Missed':>10} {'Recall':>10} {'Max score':>12}")

grand_total = 0
grand_detected = 0

for label, (total, detected, missed, max_score) in sorted(totals.items()):
    recall = 100 * detected / total if total else 0
    print(
        f"{label:<25} {total:>10,} {detected:>10,} "
        f"{missed:>10,} {recall:>9.2f}% {max_score:>12.6f}"
    )
    grand_total += total
    grand_detected += detected

print(f"\nOverall: {grand_detected:,}/{grand_total:,} flagged "
      f"({100 * grand_detected / grand_total:.2f}%)")
