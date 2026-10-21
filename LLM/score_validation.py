
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

DATA_PATH = "Dataset/validation_set.csv"
MODEL_PATH = "Deep_SVDD.pt"
THRESHOLD = 0.06141921  # Provisional; not yet calibrated

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

print("Loading validation sample...")
df = pd.read_csv(DATA_PATH, nrows=2000)

missing = [name for name in FEATURE_NAMES if name not in df.columns]
if missing:
    raise ValueError(
        f"Missing features: {missing}\n"
        f"Actual columns: {df.columns.tolist()}"
    )

X = df[FEATURE_NAMES].apply(
    pd.to_numeric, errors="raise"
).to_numpy(dtype=np.float32)

print("Loading checkpoint...")
checkpoint = torch.load(
    MODEL_PATH, map_location="cpu", weights_only=False
)

model = DeepSVDD()
model.load_state_dict(checkpoint["state_dict"])
model.eval()

center = torch.as_tensor(
    checkpoint["center"], dtype=torch.float32
).reshape(1, -1)

with torch.no_grad():
    z = model(torch.from_numpy(X))
    scores = ((z - center) ** 2).sum(dim=1).numpy()

print("\nValidation sample score summary:")
print(
    pd.Series(scores).describe(
        percentiles=[0.50, 0.90, 0.95, 0.99]
    ).to_string()
)

above = scores > THRESHOLD
print(f"\nProvisional threshold: {THRESHOLD}")
print(f"Rows above threshold: {above.sum()} / {len(scores)}")
print(f"Benign false-positive rate in this sample: {above.mean():.4%}")

result = df[["label", "binary_label"]].copy()
result["svdd_score"] = scores

print("\nHighest-scoring benign rows:")
print(result.nlargest(10, "svdd_score").to_string(index=False))

