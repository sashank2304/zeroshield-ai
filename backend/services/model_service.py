from pathlib import Path

import numpy as np
import torch
from torch import nn


class DeepSVDDNetwork(nn.Module):
    """Architecture copied from the existing Deep SVDD evaluation implementation."""
    def __init__(self, input_dim: int, hidden: list[int], latent_dim: int):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden:
            layers += [nn.Linear(previous, width, bias=False), nn.ReLU()]
            previous = width
        layers += [nn.Linear(previous, latent_dim, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Detector:
    def __init__(self, root: Path):
        self.loaded = False
        self.device = torch.device("cpu")
        self.model_path = root / "model_training" / "outputs" / "Deep_SVDD_Tuned.pt"
        self.features = []
        self.hidden = []
        self.latent_dim = 0
        self.threshold = 0.06141921002417763
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.features = list(checkpoint["features"])
            self.hidden = list(checkpoint["hidden"])
            self.latent_dim = int(checkpoint["latent_dim"])
            self.threshold = float(checkpoint["threshold"])
            if len(self.features) != 43:
                raise ValueError("Checkpoint feature count is not 43")
            self.center = checkpoint["center"].to(self.device).float()
            self.model = DeepSVDDNetwork(len(self.features), self.hidden, self.latent_dim).to(self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self.loaded = True
        except Exception:
            import logging
            logging.exception("Could not load the existing Deep SVDD checkpoint")

    def score(self, vector: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.from_numpy(vector.reshape(1, -1).astype(np.float32)).to(self.device)
            return float(((self.model(x) - self.center) ** 2).sum(dim=1).item())

    def analyze(self, values: dict[str, float]):
        vector = np.asarray([values[name] for name in self.features], dtype=np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("All features must be finite numeric values")
        score = self.score(vector)
        contributions = []
        for index, name in enumerate(self.features):
            masked = vector.copy()
            masked[index] = 0.0
            contribution = score - self.score(masked)
            contributions.append({"feature": name, "value": float(vector[index]), "contribution": contribution, "rank": 0})
        contributions.sort(key=lambda item: item["contribution"], reverse=True)
        for rank, item in enumerate(contributions, start=1):
            item["rank"] = rank
        return {"anomaly_score": score, "threshold": self.threshold, "detected": score >= self.threshold, "detection": "ANOMALOUS" if score >= self.threshold else "NORMAL", "top_features": contributions[:10], "all_contributions": contributions, "xai_method": "Feature Perturbation XAI", "xai_baseline": "0 in the prepared StandardScaler space"}
