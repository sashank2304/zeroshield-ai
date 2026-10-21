import json
from pathlib import Path


class SampleService:
    """Loads only the deliberately small checked-in demo cache; never scans source CSVs."""
    def __init__(self, root: Path, features: list[str]):
        self.features = features
        self.path = root / "backend" / "data" / "demo_samples.json"
        try:
            self.samples = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.samples = {}

    def available(self):
        return {"samples": [{"category": category, "available": category in self.samples, "features": self.samples.get(category)} for category in ["BenignTraffic", "Backdoor_Malware", "BrowserHijacking", "CommandInjection", "Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain", "SqlInjection", "Uploading_Attack", "XSS"]]}
