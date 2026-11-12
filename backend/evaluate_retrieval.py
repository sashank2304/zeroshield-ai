"""Measure whether MITRE retrieval returns a relevant technique, using the app's own retrieval code.

    python evaluate_retrieval.py

Data: model_training/outputs/Deep_SVDD_XAI_per_sample.csv (200 samples x top-10 XAI features of the tuned
detector, with dataset labels). The label is used ONLY to score the answer, never to build the query.
"Relevant" is generous: any technique or sub-technique in the family listed in EXPECTED counts.

The samples are split by position within each label. The tuning half chose MIN_SIMILARITY; the test half
was evaluated once with that value. Flow statistics cannot identify application-layer attacks, so web
families are expected to score near zero; that is a property of the data, not a bug to tune away.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from services.mitre_service import MIN_SIMILARITY, MitreService  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "Mirai-greeth_flood": {"T1498", "T1499", "T1584"}, "Mirai-udpplain": {"T1498", "T1499", "T1584"},
    "Mirai-greip_flood": {"T1498", "T1499", "T1584"}, "Backdoor_Malware": {"T1505", "T1190"},
    "BrowserHijacking": {"T1185", "T1189", "T1190"}, "CommandInjection": {"T1190", "T1059"},
    "SqlInjection": {"T1190"}, "Uploading_Attack": {"T1190", "T1505"}, "XSS": {"T1190", "T1189"},
}


def load_samples(path=ROOT / "model_training" / "outputs" / "Deep_SVDD_XAI_per_sample.csv"):
    frame = pd.read_csv(path)
    seen = defaultdict(int)
    for ident, g in frame.groupby("identifier"):   # sorted by identifier: the same split the similarity floor was tuned on
        g = g.sort_values("rank")
        label = g.attack_label.iloc[0]
        if label not in EXPECTED:
            continue
        half = "tuning" if seen[label] % 2 == 0 else "test"
        seen[label] += 1
        yield {"label": label, "half": half, "detected": bool(g.detected.iloc[0]),
               "analysis": {"anomaly_score": float(g.anomaly_score.iloc[0]), "threshold": float(g.threshold.iloc[0]),
                            "top_features": [{"feature": r.feature, "value": float(r.feature_value_scaled), "contribution": float(r.contribution)}
                                             for r in g.itertuples()]}}


def evaluate(service=None):
    service = service or MitreService(ROOT)
    rows = []
    for s in load_samples():
        techniques = service.retrieve_behavior(s["analysis"]).get("techniques", [])
        ids = [t["technique_id"] for t in techniques]
        rows.append({**{k: s[k] for k in ("label", "half", "detected")}, "answered": bool(ids),
                     "hit": any(i.split(".")[0] in EXPECTED[s["label"]] for i in ids), "top1": ids[0] if ids else None})
    return pd.DataFrame(rows)


def summarise(df, name):
    answered = df[df.answered]
    return {"subset": name, "samples": len(df), "answered": int(len(answered)), "relevant_in_top5": int(df.hit.sum()),
            "relevant_share_of_all": round(df.hit.sum() / max(len(df), 1), 3),
            "precision_when_answering": round(answered.hit.sum() / max(len(answered), 1), 3),
            "most_common_top1": Counter(df.top1.dropna()).most_common(1)}


def main():
    df = evaluate()
    out = {"similarity_floor": MIN_SIMILARITY, "results": [
        summarise(df[df.half == "test"], "test half, all samples"),
        summarise(df[(df.half == "test") & df.detected], "test half, detector-flagged"),
        summarise(df[df.detected], "both halves, detector-flagged"),
    ], "per_family_flagged_test_half": {
        lab: {"flagged": int(len(g)), "answered": int(g.answered.sum()), "relevant": int(g.hit.sum())}
        for lab, g in df[(df.half == "test") & df.detected].groupby("label")}}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
