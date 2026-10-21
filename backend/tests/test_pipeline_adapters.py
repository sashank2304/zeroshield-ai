import os
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from services.llm_service import reason
from services.mitre_service import MitreService
from services.verification_service import verify_record

ROOT = Path(__file__).resolve().parents[2]


class PipelineAdapterTests(unittest.TestCase):
    def test_behavior_query_uses_observed_features_and_traceable_records(self):
        service = MitreService(ROOT)
        self.assertIsNotNone(service.index)
        result = service.retrieve_behavior({"anomaly_score": .5, "threshold": .1,
            "top_features": [{"feature": "SYNRatio", "value": .9, "contribution": .2}]})
        self.assertIn("SYNRatio", result["query"])
        self.assertIn("0.9", result["query"])
        for item in result["techniques"]:
            self.assertTrue(item["technique_id"].startswith("T"))
            self.assertTrue(item["technique_name"])
            self.assertTrue(item["source"])

    def test_empty_retrieval_is_explicit(self):
        import numpy as np
        class Scores:
            def toarray(self): return np.zeros((1, 1))
        class Query:
            def __matmul__(self, _): return Scores()
        class Matrix:
            @property
            def T(self): return self
        class Vectorizer:
            def transform(self, _): return Query()
        service = MitreService(ROOT)
        service.index = {"vectorizer": Vectorizer(), "matrix": Matrix(), "documents": []}
        result = service.retrieve_behavior({"anomaly_score": 0, "threshold": 1, "top_features": []})
        self.assertEqual(result["status"], "empty_retrieval")
        self.assertEqual(result["techniques"], [])

    def test_stage5_missing_key_returns_labeled_schema(self):
        with patch.dict(os.environ, {}, clear=True):
            result = reason({"detected": False, "anomaly_score": .1, "threshold": .2, "top_features": []}, None, {"message": "No retrieval", "techniques": []})
        self.assertEqual(result["mode"], "Demo Fallback Reasoning")
        self.assertEqual(set(result["report"]), {"incident_summary", "evidence", "attack_context", "interpretation", "recommended_action", "limitations"})

    def test_stage5_invalid_json_falls_back_with_label(self):
        import openai
        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{invalid"))]))))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}), patch.object(openai, "OpenAI", return_value=fake), self.assertLogs(level="ERROR"):
            result = reason({"detected": False, "anomaly_score": .1, "threshold": .2, "top_features": []}, None, {"message": "No retrieval", "techniques": []})
        self.assertEqual(result["mode"], "Demo Fallback Reasoning")

    def test_stage6_rejects_xai_feature_outside_supplied_evidence(self):
        analysis = {"anomaly_score": 2., "threshold": 1., "detected": True,
                    "feature_names": ["SYNRatio"], "all_feature_names": ["SYNRatio", "PacketDensity"]}
        report = {"incident_summary": "Anomaly score exceeded threshold.", "evidence": ["Score 2.000000; threshold 1.000000. PacketDensity is high."],
                  "attack_context": "No ATT&CK mapping.", "interpretation": "Anomaly.", "recommended_action": "Review.", "limitations": "Flow-level only."}
        result = verify_record(analysis, {"techniques": []}, {"report": report})
        self.assertNotEqual(result["verdict"], "VERIFIED")
        self.assertIn("feature_claims_supported", result["failed_checks"])

    def test_stage6_rejects_contradictory_numeric_claim(self):
        analysis = {"anomaly_score": 2., "threshold": 1., "detected": True,
                    "feature_names": ["SYNRatio"], "all_feature_names": ["SYNRatio"],
                    "top_features": [{"feature": "SYNRatio", "value": .8, "contribution": .2}]}
        report = {"incident_summary": "Anomaly score exceeded the threshold.",
                  "evidence": ["Score 2.0; threshold 1.0; SYNRatio contribution was 99.0."],
                  "attack_context": "No ATT&CK mapping.", "interpretation": "Anomaly.",
                  "recommended_action": "Review.", "limitations": "Flow-level only."}
        result = verify_record(analysis, {"techniques": []}, {"report": report})
        self.assertIn("numeric_claims_grounded", result["failed_checks"])

    def test_stage6_rejects_wrong_perturbation_direction(self):
        analysis = {"anomaly_score": 2., "threshold": 1., "detected": True,
                    "feature_names": ["SYNRatio"], "all_feature_names": ["SYNRatio"],
                    "top_features": [{"feature": "SYNRatio", "value": .8, "contribution": .2}]}
        report = {"incident_summary": "Anomaly score exceeded the threshold.",
                  "evidence": ["Score 2.0; threshold 1.0. Perturbing SYNRatio increased the anomaly score."],
                  "attack_context": "No ATT&CK mapping.", "interpretation": "Anomaly.",
                  "recommended_action": "Review.", "limitations": "Flow-level only."}
        result = verify_record(analysis, {"techniques": []}, {"report": report})
        self.assertIn("feature_directions_consistent", result["failed_checks"])


if __name__ == "__main__": unittest.main()
