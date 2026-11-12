"""Stage 4 retrieval must describe behaviour in words ATT&CK uses, vary with the evidence, stay on
network-observable techniques, and abstain when there is nothing to say. Uses the repo's labelled
XAI evidence (Deep_SVDD_XAI_per_sample.csv); labels score the answers and never enter a query."""
import json
import unittest
from pathlib import Path

from evaluate_retrieval import evaluate, load_samples
from services.mitre_service import (BEHAVIOR_ONTOLOGY, MIN_SIMILARITY, NETWORK_TACTICS, MitreService,
                                    build_behavior_query)

ROOT = Path(__file__).resolve().parents[2]


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = MitreService(ROOT)
        cls.samples = list(load_samples())
        cls.frame = evaluate(cls.service)

    def test_ontology_covers_every_project_feature(self):
        features = json.loads((ROOT / "model_training" / "outputs" / "zero_shield_feature_columns.json").read_text())
        self.assertEqual([f for f in features if f not in BEHAVIOR_ONTOLOGY], [])

    def test_query_is_plain_language_without_feature_names_or_numbers(self):
        # The old query was feature names and numbers, which share no words with ATT&CK text.
        for s in self.samples:
            query, _ = build_behavior_query(s["analysis"]["top_features"])
            self.assertNotRegex(query, r"[\d_]", query)

    def test_flood_alerts_get_a_relevant_denial_of_service_technique(self):
        flagged = self.frame[self.frame.detected & self.frame.label.str.startswith("Mirai")]
        self.assertGreaterEqual(flagged.hit.mean(), 0.5, "measured 62% of 60 flagged Mirai samples")

    def test_results_vary_with_the_evidence_and_are_not_dominated_by_one_technique(self):
        queries = {build_behavior_query(s["analysis"]["top_features"])[0] for s in self.samples} - {""}
        self.assertGreater(len(queries), 20, "measured 49 distinct queries")
        share = self.frame[self.frame.detected].top1.value_counts(normalize=True).iloc[0]
        self.assertLess(share, 0.7, "the old query gave one irrelevant technique to 88% of alerts")

    def test_candidates_are_network_observable_and_above_the_similarity_floor(self):
        checked = 0
        for s in self.samples[:80]:
            for t in self.service.retrieve_behavior(s["analysis"]).get("techniques", []):
                checked += 1
                self.assertTrue(set(t["tactics"]) & NETWORK_TACTICS, t["technique_id"])
                self.assertGreaterEqual(t["similarity"], MIN_SIMILARITY)
        self.assertGreater(checked, 0)

    def test_host_only_techniques_are_never_returned(self):
        # T1547.014 (Active Setup, persistence) was the old top result for 88% of alerts.
        for s in self.samples:
            ids = {t["technique_id"] for t in self.service.retrieve_behavior(s["analysis"]).get("techniques", [])}
            self.assertNotIn("T1547.014", ids)

    def test_abstains_when_nothing_is_outside_the_benign_range(self):
        result = self.service.retrieve_behavior({"anomaly_score": .2, "threshold": .1, "top_features": [
            {"feature": "Rate", "value": 0.4, "contribution": .2}, {"feature": "SYNRatio", "value": -0.3, "contribution": .1}]})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["techniques"], [])

    def test_an_absent_protocol_and_a_negative_contribution_add_nothing(self):
        absent_protocol = [{"feature": "HTTPS", "value": -3.0, "contribution": .2}]
        negative = [{"feature": "Rate", "value": 5.0, "contribution": -.2}]
        self.assertEqual(build_behavior_query(absent_protocol)[0], "")
        self.assertEqual(build_behavior_query(negative)[0], "")

    def test_result_records_the_evidence_behind_the_query_and_the_reliability_note(self):
        result = self.service.retrieve_behavior(self.samples[0]["analysis"])
        self.assertIn("reliability_note", result)
        self.assertEqual(result["similarity_floor"], MIN_SIMILARITY)
        for item in result["evidence_features"]:
            self.assertIn(item["feature"], BEHAVIOR_ONTOLOGY)


if __name__ == "__main__": unittest.main()
