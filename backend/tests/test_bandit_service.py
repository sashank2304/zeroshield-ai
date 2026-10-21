import tempfile
import unittest
import json
import sqlite3
from pathlib import Path

from services.bandit_service import ACTIONS, BanditService, build_context, fixed_rule_action, random_action, simulate_outcome


class BanditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = BanditService(Path(self.tmp.name) / "ledger.sqlite3", seed=7)
        self.alert = {"alert_id": "a1", "anomaly_score": .2, "threshold": .1, "detected": True,
                      "verification_confidence": 1., "xai_evidence": [{"contribution": .04}], "retrieved_techniques": []}

    def tearDown(self): self.tmp.cleanup()

    def test_context_excludes_label_and_outcome(self):
        first = build_context({**self.alert, "ground_truth_attack": True, "reward": -100, "observed_outcome": "attack"})
        second = build_context(self.alert)
        self.assertEqual(first, second)

    def test_actions_and_selection(self):
        decision = self.service.select(self.alert)
        self.assertEqual(set(decision["action_scores"]), set(ACTIONS))
        self.assertIn(decision["selected_action"], ACTIONS)

    def test_auto_remediation_safety_gate(self):
        benign = {**self.alert, "detected": False}
        decision = self.service.select(benign)
        self.assertNotEqual(decision["selected_action"], "AUTO_REMEDIATE")
        self.assertFalse(decision["auto_remediation_eligible"])

    def test_feedback_updates_once(self):
        rec = self.service.record(self.alert)
        result = self.service.feedback("a1", True, "APPROVED" if rec["selected_action"] == "CONFIRM_TO_ACT" else None)
        self.assertEqual(result["policy_update_status"], "UPDATED")
        self.assertFalse(self.service.update("a1", rec["context_features"], rec["selected_action"], result["reward"]))
        self.assertEqual(self.service.updates, 1)

    def test_approval_waits(self):
        rec = {**self.alert, "alert_id": "approval"}
        selected = self.service.record(rec)
        if selected["selected_action"] != "CONFIRM_TO_ACT":
            c = sqlite3.connect(self.service.db_path)
            try:
                selected["selected_action"] = "CONFIRM_TO_ACT"
                selected["approval_status"] = "PENDING"
                c.execute("UPDATE alerts SET record_json=? WHERE alert_id='approval'", (json.dumps(selected),))
                c.commit()
            finally: c.close()
        with self.assertRaises(ValueError): self.service.feedback("approval", True)

    def test_rejected_approval_penalizes_uncontained_attack(self):
        alert = {**self.alert, "alert_id": "reject", "detected": False}
        decision = self.service.select(alert)
        c = sqlite3.connect(self.service.db_path)
        try:
            forced = {**alert, **decision, "selected_action": "CONFIRM_TO_ACT", "verification_result": "VERIFIED",
                      "approval_status": "PENDING", "policy_update_status": "WAITING_FOR_FEEDBACK", "reward": None}
            c.execute("INSERT INTO alerts VALUES(?,?,?)", ("reject", json.dumps(forced), "now"))
            c.commit()
        finally: c.close()
        result = self.service.feedback("reject", True, "REJECTED")
        self.assertEqual(result["reward"], -6)

    def test_reward_table(self):
        self.assertEqual(simulate_outcome("AUTO_REMEDIATE", False)["reward"], -12)
        self.assertEqual(simulate_outcome("MONITOR_ONLY", True)["reward"], -6)

    def test_baselines_are_deterministic(self):
        self.assertEqual(fixed_rule_action({"verification_result": "VERIFIED", "detected": False}), "MONITOR_ONLY")
        self.assertEqual(fixed_rule_action({"verification_result": "REJECTED", "detected": True}), "HUMAN_REVIEW")
        self.assertEqual(random_action(42, 3), random_action(42, 3))

    def test_duplicate_feedback_is_idempotent(self):
        self.service.record(self.alert)
        first = self.service.feedback("a1", True)
        second = self.service.feedback("a1", False)
        self.assertEqual(first["reward"], second["reward"])
        self.assertEqual(self.service.updates, 1)

    def test_policy_changes_after_observed_reward(self):
        alert = {**self.alert, "alert_id": "learn-1", "detected": True}
        before = self.service.select(alert, explore=False)["action_scores"]
        chosen = self.service.record(alert)
        outcome = self.service.feedback("learn-1", True)
        after = self.service.select(alert, explore=False)["action_scores"]
        self.assertEqual(outcome["policy_update_status"], "UPDATED")
        self.assertNotEqual(before[chosen["selected_action"]], after[chosen["selected_action"]])


if __name__ == "__main__": unittest.main()
