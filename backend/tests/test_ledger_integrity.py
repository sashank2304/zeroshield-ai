"""Regression tests for the Stage 8-9 ledger: atomic feedback, audit history,
versioning, reward table design and concurrency."""
import json
import sqlite3
import tempfile
import threading
from contextlib import closing
import unittest
from pathlib import Path
from unittest.mock import patch

from services.bandit_service import (
    ACTIONS, REWARD_TABLES, REWARD_VERSION, SCHEMA_VERSION, BanditService,
    build_context, decision_regions,
)


def make_alert(alert_id="a1", detected=True):
    return {"alert_id": alert_id, "anomaly_score": .2, "threshold": .1, "detected": detected,
            "verification_result": "VERIFIED", "verification_confidence": 1.,
            "xai_evidence": [{"contribution": .04}], "retrieved_techniques": []}


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = BanditService(Path(self.tmp.name) / "ledger.sqlite3", seed=7)

    def tearDown(self): self.tmp.cleanup()

    def force_action(self, alert_id, action, **extra):
        """Store a record with a chosen action (the policy would otherwise pick its own)."""
        alert = make_alert(alert_id)
        decision = self.service.select(alert)
        rec = {**alert, **decision, "selected_action": action, "policy_update_status": "WAITING_FOR_FEEDBACK",
               "reward": None, "approval_status": "PENDING" if action == "CONFIRM_TO_ACT" else "NOT_REQUIRED"}
        with closing(sqlite3.connect(self.service.db_path)) as c, c:  # close: an open handle locks the file on Windows
            c.execute("INSERT INTO alerts VALUES(?,?,?)", (alert_id, json.dumps(rec), "now"))
        return rec

    # ---- the stuck-feedback bug ------------------------------------------
    def test_failed_policy_update_rolls_back_and_can_be_retried(self):
        self.service.record(make_alert())
        with patch.object(BanditService, "_apply_update", side_effect=RuntimeError("crash")):
            with self.assertRaises(RuntimeError): self.service.feedback("a1", True)
        stored = self.service.get_alert("a1")
        self.assertIsNone(stored["reward"])                       # nothing half-written
        self.assertEqual(stored["policy_update_status"], "WAITING_FOR_FEEDBACK")
        self.assertEqual(self.service.updates, 0)
        result = self.service.feedback("a1", True)                # retry works
        self.assertEqual(result["policy_update_status"], "UPDATED")
        self.assertEqual(self.service.updates, 1)
        self.assertIsNotNone(result["reward"])

    def test_record_stranded_by_old_two_step_write_is_repaired(self):
        rec = self.service.record(make_alert())
        stranded = {**rec, "reward": 5, "observed_outcome": "attack", "policy_update_status": "WAITING_FOR_UPDATE"}
        with closing(sqlite3.connect(self.service.db_path)) as c, c:  # close: an open handle locks the file on Windows
            c.execute("UPDATE alerts SET record_json=? WHERE alert_id='a1'", (json.dumps(stranded),))
        result = self.service.feedback("a1", False)               # label must NOT change the stored reward
        self.assertEqual(result["policy_update_status"], "UPDATED")
        self.assertEqual(result["reward"], 5)
        self.assertEqual(self.service.updates, 1)

    # ---- audit trail and versions ------------------------------------------
    def test_history_versions_and_timestamps_are_recorded(self):
        self.force_action("c1", "CONFIRM_TO_ACT")
        self.service.decide_approval("c1", "APPROVED")
        rec = self.service.feedback("c1", True)
        names = [e["event"] for e in rec["history"]]
        self.assertEqual(names, ["APPROVAL_DECIDED", "OUTCOME_OBSERVED", "POLICY_UPDATED"])
        for event in rec["history"]: self.assertTrue(event["at"])
        self.assertEqual(rec["reward_version"], REWARD_VERSION)
        self.assertTrue(rec["feedback_at"]); self.assertTrue(rec["reviewed_at"]); self.assertTrue(rec["policy_updated_at"])
        self.assertTrue(rec["updated_policy_version"].endswith("+1"))

    def test_new_record_has_schema_version_and_selection_event(self):
        rec = self.service.record(make_alert())
        self.assertEqual(rec["schema_version"], SCHEMA_VERSION)
        self.assertEqual(rec["history"][0]["event"], "ACTION_SELECTED")
        self.assertIn("policy_state_at_selection", rec)

    def test_duplicate_feedback_adds_no_second_reward_or_event(self):
        self.service.record(make_alert())
        first = self.service.feedback("a1", True)
        second = self.service.feedback("a1", False)
        self.assertEqual(first["reward"], second["reward"])
        self.assertEqual(len(second["history"]), len(first["history"]))
        self.assertEqual(self.service.updates, 1)

    # ---- approval rewards --------------------------------------------------
    def test_rejected_benign_costs_only_analyst_time_but_approved_benign_costs_more(self):
        self.force_action("rej", "CONFIRM_TO_ACT"); self.force_action("app", "CONFIRM_TO_ACT")
        self.assertEqual(self.service.feedback("rej", False, "REJECTED")["reward"], -2)
        self.assertEqual(self.service.feedback("app", False, "APPROVED")["reward"], -3)

    # ---- reward table design -------------------------------------------------
    def test_v1_table_never_prefers_human_review(self):
        actions = {r["action"] for r in decision_regions(REWARD_TABLES["designed-simulation-v1"])}
        self.assertNotIn("HUMAN_REVIEW", actions)

    def test_v2_table_gives_every_action_a_region(self):
        regions = decision_regions(REWARD_TABLES["designed-simulation-v2"])
        self.assertEqual([r["action"] for r in regions],
                         ["MONITOR_ONLY", "HUMAN_REVIEW", "CONFIRM_TO_ACT", "AUTO_REMEDIATE"])
        edges = [r["from"] for r in regions[1:]]
        for got, want in zip(edges, (3 / 11, 0.5, 9 / 11)):
            self.assertAlmostEqual(got, want, delta=0.002)

    # ---- labels never reach the policy -----------------------------------------
    def test_policy_context_ignores_label_like_fields(self):
        base = make_alert()
        noisy = {**base, "sample_id": "Mirai-udpplain", "attack_family": "Mirai", "label": "x",
                 "ground_truth_attack": True, "observed_outcome": "attack", "reward": 5}
        self.assertEqual(build_context(base), build_context(noisy))

    # ---- concurrency ---------------------------------------------------------------
    def run_threads(self, fn, n=8):
        results, errors = [], []
        def work():
            try: results.append(fn())
            except ValueError as exc: errors.append(str(exc))
        threads = [threading.Thread(target=work) for _ in range(n)]
        for t in threads: t.start()
        for t in threads: t.join()
        return results, errors

    def test_concurrent_approvals_only_one_wins(self):
        self.force_action("race", "CONFIRM_TO_ACT")
        results, errors = self.run_threads(lambda: self.service.decide_approval("race", "APPROVED"))
        self.assertEqual(len(results), 1); self.assertEqual(len(errors), 7)
        history = self.service.get_alert("race")["history"]
        self.assertEqual([e["event"] for e in history].count("APPROVAL_DECIDED"), 1)

    def test_concurrent_feedback_updates_policy_once(self):
        self.service.record(make_alert("fb"))
        results, errors = self.run_threads(lambda: self.service.feedback("fb", True))
        self.assertEqual(errors, [])
        self.assertEqual({r["reward"] for r in results}, {results[0]["reward"]})
        self.assertEqual(self.service.updates, 1)
        events = [e["event"] for e in self.service.get_alert("fb")["history"]]
        self.assertEqual(events.count("POLICY_UPDATED"), 1)

    def test_policy_survives_restart(self):
        self.service.record(make_alert()); self.service.feedback("a1", True)
        reopened = BanditService(self.service.db_path, seed=7)
        self.assertEqual(reopened.updates, 1)

    def test_pagination_and_get_alert(self):
        for i in range(5): self.service.record(make_alert(f"p{i}"))
        self.assertEqual(len(self.service.list_alerts()), 5)
        self.assertEqual(len(self.service.list_alerts(limit=2)), 2)
        with self.assertRaises(KeyError): self.service.get_alert("missing")


if __name__ == "__main__": unittest.main()
