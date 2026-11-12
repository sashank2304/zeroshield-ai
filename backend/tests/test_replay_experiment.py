"""The replay must route only detected flows, evaluate each alert once, keep labels out of the
policy's context, and be reproducible. Uses a synthetic stream, so it needs no PyTorch."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import run_response_experiment as replay
from services.bandit_service import CONTEXT_FEATURES, REWARD_VERSION

TIMING_KEYS = {"policy_training_update_seconds", "total_elapsed_seconds"}


def make_stream(n=60, seed=1):
    rng = random.Random(seed)
    stream = []
    for i in range(n):
        attack = rng.random() < 0.6
        detected = rng.random() < (0.9 if attack else 0.3)
        score = rng.uniform(.15, .5) if attack else rng.uniform(.02, .12)
        stream.append({"alert_id": f"alert-{i:03d}", "sample_id": "x", "anomaly_score": score, "threshold": .0614,
                       "detected": detected, "verification_result": "VERIFIED",
                       "verification_confidence": 1.0 if rng.random() < .8 else .8,
                       "xai_evidence": [{"feature": "SYNRatio", "contribution": rng.uniform(.01, .3)}],
                       "retrieved_techniques": [{"technique_id": "T1498.001"}] if rng.random() < .7 else [],
                       "_simulator_attack": attack})
    return stream


def stable(result):
    return json.loads(json.dumps({**{k: v for k, v in result.items() if k not in TIMING_KEYS},
                                  "metrics": [{k: v for k, v in m.items() if k not in TIMING_KEYS} for m in result["metrics"]]}))


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.out = Path(self.tmp.name)
        self.stream = make_stream()

    def tearDown(self): self.tmp.cleanup()

    def run_it(self, name="run", **kw):
        return replay.run_replay(self.stream, self.out / name, **kw)

    def test_only_detected_flows_reach_the_policy(self):
        result = self.run_it()
        detected = [a for a in self.stream if a["detected"]]
        self.assertEqual(result["routed_alerts"], len(detected))
        self.assertEqual(result["not_routed"]["unflagged_attack_flows"], sum(a["_simulator_attack"] for a in self.stream if not a["detected"]))
        self.assertEqual(result["not_routed"]["unflagged_benign_flows"], sum(not a["_simulator_attack"] for a in self.stream if not a["detected"]))
        for m in result["metrics"]: self.assertEqual(m["alerts_processed"], len(detected))

    def test_every_routed_alert_is_evaluated_exactly_once_by_every_policy(self):
        self.run_it()
        rows = [json.loads(l) for l in (self.out / "run" / "action_ledger.jsonl").read_text().splitlines()]
        evaluated = [r for r in rows if r.get("phase") == "evaluation"]
        detected_ids = sorted(a["alert_id"] for a in self.stream if a["detected"])
        for policy in replay.POLICIES:
            self.assertEqual(sorted(r["alert"]["alert_id"] for r in evaluated if r["policy"] == policy), detected_ids)

    def test_policy_never_sees_simulator_labels(self):
        self.run_it()
        rows = [json.loads(l) for l in (self.out / "run" / "action_ledger.jsonl").read_text().splitlines()]
        for r in rows:
            alert = r.get("alert", r)
            if r.get("phase") == "evaluation":
                self.assertFalse([k for k in alert if k.startswith("_")])
                if r["policy"].startswith("Contextual"):
                    self.assertEqual(list(r["context_features"]), CONTEXT_FEATURES)
            else:
                self.assertNotIn("_simulator_attack", r)

    def test_same_seed_gives_identical_results(self):
        a, b = self.run_it("a", seed=11), self.run_it("b", seed=11)
        self.assertEqual(stable(a), stable(b))

    def test_different_seed_changes_the_split(self):
        a, b = self.run_it("a", seed=11), self.run_it("b", seed=12)
        self.assertNotEqual(a["metrics"][0]["per_fold_mean_reward"], b["metrics"][0]["per_fold_mean_reward"])

    def test_artifacts_record_reward_version_and_decision_regions(self):
        result = self.run_it()
        self.assertEqual(result["reward_version"], REWARD_VERSION)
        table = json.loads((self.out / "run" / "reward_table.json").read_text())
        self.assertEqual(table["version"], REWARD_VERSION)
        self.assertEqual({r["action"] for r in table["best_action_by_attack_probability"]},
                         {"MONITOR_ONLY", "HUMAN_REVIEW", "CONFIRM_TO_ACT", "AUTO_REMEDIATE"})
        for name in ("policy_comparison_metrics.json", "experiment_config.json", "baseline_definitions.json",
                     "context_schema.json", "approval_queue_statistics.json", "learning_curve.svg",
                     "action_distribution.svg", "summary.md"):
            self.assertTrue((self.out / "run" / name).exists(), name)
        self.assertIn("never became alerts", (self.out / "run" / "summary.md").read_text())

    def test_training_records_carry_versions_and_history(self):
        self.run_it()
        rows = [json.loads(l) for l in (self.out / "run" / "action_ledger.jsonl").read_text().splitlines()]
        training = [r for r in rows if r.get("phase") == "training"]
        self.assertTrue(training)
        for r in training:
            self.assertEqual(r["reward_version"], REWARD_VERSION)
            self.assertEqual(r["policy_update_status"], "UPDATED")
            self.assertEqual([e["event"] for e in r["history"]][-2:], ["OUTCOME_OBSERVED", "POLICY_UPDATED"])

    def test_too_few_detected_alerts_is_a_clear_error(self):
        for a in self.stream[1:]: a["detected"] = False
        self.stream[0]["detected"] = True
        with self.assertRaises(SystemExit): self.run_it()

    def test_fewer_alerts_than_folds_reduces_the_fold_count(self):
        for i, a in enumerate(self.stream): a["detected"] = i < 3
        self.assertEqual(self.run_it(folds=5)["folds"], 3)


if __name__ == "__main__": unittest.main()
