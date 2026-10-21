"""Stage 7-9 contextual response policy and local audit ledger.

LinUCB uses only pre-action evidence. Ground-truth labels are accepted only by
simulate_outcome(), after an action has been selected.
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ACTIONS = ("AUTO_REMEDIATE", "CONFIRM_TO_ACT", "HUMAN_REVIEW", "MONITOR_ONLY")
POLICY_VERSION = "linucb-v1"
REWARD_VERSION = "designed-simulation-v1"
REWARDS = {
    "attack": {"AUTO_REMEDIATE": 5, "CONFIRM_TO_ACT": 3, "HUMAN_REVIEW": 2, "MONITOR_ONLY": -6},
    "benign": {"AUTO_REMEDIATE": -12, "CONFIRM_TO_ACT": -2, "HUMAN_REVIEW": -2, "MONITOR_ONLY": 1},
}
REJECTED_APPROVAL_REWARDS = {"attack": -6, "benign": -2}
CONTEXT_FEATURES = ["intercept", "normalized_score", "detected", "verification_confidence",
                    "xai_feature_fraction", "leading_xai_magnitude", "mitre_context_present"]


def fixed_rule_action(alert: dict) -> str:
    if alert.get("verification_result") != "VERIFIED": return "HUMAN_REVIEW"
    if alert.get("detected") and float(alert.get("verification_confidence", 0)) >= 0.9: return "AUTO_REMEDIATE"
    return "HUMAN_REVIEW" if alert.get("detected") else "MONITOR_ONLY"


def random_action(seed: int, index: int) -> str:
    return random.Random(seed + index).choice(ACTIONS)


def always_human_action(_alert: dict) -> str:
    return "HUMAN_REVIEW"


def build_context(alert: dict) -> list[float]:
    """Stable context encoding. No labels, outcomes, reward, or post-action fields."""
    score, threshold = float(alert["anomaly_score"]), float(alert["threshold"])
    xai = alert.get("xai_evidence", [])
    leading = max((abs(float(x.get("contribution", 0))) for x in xai), default=0.0)
    return [1.0, max(-1.0, min(5.0, score / max(threshold, 1e-12))),
            float(bool(alert.get("detected"))), float(alert.get("verification_confidence", 0)),
            min(1.0, len(xai) / 10), min(5.0, leading),
            float(bool(alert.get("retrieved_techniques")))]


def simulate_outcome(action: str, ground_truth_attack: bool) -> dict:
    if action not in ACTIONS:
        raise ValueError("Unknown response action")
    status = ("SIMULATED_REMEDIATION" if action == "AUTO_REMEDIATE" else
              "PENDING_HUMAN_APPROVAL" if action == "CONFIRM_TO_ACT" else
              "ROUTED_FOR_REVIEW" if action == "HUMAN_REVIEW" else "MONITORING")
    return {"execution_status": status, "proposed_response": "Simulated network-flow containment" if action in ACTIONS[:2] else "Analyst triage" if action == "HUMAN_REVIEW" else "Continue observation",
            "approval_required": action == "CONFIRM_TO_ACT", "approval_status": "PENDING" if action == "CONFIRM_TO_ACT" else "NOT_REQUIRED",
            "observed_outcome": "attack" if ground_truth_attack else "benign",
            "reward": REWARDS["attack" if ground_truth_attack else "benign"][action]}


class BanditService:
    def __init__(self, db_path: Path, seed: int = 20261020, alpha: float = 0.25):
        self.db_path, self.seed, self.alpha = Path(db_path), seed, alpha
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self._feedback_lock = threading.Lock()
        self._init_db()
        self.A = {a: [[float(i == j) for j in range(len(CONTEXT_FEATURES))] for i in range(len(CONTEXT_FEATURES))] for a in ACTIONS}
        self.b = {a: [0.0] * len(CONTEXT_FEATURES) for a in ACTIONS}
        self.updates = 0
        self._load_policy()

    def _connect(self):
        c = sqlite3.connect(self.db_path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_db(self):
        with self._connection() as c:
            c.execute("CREATE TABLE IF NOT EXISTS alerts(alert_id TEXT PRIMARY KEY, record_json TEXT NOT NULL, created_at TEXT NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS policy(id INTEGER PRIMARY KEY CHECK(id=1), state_json TEXT NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            c.execute("INSERT OR IGNORE INTO metadata VALUES('seed', ?)", (str(self.seed),))

    def _load_policy(self):
        with self._connection() as c:
            row = c.execute("SELECT state_json FROM policy WHERE id=1").fetchone()
        if row:
            state = json.loads(row[0]); self.A, self.b, self.updates = state["A"], state["b"], state["updates"]

    def _save_policy(self):
        state = json.dumps({"A": self.A, "b": self.b, "updates": self.updates})
        with self._connection() as c:
            c.execute("INSERT INTO policy(id,state_json) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET state_json=excluded.state_json", (state,))

    @staticmethod
    def _solve(matrix, vector):
        # Small deterministic Gauss-Jordan solve; no RL dependency required.
        n = len(vector); m = [list(map(float, matrix[i])) + [float(vector[i])] for i in range(n)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
            m[col], m[pivot] = m[pivot], m[col]
            d = m[col][col]
            if abs(d) < 1e-12: raise ValueError("Singular policy matrix")
            m[col] = [v / d for v in m[col]]
            for r in range(n):
                if r != col:
                    f = m[r][col]; m[r] = [x - f*y for x, y in zip(m[r], m[col])]
        return [m[i][-1] for i in range(n)]

    def select(self, alert: dict, explore: bool = True):
        x = build_context(alert); scores = {}
        for action in ACTIONS:
            theta = self._solve(self.A[action], self.b[action])
            mean = sum(a*b for a, b in zip(theta, x))
            inv_x = self._solve(self.A[action], x)
            uncertainty = math.sqrt(max(0.0, sum(a*b for a, b in zip(x, inv_x))))
            scores[action] = mean + (self.alpha * uncertainty if explore else 0.0)
        # Deployment safety constraint: a faithful report is not evidence for remediation
        # when the detector did not flag the flow or verification confidence is low.
        auto_eligible = bool(alert.get("detected")) and float(alert.get("verification_confidence", 0)) >= 0.85
        if not auto_eligible:
            scores["AUTO_REMEDIATE"] = -1_000_000.0
        action = max(ACTIONS, key=lambda a: (scores[a], -ACTIONS.index(a)))
        return {"context_features": dict(zip(CONTEXT_FEATURES, x)), "action_scores": scores,
                "selected_action": action, "selection_mode": "exploration-enabled LinUCB" if explore else "greedy LinUCB",
                "policy_version": POLICY_VERSION, "auto_remediation_eligible": auto_eligible,
                "reason": f"Highest LinUCB score ({scores[action]:.4f}) among eligible actions." + (" AUTO_REMEDIATE was safety-masked because detection was negative or verification confidence was below 0.85." if not auto_eligible else "")}

    def update(self, alert_id: str, context: dict, action: str, reward: float):
        with self._connection() as c:
            row = c.execute("SELECT record_json FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            if not row: raise KeyError(alert_id)
            record = json.loads(row[0])
            if record.get("policy_update_status") == "UPDATED": return False
            if record["selected_action"] != action: raise ValueError("Feedback action differs from selected action")
            x = [float(context[k]) for k in CONTEXT_FEATURES]
            for i in range(len(x)):
                self.b[action][i] += reward * x[i]
                for j in range(len(x)): self.A[action][i][j] += x[i] * x[j]
            self.updates += 1
            record.update(reward=reward, policy_update_status="UPDATED", updated_policy_version=f"{POLICY_VERSION}+{self.updates}")
            c.execute("UPDATE alerts SET record_json=? WHERE alert_id=?", (json.dumps(record), alert_id))
        self._save_policy()
        return True

    def record(self, alert: dict):
        decision = self.select(alert)
        record = {**alert, **decision, "approval_status": "PENDING" if decision["selected_action"] == "CONFIRM_TO_ACT" else "NOT_REQUIRED",
                  "proposed_response": "Simulated network-flow containment" if decision["selected_action"] in ACTIONS[:2] else "Analyst triage" if decision["selected_action"] == "HUMAN_REVIEW" else "Continue observation",
                  "approval_required": decision["selected_action"] == "CONFIRM_TO_ACT",
                  "simulated_execution_status": "PENDING_HUMAN_APPROVAL" if decision["selected_action"] == "CONFIRM_TO_ACT" else "SIMULATED_NO_REAL_ACTION",
                  "execution_timestamp": datetime.now(timezone.utc).isoformat(), "observed_outcome": None, "reward": None,
                  "policy_update_status": "WAITING_FOR_FEEDBACK"}
        with self._connection() as c:
            c.execute("INSERT OR IGNORE INTO alerts VALUES(?,?,?)", (record["alert_id"], json.dumps(record), datetime.now(timezone.utc).isoformat()))
            existing = c.execute("SELECT record_json FROM alerts WHERE alert_id=?", (record["alert_id"],)).fetchone()
        self._save_policy()
        return json.loads(existing[0])

    def feedback(self, alert_id: str, ground_truth_attack: bool, reviewer_choice: str | None = None):
        with self._feedback_lock:
            return self._feedback(alert_id, ground_truth_attack, reviewer_choice)

    def _feedback(self, alert_id: str, ground_truth_attack: bool, reviewer_choice: str | None = None):
        with self._connection() as c:
            row = c.execute("SELECT record_json FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            if not row: raise KeyError(alert_id)
            rec = json.loads(row[0])
            if rec.get("reward") is not None: return rec
            approval = reviewer_choice or rec.get("approval_status")
            if rec["selected_action"] == "CONFIRM_TO_ACT" and approval not in ("APPROVED", "REJECTED"):
                raise ValueError("Human approval or rejection is required before feedback")
            outcome = simulate_outcome(rec["selected_action"], ground_truth_attack)
            if approval in ("APPROVED", "REJECTED"):
                outcome["approval_status"] = approval
                outcome["execution_status"] = "SIMULATED_REMEDIATION" if approval == "APPROVED" else "REJECTED_BY_ANALYST"
            rec.update(simulated_execution_status=outcome["execution_status"], approval_status=outcome["approval_status"],
                       observed_outcome=outcome["observed_outcome"], reward=outcome["reward"],
                       outcome_source="offline simulator ground-truth label", policy_update_status="WAITING_FOR_UPDATE")
            if approval in ("APPROVED", "REJECTED"):
                rec["reviewer_choice"] = approval
                rec.setdefault("reviewed_at", datetime.now(timezone.utc).isoformat())
                # Rejection leaves a genuine attack uncontained; benign approval workload still costs -2.
                if approval == "REJECTED" and ground_truth_attack:
                    rec["reward"] = REJECTED_APPROVAL_REWARDS["attack"]
            c.execute("UPDATE alerts SET record_json=? WHERE alert_id=?", (json.dumps(rec), alert_id))
        self.update(alert_id, rec["context_features"], rec["selected_action"], rec["reward"])
        with self._connection() as c:
            return json.loads(c.execute("SELECT record_json FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()[0])

    def decide_approval(self, alert_id: str, choice: str):
        if choice not in ("APPROVED", "REJECTED"):
            raise ValueError("Choice must be APPROVED or REJECTED")
        with self._connection() as c:
            row = c.execute("SELECT record_json FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            if not row: raise KeyError(alert_id)
            record = json.loads(row[0])
            if record.get("verification_result") != "VERIFIED": raise ValueError("Approval cannot bypass verification")
            if record.get("selected_action") != "CONFIRM_TO_ACT": raise ValueError("Alert does not require approval")
            if record.get("approval_status") != "PENDING": raise ValueError("Approval was already decided")
            record.update(approval_status=choice, reviewer_choice=choice,
                          reviewed_at=datetime.now(timezone.utc).isoformat(),
                          simulated_execution_status="SIMULATED_REMEDIATION" if choice == "APPROVED" else "REJECTED_BY_ANALYST")
            c.execute("UPDATE alerts SET record_json=? WHERE alert_id=?", (json.dumps(record), alert_id))
            return record

    def list_alerts(self):
        with self._connection() as c: rows = c.execute("SELECT record_json FROM alerts ORDER BY created_at DESC").fetchall()
        return [json.loads(r[0]) for r in rows]

    def reset_policy(self):
        self.A = {a: [[float(i == j) for j in range(len(CONTEXT_FEATURES))] for i in range(len(CONTEXT_FEATURES))] for a in ACTIONS}
        self.b = {a: [0.0] * len(CONTEXT_FEATURES) for a in ACTIONS}; self.updates = 0; self._save_policy()

