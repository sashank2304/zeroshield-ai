"""Stage 7-9 contextual response policy and local audit ledger.

LinUCB uses only pre-action evidence. Ground-truth labels are accepted only by
simulate_outcome(), after an action has been selected.

Ledger guarantees (schema alert-record-v2):
  * Every state change (action, approval, outcome, policy update) is appended to
    the alert's own ``history`` list with a timestamp, in the same ledger record.
  * Outcome + reward + policy update are committed in ONE SQLite transaction. If
    anything fails, nothing is written and the same feedback can be retried; the
    policy is never left half-updated.
  * Feedback is idempotent: once ``policy_update_status == "UPDATED"`` repeat
    feedback is ignored. Records stranded by the old two-step write (reward saved,
    policy never updated) are repaired by the next feedback call.
  * The reward table version used is stored on every rewarded record.
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ACTIONS = ("AUTO_REMEDIATE", "CONFIRM_TO_ACT", "HUMAN_REVIEW", "MONITOR_ONLY")
POLICY_VERSION = "linucb-v1"
SCHEMA_VERSION = "alert-record-v2"

# Designed simulation trade-offs, not measured operational costs.
# v1 is kept so earlier results stay reproducible. Under v1, CONFIRM_TO_ACT is at least
# as good as HUMAN_REVIEW for BOTH outcomes, so HUMAN_REVIEW is never the best action.
# v2 makes approving a containment of a benign flow cost more than plain triage (-3 vs -2):
# it wastes the same analyst time and risks a wrongful block being approved. That opens a
# band of attack probabilities where HUMAN_REVIEW is optimal. See decision_regions().
REWARD_TABLES = {
    "designed-simulation-v1": {
        "attack": {"AUTO_REMEDIATE": 5, "CONFIRM_TO_ACT": 3, "HUMAN_REVIEW": 2, "MONITOR_ONLY": -6},
        "benign": {"AUTO_REMEDIATE": -12, "CONFIRM_TO_ACT": -2, "HUMAN_REVIEW": -2, "MONITOR_ONLY": 1},
    },
    "designed-simulation-v2": {
        "attack": {"AUTO_REMEDIATE": 5, "CONFIRM_TO_ACT": 3, "HUMAN_REVIEW": 2, "MONITOR_ONLY": -6},
        "benign": {"AUTO_REMEDIATE": -12, "CONFIRM_TO_ACT": -3, "HUMAN_REVIEW": -2, "MONITOR_ONLY": 1},
    },
}
REWARD_VERSION = "designed-simulation-v2"
REWARDS = REWARD_TABLES[REWARD_VERSION]
# A rejected approval: no containment was performed, so no wrongful-block risk was realised.
# An attack stays uncontained (-6); a benign flow only costs the analyst's approval time (-2).
REJECTED_APPROVAL_REWARDS = {"attack": -6, "benign": -2}
CONTEXT_FEATURES = ["intercept", "normalized_score", "detected", "verification_confidence",
                    "xai_feature_fraction", "leading_xai_magnitude", "mitre_context_present"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_reward(action: str, p_attack: float, table: dict | None = None) -> float:
    t = table or REWARDS
    return p_attack * t["attack"][action] + (1 - p_attack) * t["benign"][action]


def decision_regions(table: dict | None = None, steps: int = 1000) -> list[dict]:
    """Which action maximises expected reward at each probability that the flow is an attack.
    Ties go to the earlier action in ACTIONS, as in the policy. Used for docs and tests."""
    regions: list[dict] = []
    for i in range(steps + 1):
        p = i / steps
        best = max(ACTIONS, key=lambda a: (expected_reward(a, p, table), -ACTIONS.index(a)))
        if regions and regions[-1]["action"] == best:
            regions[-1]["to"] = p
        else:
            regions.append({"action": best, "from": p, "to": p})
    return regions


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


def simulate_outcome(action: str, ground_truth_attack: bool, reward_version: str = REWARD_VERSION) -> dict:
    if action not in ACTIONS:
        raise ValueError("Unknown response action")
    table = REWARD_TABLES[reward_version]
    status = ("SIMULATED_REMEDIATION" if action == "AUTO_REMEDIATE" else
              "PENDING_HUMAN_APPROVAL" if action == "CONFIRM_TO_ACT" else
              "ROUTED_FOR_REVIEW" if action == "HUMAN_REVIEW" else "MONITORING")
    return {"execution_status": status, "proposed_response": "Simulated network-flow containment" if action in ACTIONS[:2] else "Analyst triage" if action == "HUMAN_REVIEW" else "Continue observation",
            "approval_required": action == "CONFIRM_TO_ACT", "approval_status": "PENDING" if action == "CONFIRM_TO_ACT" else "NOT_REQUIRED",
            "observed_outcome": "attack" if ground_truth_attack else "benign",
            "reward": table["attack" if ground_truth_attack else "benign"][action],
            "reward_version": reward_version}


def _event(record: dict, name: str, at: str | None = None, **data) -> None:
    """Append one audit event to the alert's own history (same ledger record)."""
    record.setdefault("history", []).append({"event": name, "at": at or _utc_now(), **data})


def _identity_state():
    n = len(CONTEXT_FEATURES)
    A = {a: [[float(i == j) for j in range(n)] for i in range(n)] for a in ACTIONS}
    b = {a: [0.0] * n for a in ACTIONS}
    return A, b


class BanditService:
    def __init__(self, db_path: Path, seed: int = 20261020, alpha: float = 0.25):
        self.db_path, self.seed, self.alpha = Path(db_path), seed, alpha
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self._feedback_lock = threading.Lock()
        self._init_db()
        self.A, self.b = _identity_state()
        self.updates = 0
        self._load_policy()

    # ---- storage ---------------------------------------------------------
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

    @contextmanager
    def _tx(self):
        """One atomic read-modify-write transaction. Takes the write lock up front so two
        callers cannot both pass a status check, and rolls everything back on any error."""
        c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        try:
            c.execute("BEGIN IMMEDIATE")
            yield c
            c.execute("COMMIT")
        except BaseException:
            try: c.execute("ROLLBACK")
            except sqlite3.Error: pass
            raise
        finally:
            c.close()

    def _init_db(self):
        with self._connection() as c:
            c.execute("CREATE TABLE IF NOT EXISTS alerts(alert_id TEXT PRIMARY KEY, record_json TEXT NOT NULL, created_at TEXT NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS policy(id INTEGER PRIMARY KEY CHECK(id=1), state_json TEXT NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            c.execute("INSERT OR IGNORE INTO metadata VALUES('seed', ?)", (str(self.seed),))

    @staticmethod
    def _read_policy(c):
        row = c.execute("SELECT state_json FROM policy WHERE id=1").fetchone()
        if row:
            state = json.loads(row[0])
            return state["A"], state["b"], state["updates"]
        A, b = _identity_state()
        return A, b, 0

    @staticmethod
    def _write_policy(c, A, b, updates):
        c.execute("INSERT INTO policy(id,state_json) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET state_json=excluded.state_json",
                  (json.dumps({"A": A, "b": b, "updates": updates}),))

    def _load_policy(self):
        with self._connection() as c:
            self.A, self.b, self.updates = self._read_policy(c)

    def _save_policy(self):
        with self._tx() as c:
            self._write_policy(c, self.A, self.b, self.updates)

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

    # ---- policy ----------------------------------------------------------
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

    def _apply_update(self, c, rec: dict, context: dict | None = None, reward: float | None = None):
        """Update LinUCB from one rewarded record, inside the caller's transaction.
        Reads the policy from the database (not memory) so it is safe across processes."""
        action = rec["selected_action"]
        context = context if context is not None else rec["context_features"]
        reward = float(rec["reward"] if reward is None else reward)
        A, b, updates = self._read_policy(c)
        x = [float(context[k]) for k in CONTEXT_FEATURES]
        for i in range(len(x)):
            b[action][i] += reward * x[i]
            for j in range(len(x)): A[action][i][j] += x[i] * x[j]
        updates += 1
        self._write_policy(c, A, b, updates)
        version = f"{POLICY_VERSION}+{updates}"
        now = _utc_now()
        rec.update(policy_update_status="UPDATED", updated_policy_version=version, policy_updated_at=now)
        _event(rec, "POLICY_UPDATED", at=now, policy_version=version)

    def update(self, alert_id: str, context: dict, action: str, reward: float):
        """Apply one policy update for an alert. Returns False if it was already applied."""
        with self._tx() as c:
            row = c.execute("SELECT record_json FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            if not row: raise KeyError(alert_id)
            record = json.loads(row[0])
            if record.get("policy_update_status") == "UPDATED": return False
            if record["selected_action"] != action: raise ValueError("Feedback action differs from selected action")
            record["reward"] = reward
            record.setdefault("reward_version", REWARD_VERSION)
            self._apply_update(c, record, context=context, reward=reward)
            c.execute("UPDATE alerts SET record_json=? WHERE alert_id=?", (json.dumps(record), alert_id))
        self._load_policy()
        return True

    # ---- ledger ----------------------------------------------------------
    def record(self, alert: dict):
        decision = self.select(alert)
        now = _utc_now()
        action = decision["selected_action"]
        record = {**alert, **decision, "schema_version": SCHEMA_VERSION,
                  "policy_state_at_selection": f"{POLICY_VERSION}+{self.updates}",
                  "approval_status": "PENDING" if action == "CONFIRM_TO_ACT" else "NOT_REQUIRED",
                  "proposed_response": "Simulated network-flow containment" if action in ACTIONS[:2] else "Analyst triage" if action == "HUMAN_REVIEW" else "Continue observation",
                  "approval_required": action == "CONFIRM_TO_ACT",
                  "simulated_execution_status": "PENDING_HUMAN_APPROVAL" if action == "CONFIRM_TO_ACT" else "SIMULATED_NO_REAL_ACTION",
                  "execution_timestamp": now, "observed_outcome": None, "reward": None, "reward_version": None,
                  "policy_update_status": "WAITING_FOR_FEEDBACK"}
        _event(record, "ACTION_SELECTED", at=now, action=action, policy_version=decision["policy_version"],
               policy_state=record["policy_state_at_selection"], approval_required=record["approval_required"])
        with self._tx() as c:
            c.execute("INSERT OR IGNORE INTO alerts VALUES(?,?,?)", (record["alert_id"], json.dumps(record), now))
            existing = c.execute("SELECT record_json FROM alerts WHERE alert_id=?", (record["alert_id"],)).fetchone()
        return json.loads(existing[0])

    def feedback(self, alert_id: str, ground_truth_attack: bool, reviewer_choice: str | None = None):
        with self._feedback_lock:
            return self._feedback(alert_id, ground_truth_attack, reviewer_choice)

    def _feedback(self, alert_id: str, ground_truth_attack: bool, reviewer_choice: str | None = None):
        with self._tx() as c:
            row = c.execute("SELECT record_json FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            if not row: raise KeyError(alert_id)
            rec = json.loads(row[0])
            if rec.get("policy_update_status") == "UPDATED":
                return rec                       # duplicate feedback: no second reward, no second update
            if rec.get("reward") is None:
                approval = reviewer_choice or rec.get("approval_status")
                if rec["selected_action"] == "CONFIRM_TO_ACT" and approval not in ("APPROVED", "REJECTED"):
                    raise ValueError("Human approval or rejection is required before feedback")
                outcome = simulate_outcome(rec["selected_action"], ground_truth_attack)
                reward = outcome["reward"]
                if approval in ("APPROVED", "REJECTED"):
                    outcome["approval_status"] = approval
                    outcome["execution_status"] = "SIMULATED_REMEDIATION" if approval == "APPROVED" else "REJECTED_BY_ANALYST"
                    if approval == "REJECTED":
                        reward = REJECTED_APPROVAL_REWARDS["attack" if ground_truth_attack else "benign"]
                now = _utc_now()
                rec.update(simulated_execution_status=outcome["execution_status"], approval_status=outcome["approval_status"],
                           observed_outcome=outcome["observed_outcome"], reward=reward, reward_version=REWARD_VERSION,
                           outcome_source="offline simulator ground-truth label", feedback_at=now,
                           policy_update_status="WAITING_FOR_UPDATE")
                if approval in ("APPROVED", "REJECTED"):
                    rec["reviewer_choice"] = approval
                    rec.setdefault("reviewed_at", now)
                _event(rec, "OUTCOME_OBSERVED", at=now, outcome=outcome["observed_outcome"], reward=reward,
                       reward_version=REWARD_VERSION)
            # else: a record stranded by the old two-step write (reward saved, policy not updated);
            # fall through and apply the missing update using the stored reward.
            self._apply_update(c, rec)
            c.execute("UPDATE alerts SET record_json=? WHERE alert_id=?", (json.dumps(rec), alert_id))
        self._load_policy()
        return rec

    def decide_approval(self, alert_id: str, choice: str):
        if choice not in ("APPROVED", "REJECTED"):
            raise ValueError("Choice must be APPROVED or REJECTED")
        with self._tx() as c:
            row = c.execute("SELECT record_json FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            if not row: raise KeyError(alert_id)
            record = json.loads(row[0])
            if record.get("verification_result") != "VERIFIED": raise ValueError("Approval cannot bypass verification")
            if record.get("selected_action") != "CONFIRM_TO_ACT": raise ValueError("Alert does not require approval")
            if record.get("approval_status") != "PENDING": raise ValueError("Approval was already decided")
            now = _utc_now()
            record.update(approval_status=choice, reviewer_choice=choice, reviewed_at=now,
                          simulated_execution_status="SIMULATED_REMEDIATION" if choice == "APPROVED" else "REJECTED_BY_ANALYST")
            _event(record, "APPROVAL_DECIDED", at=now, choice=choice)
            c.execute("UPDATE alerts SET record_json=? WHERE alert_id=?", (json.dumps(record), alert_id))
        return record

    def list_alerts(self, limit: int | None = None, offset: int = 0):
        sql = "SELECT record_json FROM alerts ORDER BY created_at DESC"
        args: tuple = ()
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"; args = (int(limit), int(offset))
        with self._connection() as c: rows = c.execute(sql, args).fetchall()
        return [json.loads(r[0]) for r in rows]

    def get_alert(self, alert_id: str):
        with self._connection() as c:
            row = c.execute("SELECT record_json FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
        if not row: raise KeyError(alert_id)
        return json.loads(row[0])

    def reset_policy(self):
        self.A, self.b = _identity_state(); self.updates = 0
        self._save_policy()
