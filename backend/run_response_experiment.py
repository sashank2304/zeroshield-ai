"""Small reproducible response-policy replay on the ten cached prepared samples.

Run from backend with .venv: python run_response_experiment.py
The stream is too small for generalization claims; all outputs say so.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ZeroShield_Verification_Handoff"))
from services.bandit_service import (  # noqa: E402
    ACTIONS, CONTEXT_FEATURES, REJECTED_APPROVAL_REWARDS, REWARDS, BanditService, always_human_action, build_context,
    fixed_rule_action, simulate_outcome,
)
from services.llm_service import reason  # noqa: E402
from services.mitre_service import MitreService  # noqa: E402
from services.model_service import Detector  # noqa: E402
from services.verification_service import verify_record  # noqa: E402

SEED = 20261020
OUT_ROOT = ROOT / "backend" / "experiments" / "contextual_bandit_replay_v1"


def alert_stream():
    # Use the documented ten-row cache only; do not scan the large source datasets.
    samples = json.loads((ROOT / "backend" / "data" / "demo_samples.json").read_text(encoding="utf-8"))
    detector, mitre = Detector(ROOT), MitreService(ROOT)
    stream = []
    for category, features in samples.items():
        result = detector.analyze(features)
        mapping = mitre.retrieve_behavior(result)
        # Keep the offline experiment explicitly independent of API credentials/network.
        reasoning = reason(result, None, mapping)
        verified = verify_record({**result, "feature_names": [x["feature"] for x in result["top_features"]], "all_feature_names": [x["feature"] for x in result["all_contributions"]]}, mapping, reasoning)
        stream.append({
            "alert_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"zeroshield-demo:{category}")),
            "sample_id": category, "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_summary": {x["feature"]: x["value"] for x in result["top_features"]},
            "anomaly_score": result["anomaly_score"], "threshold": result["threshold"], "detected": result["detected"],
            "xai_evidence": result["top_features"], "retrieval_query": mapping.get("query"),
            "retrieved_techniques": mapping.get("techniques", []), "llm_explanation": reasoning,
            "verification_result": verified["verdict"], "verification_checks": verified["checks"],
            "verification_confidence": verified["verification_confidence"],
            "_simulator_attack": category != "BenignTraffic",
        })
    return stream


def score_policy(name, decisions, stream):
    rewards, counts, benign_actions, attack_actions = [], Counter(), [], []
    benign_remediations = attack_remediations = attack_human = attack_monitor = benign_human = misses = 0
    queue = 0
    for alert, action in zip(stream, decisions):
        attack = alert["_simulator_attack"]
        outcome = simulate_outcome(action, attack)
        rewards.append(outcome["reward"]); counts[action] += 1
        if action == "CONFIRM_TO_ACT": queue += 1
        (attack_actions if attack else benign_actions).append(action)
        if attack and action == "AUTO_REMEDIATE": attack_remediations += 1
        if attack and action == "HUMAN_REVIEW": attack_human += 1
        if attack and action == "MONITOR_ONLY": attack_monitor += 1
        if attack and action == "MONITOR_ONLY": misses += 1
        if not attack and action == "AUTO_REMEDIATE": benign_remediations += 1
        if not attack and action == "HUMAN_REVIEW": benign_human += 1
    n = len(stream)
    return {"policy": name, "alerts_processed": n, "mean_reward_per_alert": sum(rewards) / max(n, 1),
            "cumulative_reward": sum(rewards), "action_distribution": {a: counts[a] for a in ACTIONS},
            "action_selection_rate": {a: counts[a] / max(n, 1) for a in ACTIONS},
            "benign_remediation_rate": benign_remediations / max(len(benign_actions), 1),
            "attack_remediation_rate": attack_remediations / max(len(attack_actions), 1),
            "attack_human_review_rate": attack_human / max(len(attack_actions), 1),
            "attack_monitor_only_rate": attack_monitor / max(len(attack_actions), 1),
            "benign_human_review_rate": benign_human / max(len(benign_actions), 1),
            "unnecessary_remediation_count": benign_remediations, "missed_attack_count": misses,
            "approval_queue_size": queue}


def svg_chart(path: Path, title: str, labels: list[str], values: list[float]):
    width, height, margin = 760, 300, 45
    high, low = max(values + [1]), min(values + [0])
    span = max(high - low, 1)
    points = []
    for i, value in enumerate(values):
        x = margin + i * (width - 2 * margin) / max(len(values) - 1, 1)
        y = height - margin - (value - low) / span * (height - 2 * margin)
        points.append(f"{x:.1f},{y:.1f}")
    labels_svg = "".join(f'<text x="{margin+i*(width-2*margin)/max(len(labels)-1,1):.1f}" y="{height-12}" fill="#9aa" font-size="10">{label}</text>' for i, label in enumerate(labels))
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="#101820"/><text x="20" y="25" fill="white" font-size="16">{title}</text><polyline points="{" ".join(points)}" fill="none" stroke="#40cbbb" stroke-width="3"/>{labels_svg}</svg>', encoding="utf-8")


def svg_bars(path: Path, title: str, labels: list[str], values: list[float]):
    width, height, left, bottom = 760, 300, 50, 55
    maximum = max(values + [1])
    slot = (width - left - 20) / max(len(values), 1)
    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        bar_height = (height - bottom - 50) * value / maximum
        x = left + index * slot + slot * .2
        y = height - bottom - bar_height
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot*.6:.1f}" height="{bar_height:.1f}" fill="#40cbbb"/><text x="{x:.1f}" y="{height-30}" fill="#9aa" font-size="10">{label}</text><text x="{x:.1f}" y="{y-5:.1f}" fill="white" font-size="11">{value:g}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="#101820"/><text x="20" y="25" fill="white" font-size="16">{title}</text>{"".join(bars)}</svg>', encoding="utf-8")


def main():
    os.environ.pop("OPENAI_API_KEY", None)
    started = time.perf_counter(); stream = alert_stream()
    # With one benign cached row, reserve it and four attack categories for evaluation.
    # The other five attack categories form replay training; this is illustrative only.
    evaluation = [a for a in stream if a["sample_id"] == "BenignTraffic"] + [a for a in stream if a["sample_id"] != "BenignTraffic"][-4:]
    training = [a for a in stream if a not in evaluation]
    OUT = OUT_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    OUT.mkdir(parents=True, exist_ok=False)
    bandit = BanditService(OUT / "training_policy.sqlite3", seed=SEED)
    # An explicit replay reset starts a new policy version; the experiment ledger is retained.
    bandit.reset_policy()
    learning_curve, training_ledger = [], []
    training_started = time.perf_counter()
    for alert in training:
        selected = bandit.select(alert, explore=True)
        internal = {k: v for k, v in alert.items() if not k.startswith("_")}
        bandit.record({**internal, **selected})
        result = bandit.feedback(alert["alert_id"], bool(alert["_simulator_attack"]),
                                 "APPROVED" if selected["selected_action"] == "CONFIRM_TO_ACT" else None)
        training_ledger.append(result); learning_curve.append(sum(x["reward"] for x in training_ledger))
    training_seconds = time.perf_counter() - training_started

    policies = {"Contextual bandit (greedy held-out)": [], "Fixed rules": [], "Always HUMAN_REVIEW": [], "Random (seeded)": []}
    randomizer = random.Random(SEED)
    evaluation_traces = []
    for i, alert in enumerate(evaluation):
        # The policy sees verified alert fields only; held-out label is used afterward for scoring.
        clean = {k: v for k, v in alert.items() if not k.startswith("_")}
        choices = {
            "Contextual bandit (greedy held-out)": bandit.select(clean, explore=False),
            "Fixed rules": {"selected_action": fixed_rule_action(clean)},
            "Always HUMAN_REVIEW": {"selected_action": always_human_action(clean)},
            "Random (seeded)": {"selected_action": randomizer.choice(ACTIONS)},
        }
        for policy_name, decision in choices.items():
            action = decision["selected_action"]
            policies[policy_name].append(action)
            outcome = simulate_outcome(action, bool(alert["_simulator_attack"]))
            evaluation_traces.append({"alert": clean, "policy": policy_name,
                "context_features": decision.get("context_features", dict(zip(CONTEXT_FEATURES, build_context(clean)))),
                "action_scores": decision.get("action_scores"), "selected_action": action,
                "policy_version": decision.get("policy_version", "baseline-v1"),
                "selection_mode": decision.get("selection_mode", policy_name),
                "selection_reason": decision.get("reason", "Action selected by the documented baseline."),
                "simulated_execution_status": outcome["execution_status"],
                "approval_status": "SIMULATOR_ASSUMES_APPROVED" if action == "CONFIRM_TO_ACT" else "NOT_REQUIRED",
                "observed_outcome": outcome["observed_outcome"], "outcome_source": "offline simulator label; read after selection",
                "reward": outcome["reward"], "reward_version": "designed-simulation-v1", "policy_update_status": "EVALUATION_NO_UPDATE"})
    metrics = [score_policy(name, decisions, evaluation) for name, decisions in policies.items()]
    for row in metrics: row["policy_training_update_seconds"] = training_seconds if row["policy"].startswith("Contextual") else None
    with (OUT / "action_ledger.jsonl").open("w", encoding="utf-8") as f:
        for item in training_ledger: f.write(json.dumps(item) + "\n")
        for item in evaluation_traces: f.write(json.dumps(item) + "\n")
    (OUT / "policy_comparison_metrics.json").write_text(json.dumps({"metrics": metrics, "training_updates": len(training), "evaluation_alerts": len(evaluation), "total_elapsed_seconds": time.perf_counter()-started}, indent=2), encoding="utf-8")
    (OUT / "reward_table.json").write_text(json.dumps({"version": "designed-simulation-v1", "table": REWARDS, "rejected_approval_outcome": REJECTED_APPROVAL_REWARDS, "claim": "Designed trade-off only; not real operational cost or optimal policy."}, indent=2), encoding="utf-8")
    (OUT / "context_schema.json").write_text(json.dumps({"version": "context-v1", "features": CONTEXT_FEATURES, "no_labels_or_future_outcomes": True}, indent=2), encoding="utf-8")
    (OUT / "baseline_definitions.json").write_text(json.dumps({"fixed": "VERIFIED + detected + confidence>=0.9 -> AUTO_REMEDIATE; else detected -> HUMAN_REVIEW; else MONITOR_ONLY", "always_human": "HUMAN_REVIEW", "random": {"method": "random.Random(seed).choice(ACTIONS)", "seed": SEED}}, indent=2), encoding="utf-8")
    (OUT / "approval_queue_statistics.json").write_text(json.dumps({m["policy"]: {"queued": m["approval_queue_size"], "live_approval_simulated": False} for m in metrics}, indent=2), encoding="utf-8")
    (OUT / "experiment_config.json").write_text(json.dumps({"seed": SEED, "algorithm": "LinUCB", "alpha": bandit.alpha,
        "reward_version": "designed-simulation-v1", "reward_table": REWARDS,
        "context_schema": {"version": "context-v1", "features": CONTEXT_FEATURES},
        "baseline_definitions": {"fixed": "VERIFIED + detected + confidence>=0.9 -> AUTO_REMEDIATE; else detected -> HUMAN_REVIEW; else MONITOR_ONLY", "always_human": "HUMAN_REVIEW for every alert", "random": "Python random.Random(seed=20261020), one draw per held-out alert"},
        "split": "5 attack categories train; BenignTraffic plus 4 attack categories held out",
        "stream": "ten cached project samples; first 5 non-benign categories train, remaining 4 non-benign plus benign evaluate",
        "approval_simulator_assumption": "CONFIRM_TO_ACT is assumed approved offline so its reward can be observed; this is not live approval.",
        "limitations": "Tiny category cache, one benign row, category-level split only; no generalization or statistical significance claim.", "labels_in_context": False}, indent=2), encoding="utf-8")
    svg_chart(OUT / "learning_curve.svg", "Training replay cumulative reward", [str(i+1) for i in range(len(learning_curve))], learning_curve)
    svg_bars(OUT / "action_distribution.svg", "Held-out action distribution: bandit", list(ACTIONS), [metrics[0]["action_distribution"][a] for a in ACTIONS])
    (OUT / "summary.md").write_text("# Stage 7 replay summary\n\nThis is a ten-row cached demo experiment, not a representative evaluation. Five attack categories trained LinUCB; the benign cache row and four attack categories were held out. All policies used the identical held-out stream and reward table. See `policy_comparison_metrics.json` and the action ledger. The reward values are designed simulation trade-offs, not real-world costs or optimality. This response policy does not change or improve Deep SVDD detection accuracy. No statistical significance is claimed. Asset criticality and analyst history were unavailable and not fabricated. A CONFIRM_TO_ACT in the training simulator is assumed approved to permit outcome/reward observation; live alerts remain pending actual analyst approval.\n", encoding="utf-8")
    print(json.dumps({"output_directory": str(OUT), "training_updates": len(training), "metrics": metrics}, indent=2))


if __name__ == "__main__": main()
