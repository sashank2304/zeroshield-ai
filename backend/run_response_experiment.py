"""Reproducible replay of the response policy on prepared alerts.

    python run_response_experiment.py                      # ten cached demo samples (too small for claims)
    python run_response_experiment.py --stream alerts.jsonl --folds 5 --seed 20261020

What the replay does, and what it deliberately does not do:
  * Only flows the detector FLAGGED are routed to the response policy. A flow below the
    threshold never becomes an alert in a real system, so no policy acts on it. Unflagged
    flows are counted separately (detector misses / true negatives), not scored as responses.
  * Routed alerts are split into k folds. For each fold a fresh policy trains (with
    exploration and feedback) on the other folds, then is evaluated greedily on the held-out
    fold. Every routed alert is evaluated exactly once. Baselines see the same held-out alerts.
  * The simulator label is read only AFTER an action is chosen, to compute reward. It is never
    part of the policy's context. A CONFIRM_TO_ACT is assumed approved so a reward can be observed.
  * Rewards come from a designed table (see reward_table.json), not measured operational cost.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from services.bandit_service import (  # noqa: E402
    ACTIONS, CONTEXT_FEATURES, REJECTED_APPROVAL_REWARDS, REWARD_VERSION, REWARDS, BanditService,
    always_human_action, build_context, decision_regions, fixed_rule_action, simulate_outcome,
)

SEED = 20261020
OUT_ROOT = ROOT / "backend" / "experiments" / "contextual_bandit_replay_v1"
POLICIES = ("Contextual bandit (greedy held-out)", "Fixed rules", "Always HUMAN_REVIEW", "Random (seeded)")
BASELINE_TEXT = {"fixed": "VERIFIED + detected + confidence>=0.9 -> AUTO_REMEDIATE; else detected -> HUMAN_REVIEW; else MONITOR_ONLY",
                 "always_human": "HUMAN_REVIEW for every alert",
                 "random": "Python random.Random(seed), one draw per held-out alert"}


def demo_alert_stream():
    """Build alerts from the ten cached prepared samples. Needs PyTorch (loads the detector)."""
    from services.llm_service import reason
    from services.mitre_service import MitreService
    from services.model_service import Detector
    from services.verification_service import verify_record
    samples = json.loads((ROOT / "backend" / "data" / "demo_samples.json").read_text(encoding="utf-8"))
    detector, mitre = Detector(ROOT), MitreService(ROOT)
    stream = []
    for category, features in samples.items():
        result = detector.analyze(features)
        mapping = mitre.retrieve_behavior(result)
        reasoning = reason(result, None, mapping)   # offline: no API key, labelled template reasoning
        verified = verify_record({**result, "feature_names": [x["feature"] for x in result["top_features"]],
                                  "all_feature_names": [x["feature"] for x in result["all_contributions"]]}, mapping, reasoning)
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


def load_stream(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def route(stream):
    """Split into alerts the response stage would actually receive, and flows it never sees."""
    routed = sorted((a for a in stream if a["detected"]), key=lambda a: a["alert_id"])
    unflagged = [a for a in stream if not a["detected"]]
    return routed, {"unflagged_attack_flows": sum(a["_simulator_attack"] for a in unflagged),
                    "unflagged_benign_flows": sum(not a["_simulator_attack"] for a in unflagged),
                    "note": "Flows below the detector threshold never become alerts. They are detector misses or "
                            "true negatives, not response decisions, and are excluded from policy scoring."}


def make_folds(alerts, k, seed):
    order = list(alerts)
    random.Random(seed).shuffle(order)
    k = max(2, min(k, len(order)))
    return [order[i::k] for i in range(k)]


def clean(alert):
    return {k: v for k, v in alert.items() if not k.startswith("_")}


def score_policy(name, decisions, evaluation):
    rewards, counts = [], Counter()
    benign_remediations = attack_remediations = attack_human = attack_monitor = benign_human = 0
    n_attack = n_benign = queue = 0
    for alert, action in zip(evaluation, decisions):
        attack = alert["_simulator_attack"]
        rewards.append(simulate_outcome(action, attack)["reward"]); counts[action] += 1
        n_attack += attack; n_benign += not attack
        queue += action == "CONFIRM_TO_ACT"
        if attack and action == "AUTO_REMEDIATE": attack_remediations += 1
        if attack and action == "HUMAN_REVIEW": attack_human += 1
        if attack and action == "MONITOR_ONLY": attack_monitor += 1
        if not attack and action == "AUTO_REMEDIATE": benign_remediations += 1
        if not attack and action == "HUMAN_REVIEW": benign_human += 1
    n = len(evaluation)
    return {"policy": name, "alerts_processed": n, "mean_reward_per_alert": sum(rewards) / max(n, 1),
            "cumulative_reward": sum(rewards), "action_distribution": {a: counts[a] for a in ACTIONS},
            "action_selection_rate": {a: counts[a] / max(n, 1) for a in ACTIONS},
            "benign_remediation_rate": benign_remediations / max(n_benign, 1),
            "attack_remediation_rate": attack_remediations / max(n_attack, 1),
            "attack_human_review_rate": attack_human / max(n_attack, 1),
            "attack_monitor_only_rate": attack_monitor / max(n_attack, 1),
            "benign_human_review_rate": benign_human / max(n_benign, 1),
            "unnecessary_remediation_count": benign_remediations, "missed_attack_count": attack_monitor,
            "approval_queue_size": queue}


def svg_chart(path: Path, title: str, labels, values):
    width, height, margin = 760, 300, 45
    high, low = max(list(values) + [1]), min(list(values) + [0])
    span = max(high - low, 1)
    points = []
    for i, value in enumerate(values):
        x = margin + i * (width - 2 * margin) / max(len(values) - 1, 1)
        y = height - margin - (value - low) / span * (height - 2 * margin)
        points.append(f"{x:.1f},{y:.1f}")
    labels_svg = "".join(f'<text x="{margin+i*(width-2*margin)/max(len(labels)-1,1):.1f}" y="{height-12}" fill="#9aa" font-size="10">{label}</text>' for i, label in enumerate(labels))
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="#101820"/><text x="20" y="25" fill="white" font-size="16">{title}</text><polyline points="{" ".join(points)}" fill="none" stroke="#40cbbb" stroke-width="3"/>{labels_svg}</svg>', encoding="utf-8")


def svg_bars(path: Path, title: str, labels, values):
    width, height, left, bottom = 760, 300, 50, 55
    maximum = max(list(values) + [1])
    slot = (width - left - 20) / max(len(values), 1)
    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        bar_height = (height - bottom - 50) * value / maximum
        x = left + index * slot + slot * .2
        y = height - bottom - bar_height
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot*.6:.1f}" height="{bar_height:.1f}" fill="#40cbbb"/><text x="{x:.1f}" y="{height-30}" fill="#9aa" font-size="10">{label}</text><text x="{x:.1f}" y="{y-5:.1f}" fill="white" font-size="11">{value:g}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="#101820"/><text x="20" y="25" fill="white" font-size="16">{title}</text>{"".join(bars)}</svg>', encoding="utf-8")


def run_replay(stream, out_dir: Path, seed: int = SEED, folds: int = 5, stream_source: str = "demo cache"):
    """Run the k-fold replay on `stream` and write all artifacts to `out_dir`. Returns the metrics dict."""
    started = time.perf_counter()
    routed, not_routed = route(stream)
    if len(routed) < 2:
        raise SystemExit(f"Only {len(routed)} detected alert(s); at least 2 are needed to train and evaluate.")
    fold_sets = make_folds(routed, folds, seed)
    out_dir.mkdir(parents=True, exist_ok=False)

    evaluation, decisions = [], {name: [] for name in POLICIES}
    ledger, traces, per_fold = [], [], {name: [] for name in POLICIES}
    learning_curve, final_versions = [], []
    randomizer = random.Random(seed)
    train_seconds = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for fold_index, test in enumerate(fold_sets):
            train = [a for i, f in enumerate(fold_sets) if i != fold_index for a in f]
            bandit = BanditService(Path(tmp) / f"fold{fold_index}.sqlite3", seed=seed + fold_index)
            t0, curve = time.perf_counter(), []
            for alert in train:
                record = bandit.record(clean(alert))
                choice = "APPROVED" if record["selected_action"] == "CONFIRM_TO_ACT" else None
                done = bandit.feedback(alert["alert_id"], bool(alert["_simulator_attack"]), choice)
                ledger.append({**done, "fold": fold_index, "phase": "training"})
                curve.append((curve[-1] if curve else 0) + done["reward"])
            train_seconds += time.perf_counter() - t0
            if fold_index == 0: learning_curve = curve
            final_versions.append(f"linucb-v1+{bandit.updates}")

            fold_actions = {name: [] for name in POLICIES}
            for alert in test:
                c = clean(alert)    # the label is not in this dict; it is read only after the choice, below
                choice = {POLICIES[0]: bandit.select(c, explore=False), POLICIES[1]: {"selected_action": fixed_rule_action(c)},
                          POLICIES[2]: {"selected_action": always_human_action(c)}, POLICIES[3]: {"selected_action": randomizer.choice(ACTIONS)}}
                evaluation.append(alert)
                for name, decision in choice.items():
                    action = decision["selected_action"]
                    decisions[name].append(action); fold_actions[name].append(action)
                    outcome = simulate_outcome(action, bool(alert["_simulator_attack"]))
                    traces.append({"phase": "evaluation", "fold": fold_index, "alert": c, "policy": name,
                                   "context_features": decision.get("context_features", dict(zip(CONTEXT_FEATURES, build_context(c)))),
                                   "action_scores": decision.get("action_scores"), "selected_action": action,
                                   "policy_version": decision.get("policy_version", "baseline-v1"),
                                   "selection_mode": decision.get("selection_mode", name),
                                   "simulated_execution_status": outcome["execution_status"],
                                   "approval_status": "SIMULATOR_ASSUMES_APPROVED" if action == "CONFIRM_TO_ACT" else "NOT_REQUIRED",
                                   "observed_outcome": outcome["observed_outcome"],
                                   "outcome_source": "offline simulator label; read after selection",
                                   "reward": outcome["reward"], "reward_version": REWARD_VERSION,
                                   "policy_update_status": "EVALUATION_NO_UPDATE"})
            for name in POLICIES:
                rewards = [simulate_outcome(act, bool(a["_simulator_attack"]))["reward"] for a, act in zip(test, fold_actions[name])]
                per_fold[name].append(sum(rewards) / len(rewards))

    metrics = [score_policy(name, decisions[name], evaluation) for name in POLICIES]
    for row in metrics:
        row["per_fold_mean_reward"] = per_fold[row["policy"]]
        row["fold_mean_of_means"] = statistics.fmean(per_fold[row["policy"]])
        row["fold_std_of_means"] = statistics.stdev(per_fold[row["policy"]]) if len(per_fold[row["policy"]]) > 1 else None
        row["policy_training_update_seconds"] = train_seconds if row["policy"].startswith("Contextual") else None
    result = {"metrics": metrics, "reward_version": REWARD_VERSION, "seed": seed, "folds": len(fold_sets),
              "routed_alerts": len(routed), "evaluation_alerts": len(evaluation), "not_routed": not_routed,
              "final_policy_version_per_fold": final_versions, "stream_source": stream_source,
              "total_elapsed_seconds": time.perf_counter() - started}

    with (out_dir / "action_ledger.jsonl").open("w", encoding="utf-8") as f:
        for item in ledger + traces: f.write(json.dumps(item) + "\n")
    (out_dir / "policy_comparison_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "reward_table.json").write_text(json.dumps({
        "version": REWARD_VERSION, "table": REWARDS, "rejected_approval_outcome": REJECTED_APPROVAL_REWARDS,
        "best_action_by_attack_probability": decision_regions(),
        "claim": "Designed trade-off only; not real operational cost or optimal policy."}, indent=2), encoding="utf-8")
    (out_dir / "context_schema.json").write_text(json.dumps({"version": "context-v1", "features": CONTEXT_FEATURES, "no_labels_or_future_outcomes": True}, indent=2), encoding="utf-8")
    (out_dir / "baseline_definitions.json").write_text(json.dumps({**BASELINE_TEXT, "seed": seed}, indent=2), encoding="utf-8")
    (out_dir / "approval_queue_statistics.json").write_text(json.dumps({m["policy"]: {"queued": m["approval_queue_size"], "live_approval_simulated": False} for m in metrics}, indent=2), encoding="utf-8")
    (out_dir / "experiment_config.json").write_text(json.dumps({
        "seed": seed, "algorithm": "LinUCB", "alpha": 0.25, "reward_version": REWARD_VERSION, "reward_table": REWARDS,
        "context_schema": {"version": "context-v1", "features": CONTEXT_FEATURES}, "baseline_definitions": BASELINE_TEXT,
        "split": f"{len(fold_sets)}-fold over detected alerts only; each alert evaluated once on a policy that did not train on it",
        "stream": stream_source, "routed_alerts": len(routed), "not_routed": not_routed,
        "approval_simulator_assumption": "CONFIRM_TO_ACT is assumed approved offline so its reward can be observed; this is not live approval.",
        "labels_in_context": False}, indent=2), encoding="utf-8")
    svg_chart(out_dir / "learning_curve.svg", "Training replay cumulative reward (fold 0)", [str(i + 1) for i in range(len(learning_curve))], learning_curve)
    svg_bars(out_dir / "action_distribution.svg", "Held-out action distribution: bandit", list(ACTIONS), [metrics[0]["action_distribution"][a] for a in ACTIONS])
    bandit_row, fixed_row = metrics[0], metrics[1]
    (out_dir / "summary.md").write_text(
        f"# Stage 7 replay summary\n\n"
        f"Stream: {stream_source}. {len(stream)} flows, of which {len(routed)} were flagged by the detector and routed to the response policy; "
        f"{not_routed['unflagged_attack_flows']} unflagged attack flow(s) and {not_routed['unflagged_benign_flows']} unflagged benign flow(s) never became alerts and are not scored.\n\n"
        f"Evaluation: {len(fold_sets)}-fold, every routed alert evaluated once by a policy that did not train on it. Reward table: {REWARD_VERSION}. Seed: {seed}.\n\n"
        f"Bandit mean reward {bandit_row['mean_reward_per_alert']:.2f} per alert (fold std {bandit_row['fold_std_of_means'] if bandit_row['fold_std_of_means'] is None else round(bandit_row['fold_std_of_means'], 2)}); "
        f"fixed rules {fixed_row['mean_reward_per_alert']:.2f}. See policy_comparison_metrics.json.\n\n"
        "The reward values are designed simulation trade-offs, not real-world costs. This response policy does not change or improve detector accuracy. "
        "No statistical significance is claimed, and a stream this small supports no general performance claim. "
        "Asset criticality and analyst history are unavailable and were not fabricated. "
        "CONFIRM_TO_ACT is assumed approved in the simulator; live alerts stay pending until an analyst decides.\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Replay the response policy on prepared alerts.")
    parser.add_argument("--stream", type=Path, help="JSONL of prepared alerts (same fields as the demo stream, plus _simulator_attack)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    os.environ.pop("OPENAI_API_KEY", None)   # keep the experiment independent of credentials and network
    stream = load_stream(args.stream) if args.stream else demo_alert_stream()
    out = OUT_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result = run_replay(stream, out, args.seed, args.folds, str(args.stream) if args.stream else "ten cached demo samples")
    print(json.dumps({"output_directory": str(out), "routed_alerts": result["routed_alerts"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
