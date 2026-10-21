from __future__ import annotations

import logging
import uuid
import subprocess
import sys
import threading
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from services.model_service import Detector
from services.mitre_service import MitreService
from services.sample_service import SampleService
from services.llm_service import reason
from schemas.analysis import AnalysisRequest
from services.bandit_service import BanditService
from services.verification_service import verify_record

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
services: dict = {}
replay_process = None
replay_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector = Detector(ROOT)
    services.update(detector=detector, samples=SampleService(ROOT, detector.features), mitre=MitreService(ROOT),
                    bandit=BanditService(ROOT / "backend" / "data" / "zeroshield_ledger.sqlite3"))
    yield


app = FastAPI(title="ZeroShield AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    d = services.get("detector")
    if not d or not d.loaded:
        return {"status": "degraded", "model": "Deep SVDD", "threshold": 0.0614192100, "features": 43}
    return {"status": "ok", "model": "Deep SVDD", "threshold": d.threshold, "features": len(d.features)}


@app.get("/api/model-info")
def model_info():
    d = services["detector"]
    return {"model": "Deep SVDD", "architecture": {"input": len(d.features), "hidden": d.hidden, "latent": d.latent_dim}, "feature_count": len(d.features), "features": d.features, "threshold": d.threshold, "model_loaded": d.loaded, "scaler_loaded": False, "preprocessing": "Input must use the project's existing StandardScaler-prepared feature space; no scaler artifact was present.", "xai_available": d.loaded, "mitre_available": services["mitre"].available, "mitre_behavior_index_available": services["mitre"].index is not None, "llm_status": "OpenAI configured" if __import__("os").getenv("OPENAI_API_KEY") else "Demo Fallback Reasoning", "response_actions": ["AUTO_REMEDIATE", "CONFIRM_TO_ACT", "HUMAN_REVIEW", "MONITOR_ONLY"]}


@app.get("/api/demo-samples")
def demo_samples():
    return services["samples"].available()


def analyze(features: dict[str, float], label: str | None = None):
    d = services["detector"]
    if not d.loaded:
        raise HTTPException(503, "Deep SVDD model is unavailable. Check backend logs and model file.")
    if len(features) != len(d.features) or set(features) != set(d.features):
        missing = [x for x in d.features if x not in features]
        extra = [x for x in features if x not in d.features]
        raise HTTPException(422, {"message": "Expected exactly the 43 project features.", "missing": missing, "extra": extra})
    try:
        result = d.analyze(features)
        mapping = services["mitre"].retrieve_behavior(result)
        reasoning = reason(result, None, mapping)
        verification = verify_record({**result, "feature_names": [x["feature"] for x in result["top_features"]], "all_feature_names": [x["feature"] for x in result["all_contributions"]]}, mapping, reasoning)
        alert_id = str(uuid.uuid4())
        alert = {"alert_id": alert_id, "sample_id": label or "user_input", "timestamp": datetime.now(timezone.utc).isoformat(),
                 "feature_summary": {x["feature"]: x["value"] for x in result["top_features"]},
                 "anomaly_score": result["anomaly_score"], "threshold": result["threshold"], "detected": result["detected"],
                 "xai_evidence": result["top_features"], "retrieval_query": mapping.get("query"), "retrieved_techniques": mapping.get("techniques", []),
                 "llm_explanation": reasoning, "verification_result": verification["verdict"],
                 "verification_checks": verification["checks"], "verification_confidence": verification["verification_confidence"]}
        # No response policy runs on a failed/uncertain Stage 6 record.
        if verification["verdict"] == "VERIFIED":
            bandit_record = services["bandit"].record(alert)
        else:
            bandit_record = {"selected_action": "HUMAN_REVIEW", "policy_version": "safety-fallback-v1",
                             "action_scores": {}, "context_features": {}, "selection_mode": "verification safety fallback",
                             "reason": "Stage 6 did not verify this alert; response selection is restricted to human review.",
                             "approval_status": "NOT_REQUIRED", "simulated_execution_status": "NOT_EXECUTED_UNVERIFIED",
                             "observed_outcome": None, "reward": None, "policy_update_status": "NOT_ELIGIBLE"}
        result.update(alert_id=alert_id, attack_family=label, mitre=mapping, reasoning=reasoning,
                      verification=verification, bandit=bandit_record)
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception:
        logging.exception("Analysis failed")
        raise HTTPException(500, "Analysis failed. See backend terminal for details.")


@app.post("/api/analyze")
def analyze_request(request: AnalysisRequest):
    return analyze(request.features, request.attack_family)


@app.post("/api/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(415, "Upload a CSV file.")
    try:
        import pandas as pd
        frame = pd.read_csv(file.file, nrows=2)
        d = services["detector"]
        if len(frame) != 1:
            raise HTTPException(422, "CSV must contain exactly one flow row.")
        result = analyze({name: float(frame.iloc[0][name]) for name in d.features if name in frame.columns})
        return result
    except HTTPException:
        raise
    except Exception:
        logging.exception("CSV analysis failed")
        raise HTTPException(422, "CSV must contain one row with all 43 named feature columns and numeric values.")


@app.get("/api/response/alerts")
def response_alerts():
    return {"alerts": services["bandit"].list_alerts()}


@app.get("/api/response/approvals")
def response_approvals():
    return {"pending": [x for x in services["bandit"].list_alerts() if x.get("approval_status") == "PENDING"]}


@app.post("/api/response/approval/{alert_id}")
def response_approval(alert_id: str, payload: dict):
    try:
        return services["bandit"].decide_approval(alert_id, str(payload.get("choice", "")).upper())
    except KeyError as exc:
        raise HTTPException(404, "Alert not found.") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/response/feedback/{alert_id}")
def response_feedback(alert_id: str, payload: dict):
    try:
        return services["bandit"].feedback(alert_id, bool(payload["ground_truth_attack"]), payload.get("reviewer_choice"))
    except KeyError as exc:
        raise HTTPException(404, "Alert not found or ground_truth_attack was not supplied.") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/response/reset-policy")
def reset_policy():
    services["bandit"].reset_policy()
    return {"status": "policy_reset", "policy_version": "linucb-v1", "ledger_preserved": True}


@app.get("/api/response/experiment")
def response_experiment():
    root = ROOT / "backend" / "experiments" / "contextual_bandit_replay_v1"
    runs = sorted((p for p in root.glob("run_*") if p.is_dir()), key=lambda p: p.name, reverse=True) if root.exists() else []
    for run in runs:
        metrics = run / "policy_comparison_metrics.json"
        if metrics.exists():
            import json
            return {"status": "completed", "run_id": run.name,
                    "summary_path": str(run / "summary.md"), "metrics": json.loads(metrics.read_text(encoding="utf-8"))}
    return {"status": "not_run", "metrics": None}


@app.get("/api/response/experiment/{plot_name}.svg")
def response_experiment_plot(plot_name: str):
    if plot_name not in ("learning_curve", "action_distribution"):
        raise HTTPException(404, "Unknown experiment plot.")
    root = ROOT / "backend" / "experiments" / "contextual_bandit_replay_v1"
    runs = sorted((p for p in root.glob("run_*") if p.is_dir()), key=lambda p: p.name, reverse=True) if root.exists() else []
    for run in runs:
        path = run / f"{plot_name}.svg"
        if path.exists(): return FileResponse(path, media_type="image/svg+xml")
    raise HTTPException(404, "No completed replay plot is available.")


@app.post("/api/response/replay/start")
def start_response_replay():
    global replay_process
    with replay_lock:
        if replay_process is not None and replay_process.poll() is None:
            raise HTTPException(409, "A replay job is already running.")
        log_path = ROOT / "backend" / "experiments" / "replay_job.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log:
            replay_process = subprocess.Popen([sys.executable, "run_response_experiment.py"],
                cwd=ROOT / "backend", stdout=log, stderr=subprocess.STDOUT)
        return {"status": "running", "pid": replay_process.pid, "log": str(log_path)}


@app.get("/api/response/replay/status")
def response_replay_status():
    process = replay_process
    if process is None: return {"status": "idle"}
    code = process.poll()
    return {"status": "running" if code is None else "completed" if code == 0 else "failed", "exit_code": code}
