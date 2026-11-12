"""API-level tests for the response endpoints: validation, access control, safe index loading.
The detector is stubbed so these run without PyTorch."""
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from services.bandit_service import BanditService
from services.mitre_service import MitreService

ROOT = Path(__file__).resolve().parents[2]


class _StubDetector:
    loaded = False
    def __init__(self, *_a, **_k): self.features = []


with patch.dict(sys.modules, {"services.model_service": types.SimpleNamespace(Detector=_StubDetector)}):
    import main  # noqa: E402  (imported with the detector stubbed)


def alert(alert_id):
    return {"alert_id": alert_id, "anomaly_score": .2, "threshold": .1, "detected": True,
            "verification_result": "VERIFIED", "verification_confidence": 1.,
            "xai_evidence": [{"contribution": .04}], "retrieved_techniques": []}


class ResponseApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        main.services["bandit"] = BanditService(Path(self.tmp.name) / "ledger.sqlite3", seed=3)
        main.services["detector"] = _StubDetector()
        self.local = TestClient(main.app)                                   # host "testclient" = local
        self.remote = TestClient(main.app, client=("203.0.113.9", 5000))    # not the local machine

    def tearDown(self):
        main.services.clear(); self.tmp.cleanup()

    def stored(self, alert_id): return main.services["bandit"].get_alert(alert_id)

    # ---- validation ------------------------------------------------------
    def test_feedback_string_false_is_a_benign_outcome_not_an_attack(self):
        main.services["bandit"].record(alert("f1"))
        r = self.local.post("/api/response/feedback/f1", json={"ground_truth_attack": "false"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["observed_outcome"], "benign")

    def test_feedback_rejects_bad_or_missing_label(self):
        main.services["bandit"].record(alert("f2"))
        self.assertEqual(self.local.post("/api/response/feedback/f2", json={"ground_truth_attack": "maybe"}).status_code, 422)
        self.assertEqual(self.local.post("/api/response/feedback/f2", json={}).status_code, 422)
        self.assertIsNone(self.stored("f2")["reward"])

    def test_feedback_unknown_alert_is_404(self):
        self.assertEqual(self.local.post("/api/response/feedback/nope", json={"ground_truth_attack": True}).status_code, 404)

    def test_approval_accepts_lowercase_and_rejects_nonsense(self):
        self.assertEqual(self.local.post("/api/response/approval/x", json={"choice": "MAYBE"}).status_code, 422)
        self.assertEqual(self.local.post("/api/response/approval/missing", json={"choice": "approved"}).status_code, 404)

    # ---- access control --------------------------------------------------
    def test_remote_clients_cannot_change_state(self):
        main.services["bandit"].record(alert("r1"))
        for method, url, body in (("post", "/api/response/feedback/r1", {"ground_truth_attack": True}),
                                  ("post", "/api/response/approval/r1", {"choice": "APPROVED"}),
                                  ("post", "/api/response/reset-policy", None),
                                  ("post", "/api/response/replay/start", None)):
            r = getattr(self.remote, method)(url, json=body) if body is not None else getattr(self.remote, method)(url)
            self.assertEqual(r.status_code, 403, url)
        self.assertIsNone(self.stored("r1")["reward"])

    def test_local_client_can_reset_policy(self):
        self.assertEqual(self.local.post("/api/response/reset-policy").json()["status"], "policy_reset")

    def test_admin_token_is_required_and_enables_remote_use(self):
        main.services["bandit"].record(alert("t1"))
        with patch.dict(os.environ, {"ZEROSHIELD_ADMIN_TOKEN": "s3cret"}):
            body = {"ground_truth_attack": True}
            self.assertEqual(self.local.post("/api/response/feedback/t1", json=body).status_code, 401)   # even locally
            self.assertEqual(self.remote.post("/api/response/feedback/t1", json=body, headers={"X-Admin-Token": "wrong"}).status_code, 401)
            self.assertEqual(self.remote.post("/api/response/feedback/t1", json=body, headers={"X-Admin-Token": "s3cret"}).status_code, 200)

    def test_read_only_endpoints_stay_open(self):
        self.assertEqual(self.remote.get("/api/response/alerts").status_code, 200)

    # ---- health ------------------------------------------------------------
    def test_degraded_health_does_not_invent_a_threshold(self):
        body = self.local.get("/api/health").json()
        self.assertEqual(body["status"], "degraded")
        self.assertIsNone(body["threshold"])


class IndexIntegrityTests(unittest.TestCase):
    def make_root(self, sidecar_text):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        store = Path(tmp.name) / "LLM" / "rag_store"; store.mkdir(parents=True)
        shutil.copy(ROOT / "LLM" / "rag_store" / "mitre_tfidf.pkl", store / "mitre_tfidf.pkl")
        if sidecar_text is not None: (store / "mitre_tfidf.pkl.sha256").write_text(sidecar_text)
        return Path(tmp.name)

    def test_matching_hash_loads(self):
        digest = (ROOT / "LLM" / "rag_store" / "mitre_tfidf.pkl.sha256").read_text()
        self.assertIsNotNone(MitreService(self.make_root(digest)).index)

    def test_tampered_hash_refuses_to_unpickle(self):
        service = MitreService(self.make_root("0" * 64 + "  mitre_tfidf.pkl\n"))
        self.assertIsNone(service.index)
        self.assertIn("hash mismatch", service.index_error)
        self.assertEqual(service.retrieve_behavior({"anomaly_score": 1, "threshold": 1, "top_features": []})["status"], "retrieval_unavailable")

    def test_missing_sidecar_still_loads_with_a_warning(self):
        with self.assertLogs(level="WARNING"):
            self.assertIsNotNone(MitreService(self.make_root(None)).index)

    def test_committed_index_matches_its_recorded_hash(self):
        self.assertIsNotNone(MitreService(ROOT).index)


if __name__ == "__main__": unittest.main()
