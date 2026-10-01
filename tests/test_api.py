"""API tests in fast mode (ML only). Skipped automatically if models aren't trained."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LLM_PROVIDER", "none")
os.environ.setdefault("WEB_SEARCH_ENABLED", "false")
os.environ["DB_PATH"] = str(Path(__file__).parent / "_test.db")

MODELS = Path(__file__).resolve().parents[1] / "models"
pytestmark = pytest.mark.skipif(not (MODELS / "tfidf_lr.joblib").exists(), reason="models not trained")


@pytest.fixture(scope="module")
def client():
    from backend.api.main import app

    with TestClient(app) as c:
        yield c
    db = Path(os.environ["DB_PATH"])
    if db.exists():
        db.unlink()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "llm" in body and "models_loaded" in body


def test_validation(client):
    assert client.post("/analyze", json={}).status_code == 422
    assert client.post("/analyze", json={"text": "   "}).status_code == 422


def test_fast_analyze_and_persistence(client):
    r = client.post("/analyze", json={"text": "BREAKING: miracle cure EXPOSED, doctors furious!!",
                                      "mode": "fast"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] in {"fake", "real", "uncertain"}
    assert 0 <= body["fake_probability"] <= 1
    assert body["ml"]["models"]
    assert body["llm"] is None
    names = [a["name"] for a in body["agents"]]
    assert names[:2] == ["ingest", "classifier"] and names[-1] == "judge"

    got = client.get(f"/analysis/{body['id']}")
    assert got.status_code == 200 and got.json()["id"] == body["id"]
    hist = client.get("/history").json()["items"]
    assert hist and hist[0]["id"] == body["id"]

    fb = client.post("/feedback", json={"analysis_id": body["id"], "rating": "correct"})
    assert fb.status_code == 200
    usage = client.get("/metrics/usage").json()
    assert usage["analyses"] >= 1 and usage["feedback"]["correct"] >= 1


def test_batch(client):
    r = client.post("/analyze/batch", json={"texts": ["a plain sentence", "SHOCKING!!! secret cure"]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_stream_emits_steps_and_result(client):
    with client.stream("POST", "/analyze/stream",
                       json={"text": "The city council met on Tuesday.", "mode": "fast"}) as r:
        assert r.status_code == 200
        raw = "".join(r.iter_text())
    assert "event: step" in raw and "event: result" in raw
    assert '"agent": "judge"' in raw
