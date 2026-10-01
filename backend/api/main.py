"""VeriFact API — agentic fake-news detection."""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.agents.graph import run_pipeline
from backend.api.schemas import AnalyzeRequest, BatchRequest, FeedbackRequest
from backend.core import config
from backend.ml.predictor import get_predictor
from backend.services import llm, store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("verifact")

EXAMPLES = [
    {"label": "Viral health claim",
     "text": "BREAKING: Scientists confirm that drinking lemon water every morning cures cancer "
             "within 30 days. Doctors are FURIOUS and don't want you to know this!"},
    {"label": "Political statement",
     "text": "The unemployment rate fell to 3.7 percent last month, the lowest level in half a century, "
             "according to the Bureau of Labor Statistics."},
    {"label": "Celebrity gossip",
     "text": "Taylor Swift secretly married in a private ceremony last weekend, sources close to the "
             "singer reveal — and you won't believe who walked her down the aisle."},
    {"label": "Science news",
     "text": "NASA's James Webb Space Telescope has detected carbon dioxide in the atmosphere of an "
             "exoplanet for the first time, the agency announced on Thursday."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init()
    # Load models off the event loop so the port binds immediately.
    threading.Thread(target=lambda: get_predictor().load(), daemon=True).start()
    yield


app = FastAPI(title=config.APP_NAME, version=config.VERSION, lifespan=lifespan,
              description="Agentic fake-news detection: deep-learning ensemble + LLM fact-checking agents.")
app.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS, allow_methods=["*"],
                   allow_headers=["*"])


def require_key(x_api_key: str | None = Header(default=None)):
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(401, "invalid API key")


# ----------------------------------------------------------------- helpers
def _package(state: dict, mode: str, started: float, url: str | None) -> dict:
    text = state["text"]
    llm_out = state.get("llm_out")
    return {
        "id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "input": {"text": text[:5000], "url": url, "title": state.get("title"),
                  "domain": state.get("domain"), "word_count": len(text.split())},
        "verdict": state["verdict"],
        "fake_probability": round(state["fake_probability"], 4),
        "confidence": round(state["confidence"], 4),
        "ml": state["ml"],
        "claims": [{k: v for k, v in c.items()} for c in state.get("claims", [])],
        "source": state["source"],
        "llm": ({"provider": llm_out.get("provider", ""), "model": llm_out.get("model", ""),
                 "summary": llm_out.get("summary", ""), "red_flags": llm_out.get("red_flags", []),
                 "reasoning": llm_out.get("reasoning", "")}
                if llm_out and "fake_probability" in llm_out else None),
        "agents": state.get("agents", []),
        "latency_ms": int((time.time() - started) * 1000),
    }


def _analyze_sync(req: AnalyzeRequest, emit=None) -> dict:
    started = time.time()
    try:
        state = run_pipeline(text=req.text, url=req.url, mode=req.mode, emit=emit)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result = _package(state, req.mode, started, req.url)
    store.save_analysis(result)
    return result


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    pred = get_predictor()
    return {"status": "ok", "version": config.VERSION, "models_loaded": pred.loaded_models(),
            "models_ready": pred._loaded,
            "llm": {**llm.describe(), "ok": llm.available()},
            "index": {"analyses": store.usage()["analyses"]}}


@app.get("/examples")
def examples():
    return {"examples": EXAMPLES}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    return await asyncio.to_thread(_analyze_sync, req)


@app.post("/analyze/stream")
async def analyze_stream(req: AnalyzeRequest):
    """Server-sent events: one `step` event per agent transition, then `result`."""
    q: queue.Queue = queue.Queue()

    def worker():
        try:
            result = _analyze_sync(req, emit=lambda ev: q.put(("step", ev)))
            q.put(("result", result))
        except HTTPException as exc:
            q.put(("error", {"detail": exc.detail}))
        except Exception as exc:  # noqa: BLE001
            log.exception("stream failed")
            q.put(("error", {"detail": str(exc)}))
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        while True:
            item = await asyncio.to_thread(q.get)
            if item is None:
                break
            event, data = item
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/analyze/batch")
async def analyze_batch(req: BatchRequest):
    """ML-only scoring for many texts (used by the MCP server and evaluations)."""
    preds = await asyncio.to_thread(get_predictor().predict_batch, req.texts)
    return {"results": [{"text": t[:200], "fake_probability": p["ensemble_prob"],
                         "verdict": "fake" if p["ensemble_prob"] >= 0.62 else "real" if p["ensemble_prob"] <= 0.38 else "uncertain",
                         "models": p["models"]} for t, p in zip(req.texts, preds)]}


@app.get("/analysis/{analysis_id}")
def get_analysis(analysis_id: str):
    item = store.get_analysis(analysis_id)
    if not item:
        raise HTTPException(404, "analysis not found")
    return item


@app.get("/history")
def history(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": store.history(limit)}


@app.post("/feedback", dependencies=[Depends(require_key)])
def feedback(req: FeedbackRequest):
    if not store.get_analysis(req.analysis_id):
        raise HTTPException(404, "analysis not found")
    store.add_feedback(req.analysis_id, req.rating, req.comment)
    return {"ok": True}


@app.get("/metrics/models")
def metrics_models():
    path = Path(config.MODELS_DIR) / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "models not trained yet")
    return json.loads(path.read_text())


@app.get("/metrics/usage")
def metrics_usage():
    return store.usage()
