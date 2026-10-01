"""LangGraph multi-agent pipeline.

    ingest -> classifier -> claim_extractor -> evidence_retriever
           -> fact_checker -> source_credibility -> judge

* `fast` mode runs ingest -> classifier -> source_credibility -> judge only.
* `deep` mode adds the LLM agents; if the LLM is unreachable the graph routes
  around them and the judge explains the degraded verdict.

Every node appends an AgentTrace and, when a callback is supplied, emits
step events so the API can stream progress to the UI.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, StateGraph

from backend.core import config
from backend.ml.predictor import get_predictor
from backend.services import credibility, evidence, ingest, llm

log = logging.getLogger(__name__)
StepCallback = Callable[[dict[str, Any]], None]


class AgentTrace(TypedDict):
    name: str
    status: Literal["done", "error", "skipped"]
    elapsed_ms: int
    summary: str


class State(TypedDict, total=False):
    # inputs
    raw_text: str | None
    url: str | None
    mode: Literal["fast", "deep"]
    llm_ok: bool
    emit: StepCallback | None
    # produced
    text: str
    title: str | None
    domain: str | None
    ml: dict
    claims: list[dict]
    source: dict
    llm_out: dict | None
    verdict: str
    fake_probability: float
    confidence: float
    agents: list[AgentTrace]
    error: str | None


# --------------------------------------------------------------- utilities
def _emit(state: State, agent: str, status: str, summary: str, elapsed: float | None = None,
          payload: Any = None):
    cb = state.get("emit")
    if cb:
        cb({"agent": agent, "status": status, "summary": summary,
            "elapsed_ms": None if elapsed is None else int(elapsed * 1000), "payload": payload})


def _node(name: str):
    """Decorator: timing, tracing, streaming and error containment for a node."""
    def wrap(fn):
        def run(state: State) -> dict:
            _emit(state, name, "running", f"{name} started")
            t0 = time.time()
            try:
                out = fn(state)
                summary = out.pop("_summary", f"{name} done")
                status = "done"
            except Exception as exc:  # noqa: BLE001 - one agent failing must not kill the run
                log.exception("agent %s failed", name)
                out = {"_err": str(exc)}
                summary = f"{name} failed: {exc}"[:300]
                status = "error"
            el = time.time() - t0
            trace: AgentTrace = {"name": name, "status": status, "elapsed_ms": int(el * 1000),
                                 "summary": summary}
            payload = out.pop("_payload", None)
            err = out.pop("_err", None)
            out["agents"] = state.get("agents", []) + [trace]
            if err and name == "ingest":
                out["error"] = err
            _emit(state, name, status, summary, el, payload)
            return out
        return run
    return wrap


def _trunc(text: str, n: int = 3500) -> str:
    return text if len(text) <= n else text[:n] + " …"


# ------------------------------------------------------------------- nodes
@_node("ingest")
def ingest_node(state: State) -> dict:
    if state.get("url"):
        art = ingest.fetch_article(state["url"])
        return {"text": art.text, "title": art.title, "domain": art.domain,
                "_summary": f"Fetched {art.domain} — {len(art.text.split())} words",
                "_payload": {"title": art.title, "domain": art.domain}}
    text = (state.get("raw_text") or "").strip()
    if not text:
        raise ValueError("no text or URL supplied")
    return {"text": text, "title": None, "domain": None,
            "_summary": f"Received {len(text.split())} words of text"}


@_node("classifier")
def classifier_node(state: State) -> dict:
    pred = get_predictor().predict(state["text"])
    p = pred["ensemble_prob"]
    return {"ml": pred,
            "_summary": f"Ensemble fake-probability {p:.0%} across {len(pred['models'])} model(s)",
            "_payload": {"ensemble_prob": p, "models": pred["models"]}}


CLAIM_SYS = ("You are a fact-checking assistant. Extract the most important, concrete, "
             "checkable factual claims from the article. Skip opinions and vague statements. "
             "Keep each claim self-contained (include names, numbers, dates).")
CLAIM_SCHEMA = '{"claims": ["claim 1", "claim 2"], "summary": "one-sentence summary of the article"}'


@_node("claim_extractor")
def claim_node(state: State) -> dict:
    out = llm.chat_json(CLAIM_SYS, f"ARTICLE:\n{_trunc(state['text'])}\n\nReturn at most "
                        f"{config.MAX_CLAIMS} claims.", CLAIM_SCHEMA)
    claims = [str(c).strip() for c in out.get("claims", []) if str(c).strip()][:config.MAX_CLAIMS]
    if not claims:
        claims = [state["text"][:200]]
    return {"claims": [{"claim": c, "verdict": "unverified", "confidence": 0.0, "evidence": []}
                       for c in claims],
            "llm_out": {"summary": str(out.get("summary", ""))},
            "_summary": f"Extracted {len(claims)} checkable claim(s)",
            "_payload": {"claims": claims}}


@_node("evidence_retriever")
def evidence_node(state: State) -> dict:
    claims = [dict(c) for c in state.get("claims", [])]
    total = 0
    for c in claims:
        ev = evidence.search_evidence(c["claim"])
        c["evidence"] = [{"title": e["title"], "url": e["url"], "snippet": e["snippet"],
                          "stance": "neutral", "is_fact_checker": e.get("is_fact_checker", False)}
                         for e in ev]
        total += len(ev)
    fc = sum(1 for c in claims for e in c["evidence"] if e.get("is_fact_checker"))
    return {"claims": claims,
            "_summary": f"Found {total} evidence snippet(s), {fc} from fact-checkers",
            "_payload": {"evidence_count": total, "fact_checker_hits": fc}}


FACT_SYS = ("You are a rigorous fact-checker. For each claim, compare it against the evidence "
            "snippets. Decide 'supported' only if evidence clearly confirms it, 'refuted' if evidence "
            "clearly contradicts it or a fact-checker rates it false, otherwise 'unverified'. "
            "Label each evidence item's stance. Then assess the article overall: list concrete "
            "misinformation red flags (fabricated quotes, missing sources, sensational framing, "
            "satire, outdated events presented as new) and give a probability that the article is "
            "fake. Be calibrated: lack of evidence is not proof of falsehood.")
FACT_SCHEMA = ('{"claims": [{"claim": "...", "verdict": "supported|refuted|unverified", '
               '"confidence": 0.0, "evidence_stances": ["supports|refutes|neutral", ...]}], '
               '"red_flags": ["..."], "fake_probability": 0.0, "reasoning": "2-4 sentences"}')


@_node("fact_checker")
def fact_node(state: State) -> dict:
    claims = [dict(c) for c in state.get("claims", [])]
    packet = [{"claim": c["claim"],
               "evidence": [{"i": i, "source": e["url"], "title": e["title"], "snippet": e["snippet"]}
                            for i, e in enumerate(c["evidence"])]} for c in claims]
    user = (f"ARTICLE (truncated):\n{_trunc(state['text'], 2500)}\n\n"
            f"CLAIMS WITH EVIDENCE:\n{json.dumps(packet, ensure_ascii=False)}")
    out = llm.chat_json(FACT_SYS, user, FACT_SCHEMA)
    by_claim = {str(c.get("claim", "")).strip(): c for c in out.get("claims", [])}
    for idx, c in enumerate(claims):
        r = by_claim.get(c["claim"]) or (out.get("claims") or [{}])[idx] if idx < len(out.get("claims") or []) else {}
        v = str(r.get("verdict", "unverified")).lower()
        c["verdict"] = v if v in {"supported", "refuted", "unverified"} else "unverified"
        try:
            c["confidence"] = max(0.0, min(1.0, float(r.get("confidence", 0.5))))
        except (TypeError, ValueError):
            c["confidence"] = 0.5
        stances = r.get("evidence_stances") or []
        for i, e in enumerate(c["evidence"]):
            s = str(stances[i]).lower() if i < len(stances) else "neutral"
            e["stance"] = s if s in {"supports", "refutes", "neutral"} else "neutral"
    try:
        p = max(0.0, min(1.0, float(out.get("fake_probability", 0.5))))
    except (TypeError, ValueError):
        p = 0.5
    prev = state.get("llm_out") or {}
    llm_out = {**prev, **llm.describe(), "fake_probability": p,
               "red_flags": [str(f) for f in out.get("red_flags", [])][:8],
               "reasoning": str(out.get("reasoning", ""))}
    refuted = sum(c["verdict"] == "refuted" for c in claims)
    supported = sum(c["verdict"] == "supported" for c in claims)
    return {"claims": claims, "llm_out": llm_out,
            "_summary": f"{supported} supported, {refuted} refuted, "
                        f"{len(claims) - supported - refuted} unverified; LLM fake-prob {p:.0%}",
            "_payload": {"fake_probability": p, "red_flags": llm_out["red_flags"]}}


@_node("source_credibility")
def source_node(state: State) -> dict:
    src = credibility.assess(state.get("domain"))
    return {"source": src, "_summary": f"Source credibility: {src['credibility']}"
            + (f" ({src['score']:.2f})" if src["score"] is not None else ""),
            "_payload": src}


def _fuse(state: State) -> tuple[float, float, str, list[str]]:
    """Weighted logit fusion of ML ensemble, LLM judgement, claim verdicts and
    source credibility. Returns (fake_prob, confidence, verdict, notes)."""
    import math

    def logit(p):
        p = min(max(p, 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))

    notes: list[str] = []
    parts: list[tuple[float, float]] = [(logit(state["ml"]["ensemble_prob"]), 1.0)]
    llm_out = state.get("llm_out")
    if llm_out and "fake_probability" in llm_out:
        parts.append((logit(llm_out["fake_probability"]), 1.2))
        notes.append("LLM fact-check weighted 1.2x")
    claims = state.get("claims", [])
    refuted = [c for c in claims if c["verdict"] == "refuted"]
    supported = [c for c in claims if c["verdict"] == "supported"]
    if claims:
        if refuted:
            parts.append((2.0 * min(len(refuted), 2), 1.0))
            notes.append(f"{len(refuted)} claim(s) refuted by evidence")
        if supported and not refuted:
            parts.append((-1.2 * min(len(supported), 2), 1.0))
            notes.append(f"{len(supported)} claim(s) supported by evidence")
    src = state.get("source") or {}
    if src.get("score") is not None:
        parts.append(((0.5 - src["score"]) * 3.0, 0.8))
        notes.append(f"source credibility {src['credibility']}")
    z = sum(l * w for l, w in parts) / sum(w for _, w in parts)
    p = 1 / (1 + math.exp(-z))

    # Confidence: distance from 0.5 plus agreement between signals.
    signals = [state["ml"]["ensemble_prob"]] + ([llm_out["fake_probability"]] if llm_out and "fake_probability" in llm_out else [])
    agree = 1.0 - (max(signals) - min(signals)) if len(signals) > 1 else 0.7
    conf = min(1.0, 0.55 * abs(p - 0.5) * 2 + 0.45 * agree)
    if refuted:
        conf = max(conf, 0.75)
    verdict = "fake" if p >= 0.62 else "real" if p <= 0.38 else "uncertain"
    if verdict == "uncertain":
        conf = min(conf, 0.6)
    return p, conf, verdict, notes


@_node("judge")
def judge_node(state: State) -> dict:
    p, conf, verdict, notes = _fuse(state)
    degraded = state["mode"] == "deep" and not state.get("llm_ok", True)
    if degraded:
        notes.append("LLM unavailable — verdict from ML ensemble and source signals only")
    return {"fake_probability": p, "confidence": conf, "verdict": verdict,
            "_summary": f"Verdict {verdict.upper()} (p_fake={p:.0%}, confidence {conf:.0%})"
                        + (f" — {'; '.join(notes)}" if notes else ""),
            "_payload": {"verdict": verdict, "fake_probability": p, "confidence": conf}}


# -------------------------------------------------------------------- graph
def _route_after_ingest(state: State) -> str:
    return "end" if state.get("error") else "classifier"


def _route_after_classifier(state: State) -> str:
    if state["mode"] == "deep" and state.get("llm_ok", False):
        return "claim_extractor"
    return "source_credibility"


def _route_after_claims(state: State) -> str:
    # If claim extraction errored we still try evidence on the raw text claim.
    return "evidence_retriever" if state.get("claims") else "source_credibility"


def build_graph():
    g = StateGraph(State)
    g.add_node("ingest", ingest_node)
    g.add_node("classifier", classifier_node)
    g.add_node("claim_extractor", claim_node)
    g.add_node("evidence_retriever", evidence_node)
    g.add_node("fact_checker", fact_node)
    g.add_node("source_credibility", source_node)
    g.add_node("judge", judge_node)

    g.set_entry_point("ingest")
    g.add_conditional_edges("ingest", _route_after_ingest, {"classifier": "classifier", "end": END})
    g.add_conditional_edges("classifier", _route_after_classifier,
                            {"claim_extractor": "claim_extractor",
                             "source_credibility": "source_credibility"})
    g.add_conditional_edges("claim_extractor", _route_after_claims,
                            {"evidence_retriever": "evidence_retriever",
                             "source_credibility": "source_credibility"})
    g.add_edge("evidence_retriever", "fact_checker")
    g.add_edge("fact_checker", "source_credibility")
    g.add_edge("source_credibility", "judge")
    g.add_edge("judge", END)
    return g.compile()


GRAPH = build_graph()


def run_pipeline(text: str | None = None, url: str | None = None, mode: str = "deep",
                 emit: StepCallback | None = None) -> State:
    llm_ok = llm.available() if mode == "deep" else False
    if mode == "deep" and not llm_ok:
        _emit({"emit": emit}, "claim_extractor", "error", "LLM unavailable — skipping LLM agents")
    state: State = {"raw_text": text, "url": url, "mode": mode, "llm_ok": llm_ok,  # type: ignore[typeddict-item]
                    "emit": emit, "agents": []}
    result = GRAPH.invoke(state)
    if result.get("error"):
        raise ValueError(result["error"])
    return result
