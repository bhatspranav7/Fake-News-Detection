"""MCP server exposing VeriFact to Claude (Desktop / Code) and other MCP clients.

    pip install "mcp[cli]" requests
    claude mcp add verifact -- python -m mcp_server.server
    # or point at a deployed API:
    VERIFACT_API_URL=https://verifact-api.onrender.com python -m mcp_server.server

Claude Desktop config (claude_desktop_config.json):
    {"mcpServers": {"verifact": {"command": "python", "args": ["-m", "mcp_server.server"],
                    "cwd": "<repo>", "env": {"VERIFACT_API_URL": "http://127.0.0.1:8000"}}}}
"""
from __future__ import annotations

import os

import requests
from mcp.server.fastmcp import FastMCP

API = os.getenv("VERIFACT_API_URL", "http://127.0.0.1:8000").rstrip("/")
mcp = FastMCP("verifact", instructions="Fact-check news text or URLs with the VeriFact agentic pipeline.")


def _fmt(r: dict) -> str:
    lines = [f"Verdict: {r['verdict'].upper()}  (fake probability {r['fake_probability']:.0%}, "
             f"confidence {r['confidence']:.0%})"]
    ml = r.get("ml", {})
    if ml.get("models"):
        lines.append("ML models: " + ", ".join(f"{k}={v:.0%}" for k, v in ml["models"].items()))
    if ml.get("style_flags"):
        lines.append("Style flags: " + "; ".join(ml["style_flags"]))
    src = r.get("source") or {}
    if src.get("domain"):
        lines.append(f"Source {src['domain']}: {src['credibility']} credibility — {' '.join(src.get('notes', []))}")
    for c in r.get("claims", []):
        lines.append(f"- Claim [{c['verdict']}]: {c['claim']}")
        for e in c.get("evidence", [])[:3]:
            lines.append(f"    · ({e['stance']}) {e['title']} — {e['url']}")
    if r.get("llm"):
        lines.append(f"LLM red flags: {', '.join(r['llm'].get('red_flags', [])) or 'none'}")
        lines.append(f"LLM reasoning: {r['llm'].get('reasoning', '')}")
    lines.append(f"Analysis id: {r['id']}  ({r['latency_ms']} ms)")
    return "\n".join(lines)


@mcp.tool()
def check_news(text: str = "", url: str = "", mode: str = "deep") -> str:
    """Analyse a news article / headline (give `text`) or a web page (`url`).
    mode='fast' = ML ensemble only (~1s); 'deep' = + LLM claim extraction,
    web evidence retrieval and fact-checking (slower)."""
    body = {"mode": mode}
    if url:
        body["url"] = url
    else:
        body["text"] = text
    r = requests.post(f"{API}/analyze", json=body, timeout=180)
    if not r.ok:
        return f"VeriFact error {r.status_code}: {r.text[:300]}"
    return _fmt(r.json())


@mcp.tool()
def score_headlines(headlines: list[str]) -> str:
    """Quickly score many headlines with the ML ensemble (no LLM). Returns one line per headline."""
    r = requests.post(f"{API}/analyze/batch", json={"texts": headlines}, timeout=120)
    r.raise_for_status()
    return "\n".join(f"{x['fake_probability']:.0%} {x['verdict']:9s} {x['text']}" for x in r.json()["results"])


@mcp.tool()
def model_metrics() -> str:
    """Held-out test metrics of the trained models (LIAR + FakeNewsNet)."""
    r = requests.get(f"{API}/metrics/models", timeout=30)
    r.raise_for_status()
    m = r.json()
    out = []
    for name, v in list(m["models"].items()) + [("ensemble", m["ensemble"])]:
        out.append(f"{name:12s} acc={v['accuracy']:.3f} f1={v['f1']:.3f} auc={v['roc_auc']:.3f} n={v['n']}")
    return "\n".join(out)


@mcp.tool()
def recent_analyses(limit: int = 10) -> str:
    """List recent analyses stored by the API."""
    r = requests.get(f"{API}/history", params={"limit": limit}, timeout=30)
    r.raise_for_status()
    return "\n".join(f"{i['created_at'][:19]} {i['verdict']:9s} {i['fake_probability']:.0%} {i['snippet'][:80]}"
                     for i in r.json()["items"]) or "no analyses yet"


if __name__ == "__main__":
    mcp.run()
