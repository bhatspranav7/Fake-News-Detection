"""Evidence retrieval: DuckDuckGo web search (no API key) + Wikipedia summaries.

Fact-checking sites are boosted so PolitiFact / Snopes / Reuters Fact Check
results surface first when they exist for a claim.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from backend.core import config
from backend.services.ingest import domain_of

log = logging.getLogger(__name__)

FACT_CHECKERS = ["politifact.com", "snopes.com", "factcheck.org", "apnews.com", "reuters.com",
                 "fullfact.org", "afp.com", "boomlive.in", "altnews.in", "leadstories.com",
                 "checkyourfact.com", "usatoday.com", "bbc.com", "bbc.co.uk"]


def _ddg(query: str, k: int) -> list[dict]:
    try:
        from ddgs import DDGS

        with DDGS(timeout=config.FETCH_TIMEOUT) as d:
            rows = list(d.text(query, max_results=k, safesearch="moderate"))
    except Exception as exc:  # noqa: BLE001
        log.warning("ddg search failed: %s", exc)
        return []
    out = []
    for r in rows:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        out.append({"title": r.get("title", "")[:200], "url": url,
                    "snippet": re.sub(r"\s+", " ", r.get("body", ""))[:400],
                    "domain": domain_of(url)})
    return out


def _wikipedia(query: str) -> list[dict]:
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action": "query", "list": "search", "srsearch": query,
                                 "srlimit": 2, "format": "json"},
                         headers={"User-Agent": "VeriFact/1.0 (fake-news research)"},
                         timeout=config.FETCH_TIMEOUT)
        hits = r.json().get("query", {}).get("search", [])
        out = []
        for h in hits:
            title = h["title"]
            s = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/" +
                             requests.utils.quote(title.replace(" ", "_")),
                             headers={"User-Agent": "VeriFact/1.0"}, timeout=config.FETCH_TIMEOUT)
            if not s.ok:
                continue
            j = s.json()
            out.append({"title": f"Wikipedia: {j.get('title', title)}",
                        "url": j.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        "snippet": (j.get("extract") or "")[:400], "domain": "wikipedia.org"})
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("wikipedia failed: %s", exc)
        return []


def search_evidence(claim: str, k: int | None = None) -> list[dict]:
    """Return up to k de-duplicated evidence snippets, fact-checkers first."""
    k = k or config.EVIDENCE_PER_CLAIM
    if not config.WEB_SEARCH_ENABLED:
        return []
    short = " ".join(claim.split()[:24])
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_general = ex.submit(_ddg, short, k + 2)
        f_fact = ex.submit(_ddg, f"{short} fact check", 3)
        f_wiki = ex.submit(_wikipedia, short)
        general, fact, wiki = f_general.result(), f_fact.result(), f_wiki.result()

    seen, out = set(), []
    for item in fact + general + wiki:
        key = item["url"].split("#")[0].rstrip("/")
        if not key or key in seen:
            continue
        seen.add(key)
        item["is_fact_checker"] = any(item["domain"].endswith(fc) for fc in FACT_CHECKERS)
        out.append(item)
    out.sort(key=lambda x: (not x["is_fact_checker"], x["domain"] != "wikipedia.org"))
    return out[:k]
