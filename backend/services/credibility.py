"""Source-credibility heuristics for a domain.

A small curated list (wire services, major broadsheets, fact-checkers) plus
structural signals: suspicious TLDs, look-alike names of real outlets, blog
hosts, and a lookup against the domains seen labelled fake/real in
FakeNewsNet during training.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from backend.core import config

HIGH = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com", "washingtonpost.com",
    "theguardian.com", "wsj.com", "ft.com", "bloomberg.com", "economist.com", "npr.org",
    "pbs.org", "nature.com", "science.org", "who.int", "cdc.gov", "nasa.gov", "gov.uk",
    "europa.eu", "un.org", "politifact.com", "snopes.com", "factcheck.org", "fullfact.org",
    "afp.com", "aljazeera.com", "cbsnews.com", "abcnews.go.com", "nbcnews.com", "cnn.com",
    "thehindu.com", "indianexpress.com", "ndtv.com", "hindustantimes.com", "timesofindia.indiatimes.com",
    "altnews.in", "boomlive.in", "wikipedia.org", "theatlantic.com", "time.com", "forbes.com",
    "usatoday.com", "latimes.com", "cnbc.com", "axios.com", "propublica.org", "nytimes.com",
}
MEDIUM = {
    "foxnews.com", "nypost.com", "dailymail.co.uk", "huffpost.com", "buzzfeednews.com", "vox.com",
    "msnbc.com", "newsweek.com", "businessinsider.com", "thesun.co.uk", "mirror.co.uk",
    "express.co.uk", "people.com", "eonline.com", "tmz.com", "usmagazine.com", "etonline.com",
    "hollywoodlife.com", "medium.com", "substack.com", "yahoo.com", "msn.com",
}
LOW = {
    "infowars.com", "naturalnews.com", "beforeitsnews.com", "worldnewsdailyreport.com",
    "theonion.com", "empirenews.net", "yournewswire.com", "newspunch.com", "realrawnews.com",
    "clickhole.com", "babylonbee.com", "dailybuzzlive.com", "now8news.com", "react365.com",
    "thelastlineofdefense.org", "abcnews.com.co", "cnn-trending.com", "usatoday.com.co",
}
SUSPICIOUS_TLDS = (".co", ".info", ".biz", ".xyz", ".top", ".club", ".site", ".online", ".buzz",
                   ".icu", ".click", ".news", ".press")
BLOG_HOSTS = ("blogspot.", "wordpress.com", "weebly.com", "wixsite.com", "tumblr.com",
              "medium.com", "substack.com", "livejournal.com")
LOOKALIKE = ["cnn", "bbc", "abcnews", "nytimes", "foxnews", "usatoday", "reuters", "nbcnews", "cbsnews"]

_fnn_cache: dict | None = None


def _fnn_domains() -> dict:
    """Fake/real counts per domain from FakeNewsNet, built once from the
    processed dataset (falls back to an empty map if data is absent)."""
    global _fnn_cache
    if _fnn_cache is not None:
        return _fnn_cache
    path = Path(config.MODELS_DIR) / "domain_stats.json"
    if path.exists():
        _fnn_cache = json.loads(path.read_text())
        return _fnn_cache
    try:
        import pandas as pd

        df = pd.read_parquet(Path(config.DATA_DIR) / "processed" / "unified.parquet")
        df = df[df["domain"] != ""]
        g = df.groupby("domain")["label"].agg(["sum", "count"])
        g = g[g["count"] >= 3]
        _fnn_cache = {d: {"fake": int(r["sum"]), "total": int(r["count"])} for d, r in g.iterrows()}
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(_fnn_cache))
    except Exception:  # noqa: BLE001
        _fnn_cache = {}
    return _fnn_cache


def _match(domain: str, pool: set[str]) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in pool)


def assess(domain: str | None) -> dict:
    if not domain:
        return {"domain": None, "credibility": "unknown", "score": None,
                "notes": ["No source URL supplied — judged on content only."]}
    d = domain.lower()
    notes: list[str] = []
    score = 0.5

    if _match(d, HIGH):
        score, notes = 0.9, ["Established outlet / wire service / official body with editorial standards."]
    elif _match(d, LOW):
        score, notes = 0.1, ["Domain is known for satire, hoaxes or fabricated stories."]
    elif _match(d, MEDIUM):
        score, notes = 0.55, ["Mainstream outlet with mixed reliability or strong partisan / tabloid slant."]

    stats = _fnn_domains().get(d)
    if stats:
        rate = stats["fake"] / stats["total"]
        notes.append(f"In the FakeNewsNet training data {stats['fake']}/{stats['total']} "
                     f"articles from this domain were labelled fake.")
        score = 0.6 * score + 0.4 * (1 - rate)

    if d.endswith(SUSPICIOUS_TLDS) and not _match(d, HIGH):
        score -= 0.15
        notes.append("Unusual top-level domain often used by imitation news sites.")
    base = re.sub(r"\.(com|org|net|co|uk|in|info|news)$", "", d)
    for name in LOOKALIKE:
        if name in base and not _match(d, HIGH | MEDIUM):
            score -= 0.25
            notes.append(f"Name resembles a well-known outlet ('{name}') but is not its official domain.")
            break
    if any(h in d for h in BLOG_HOSTS):
        score -= 0.1
        notes.append("Hosted on a free blogging platform — no editorial oversight.")
    if not notes:
        notes.append("Domain not in our reference lists; treat with normal caution.")

    score = max(0.0, min(1.0, score))
    level = "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"
    return {"domain": d, "credibility": level, "score": round(score, 2), "notes": notes}
