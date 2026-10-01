"""Hand-crafted linguistic / stylistic features.

These capture the *style* of misinformation (sensational punctuation,
clickbait cues, shouting, hedging) independently of the topic, which makes
the ensemble more robust to topic shift between datasets.
"""
from __future__ import annotations

import re

import numpy as np

CLICKBAIT = [
    "breaking", "shocking", "you won't believe", "unbelievable", "exposed",
    "secret", "truth", "they don't want you", "must see", "viral", "bombshell",
    "destroyed", "slams", "wow", "insane", "miracle", "cure", "hoax", "exclusive",
    "revealed", "finally", "shock", "stunning", "outrage", "banned", "leaked",
]
HEDGES = ["reportedly", "allegedly", "sources say", "claims", "rumor", "rumour",
          "may", "might", "could", "supposedly", "anonymous"]
PRONOUNS_1_2 = {"i", "me", "my", "we", "our", "us", "you", "your"}
PRONOUNS_3 = {"he", "she", "they", "them", "his", "her", "their"}
NEGATIONS = {"no", "not", "never", "none", "nothing", "nobody", "n't"}
SUPERLATIVES = ["best", "worst", "most", "least", "greatest", "biggest", "ever", "always", "never"]
NUMBERS_RE = re.compile(r"\d+(?:[.,]\d+)?%?")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
QUOTE_RE = re.compile(r"[\"“”']")

FEATURE_NAMES = [
    "n_chars", "n_words", "avg_word_len", "caps_ratio", "all_caps_words",
    "exclam", "question", "ellipsis", "quotes", "numbers", "percent_signs",
    "urls", "clickbait_hits", "hedge_hits", "superlatives", "pron_1_2",
    "pron_3", "negations", "type_token_ratio", "starts_with_caps_word",
    "long_word_ratio", "digit_ratio", "punct_ratio",
]


def featurize_one(text: str) -> np.ndarray:
    t = text or ""
    low = t.lower()
    words = re.findall(r"[A-Za-z']+", t)
    n_words = max(len(words), 1)
    n_chars = max(len(t), 1)
    letters = [c for c in t if c.isalpha()]
    caps = sum(1 for c in letters if c.isupper())
    all_caps_words = sum(1 for w in words if len(w) > 2 and w.isupper())
    tokens_low = [w.lower() for w in words]
    uniq = len(set(tokens_low))
    first = words[0] if words else ""
    punct = sum(1 for c in t if not c.isalnum() and not c.isspace())

    return np.array([
        n_chars,
        len(words),
        sum(len(w) for w in words) / n_words,
        caps / max(len(letters), 1),
        all_caps_words / n_words,
        t.count("!"),
        t.count("?"),
        t.count("..."),
        len(QUOTE_RE.findall(t)),
        len(NUMBERS_RE.findall(t)),
        t.count("%"),
        len(URL_RE.findall(t)),
        sum(low.count(c) for c in CLICKBAIT),
        sum(low.count(h) for h in HEDGES),
        sum(1 for w in tokens_low if w in SUPERLATIVES),
        sum(1 for w in tokens_low if w in PRONOUNS_1_2) / n_words,
        sum(1 for w in tokens_low if w in PRONOUNS_3) / n_words,
        sum(1 for w in tokens_low if w in NEGATIONS) / n_words,
        uniq / n_words,
        1.0 if first.isupper() and len(first) > 1 else 0.0,
        sum(1 for w in words if len(w) >= 8) / n_words,
        sum(c.isdigit() for c in t) / n_chars,
        punct / n_chars,
    ], dtype=np.float32)


def featurize(texts) -> np.ndarray:
    return np.vstack([featurize_one(t) for t in texts])


def explain_style(text: str) -> list[str]:
    """Human-readable stylistic red flags for the UI."""
    f = dict(zip(FEATURE_NAMES, featurize_one(text)))
    flags = []
    if f["clickbait_hits"] >= 1:
        flags.append("Contains clickbait / sensational wording")
    if f["caps_ratio"] > 0.3 and f["n_words"] > 3:
        flags.append("Excessive capitalisation (shouting)")
    if f["exclam"] >= 2:
        flags.append("Multiple exclamation marks")
    if f["hedge_hits"] >= 2:
        flags.append("Heavy use of hedging / unnamed sources")
    if f["superlatives"] >= 2:
        flags.append("Loaded superlatives")
    if f["numbers"] == 0 and f["n_words"] > 25:
        flags.append("No concrete figures for a long claim")
    return flags
