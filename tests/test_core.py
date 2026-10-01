"""Fast unit tests — no network, no LLM. `pytest -q`."""
from __future__ import annotations

import numpy as np
import pytest

from backend.ml.features import FEATURE_NAMES, explain_style, featurize, featurize_one
from backend.services import credibility
from backend.services.ingest import domain_of


def test_feature_vector_shape_and_names():
    v = featurize_one("BREAKING!!! You won't believe this SHOCKING secret.")
    assert v.shape == (len(FEATURE_NAMES),)
    f = dict(zip(FEATURE_NAMES, v))
    assert f["exclam"] == 3
    assert f["clickbait_hits"] >= 2
    assert 0 < f["caps_ratio"] <= 1


def test_featurize_batch():
    X = featurize(["one two", "three"])
    assert X.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(X).all()


def test_style_flags_fire_on_sensational_text():
    flags = explain_style("SHOCKING: Doctors FURIOUS as miracle cure EXPOSED!! Must see!!")
    assert any("clickbait" in f.lower() for f in flags)
    assert any("exclamation" in f.lower() for f in flags)
    assert explain_style("The council approved the budget on Tuesday.") == []


@pytest.mark.parametrize("url,expected", [
    ("https://www.reuters.com/world/x", "reuters.com"),
    ("http://cnn-trending.com/a", "cnn-trending.com"),
    ("https://user@sub.example.co.uk:8080/p", "sub.example.co.uk"),
])
def test_domain_of(url, expected):
    assert domain_of(url) == expected


def test_credibility_levels():
    assert credibility.assess("reuters.com")["credibility"] == "high"
    assert credibility.assess("worldnewsdailyreport.com")["credibility"] == "low"
    look = credibility.assess("abcnews.com.co")
    assert look["credibility"] == "low"
    assert any("resembles" in n or "satire" in n for n in look["notes"])
    assert credibility.assess(None)["credibility"] == "unknown"
    assert credibility.assess("some-random-site.org")["credibility"] in {"medium", "low", "high"}


def test_judge_fusion_monotonic():
    from backend.agents.graph import _fuse

    base = {"ml": {"ensemble_prob": 0.5}, "claims": [], "source": {"score": None}, "mode": "fast"}
    p_mid, _, v_mid, _ = _fuse(base)
    assert v_mid == "uncertain" and abs(p_mid - 0.5) < 0.05
    fake = {**base, "ml": {"ensemble_prob": 0.9},
            "claims": [{"verdict": "refuted"}], "source": {"score": 0.1, "credibility": "low"}}
    p_fake, conf, v_fake, _ = _fuse(fake)
    assert v_fake == "fake" and p_fake > 0.85 and conf >= 0.75
    real = {**base, "ml": {"ensemble_prob": 0.1},
            "claims": [{"verdict": "supported"}], "source": {"score": 0.9, "credibility": "high"}}
    p_real, _, v_real, _ = _fuse(real)
    assert v_real == "real" and p_real < 0.2


def test_llm_json_extraction():
    from backend.services.llm import _extract_json

    assert _extract_json('```json\n{"a": 1}\n```')["a"] == 1
    assert _extract_json('Sure! Here it is: {"claims": ["x"]} hope that helps')["claims"] == ["x"]
    with pytest.raises(ValueError):
        _extract_json("no json here")


def test_ollama_host_normalisation():
    from backend.core.config import _ollama_url

    assert _ollama_url("0.0.0.0") == "http://127.0.0.1:11434"
    assert _ollama_url("0.0.0.0:11434") == "http://127.0.0.1:11434"
    assert _ollama_url("http://gpu-box:11434/") == "http://gpu-box:11434"
    assert _ollama_url("") == "http://127.0.0.1:11434"
