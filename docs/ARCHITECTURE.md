# VeriFact — Architecture

## 1. Problem framing

Misinformation detection is not one problem but three stacked ones:

1. **Does this *read* like misinformation?** — style, framing, sensationalism. A supervised
   classifier trained on labelled corpora learns this well and runs in milliseconds.
2. **Is what it *says* actually true?** — needs the claims, outside evidence and reasoning.
   Only an LLM with retrieval can do this, and only slowly.
3. **Who is saying it?** — source reputation is a strong prior that no text model sees.

VeriFact answers all three and fuses them, instead of pretending a single text classifier can
"detect fake news".

## 2. Data

| Dataset | Rows used | Labels | What it contributes |
|---|---|---|---|
| **LIAR** (Wang, ACL 2017) | 12,810 PolitiFact statements, official train/valid/test | 6-way → binarised (`pants-fire/false/barely-true` = fake) | Short political claims — the hard, "sounds plausible" end of the spectrum |
| **FakeNewsNet / PolitiFact** (Shu et al., 2018) | 983 headlines + source URL | fake / real | Political news with domain information |
| **FakeNewsNet / GossipCop** | 20,741 headlines + source URL | fake / real | Entertainment news; clickbait style; most of the domain-reputation signal |

All three are merged into one schema (`text, label, dataset, domain, split`), de-duplicated
on text, and FakeNewsNet is split 80/10/10 stratified per source. LIAR keeps its official
splits so numbers remain comparable with the literature. The domain column also yields a
per-domain fake-rate table that the source-credibility agent consults at inference.

## 3. Classifier ensemble

```
text ─┬─> TF-IDF (word 1-2g + char 2-5g) ──> Logistic Regression ──┐
      ├─> MiniLM-L6 sentence embedding ⊕ 23 style features ─> XGBoost ─┤─> logit stack ─> p_ml
      └─> DistilBERT fine-tuned (2 ep, max_len 64) ─> ONNX int8 ───────┘      (LogReg meta-learner
                                                                               + 8 style features)
```

* **TF-IDF + LR** — strong lexical baseline; its coefficients give exact per-token attributions
  (`coef × tfidf`) that the UI shows as red/green chips.
* **MiniLM + style → XGBoost** — "LLM-based features": a transformer sentence encoder gives a
  semantic vector; 23 hand-crafted stylometric features (caps ratio, clickbait cue hits,
  hedges, superlatives, punctuation…) give topic-independent style signal.
* **DistilBERT** — end-to-end fine-tuned deep model; exported to ONNX and dynamically
  quantised to int8 (≈65 MB, ~4× faster on CPU) so it fits a free-tier container.
* **Stacker** — a logistic regression over the three logits + style features, fit on the
  validation split (never on train), so the weights reflect out-of-sample reliability.

Protocol: base models fit on `train`, stacker on `valid`, every reported number on `test`.
See `models/metrics.json` and the Dashboard page.

## 4. Agentic pipeline (LangGraph)

```
ingest ─> classifier ─┬─(deep & LLM up)─> claim_extractor ─> evidence_retriever ─> fact_checker ─┐
                      └─(fast / LLM down)───────────────────────────────────────────────────────┤
                                                                     source_credibility <────────┘
                                                                            └─> judge
```

| Agent | Tooling | Output |
|---|---|---|
| `ingest` | trafilatura, SSRF-guarded fetch | clean text, title, domain |
| `classifier` | ensemble above | p_ml, per-model probs, token attributions, style flags |
| `claim_extractor` | LLM (JSON mode) | ≤3 self-contained checkable claims + summary |
| `evidence_retriever` | DuckDuckGo (general + "fact check" query) + Wikipedia REST | ranked snippets, fact-checkers first |
| `fact_checker` | LLM | per-claim verdict (supported / refuted / unverified), evidence stances, red flags, p_llm |
| `source_credibility` | curated lists + FakeNewsNet domain stats + TLD / look-alike heuristics | credibility level + score + notes |
| `judge` | deterministic fusion | final verdict, probability, confidence |

Each node is wrapped by a decorator that times it, records a trace, streams a `step` event
(SSE) and contains failures — one failing agent degrades the answer instead of killing the run.
The graph itself routes around the LLM agents when the provider is down, and the judge says so.

### Fusion rule (judge)

Weighted average in logit space: `p_ml` (w=1.0), `p_llm` (w=1.2), `+2.0` per refuted claim,
`−1.2` per supported claim when none refuted, and `(0.5 − source_score)·3` (w=0.8). Confidence
combines distance from 0.5 with agreement between the ML and LLM probabilities, floored at
0.75 when evidence refutes a claim. Verdict thresholds: fake ≥ 0.62, real ≤ 0.38, else uncertain.

## 5. Serving

* **FastAPI** — `/analyze` (sync), `/analyze/stream` (SSE over POST so the UI shows the agents
  working live), `/analyze/batch`, `/history`, `/analysis/{id}`, `/feedback`, `/metrics/*`,
  `/health`. SQLite persistence. Models load in a background thread so the port binds instantly.
* **LLM provider switch** — Ollama (`llama3.2`) locally; Groq (`openai/gpt-oss-20b`, free tier)
  or any OpenAI-compatible endpoint in the cloud. All agents use a strict-JSON helper with
  repair + retry.
* **MCP server** — `check_news`, `score_headlines`, `model_metrics`, `recent_analyses` so Claude
  (or any MCP client) can call VeriFact as a tool.
* **Frontend** — React + Vite + TS on Vercel: live agent timeline, verdict gauge, model
  breakdown, token attributions, claims with evidence, source card, LLM reasoning, feedback,
  metrics dashboard, history.
* **Deploy** — Docker image (CPU torch) on Render; `render.yaml` blueprint; Vercel for the SPA.

## 6. Limitations & next steps

* LIAR is intrinsically hard (SOTA binary ≈ 0.70 acc); the ensemble's strength is the
  cross-dataset generalisation plus the evidence layer, not a leaderboard number.
* Headline-only training for FakeNewsNet (full bodies need scraping) — the ingest agent still
  classifies full article text at inference, which the models tolerate but weren't tuned on.
* Web evidence is open-web search; a curated fact-check API (Google Fact Check Tools) would
  raise precision. Image / video misinformation is out of scope.
* Planned: multilingual encoder (XLM-R), feedback-driven re-weighting of the stacker, and a
  browser extension calling `/analyze/stream`.
