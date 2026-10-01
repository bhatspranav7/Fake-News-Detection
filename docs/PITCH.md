# VeriFact — 3-minute pitch & demo script

## The one-liner
"Fake-news detectors that only look at *how* something is written get fooled by well-written
lies. VeriFact checks **how it's written, what it claims, and who's saying it** — a
deep-learning ensemble, LLM fact-checking agents with live web evidence, and source
reputation — fused into one explainable verdict, streamed to the user in real time."

## Why judges should care (map to the brief)
| Brief | What we built |
|---|---|
| Deep learning | Fine-tuned DistilBERT (exported to int8 ONNX for CPU serving) + MiniLM sentence encoder + XGBoost + TF-IDF LR, stacked with an out-of-sample meta-learner |
| LLM-based features | LLM agents extract claims, judge evidence stances, list red flags, output a calibrated probability — all consumed by the judge as features |
| LIAR + FakeNewsNet | Both used; unified schema, official LIAR splits, per-dataset metrics on the dashboard; FakeNewsNet domains also power source credibility |
| Identify misinformation from articles | URL ingestion (trafilatura), full article → claims → evidence → verdict with citations |

## Demo script (deep mode, ~90 s)
1. **Paste the "lemon water cures cancer" example** → watch the agent timeline: classifier says 88 %
   fake in 50 ms; claim extractor pulls the claim; evidence retriever finds fact-checkers and
   Wikipedia; fact checker marks it *refuted*; judge: **FAKE, 95 % confidence**. Point at the
   red token chips ("breaking", "furious") and the red flags.
2. **Paste a Reuters URL** → source credibility *high*, claims *supported*, verdict **REAL**.
   Show the per-model bars disagreeing slightly and the stacker resolving it.
3. **Paste `abcnews.com.co`-style look-alike URL** (or text with it) → credibility agent flags
   the imitation domain.
4. **Dashboard** → ensemble vs single models, per-dataset table, confusion matrix, dataset
   composition. Mention: "every number is held-out test; stacker never saw training data".
5. **Thumbs-down** a verdict → feedback agreement updates. Mention MCP: "Claude can call this
   as a tool" (`claude mcp add verifact …`).

## Fast answers to likely questions
* **Accuracy?** LIAR binary is a notoriously hard benchmark (published SOTA ≈ 0.70); we report
  honest per-dataset numbers rather than one inflated figure, and the evidence layer is what
  catches *novel* fakes the classifier has never seen.
* **Why not just the LLM?** Cost/latency (seconds vs ms), hallucination risk, and no
  calibration. The ML prior + evidence + LLM reasoning, fused, beats any one of them.
* **Why stacking on validation?** Fitting the meta-learner on training predictions would
  overweight whichever base model overfits most. Out-of-sample stacking is the honest way.
* **Scales?** Fast mode is batched ML (`/analyze/batch`); agents are stateless and parallel;
  evidence search runs threads per claim; int8 ONNX makes the transformer 4× cheaper.
* **Limitations?** Headline-level FakeNewsNet training; open-web evidence; English only.
  Roadmap in `docs/ARCHITECTURE.md`.
