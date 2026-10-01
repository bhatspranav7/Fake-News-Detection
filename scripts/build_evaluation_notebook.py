"""Generate and execute notebooks/02_model_training_and_evaluation.ipynb from
the artefacts in models/ (re-run after training to refresh numbers)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_model_training_and_evaluation.ipynb"

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# VeriFact — Model training & evaluation

How the ensemble is trained (`backend/ml/train.py`) and how it performs on the **held-out test split**.

```
text ─┬─> TF-IDF (word 1-2g + char 2-5g) ──> Logistic Regression ──┐
      ├─> MiniLM-L6 sentence embedding ⊕ 23 style features ─> XGBoost ─┤─> logit stacking ─> p(fake)
      └─> DistilBERT fine-tuned (max_len 64) ─> ONNX int8 ───────────┘   (LogReg meta-learner)
```

**Protocol.** Base models are fit on `train`; the stacker is fit on their `valid` predictions
(out-of-sample, so it learns each model's real reliability); every number below is on `test`.""")

code("""import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path.cwd().parent))
from backend.ml.data import load_unified
MODELS = Path.cwd().parent / "models"
report = json.loads((MODELS / "metrics.json").read_text())
plt.rcParams["figure.dpi"] = 110
print("protocol:", report["protocol"])
print("base models in the stack:", list(report["models"]))""")

md("## 1. Headline metrics (test set)")
code("""rows = {name: m for name, m in report["models"].items()}
rows["stacked ensemble"] = report["ensemble"]
tbl = pd.DataFrame({k: {"accuracy": v["accuracy"], "precision": v["precision"], "recall": v["recall"],
                        "f1 (fake)": v["f1"], "roc_auc": v["roc_auc"], "n": v["n"]} for k, v in rows.items()}).T
tbl.style.format({c: "{:.3f}" for c in tbl.columns if c != "n"}).background_gradient(subset=["roc_auc"], cmap="Purples")""")

code("""ax = tbl[["accuracy", "f1 (fake)", "roc_auc"]].plot(kind="bar", figsize=(8, 3.8), color=["#8b7cf6", "#22d3ee", "#f59e0b"],
                                                      title="Test-set comparison")
ax.set_ylim(0.4, 1.0); ax.tick_params(axis="x", rotation=0); ax.set_xlabel(""); plt.tight_layout()""")

md("## 2. Per-dataset breakdown\nLIAR is the hard case (short political statements, SOTA binary ≈ 0.70); GossipCop headlines are easier.")
code("""per = {name: pd.DataFrame(m["per_dataset"]).T for name, m in rows.items()}
wide = pd.concat({k: v[["accuracy", "f1", "roc_auc"]] for k, v in per.items()}, axis=1)
wide.style.format("{:.3f}")""")

code("""fig, ax = plt.subplots(figsize=(8, 3.6))
acc = pd.DataFrame({k: v["accuracy"] for k, v in per.items()})
acc.plot(kind="bar", ax=ax, title="Accuracy per dataset"); ax.set_ylim(0.4, 1); ax.tick_params(axis="x", rotation=0); ax.set_xlabel("")
plt.tight_layout()""")

md("## 3. Confusion matrix — stacked ensemble")
code("""cm = np.array(report["ensemble"]["confusion"])
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.imshow(cm, cmap="Purples")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v}\\n{v/cm.sum():.1%}", ha="center", va="center", color="black" if v < cm.max()/2 else "white")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["pred real", "pred fake"]); ax.set_yticklabels(["real", "fake"])
ax.set_title("Ensemble confusion (test)"); plt.tight_layout()""")

md("## 4. What the meta-learner learned\nStacking weights on each base model's logit (plus 8 style features). Larger = trusted more.")
code("""w = pd.Series(report["meta_weights"])
w.plot(kind="barh", color=np.where(w > 0, "#8b7cf6", "#ef4444"), figsize=(7, 3.5), title="Meta-learner coefficients")
plt.tight_layout(); w.round(3)""")

md("## 5. Calibration & threshold\nThe judge treats p ≥ 0.62 as *fake*, p ≤ 0.38 as *real*, in between *uncertain*. Reliability curve of the ensemble:")
code("""PREDS = MODELS / "preds"
df = load_unified(); test = df[df.split == "test"].reset_index(drop=True)
p = np.load(PREDS / "ensemble_test.npy"); y = test["label"].values
bins = np.linspace(0, 1, 11); idx = np.digitize(p, bins) - 1
cal = pd.DataFrame({"bin_mid": [(bins[i]+bins[i+1])/2 for i in range(10)],
                    "observed_fake_rate": [y[idx == i].mean() if (idx == i).any() else np.nan for i in range(10)],
                    "count": [(idx == i).sum() for i in range(10)]})
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.plot([0, 1], [0, 1], "--", color="gray"); ax.plot(cal.bin_mid, cal.observed_fake_rate, "o-", color="#8b7cf6")
ax.set_xlabel("predicted p(fake)"); ax.set_ylabel("observed fake rate"); ax.set_title("Reliability (test)")
plt.tight_layout(); cal""")

code("""thr = np.linspace(0.3, 0.8, 11)
pd.DataFrame({"threshold": thr,
              "accuracy": [((p >= t) == y).mean() for t in thr],
              "fake_recall": [((p >= t) & (y == 1)).sum() / (y == 1).sum() for t in thr],
              "fake_precision": [((p >= t) & (y == 1)).sum() / max((p >= t).sum(), 1) for t in thr]}).round(3)""")

md("## 6. Qualitative check — explanations the API returns")
code("""from backend.ml.predictor import Predictor
pred = Predictor().load()
examples = [
    "BREAKING: Scientists CONFIRM lemon water cures cancer in 30 days — doctors don't want you to know!",
    "The unemployment rate fell to 3.7 percent last month, according to the Bureau of Labor Statistics.",
    "Taylor Swift secretly married in a private ceremony, sources close to the singer reveal.",
    "NASA's James Webb telescope detected carbon dioxide in an exoplanet atmosphere, the agency said Thursday.",
]
for t in examples:
    r = pred.predict(t)
    toks = ", ".join(f"{d['token']}({d['weight']:+.2f})" for d in r["top_tokens"][:5])
    print(f"{r['ensemble_prob']:.0%} fake | {t[:70]}…\\n    models={ {k: round(v, 2) for k, v in r['models'].items()} }\\n    tokens: {toks}\\n    flags: {r['style_flags']}")""")

md("""## Takeaways

* Stacking lifts ROC-AUC above every single model — the models make *different* mistakes (lexical vs semantic vs style).
* Honest per-dataset reporting: strong on GossipCop, mid-60s on LIAR like the literature — which is exactly why the
  system adds **evidence retrieval + LLM fact-checking + source credibility** on top of the classifier.
* Token attributions and style flags make every prediction explainable in the UI.""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
nbf.write(nb, OUT)
print("written", OUT)
