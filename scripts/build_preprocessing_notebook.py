"""Generate and execute notebooks/01_data_preprocessing.ipynb.

The notebook is the human-readable record of how LIAR and FakeNewsNet are
turned into the unified training set; this script keeps it reproducible.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_data_preprocessing.ipynb"
OUT.parent.mkdir(exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# VeriFact — Dataset preprocessing

Turning two public benchmarks into one clean, de-duplicated, split dataset for the fake-news ensemble.

| Dataset | Source | What it is |
|---|---|---|
| **LIAR** | Wang (2017), *"Liar, Liar Pants on Fire"*, ACL | 12.8k short political statements from PolitiFact with a 6-point truth rating and speaker metadata |
| **FakeNewsNet** | Shu et al. (2018) | News headlines + source URLs labelled fake/real by **PolitiFact** (politics) and **GossipCop** (entertainment) |

Pipeline: **download → load → clean → label → de-duplicate → split → feature sanity-check → save**.""")

code("""import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path.cwd().parent))
from backend.ml import data
from backend.ml.features import featurize, FEATURE_NAMES
pd.set_option("display.max_colwidth", 120)
plt.rcParams["figure.dpi"] = 110
RAW = data.RAW
print("raw data folder:", RAW)""")

md("## 1. Raw files\n`scripts/download_data.py` fetches LIAR (official zip) and the four FakeNewsNet CSVs.")
code("""for p in sorted(RAW.rglob("*.*")):
    print(f"{p.relative_to(RAW)!s:40s} {p.stat().st_size/1e6:6.1f} MB")""")

md("### 1a. LIAR — raw TSV (14 columns, no header)")
code("""liar_raw = pd.read_csv(RAW/"liar"/"train.tsv", sep="\\t", header=None, names=data.LIAR_COLS,
                       quoting=3, dtype=str, keep_default_na=False)
print(liar_raw.shape)
liar_raw.head(3)""")

code("""liar_raw["fine_label"].value_counts().plot(kind="barh", color="#7c3aed", title="LIAR train: 6-way labels");
plt.xlabel("statements"); plt.tight_layout()""")

md("### 1b. FakeNewsNet — raw CSVs (`id, news_url, title, tweet_ids`)")
code("""fnn_raw = pd.read_csv(RAW/"fakenewsnet"/"politifact_fake.csv", dtype=str, keep_default_na=False)
print(fnn_raw.shape); fnn_raw[["id","news_url","title"]].head(3)""")

md("""## 2. Cleaning & unification

Decisions (implemented in `backend/ml/data.py` so training and the API share one code path):

* **Text cleaning** — collapse whitespace; keep casing and punctuation (shouting and `!!!` are *signal* for the style features).
* **LIAR binarisation** — `pants-fire / false / barely-true → fake (1)`, `half-true / mostly-true / true → real (0)` (standard in the literature).
* **FakeNewsNet** — the `title` is the text; the `news_url` is parsed to a bare domain which later feeds the source-credibility agent.
* **Drop** empty rows and titles shorter than 10 characters (ids/garbage); **de-duplicate** on exact text.
* **Splits** — LIAR keeps its *official* train/valid/test; FakeNewsNet is split 80/10/10 **stratified per source** so every source appears in test.""")
code("""df = data.load_unified(cache=False)
print(df.shape)
df.sample(6, random_state=3)[["text","label","fine_label","dataset","domain","split"]]""")

md("## 3. Dataset composition")
code("""summary = data.summary(df)
print(json.dumps(summary, indent=2))""")

code("""fig, ax = plt.subplots(1, 2, figsize=(11, 4))
comp = df.groupby(["dataset","label"]).size().unstack(fill_value=0).rename(columns={0:"real",1:"fake"})
comp.plot(kind="bar", stacked=True, ax=ax[0], color=["#34d399","#ef4444"], title="Rows per dataset (fake vs real)")
ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=0)
df.groupby(["dataset","split"]).size().unstack(fill_value=0)[["train","valid","test"]].plot(
    kind="bar", ax=ax[1], title="Split sizes per dataset", color=["#6366f1","#a78bfa","#fbbf24"])
ax[1].set_xlabel(""); ax[1].tick_params(axis="x", rotation=0)
plt.tight_layout()""")

md("Class balance is **32 % fake overall** but differs by source — the models use class weights so GossipCop's 76 % real share does not swamp the LIAR signal.")

md("## 4. Text length")
code("""df["n_words"] = df["text"].str.split().str.len()
fig, ax = plt.subplots(figsize=(9, 3.8))
for name, g in df.groupby("dataset"):
    ax.hist(g["n_words"].clip(upper=60), bins=60, alpha=0.55, label=f"{name} (median {int(g['n_words'].median())})")
ax.set_xlabel("words per item (clipped at 60)"); ax.set_ylabel("items"); ax.legend(); ax.set_title("Text length by dataset")
plt.tight_layout()
df.groupby("dataset")["n_words"].describe()[["mean","50%","max"]]""")

md("Items are short (median 11–18 words) → `max_len=64` tokens for DistilBERT loses nothing, and long articles at inference are scored as *lead + body chunks* (see `agents/graph.py`).")

md("## 5. Source domains (FakeNewsNet)\nThe domain column is also aggregated into `models/domain_stats.json` — a per-domain fake rate the credibility agent consults.")
code("""dom = df[df["domain"] != ""].groupby("domain")["label"].agg(fake="sum", total="count")
dom["fake_rate"] = dom["fake"] / dom["total"]
top = dom[dom["total"] >= 30].sort_values("total", ascending=False).head(15)
top.style.format({"fake_rate": "{:.0%}"}).background_gradient(subset=["fake_rate"], cmap="Reds")""")

md("## 6. Stylistic features — do they separate the classes?\n23 hand-crafted features (`backend/ml/features.py`) capture *how* something is written, independent of topic.")
code("""sample = df.sample(6000, random_state=0)
F = pd.DataFrame(featurize(sample["text"]), columns=FEATURE_NAMES)
F["label"] = sample["label"].values
means = F.groupby("label").mean().T.rename(columns={0:"real",1:"fake"})
means["ratio fake/real"] = (means["fake"] + 1e-6) / (means["real"] + 1e-6)
means.sort_values("ratio fake/real", ascending=False).head(10).style.format("{:.3f}")""")

code("""fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a, col in zip(ax, ["caps_ratio", "exclam", "clickbait_hits"]):
    for lab, color in ((0, "#34d399"), (1, "#ef4444")):
        a.hist(F[F.label == lab][col].clip(upper=F[col].quantile(0.99)), bins=25, alpha=0.6,
               density=True, color=color, label="real" if lab == 0 else "fake")
    a.set_title(col); a.legend()
plt.tight_layout()""")

md("## 7. Leakage & sanity checks")
code("""tr = set(df[df.split=="train"]["text"]); te = set(df[df.split=="test"]["text"]); va = set(df[df.split=="valid"]["text"])
print("train∩test overlap:", len(tr & te), "| train∩valid:", len(tr & va), "| duplicates:", df["text"].duplicated().sum())
print("empty texts:", (df["text"].str.len()==0).sum(), "| label values:", sorted(df["label"].unique()))
assert not (tr & te) and not df["text"].duplicated().any()""")

md("## 8. Saved artefact\n`data/processed/unified.parquet` is what `backend/ml/train.py` consumes. Columns:")
code("""df.drop(columns="n_words").dtypes""")

md("""## Summary

* **34,534** unique labelled items from 3 sources, **32 % fake**, zero train/test overlap.
* Unified binary label, original fine labels preserved for analysis.
* Official LIAR splits + stratified FakeNewsNet splits → honest, comparable evaluation.
* Domain table and stylistic features derived here are reused by the serving pipeline.

Next: `02_model_training` — TF-IDF+LR, MiniLM+XGBoost, DistilBERT, stacked ensemble (`python -m backend.ml.train all`).""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
nbf.write(nb, OUT)
print("written", OUT)
