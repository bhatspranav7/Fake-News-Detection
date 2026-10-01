"""Train the fake-news ensemble.

Stages (run individually so the slow transformer can train in parallel):

  python -m backend.ml.train tfidf        # TF-IDF word+char n-grams -> LogisticRegression
  python -m backend.ml.train embed        # MiniLM sentence embeddings + style feats -> XGBoost
  python -m backend.ml.train transformer  # fine-tune DistilBERT, export ONNX int8
  python -m backend.ml.train stack        # stacked meta-learner over the three + report
  python -m backend.ml.train all

Protocol: base models fit on `train`, the stacker fits on `valid` predictions,
every number in models/metrics.json is measured on the held-out `test` split.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import FeatureUnion, Pipeline

from backend.ml.data import load_unified, summary
from backend.ml.features import FEATURE_NAMES, featurize

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
PREDS = MODELS / "preds"
MODELS.mkdir(exist_ok=True)
PREDS.mkdir(exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL = "distilbert-base-uncased"
MAX_LEN = 64
SEED = 42


# ----------------------------------------------------------------- helpers
def splits():
    df = load_unified()
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test")}


def metrics(y_true, prob, name: str, datasets=None) -> dict:
    pred = (prob >= 0.5).astype(int)
    out = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "confusion": confusion_matrix(y_true, pred).tolist(),
        "n": int(len(y_true)),
    }
    if datasets is not None:
        out["per_dataset"] = {}
        for ds in sorted(set(datasets)):
            m = np.asarray(datasets) == ds
            out["per_dataset"][ds] = {
                "accuracy": float(accuracy_score(y_true[m], pred[m])),
                "f1": float(f1_score(y_true[m], pred[m], zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true[m], prob[m])) if len(set(y_true[m])) > 1 else None,
                "n": int(m.sum()),
            }
    return out


def save_preds(name: str, split: str, prob: np.ndarray):
    np.save(PREDS / f"{name}_{split}.npy", prob.astype(np.float32))


# ------------------------------------------------------------- stage: tfidf
def train_tfidf():
    d = splits()
    t0 = time.time()
    pipe = Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                     sublinear_tf=True, strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                     max_features=300_000, sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced",
                                   solver="liblinear")),
    ])
    pipe.fit(d["train"]["text"], d["train"]["label"])
    for s in ("valid", "test"):
        save_preds("tfidf", s, pipe.predict_proba(d[s]["text"])[:, 1])
    joblib.dump(pipe, MODELS / "tfidf_lr.joblib", compress=3)
    m = metrics(d["test"]["label"].values, np.load(PREDS / "tfidf_test.npy"), "tfidf_lr",
                d["test"]["dataset"].values)
    m["train_seconds"] = round(time.time() - t0, 1)
    (MODELS / "metrics_tfidf.json").write_text(json.dumps(m, indent=2))
    print("tfidf", {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})


# ------------------------------------------------------------- stage: embed
def embed_texts(texts, model=None):
    from sentence_transformers import SentenceTransformer

    model = model or SentenceTransformer(EMBED_MODEL, device="cpu")
    return model.encode(list(texts), batch_size=128, show_progress_bar=False,
                        normalize_embeddings=True)


def train_embed():
    import xgboost as xgb
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import StandardScaler

    d = splits()
    t0 = time.time()
    st = SentenceTransformer(EMBED_MODEL, device="cpu")
    X = {}
    for s, df in d.items():
        cache = PREDS / f"emb_{s}.npy"
        if cache.exists():
            emb = np.load(cache)
        else:
            emb = embed_texts(df["text"], st)
            np.save(cache, emb)
        X[s] = np.hstack([emb, featurize(df["text"])])
    scaler = StandardScaler().fit(X["train"])
    Xs = {s: scaler.transform(x) for s, x in X.items()}
    pos = d["train"]["label"].mean()
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=(1 - pos) / pos,
        eval_metric="logloss", n_jobs=8, random_state=SEED, early_stopping_rounds=40,
    )
    clf.fit(Xs["train"], d["train"]["label"], eval_set=[(Xs["valid"], d["valid"]["label"])],
            verbose=False)
    for s in ("valid", "test"):
        save_preds("embed", s, clf.predict_proba(Xs[s])[:, 1])
    joblib.dump({"scaler": scaler, "clf": clf, "embed_model": EMBED_MODEL,
                 "feature_names": FEATURE_NAMES}, MODELS / "embed_xgb.joblib", compress=3)
    m = metrics(d["test"]["label"].values, np.load(PREDS / "embed_test.npy"), "embed_xgb",
                d["test"]["dataset"].values)
    m["train_seconds"] = round(time.time() - t0, 1)
    m["best_iteration"] = int(clf.best_iteration)
    (MODELS / "metrics_embed.json").write_text(json.dumps(m, indent=2))
    print("embed", {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})


# ------------------------------------------------------- stage: transformer
def train_transformer(epochs: int = 2, batch_size: int = 32, lr: float = 3e-5):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    d = splits()
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL, num_labels=2)

    def encode(df):
        enc = tok(list(df["text"]), truncation=True, max_length=MAX_LEN, padding="max_length",
                  return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"],
                             torch.tensor(df["label"].values, dtype=torch.long))

    train_ds = encode(d["train"])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # Class weights: fake is the minority class in the merged data.
    pos = d["train"]["label"].mean()
    weights = torch.tensor([1.0, (1 - pos) / pos], dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    @torch.no_grad()
    def predict(df):
        model.eval()
        ds = encode(df)
        out = []
        for ids, mask, _ in DataLoader(ds, batch_size=128):
            logits = model(input_ids=ids, attention_mask=mask).logits
            out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        model.train()
        return np.concatenate(out)

    t0 = time.time()
    step = 0
    best_f1, best_state = -1.0, None
    model.train()
    for ep in range(epochs):
        for ids, mask, y in loader:
            loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"ep{ep} step {step}/{total} loss {loss.item():.4f} "
                      f"{el/step:.2f}s/step eta {(total-step)*el/step/60:.1f}min", flush=True)
        pv = predict(d["valid"])
        f1 = f1_score(d["valid"]["label"], (pv >= 0.5).astype(int))
        print(f"epoch {ep} valid f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    for s in ("valid", "test"):
        save_preds("transformer", s, predict(d[s]))

    hf_dir = MODELS / "transformer_hf"
    model.save_pretrained(hf_dir)
    tok.save_pretrained(hf_dir)
    # Reload with eager attention: the SDPA path is not traceable by the exporter.
    export_onnx(AutoModelForSequenceClassification.from_pretrained(hf_dir, attn_implementation="eager"), tok)

    m = metrics(d["test"]["label"].values, np.load(PREDS / "transformer_test.npy"),
                "distilbert", d["test"]["dataset"].values)
    m["train_seconds"] = round(time.time() - t0, 1)
    m["epochs"] = epochs
    m["base_model"] = HF_MODEL
    (MODELS / "metrics_transformer.json").write_text(json.dumps(m, indent=2))
    print("transformer", {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})


def export_onnx(model, tok):
    """Export to ONNX and dynamically quantise to int8 for cheap CPU serving."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model.eval()
    out_dir = MODELS / "transformer_onnx"
    out_dir.mkdir(exist_ok=True)
    tok.save_pretrained(out_dir)
    sample = tok(["export sample"], return_tensors="pt", padding="max_length", max_length=MAX_LEN,
                 truncation=True)

    class Wrapper(torch.nn.Module):
        """Positional-arg shim: newer transformers reorder forward() kwargs."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    fp32 = out_dir / "model_fp32.onnx"
    torch.onnx.export(
        Wrapper(model), (sample["input_ids"], sample["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=18, dynamo=True,
    )
    quantize_dynamic(str(fp32), str(out_dir / "model_int8.onnx"), weight_type=QuantType.QInt8)
    fp32.unlink(); (out_dir / "model_fp32.onnx.data").unlink(missing_ok=True)  # keep only int8
    (out_dir / "config.json").write_text(json.dumps({"max_len": MAX_LEN, "labels": ["real", "fake"]}))
    print("onnx int8 ->", out_dir / "model_int8.onnx",
          f"{(out_dir / 'model_int8.onnx').stat().st_size/1e6:.1f} MB")


# ------------------------------------------------------------- stage: stack
def train_stack():
    d = splits()
    base = ["tfidf", "embed", "transformer"]
    present = [b for b in base if (PREDS / f"{b}_test.npy").exists()]
    if len(present) < 2:
        raise SystemExit(f"need at least two base models, found {present}")
    V = np.column_stack([np.load(PREDS / f"{b}_valid.npy") for b in present])
    T = np.column_stack([np.load(PREDS / f"{b}_test.npy") for b in present])
    # Logit-space stacking keeps it well-calibrated and lets the meta-learner
    # weight each model; style features give it a little extra signal.
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    Vx = np.hstack([logit(V), featurize(d["valid"]["text"])[:, :8]])
    Tx = np.hstack([logit(T), featurize(d["test"]["text"])[:, :8]])
    meta = LogisticRegression(C=1.0, max_iter=2000).fit(Vx, d["valid"]["label"])
    prob = meta.predict_proba(Tx)[:, 1]
    save_preds("ensemble", "test", prob)
    joblib.dump({"meta": meta, "base": present, "n_style": 8}, MODELS / "stacker.joblib")

    y = d["test"]["label"].values
    ds = d["test"]["dataset"].values
    report = {
        "dataset": summary(load_unified()),
        "models": {b: json.loads((MODELS / f"metrics_{'transformer' if b == 'transformer' else b}.json").read_text())
                   for b in present},
        "ensemble": metrics(y, prob, "stacked_ensemble", ds),
        "meta_weights": dict(zip(present + [f"style_{i}" for i in range(8)], meta.coef_[0].round(3).tolist())),
        "protocol": "base models fit on train; stacker fit on valid; all metrics on held-out test",
    }
    (MODELS / "metrics.json").write_text(json.dumps(report, indent=2))
    print("\n=== TEST SET ===")
    for name, m in list(report["models"].items()) + [("ensemble", report["ensemble"])]:
        print(f"{name:12s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
    for dsn, m in report["ensemble"]["per_dataset"].items():
        print(f"   {dsn:11s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} n={m['n']}")


STAGES = {"tfidf": train_tfidf, "embed": train_embed, "transformer": train_transformer,
          "stack": train_stack}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (STAGES if stage == "all" else [stage]):
        print(f"\n##### {name}")
        STAGES[name]()
