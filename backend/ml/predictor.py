"""Serve the trained ensemble: per-model probabilities, stacked fusion and
token-level explanations. Loads lazily and degrades gracefully when an
artefact is missing (e.g. transformer disabled on a tiny host)."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import joblib
import numpy as np

from backend.core import config
from backend.ml.features import explain_style, featurize

log = logging.getLogger(__name__)

MODEL_LABELS = {
    "tfidf_lr": "TF-IDF + Logistic Regression",
    "embed_xgb": "MiniLM embeddings + XGBoost",
    "distilbert": "DistilBERT (fine-tuned, ONNX int8)",
}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Predictor:
    def __init__(self, models_dir: Path | None = None):
        self.dir = Path(models_dir or config.MODELS_DIR)
        self._lock = threading.Lock()
        self._loaded = False
        self.tfidf = None
        self.embed = None
        self.st_model = None
        self.onnx = None
        self.tok = None
        self.onnx_cfg = None
        self.stacker = None
        self.metrics = None

    # ------------------------------------------------------------- loading
    def load(self) -> "Predictor":
        with self._lock:
            if self._loaded:
                return self
            if (self.dir / "tfidf_lr.joblib").exists():
                self.tfidf = joblib.load(self.dir / "tfidf_lr.joblib")
            if config.LOAD_EMBED and (self.dir / "embed_xgb.joblib").exists():
                try:
                    from sentence_transformers import SentenceTransformer

                    self.embed = joblib.load(self.dir / "embed_xgb.joblib")
                    self.st_model = SentenceTransformer(self.embed["embed_model"], device="cpu")
                except Exception as exc:  # noqa: BLE001
                    log.warning("embed model unavailable: %s", exc)
                    self.embed = None
            onnx_path = self.dir / "transformer_onnx" / "model_int8.onnx"
            if config.LOAD_TRANSFORMER and onnx_path.exists():
                try:
                    import onnxruntime as ort
                    from transformers import AutoTokenizer

                    so = ort.SessionOptions()
                    so.intra_op_num_threads = 4
                    self.onnx = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
                    self.tok = AutoTokenizer.from_pretrained(self.dir / "transformer_onnx")
                    self.onnx_cfg = json.loads((self.dir / "transformer_onnx" / "config.json").read_text())
                except Exception as exc:  # noqa: BLE001
                    log.warning("transformer unavailable: %s", exc)
                    self.onnx = None
            if (self.dir / "stacker.joblib").exists():
                self.stacker = joblib.load(self.dir / "stacker.joblib")
            if (self.dir / "metrics.json").exists():
                self.metrics = json.loads((self.dir / "metrics.json").read_text())
            self._loaded = True
            log.info("models loaded: %s", self.loaded_models())
            return self

    def loaded_models(self) -> list[str]:
        out = []
        if self.tfidf is not None:
            out.append("tfidf_lr")
        if self.embed is not None:
            out.append("embed_xgb")
        if self.onnx is not None:
            out.append("distilbert")
        if self.stacker is not None:
            out.append("stacked_ensemble")
        return out

    # ----------------------------------------------------------- inference
    def _p_tfidf(self, texts):
        return self.tfidf.predict_proba(texts)[:, 1]

    def _p_embed(self, texts):
        emb = self.st_model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                                   show_progress_bar=False)
        X = self.embed["scaler"].transform(np.hstack([emb, featurize(texts)]))
        return self.embed["clf"].predict_proba(X)[:, 1]

    def _p_onnx(self, texts):
        enc = self.tok(list(texts), truncation=True, max_length=self.onnx_cfg["max_len"],
                       padding=True, return_tensors="np")
        logits = self.onnx.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                            "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True))[:, 1]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        self.load()
        texts = [t if t.strip() else "(empty)" for t in texts]
        probs: dict[str, np.ndarray] = {}
        if self.tfidf is not None:
            probs["tfidf_lr"] = self._p_tfidf(texts)
        if self.embed is not None:
            probs["embed_xgb"] = self._p_embed(texts)
        if self.onnx is not None:
            probs["distilbert"] = self._p_onnx(texts)
        if not probs:
            raise RuntimeError("no models loaded - run `python -m backend.ml.train all`")

        ens = self._fuse(probs, texts)
        out = []
        for i, text in enumerate(texts):
            out.append({
                "ensemble_prob": float(ens[i]),
                "models": {k: float(v[i]) for k, v in probs.items()},
                "top_tokens": self.top_tokens(text),
                "style_flags": explain_style(text),
            })
        return out

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def _fuse(self, probs: dict[str, np.ndarray], texts) -> np.ndarray:
        """Stacked meta-learner when every base model it was trained on is
        present; otherwise a logit-average of whatever is loaded."""
        key_map = {"tfidf": "tfidf_lr", "embed": "embed_xgb", "transformer": "distilbert"}
        if self.stacker is not None and all(key_map[b] in probs for b in self.stacker["base"]):
            P = np.column_stack([probs[key_map[b]] for b in self.stacker["base"]])
            X = np.hstack([_logit(P), featurize(texts)[:, :self.stacker["n_style"]]])
            return self.stacker["meta"].predict_proba(X)[:, 1]
        L = np.mean([_logit(p) for p in probs.values()], axis=0)
        return 1 / (1 + np.exp(-L))

    # -------------------------------------------------------- explanations
    def top_tokens(self, text: str, k: int = 10) -> list[dict]:
        """Signed word contributions from the linear TF-IDF model: coefficient
        x tf-idf weight, which is exact for logistic regression."""
        if self.tfidf is None:
            return []
        union = self.tfidf.named_steps["tfidf"]
        clf = self.tfidf.named_steps["clf"]
        word_vec = union.transformer_list[0][1]
        X = word_vec.transform([text])
        n_word = len(word_vec.vocabulary_)
        coef = clf.coef_[0][:n_word]
        contrib = X.multiply(coef).tocoo()
        names = word_vec.get_feature_names_out()
        items = sorted(zip(contrib.col, contrib.data), key=lambda t: -abs(t[1]))[:k]
        return [{"token": str(names[c]), "weight": round(float(w), 4)} for c, w in items if abs(w) > 1e-4]


_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor
