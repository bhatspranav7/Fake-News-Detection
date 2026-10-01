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


def _session_options(ort):
    """Small-footprint ONNX Runtime settings: the memory arena and pattern
    planner pre-allocate ~30% extra RAM, which matters on a 512 MB host."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = config.ONNX_THREADS
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    return so


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
        self.embed_onnx = None
        self.embed_tok = None
        self.embed_cfg = None
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
                    self.embed = joblib.load(self.dir / "embed_xgb.joblib")
                    onnx_dir = self.dir / "embed_onnx"
                    if (onnx_dir / "model_int8.onnx").exists():
                        # Preferred: int8 ONNX encoder, no torch needed at serving time.
                        import onnxruntime as ort
                        from tokenizers import Tokenizer

                        so = _session_options(ort)
                        self.embed_onnx = ort.InferenceSession(str(onnx_dir / "model_int8.onnx"), so,
                                                               providers=["CPUExecutionProvider"])
                        self.embed_tok = Tokenizer.from_file(str(onnx_dir / "tokenizer.json"))
                        self.embed_cfg = json.loads((onnx_dir / "config.json").read_text())
                        self.embed_tok.enable_truncation(self.embed_cfg["max_len"])
                        self.embed_tok.enable_padding()
                    else:
                        from sentence_transformers import SentenceTransformer

                        self.st_model = SentenceTransformer(self.embed["embed_model"], device="cpu")
                except Exception as exc:  # noqa: BLE001
                    log.warning("embed model unavailable: %s", exc)
                    self.embed = None
            onnx_path = self.dir / "transformer_onnx" / "model_int8.onnx"
            if config.LOAD_TRANSFORMER and onnx_path.exists():
                try:
                    import onnxruntime as ort
                    from tokenizers import Tokenizer

                    so = _session_options(ort)
                    self.onnx = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
                    self.onnx_cfg = json.loads((self.dir / "transformer_onnx" / "config.json").read_text())
                    self.tok = Tokenizer.from_file(str(self.dir / "transformer_onnx" / "tokenizer.json"))
                    self.tok.enable_truncation(self.onnx_cfg["max_len"])
                    self.tok.enable_padding()
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

    @staticmethod
    def _encode(tok, texts, with_types: bool):
        encs = tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if with_types:
            feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
        return feed

    def _embed_texts(self, texts) -> np.ndarray:
        if self.embed_onnx is not None:
            out = []
            for i in range(0, len(texts), 64):
                feed = self._encode(self.embed_tok, texts[i:i + 64], with_types=True)
                hidden = self.embed_onnx.run(None, feed)[0]
                mask = feed["attention_mask"][..., None].astype(np.float32)
                emb = (hidden * mask).sum(1) / np.maximum(mask.sum(1), 1e-9)  # mean pooling
                out.append(emb / np.linalg.norm(emb, axis=1, keepdims=True))
            return np.vstack(out)
        return self.st_model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                                   show_progress_bar=False)

    def _p_embed(self, texts):
        emb = self._embed_texts(texts)
        X = self.embed["scaler"].transform(np.hstack([emb, featurize(texts)]))
        return self.embed["clf"].predict_proba(X)[:, 1]

    def _p_onnx(self, texts):
        feed = self._encode(self.tok, texts, with_types=False)
        logits = self.onnx.run(["logits"], feed)[0]
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
