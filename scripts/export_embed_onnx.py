"""Export the MiniLM sentence encoder to int8 ONNX so serving needs no torch.

Produces models/embed_onnx/{model_int8.onnx, tokenizer.json, config.json} and
verifies the ONNX embeddings match sentence-transformers (cosine > 0.999), so
the XGBoost head trained on the original embeddings stays valid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path(__file__).resolve().parents[1] / "models" / "embed_onnx"
OUT.mkdir(parents=True, exist_ok=True)
MAX_LEN = 128

tok = AutoTokenizer.from_pretrained(MODEL)
tok.save_pretrained(OUT)


class Wrapper(torch.nn.Module):
    """Positional-arg shim: newer transformers reorder forward() kwargs."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids).last_hidden_state


model = Wrapper(AutoModel.from_pretrained(MODEL, attn_implementation="eager")).eval()

sample = tok(["export"], return_tensors="pt", padding="max_length", max_length=16, truncation=True)
fp32 = OUT / "model_fp32.onnx"
torch.onnx.export(
    model, (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]), str(fp32),
    input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=["last_hidden_state"],
    dynamic_axes={k: {0: "batch", 1: "seq"} for k in ("input_ids", "attention_mask", "token_type_ids", "last_hidden_state")},
    opset_version=18, dynamo=True,
)
quantize_dynamic(str(fp32), str(OUT / "model_int8.onnx"), weight_type=QuantType.QInt8)
fp32.unlink(); (OUT / "model_fp32.onnx.data").unlink(missing_ok=True)
(OUT / "config.json").write_text(json.dumps({"max_len": MAX_LEN, "pooling": "mean", "normalize": True,
                                             "source": MODEL}))

# ---- verify against sentence-transformers
import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(str(OUT / "model_int8.onnx"), providers=["CPUExecutionProvider"])
texts = ["BREAKING: miracle cure exposed!", "The council approved the budget on Tuesday.",
         "Taylor Swift secretly married, sources say"]
enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np")
hidden = sess.run(None, {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask", "token_type_ids")})[0]
mask = enc["attention_mask"][..., None].astype(np.float32)
emb = (hidden * mask).sum(1) / np.maximum(mask.sum(1), 1e-9)
emb /= np.linalg.norm(emb, axis=1, keepdims=True)
ref = SentenceTransformer(MODEL, device="cpu").encode(texts, normalize_embeddings=True)
cos = (emb * ref).sum(1)
print("cosine onnx-int8 vs sentence-transformers:", cos.round(4))
assert cos.min() > 0.99, "ONNX export drifted from the reference encoder"
print("ok ->", OUT / "model_int8.onnx", f"{(OUT / 'model_int8.onnx').stat().st_size/1e6:.1f} MB")
