"""(Re-)export the fine-tuned DistilBERT in models/transformer_hf to int8 ONNX.

Normally `python -m backend.ml.train transformer` does this at the end of
training; run this if you only want to redo the export.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.ml.train import MODELS, export_onnx

hf_dir = MODELS / "transformer_hf"
# eager attention keeps the graph traceable by the legacy ONNX exporter
model = AutoModelForSequenceClassification.from_pretrained(hf_dir, attn_implementation="eager")
tok = AutoTokenizer.from_pretrained(hf_dir)
export_onnx(model, tok)
