"""Load LIAR and FakeNewsNet into one unified, de-duplicated binary dataset.

Unified schema
--------------
text        : the statement / headline to classify
label       : 1 = fake, 0 = real
fine_label  : original fine-grained label (LIAR 6-way, FNN fake/real)
dataset     : "liar" | "politifact" | "gossipcop"
domain      : source domain for FakeNewsNet rows, "" for LIAR
split       : train | valid | test  (LIAR keeps its official splits)
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

LIAR_COLS = [
    "id", "fine_label", "text", "subject", "speaker", "job_title", "state",
    "party", "barely_true_ct", "false_ct", "half_true_ct", "mostly_true_ct",
    "pants_fire_ct", "context",
]
# Standard binarisation used in the LIAR literature.
LIAR_FAKE = {"pants-fire", "false", "barely-true"}
LIAR_REAL = {"half-true", "mostly-true", "true"}
LIAR_6WAY = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]


def _clean(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _domain(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def load_liar() -> pd.DataFrame:
    frames = []
    for split in ("train", "valid", "test"):
        df = pd.read_csv(RAW / "liar" / f"{split}.tsv", sep="\t", header=None,
                         names=LIAR_COLS, quoting=3, dtype=str, keep_default_na=False)
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].map(_clean)
    df["label"] = df["fine_label"].map(lambda l: 1 if l in LIAR_FAKE else 0)
    df["dataset"] = "liar"
    df["domain"] = ""
    # Speaker/context are kept for the metadata-aware features.
    return df[["text", "label", "fine_label", "dataset", "domain", "split",
               "speaker", "party", "context", "subject"]]


def load_fakenewsnet(seed: int = 42) -> pd.DataFrame:
    frames = []
    for source in ("politifact", "gossipcop"):
        for fine, lab in (("fake", 1), ("real", 0)):
            df = pd.read_csv(RAW / "fakenewsnet" / f"{source}_{fine}.csv", dtype=str,
                             keep_default_na=False)
            df = df.rename(columns={"title": "text"})
            df["text"] = df["text"].map(_clean)
            df["label"] = lab
            df["fine_label"] = fine
            df["dataset"] = source
            df["domain"] = df["news_url"].map(_domain)
            frames.append(df[["text", "label", "fine_label", "dataset", "domain"]])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["text"].str.len() >= 10].drop_duplicates("text")
    for col in ("speaker", "party", "context", "subject"):
        df[col] = ""
    # Stratified 80/10/10 split per source so each one is represented in test.
    df["split"] = "train"
    for source in df["dataset"].unique():
        idx = df.index[df["dataset"] == source]
        strata = df.loc[idx, "label"]
        tr, rest = train_test_split(idx, test_size=0.2, stratify=strata, random_state=seed)
        va, te = train_test_split(rest, test_size=0.5, stratify=df.loc[rest, "label"],
                                  random_state=seed)
        df.loc[va, "split"] = "valid"
        df.loc[te, "split"] = "test"
    return df.reset_index(drop=True)


def load_unified(cache: bool = True) -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / "unified.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    df = pd.concat([load_liar(), load_fakenewsnet()], ignore_index=True)
    df = df[df["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)
    if cache:
        df.to_parquet(path, index=False)
    return df


def summary(df: pd.DataFrame) -> dict:
    out: dict = {"total": int(len(df)), "fake_share": float(df["label"].mean())}
    for ds, g in df.groupby("dataset"):
        out[ds] = {
            "rows": int(len(g)),
            "fake": int(g["label"].sum()),
            "real": int((g["label"] == 0).sum()),
            "splits": {s: int(n) for s, n in g["split"].value_counts().items()},
        }
    return out


if __name__ == "__main__":
    import json

    df = load_unified(cache=True)
    print(json.dumps(summary(df), indent=2))
    print(df.sample(5, random_state=1)[["text", "label", "dataset"]].to_string())
    print("mean words:", np.mean([len(t.split()) for t in df["text"]]))
