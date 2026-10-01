"""Download the open-source datasets used for training.

LIAR        - 12.8k short political statements labelled on a 6-point truth scale
              (Wang, 2017). https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
FakeNewsNet - PolitiFact + GossipCop news titles labelled fake/real
              (Shu et al., 2018). https://github.com/KaiDMML/FakeNewsNet

Everything lands in data/raw/. Re-running is a no-op for files that exist.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

LIAR_URLS = [
    "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
    "https://huggingface.co/datasets/ucsbnlp/liar/resolve/main/liar_dataset.zip",
]
LIAR_FILES = ["train.tsv", "valid.tsv", "test.tsv"]

FNN_BASE = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/"
FNN_FILES = [
    "politifact_fake.csv",
    "politifact_real.csv",
    "gossipcop_fake.csv",
    "gossipcop_real.csv",
]


def fetch(url: str, timeout: int = 120) -> bytes:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.content


def download_liar() -> None:
    target = RAW / "liar"
    if all((target / f).exists() for f in LIAR_FILES):
        print("LIAR already present")
        return
    target.mkdir(exist_ok=True)
    last_err: Exception | None = None
    for url in LIAR_URLS:
        try:
            print(f"LIAR <- {url}")
            blob = fetch(url)
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                for name in zf.namelist():
                    base = Path(name).name
                    if base in LIAR_FILES:
                        (target / base).write_bytes(zf.read(name))
            print("LIAR ok:", [f for f in LIAR_FILES if (target / f).exists()])
            return
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_err = exc
            print(f"  failed: {exc}")
    # Last resort: the HuggingFace parquet mirror.
    try:
        from datasets import load_dataset

        print("LIAR <- huggingface datasets (ucsbnlp/liar)")
        ds = load_dataset("ucsbnlp/liar", trust_remote_code=False)
        names = ds["train"].features["label"].names
        for split, fname in [("train", "train.tsv"), ("validation", "valid.tsv"), ("test", "test.tsv")]:
            df = ds[split].to_pandas()
            df["label"] = df["label"].map(lambda i: names[i])
            # match the official TSV column order (14 cols, no header)
            cols = [
                "id", "label", "statement", "subject", "speaker", "job_title",
                "state_info", "party_affiliation", "barely_true_counts",
                "false_counts", "half_true_counts", "mostly_true_counts",
                "pants_on_fire_counts", "context",
            ]
            df = df.reindex(columns=cols)
            df.to_csv(target / fname, sep="\t", header=False, index=False)
        print("LIAR ok (hf mirror)")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Could not download LIAR from any source: {last_err} / {exc}")


def download_fakenewsnet() -> None:
    target = RAW / "fakenewsnet"
    target.mkdir(exist_ok=True)
    for fname in FNN_FILES:
        path = target / fname
        if path.exists():
            print(f"FakeNewsNet {fname} already present")
            continue
        print(f"FakeNewsNet <- {FNN_BASE + fname}")
        path.write_bytes(fetch(FNN_BASE + fname))
    print("FakeNewsNet ok")


if __name__ == "__main__":
    download_liar()
    download_fakenewsnet()
    print("done ->", RAW)
    sys.exit(0)
