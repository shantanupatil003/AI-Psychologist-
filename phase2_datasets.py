"""
phase2_datasets.py  —  PyTorch Dataset Classes
════════════════════════════════════════════════
Phase 2 · Step 1

Defines:
  MBTIDataset     — loads mbti_{split}.csv, tokenizes, returns tensors
  EssaysDataset   — loads essays_{split}.csv, tokenizes, returns tensors
  DatasetManifest — loads manifest.json, exposes class weights as tensors

Also runs a self-test when executed directly:
  python3 phase2_datasets.py

Self-test checks:
  - Both datasets load without error
  - Batch shapes are correct
  - Token IDs are in valid range
  - Labels are in expected range
  - Class weights tensor sums to ~16 (sanity)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer

CLEAN_DIR = Path("data/clean")

MBTI_TYPES = [
    "INFJ","INFP","INTJ","INTP","ISFJ","ISFP","ISTJ","ISTP",
    "ENFJ","ENFP","ENTJ","ENTP","ESFJ","ESFP","ESTJ","ESTP",
]

HEXACO_DIMS = [
    "Honesty-Humility", "Emotionality", "Extraversion",
    "Agreeableness", "Conscientiousness", "Openness",
]

SEP  = "═" * 62
SEP2 = "─" * 62


# ══════════════════════════════════════════════════════════════
#  MANIFEST LOADER
# ══════════════════════════════════════════════════════════════

class DatasetManifest:
    """
    Loads data/clean/manifest.json and exposes:
      .mbti_class_weights      torch.FloatTensor [16]
      .mbti_dichotomy_weights  dict  axis → {A: float, B: float}
      .hexaco_label_stats      dict  dim  → {mean, std, pct_pos, derived}
      .hexaco_dim_weights      torch.FloatTensor [6]  (lower for H-H)
    """

    HEXACO_DIMS = HEXACO_DIMS

    def __init__(self, path: str = "data/clean/manifest.json"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {path}. Run phase1_clean.py first."
            )
        with open(path) as f:
            self._data = json.load(f)

        self._build_mbti_weights()
        self._build_hexaco_weights()

    # ── MBTI ──────────────────────────────────────────────────

    def _build_mbti_weights(self):
        cw = self._data["mbti"]["class_weights"]
        # Ordered by MBTI_TYPES index
        weights = [cw[t] for t in MBTI_TYPES]
        self.mbti_class_weights = torch.tensor(weights, dtype=torch.float)
        self.mbti_dichotomy_weights = self._data["mbti"]["dichotomy_weights"]

    # ── HEXACO ─────────────────────────────────────────────────

    def _build_hexaco_weights(self):
        stats = self._data["essays"]["label_stats"]
        self.hexaco_label_stats = stats

        # H-H is derived (avg of A+C) → downweight to 0.5
        # All real dims → weight 1.0
        weights = []
        for dim in HEXACO_DIMS:
            derived = stats.get(dim, {}).get("derived", False)
            weights.append(0.5 if derived else 1.0)
        self.hexaco_dim_weights = torch.tensor(weights, dtype=torch.float)

    # ── Convenience ───────────────────────────────────────────

    def summary(self):
        print(f"\n  Manifest summary")
        print(f"  {'─'*40}")
        print(f"  MBTI class weights   min={self.mbti_class_weights.min():.3f}"
              f"  max={self.mbti_class_weights.max():.3f}")
        print(f"  HEXACO dim weights   {self.hexaco_dim_weights.tolist()}")
        print(f"  Derived dims         "
              f"{[d for d in HEXACO_DIMS if self.hexaco_label_stats.get(d,{}).get('derived')]}")


# ══════════════════════════════════════════════════════════════
#  MBTI DATASET
# ══════════════════════════════════════════════════════════════

class MBTIDataset(Dataset):
    """
    Loads mbti_{split}.csv and tokenizes each post.

    Returns dict:
        input_ids      LongTensor  [max_length]
        attention_mask LongTensor  [max_length]
        label          LongTensor  scalar   (0-15, MBTI type index)

    Args:
        split       : "train" | "val" | "test"
        tokenizer   : RobertaTokenizer instance
        max_length  : int  (default 256 — posts are ~1300 words,
                       256 tokens captures the opening well and
                       is 4× faster than 512)
        clean_dir   : path to data/clean/
    """

    def __init__(
        self,
        split: str,
        tokenizer: RobertaTokenizer,
        max_length: int = 256,
        clean_dir: Path = CLEAN_DIR,
    ):
        path = Path(clean_dir) / f"mbti_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}. Run phase1_clean.py first.")

        df = pd.read_csv(path)
        self.texts  = df["posts"].astype(str).tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

        # Sanity
        assert len(self.texts) == len(self.labels), "text/label length mismatch"
        assert min(self.labels) >= 0 and max(self.labels) <= 15, "label out of range"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def label_distribution(self) -> dict:
        """Return {type: count} for this split."""
        from collections import Counter
        id2type = {i: t for i, t in enumerate(MBTI_TYPES)}
        return dict(Counter(id2type[l] for l in self.labels))


# ══════════════════════════════════════════════════════════════
#  ESSAYS DATASET
# ══════════════════════════════════════════════════════════════

class EssaysDataset(Dataset):
    """
    Loads essays_{split}.csv and tokenizes each essay.

    Returns dict:
        input_ids      LongTensor  [max_length]
        attention_mask LongTensor  [max_length]
        labels         FloatTensor [6]   (one score per HEXACO dim, in [0,1])

    Args:
        split       : "train" | "val" | "test"
        tokenizer   : RobertaTokenizer instance
        max_length  : int  (default 256 — essays are ~650 words)
        clean_dir   : path to data/clean/
    """

    def __init__(
        self,
        split: str,
        tokenizer: RobertaTokenizer,
        max_length: int = 256,
        clean_dir: Path = CLEAN_DIR,
    ):
        path = Path(clean_dir) / f"essays_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}. Run phase1_clean.py first.")

        df = pd.read_csv(path)
        self.texts  = df["text"].astype(str).tolist()
        self.labels = df[HEXACO_DIMS].values.astype(np.float32)
        self.tokenizer  = tokenizer
        self.max_length = max_length

        # Sanity
        assert self.labels.shape[1] == 6, f"Expected 6 HEXACO dims, got {self.labels.shape[1]}"
        assert self.labels.min() >= 0.0 and self.labels.max() <= 1.0, "Labels out of [0,1]"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }

    def label_stats(self) -> dict:
        """Return mean and std per HEXACO dim for this split."""
        return {
            dim: {
                "mean": float(self.labels[:, i].mean()),
                "std":  float(self.labels[:, i].std()),
            }
            for i, dim in enumerate(HEXACO_DIMS)
        }


# ══════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════

def run_self_test():
    print(f"\n{SEP}")
    print("  Phase 2 · Step 1  —  Dataset self-test")
    print(SEP)

    passed = []
    failed = []

    def check(name, condition, detail=""):
        if condition:
            passed.append(name)
            print(f"  ✅  {name}")
        else:
            failed.append(name)
            msg = f"  ❌  {name}"
            if detail:
                msg += f"\n      → {detail}"
            print(msg)

    # ── Load manifest ─────────────────────────────────────────
    print(f"\n  {SEP2}\n  Manifest\n  {SEP2}")
    try:
        manifest = DatasetManifest()
        manifest.summary()
        check("Manifest loads", True)
        check("MBTI class weights shape",
              manifest.mbti_class_weights.shape == (16,))
        check("MBTI class weights all positive",
              (manifest.mbti_class_weights > 0).all().item())
        check("HEXACO dim weights shape",
              manifest.hexaco_dim_weights.shape == (6,))
        check("H-H weight < 1.0 (downweighted)",
              manifest.hexaco_dim_weights[0].item() < 1.0)
    except Exception as e:
        check("Manifest loads", False, str(e))

    # ── Load tokenizer ────────────────────────────────────────
    print(f"\n  {SEP2}\n  Tokenizer\n  {SEP2}")
    print("  Loading roberta-base tokenizer …")
    try:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        check("Tokenizer loads", True)
        vocab_size = tokenizer.vocab_size
        print(f"  Vocab size: {vocab_size:,}")
    except Exception as e:
        check("Tokenizer loads", False, str(e))
        print("  Cannot continue without tokenizer.")
        return

    # ── MBTI Dataset ──────────────────────────────────────────
    print(f"\n  {SEP2}\n  MBTIDataset\n  {SEP2}")
    for split in ["train", "val", "test"]:
        try:
            ds = MBTIDataset(split, tokenizer, max_length=64)
            check(f"MBTI {split} loads ({len(ds):,} rows)", True)

            # Check one item
            item = ds[0]
            check(f"MBTI {split}: input_ids shape",
                  item["input_ids"].shape == (64,),
                  f"got {item['input_ids'].shape}")
            check(f"MBTI {split}: attention_mask shape",
                  item["attention_mask"].shape == (64,),
                  f"got {item['attention_mask'].shape}")
            check(f"MBTI {split}: label is scalar long",
                  item["label"].shape == () and item["label"].dtype == torch.long)
            check(f"MBTI {split}: label in [0,15]",
                  0 <= item["label"].item() <= 15,
                  f"got {item['label'].item()}")
            check(f"MBTI {split}: token IDs in vocab range",
                  item["input_ids"].max().item() < vocab_size,
                  f"max token id {item['input_ids'].max().item()} >= {vocab_size}")

            # DataLoader batch test
            loader = DataLoader(ds, batch_size=4, shuffle=False)
            batch  = next(iter(loader))
            check(f"MBTI {split}: batch input_ids shape",
                  batch["input_ids"].shape == (4, 64),
                  f"got {batch['input_ids'].shape}")
            check(f"MBTI {split}: batch labels shape",
                  batch["label"].shape == (4,),
                  f"got {batch['label'].shape}")

        except Exception as e:
            check(f"MBTI {split} loads", False, str(e))

    # ── Essays Dataset ────────────────────────────────────────
    print(f"\n  {SEP2}\n  EssaysDataset\n  {SEP2}")
    for split in ["train", "val", "test"]:
        try:
            ds = EssaysDataset(split, tokenizer, max_length=64)
            check(f"Essays {split} loads ({len(ds):,} rows)", True)

            item = ds[0]
            check(f"Essays {split}: input_ids shape",
                  item["input_ids"].shape == (64,))
            check(f"Essays {split}: labels shape",
                  item["labels"].shape == (6,),
                  f"got {item['labels'].shape}")
            check(f"Essays {split}: labels dtype float",
                  item["labels"].dtype == torch.float)
            check(f"Essays {split}: labels in [0,1]",
                  (item["labels"] >= 0).all() and (item["labels"] <= 1).all(),
                  f"got {item['labels']}")

            loader = DataLoader(ds, batch_size=4, shuffle=False)
            batch  = next(iter(loader))
            check(f"Essays {split}: batch labels shape",
                  batch["labels"].shape == (4, 6),
                  f"got {batch['labels'].shape}")

        except Exception as e:
            check(f"Essays {split} loads", False, str(e))

    # ── Final result ──────────────────────────────────────────
    print(f"\n{SEP}")
    total = len(passed) + len(failed)
    print(f"  {len(passed)}/{total} checks passed")

    if failed:
        print(f"\n  ❌  Failed:")
        for f in failed:
            print(f"      · {f}")
    else:
        print("""
  ✅  All dataset checks passed.

  Next:  python3 phase2_model.py   (Step 2 — model + training loop)
""")


if __name__ == "__main__":
    run_self_test()