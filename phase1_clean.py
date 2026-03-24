"""
phase1_clean.py  —  Data Cleaning + Splitting
═══════════════════════════════════════════════
Phase 1 · Step 2

What this script does:
  MBTI
    1. Drop posts < 20 words (1 row found in EDA)
    2. Strip ||| separators, URLs, excess whitespace
    3. Stratified train / val / test split  (70 / 15 / 15)
    4. Compute class weights for CrossEntropyLoss
    5. Compute dichotomy weights for binary heads

  Essays / HEXACO
    1. Verify labels are in {0.0, 0.5, 1.0}
    2. Strip whitespace from text
    3. Remove texts < 30 words
    4. Flag Honesty-Humility as derived (avg of A+C)
    5. Standard train / val / test split  (70 / 15 / 15)

  Both
    6. Save cleaned CSVs to  data/clean/
    7. Save manifest.json   with counts, weights, stats

Run:
    python3 phase1_clean.py

Outputs:
    data/clean/mbti_train.csv
    data/clean/mbti_val.csv
    data/clean/mbti_test.csv
    data/clean/essays_train.csv
    data/clean/essays_val.csv
    data/clean/essays_test.csv
    data/clean/manifest.json
"""

import re
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR  = Path("data")
CLEAN_DIR = DATA_DIR / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

MBTI_PATH   = DATA_DIR / "mbti_500.csv"
ESSAYS_PATH = DATA_DIR / "essays_hexaco.csv"

MBTI_TYPES = [
    "INFJ","INFP","INTJ","INTP","ISFJ","ISFP","ISTJ","ISTP",
    "ENFJ","ENFP","ENTJ","ENTP","ESFJ","ESFP","ESTJ","ESTP",
]
MBTI2ID = {t: i for i, t in enumerate(MBTI_TYPES)}

HEXACO_DIMS = [
    "Honesty-Humility", "Emotionality", "Extraversion",
    "Agreeableness", "Conscientiousness", "Openness",
]
# H-H is derived from A+C — lower confidence label
DERIVED_DIMS = {"Honesty-Humility"}

SEP  = "═" * 62
SEP2 = "─" * 62


def section(title: str):
    log.info(f"\n  {SEP2}\n  {title}\n  {SEP2}")


# ══════════════════════════════════════════════════════════════
#  TEXT CLEANING HELPERS
# ══════════════════════════════════════════════════════════════

# URLs:  http://... or https://...
_URL_RE  = re.compile(r"https?://\S+", re.IGNORECASE)
# Leftover ||| separators from raw MBTI dump
_SEP_RE  = re.compile(r"\|{2,}")
# Runs of whitespace / newlines
_WS_RE   = re.compile(r"\s+")
# Repeated punctuation  (e.g. "!!!!!" → "!")
_REPUNC  = re.compile(r"([!?.]){3,}")


def clean_text(text: str) -> str:
    """Apply all text cleaning steps in order."""
    text = str(text)
    text = _URL_RE.sub(" ", text)     # remove URLs
    text = _SEP_RE.sub(" ", text)     # remove ||| separators
    text = _REPUNC.sub(r"\1", text)   # collapse repeated punctuation
    text = _WS_RE.sub(" ", text)      # normalise whitespace
    text = text.strip()
    return text


def word_count(text: str) -> int:
    return len(str(text).split())


# ══════════════════════════════════════════════════════════════
#  MBTI CLEANING
# ══════════════════════════════════════════════════════════════

def clean_mbti() -> dict:
    section("Cleaning MBTI")

    df = pd.read_csv(MBTI_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    log.info(f"  Loaded {len(df):,} rows")

    # ── 1. Type normalisation ────────────────────────────────
    df["type"] = df["type"].str.upper().str.strip()
    before = len(df)
    df = df[df["type"].isin(MBTI_TYPES)].copy()
    log.info(f"  Removed {before - len(df)} rows with unknown types")

    # ── 2. Clean text ────────────────────────────────────────
    df["posts"] = df["posts"].apply(clean_text)

    # ── 3. Drop short posts ──────────────────────────────────
    df["_wc"] = df["posts"].apply(word_count)
    short_mask = df["_wc"] < 20
    log.info(f"  Dropping {short_mask.sum()} posts < 20 words")
    df = df[~short_mask].copy()
    df.drop(columns=["_wc"], inplace=True)

    # ── 4. Add numeric label ─────────────────────────────────
    df["label_id"] = df["type"].map(MBTI2ID)

    # ── 5. Stratified split  70 / 15 / 15 ───────────────────
    train_df, tmp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["type"]
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, random_state=42, stratify=tmp_df["type"]
    )
    log.info(f"  Split → train:{len(train_df):,}  val:{len(val_df):,}  test:{len(test_df):,}")

    # Verify all 16 types in every split
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = set(MBTI_TYPES) - set(split["type"].unique())
        if missing:
            log.warning(f"  ⚠️  {name} missing types: {missing}")
        else:
            log.info(f"  ✅  {name}: all 16 types present")

    # ── 6. Class weights ─────────────────────────────────────
    classes = np.arange(len(MBTI_TYPES))
    cw = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label_id"].values,
    )
    class_weights = {MBTI_TYPES[i]: round(float(cw[i]), 4) for i in range(len(MBTI_TYPES))}
    log.info(f"  Class weights  min={min(cw):.3f}  max={max(cw):.3f}  ratio={max(cw)/min(cw):.1f}×")

    # ── 7. Dichotomy weights ──────────────────────────────────
    total = len(train_df)
    dich_weights = {}
    for axis, (a, b) in zip(["IE","NS","TF","JP"],
                            [("I","E"),("N","S"),("T","F"),("J","P")]):
        n_a = train_df["type"].str.contains(a).sum()
        n_b = total - n_a
        # weight = n_majority / n_minority  (for the minority class)
        w_a = round(total / (2 * n_a), 4) if n_a > 0 else 1.0
        w_b = round(total / (2 * n_b), 4) if n_b > 0 else 1.0
        dich_weights[axis] = {a: w_a, b: w_b}
        log.info(f"  {axis}: {a}={n_a:,}({w_a:.2f}×)  {b}={n_b:,}({w_b:.2f}×)")

    # ── 8. Save ───────────────────────────────────────────────
    train_df.to_csv(CLEAN_DIR / "mbti_train.csv", index=False)
    val_df.to_csv(  CLEAN_DIR / "mbti_val.csv",   index=False)
    test_df.to_csv( CLEAN_DIR / "mbti_test.csv",  index=False)
    log.info(f"  Saved → data/clean/mbti_{{train,val,test}}.csv")

    return {
        "total_rows": len(df),
        "train": len(train_df),
        "val":   len(val_df),
        "test":  len(test_df),
        "class_weights": class_weights,
        "dichotomy_weights": dich_weights,
        "type_counts_train": train_df["type"].value_counts().to_dict(),
    }


# ══════════════════════════════════════════════════════════════
#  ESSAYS CLEANING
# ══════════════════════════════════════════════════════════════

def clean_essays() -> dict:
    section("Cleaning Essays / HEXACO")

    df = pd.read_csv(ESSAYS_PATH)
    log.info(f"  Loaded {len(df):,} rows")
    log.info(f"  Columns: {df.columns.tolist()}")

    # ── 1. Clean text ────────────────────────────────────────
    df["text"] = df["text"].apply(clean_text)

    # ── 2. Drop short texts ──────────────────────────────────
    df["_wc"] = df["text"].apply(word_count)
    short_mask = df["_wc"] < 30
    log.info(f"  Dropping {short_mask.sum()} texts < 30 words")
    df = df[~short_mask].copy()
    df.drop(columns=["_wc"], inplace=True)

    # ── 3. Verify & report label values ──────────────────────
    label_stats = {}
    log.info("\n  Label verification:")
    for dim in HEXACO_DIMS:
        if dim not in df.columns:
            log.warning(f"  ⚠️  {dim} missing — filling with 0.5")
            df[dim] = 0.5
            continue

        col = pd.to_numeric(df[dim], errors="coerce")
        unique_vals = sorted(col.dropna().unique())
        out_of_range = ((col < 0) | (col > 1)).sum()
        null_count   = col.isna().sum()
        derived      = dim in DERIVED_DIMS

        if out_of_range > 0:
            log.warning(f"  ⚠️  {dim}: {out_of_range} values outside [0,1] — clipping")
            col = col.clip(0.0, 1.0)

        if null_count > 0:
            med = col.median()
            log.warning(f"  ⚠️  {dim}: {null_count} nulls — filling with median {med:.3f}")
            col = col.fillna(med)

        df[dim] = col.astype(float)

        pct_pos = (col >= 0.5).mean() * 100
        tag = " [DERIVED — lower confidence]" if derived else ""
        log.info(f"  {dim:<22}: mean={col.mean():.3f}  std={col.std():.3f}"
                 f"  pos%={pct_pos:.1f}{tag}")

        label_stats[dim] = {
            "mean":     round(float(col.mean()), 4),
            "std":      round(float(col.std()),  4),
            "pct_pos":  round(float(pct_pos),    2),
            "derived":  derived,
        }

    # ── 4. Standard split  70 / 15 / 15 ─────────────────────
    train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df,  test_df = train_test_split(tmp_df, test_size=0.50, random_state=42)
    log.info(f"\n  Split → train:{len(train_df):,}  val:{len(val_df):,}  test:{len(test_df):,}")

    # ── 5. Save ───────────────────────────────────────────────
    train_df.to_csv(CLEAN_DIR / "essays_train.csv", index=False)
    val_df.to_csv(  CLEAN_DIR / "essays_val.csv",   index=False)
    test_df.to_csv( CLEAN_DIR / "essays_test.csv",  index=False)
    log.info(f"  Saved → data/clean/essays_{{train,val,test}}.csv")

    return {
        "total_rows": len(df),
        "train": len(train_df),
        "val":   len(val_df),
        "test":  len(test_df),
        "label_stats": label_stats,
        "derived_dims": list(DERIVED_DIMS),
    }


# ══════════════════════════════════════════════════════════════
#  MANIFEST
# ══════════════════════════════════════════════════════════════

def save_manifest(mbti_meta: dict, essays_meta: dict):
    manifest = {
        "phase": 1,
        "step":  2,
        "description": "Cleaned + split datasets ready for Phase 2 training",
        "mbti": mbti_meta,
        "essays": essays_meta,
        "notes": [
            "MBTI: 47x class imbalance — use class_weights in CrossEntropyLoss",
            "MBTI: I/N heavily over-represented (77% I, 86% N)",
            "Essays: binary labels 0.0/1.0 — use BCELoss",
            "Essays: Honesty-Humility is DERIVED from avg(Agreeableness, Conscientiousness)",
            "Essays: apply lower loss_weight to Honesty-Humility during training",
        ],
    }
    path = CLEAN_DIR / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"\n  Manifest saved → {path}")
    return manifest


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print(f"\n{SEP}")
    print("  Phase 1 · Step 2  —  Cleaning + Splitting")
    print(SEP)

    mbti_meta   = clean_mbti()
    essays_meta = clean_essays()
    manifest    = save_manifest(mbti_meta, essays_meta)

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  DONE  —  data/clean/ contents")
    print(SEP)

    files = sorted(CLEAN_DIR.glob("*.csv"))
    for f in files:
        rows = sum(1 for _ in open(f)) - 1
        kb   = f.stat().st_size // 1024
        print(f"  {f.name:<30} {rows:>5,} rows  {kb:>5} KB")

    print(f"\n  manifest.json  saved with class weights + label stats")

    print(f"""
  Class weight range  (MBTI):
    min = {min(mbti_meta['class_weights'].values()):.3f}  ({max(mbti_meta['class_weights'], key=mbti_meta['class_weights'].get)})
    max = {max(mbti_meta['class_weights'].values()):.3f}  ({min(mbti_meta['class_weights'], key=mbti_meta['class_weights'].get)})

  ✅  Phase 1 Step 2 complete.
  Next:  python3 phase1_verify.py   (Step 3 — final check before Phase 2)
""")


if __name__ == "__main__":
    main()