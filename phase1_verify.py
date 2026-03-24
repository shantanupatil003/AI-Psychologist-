"""
phase1_verify.py  —  Final Data Verification
══════════════════════════════════════════════
Phase 1 · Step 3  (final step)

Loads every clean CSV and manifest, then runs
hard assertions. If anything fails, it tells you
exactly what went wrong before we write a single
line of model code.

Checks:
  1.  All 6 expected files exist
  2.  No data leakage between train / val / test
  3.  All 16 MBTI types in every split
  4.  Label IDs match type strings (no off-by-one)
  5.  Class weights in manifest cover all 16 types
  6.  Essays: labels are in [0, 1], no nulls
  7.  Essays: train/val/test label distributions consistent
  8.  Text column is non-empty in every split
  9.  Manifest file is readable and complete
 10.  Row count sanity (train > val ≈ test)

Run:
    python3 phase1_verify.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_DIR = Path("data/clean")
MANIFEST  = CLEAN_DIR / "manifest.json"

MBTI_TYPES = [
    "INFJ","INFP","INTJ","INTP","ISFJ","ISFP","ISTJ","ISTP",
    "ENFJ","ENFP","ENTJ","ENTP","ESFJ","ESFP","ESTJ","ESTP",
]
MBTI2ID = {t: i for i, t in enumerate(MBTI_TYPES)}

HEXACO_DIMS = [
    "Honesty-Humility", "Emotionality", "Extraversion",
    "Agreeableness", "Conscientiousness", "Openness",
]

SEP  = "═" * 62
SEP2 = "─" * 62

# ── Tiny assertion helper ─────────────────────────────────────
passed = []
failed = []

def check(name: str, condition: bool, detail: str = ""):
    if condition:
        passed.append(name)
        print(f"  ✅  {name}")
    else:
        failed.append(name)
        msg = f"  ❌  {name}"
        if detail:
            msg += f"\n      → {detail}"
        print(msg)


# ══════════════════════════════════════════════════════════════
def main():
    print(f"\n{SEP}")
    print("  Phase 1 · Step 3  —  Verification")
    print(SEP)

    # ── 1. Files exist ────────────────────────────────────────
    print(f"\n  {SEP2}\n  1 · File existence\n  {SEP2}")
    expected_files = [
        "mbti_train.csv", "mbti_val.csv", "mbti_test.csv",
        "essays_train.csv", "essays_val.csv", "essays_test.csv",
        "manifest.json",
    ]
    for fname in expected_files:
        path = CLEAN_DIR / fname
        check(f"{fname} exists", path.exists(), f"not found at {path}")

    if any(f not in [p.name for p in CLEAN_DIR.iterdir()] for f in expected_files):
        print("\n  Cannot continue — missing files. Run phase1_clean.py first.")
        sys.exit(1)

    # ── Load all splits ───────────────────────────────────────
    mbti_train  = pd.read_csv(CLEAN_DIR / "mbti_train.csv")
    mbti_val    = pd.read_csv(CLEAN_DIR / "mbti_val.csv")
    mbti_test   = pd.read_csv(CLEAN_DIR / "mbti_test.csv")
    ess_train   = pd.read_csv(CLEAN_DIR / "essays_train.csv")
    ess_val     = pd.read_csv(CLEAN_DIR / "essays_val.csv")
    ess_test    = pd.read_csv(CLEAN_DIR / "essays_test.csv")
    with open(MANIFEST) as f:
        manifest = json.load(f)

    # ── 2. No data leakage ────────────────────────────────────
    print(f"\n  {SEP2}\n  2 · No leakage between splits\n  {SEP2}")

    mbti_train_posts = set(mbti_train["posts"])
    mbti_val_posts   = set(mbti_val["posts"])
    mbti_test_posts  = set(mbti_test["posts"])
    check("MBTI train ∩ val = ∅",
          len(mbti_train_posts & mbti_val_posts) == 0,
          f"{len(mbti_train_posts & mbti_val_posts)} shared posts")
    check("MBTI train ∩ test = ∅",
          len(mbti_train_posts & mbti_test_posts) == 0,
          f"{len(mbti_train_posts & mbti_test_posts)} shared posts")
    check("MBTI val ∩ test = ∅",
          len(mbti_val_posts & mbti_test_posts) == 0,
          f"{len(mbti_val_posts & mbti_test_posts)} shared posts")

    ess_train_texts = set(ess_train["text"])
    ess_val_texts   = set(ess_val["text"])
    ess_test_texts  = set(ess_test["text"])
    check("Essays train ∩ val = ∅",
          len(ess_train_texts & ess_val_texts) == 0,
          f"{len(ess_train_texts & ess_val_texts)} shared texts")
    check("Essays train ∩ test = ∅",
          len(ess_train_texts & ess_test_texts) == 0,
          f"{len(ess_train_texts & ess_test_texts)} shared texts")

    # ── 3. All 16 MBTI types in every split ───────────────────
    print(f"\n  {SEP2}\n  3 · MBTI type coverage\n  {SEP2}")
    for name, df in [("train", mbti_train), ("val", mbti_val), ("test", mbti_test)]:
        present = set(df["type"].unique())
        missing = set(MBTI_TYPES) - present
        check(f"MBTI {name}: all 16 types present",
              len(missing) == 0,
              f"missing: {missing}")

    # ── 4. Label IDs consistent ───────────────────────────────
    print(f"\n  {SEP2}\n  4 · Label ID consistency\n  {SEP2}")
    for name, df in [("train", mbti_train), ("val", mbti_val), ("test", mbti_test)]:
        mismatches = (df["type"].map(MBTI2ID) != df["label_id"]).sum()
        check(f"MBTI {name}: label_id matches type",
              mismatches == 0,
              f"{mismatches} mismatches")

    # ── 5. Class weights in manifest ──────────────────────────
    print(f"\n  {SEP2}\n  5 · Manifest class weights\n  {SEP2}")
    cw = manifest.get("mbti", {}).get("class_weights", {})
    check("manifest has class_weights for all 16 types",
          set(cw.keys()) == set(MBTI_TYPES),
          f"missing: {set(MBTI_TYPES) - set(cw.keys())}")
    check("all class weights > 0",
          all(v > 0 for v in cw.values()),
          f"zero/negative weights found")
    min_w = min(cw.values())
    max_w = max(cw.values())
    check("class weight range is reasonable (max/min < 100×)",
          max_w / min_w < 100,
          f"ratio = {max_w/min_w:.1f}×")
    print(f"      min={min_w:.3f}  max={max_w:.3f}  ratio={max_w/min_w:.1f}×")

    # ── 6. Essays labels in [0, 1], no nulls ─────────────────
    print(f"\n  {SEP2}\n  6 · Essays label validity\n  {SEP2}")
    for name, df in [("train", ess_train), ("val", ess_val), ("test", ess_test)]:
        for dim in HEXACO_DIMS:
            col = df[dim]
            nulls     = col.isna().sum()
            out_range = ((col < 0) | (col > 1)).sum()
            check(f"Essays {name} · {dim}: no nulls",
                  nulls == 0, f"{nulls} nulls")
            check(f"Essays {name} · {dim}: values in [0,1]",
                  out_range == 0, f"{out_range} out-of-range")

    # ── 7. Label distribution consistent across splits ────────
    print(f"\n  {SEP2}\n  7 · Essays label distribution consistency\n  {SEP2}")
    for dim in HEXACO_DIMS:
        means = {
            "train": ess_train[dim].mean(),
            "val":   ess_val[dim].mean(),
            "test":  ess_test[dim].mean(),
        }
        # Allow up to 15% deviation between splits
        mean_vals = list(means.values())
        max_dev = max(mean_vals) - min(mean_vals)
        check(f"Essays {dim}: split means within 15%",
              max_dev < 0.15,
              f"train={means['train']:.3f} val={means['val']:.3f} "
              f"test={means['test']:.3f} (spread={max_dev:.3f})")

    # ── 8. Non-empty text columns ─────────────────────────────
    print(f"\n  {SEP2}\n  8 · Non-empty text\n  {SEP2}")
    for name, df in [("train", mbti_train), ("val", mbti_val), ("test", mbti_test)]:
        empty = (df["posts"].astype(str).str.strip() == "").sum()
        check(f"MBTI {name}: no empty posts", empty == 0, f"{empty} empty")
    for name, df in [("train", ess_train), ("val", ess_val), ("test", ess_test)]:
        empty = (df["text"].astype(str).str.strip() == "").sum()
        check(f"Essays {name}: no empty texts", empty == 0, f"{empty} empty")

    # ── 9. Manifest completeness ──────────────────────────────
    print(f"\n  {SEP2}\n  9 · Manifest completeness\n  {SEP2}")
    required_keys = ["mbti", "essays", "notes"]
    for key in required_keys:
        check(f"manifest has '{key}' key", key in manifest)
    check("manifest has dichotomy_weights",
          "dichotomy_weights" in manifest.get("mbti", {}))
    check("manifest has label_stats",
          "label_stats" in manifest.get("essays", {}))
    check("H-H flagged as derived in label_stats",
          manifest.get("essays", {})
                  .get("label_stats", {})
                  .get("Honesty-Humility", {})
                  .get("derived", False))

    # ── 10. Row count sanity ──────────────────────────────────
    print(f"\n  {SEP2}\n  10 · Row count sanity\n  {SEP2}")
    check("MBTI train > val",  len(mbti_train) > len(mbti_val))
    check("MBTI val ≈ test",   abs(len(mbti_val) - len(mbti_test)) <= 10)
    check("Essays train > val", len(ess_train) > len(ess_val))
    check("Essays val ≈ test",  abs(len(ess_val) - len(ess_test)) <= 10)
    print(f"\n  MBTI   train={len(mbti_train):,}  val={len(mbti_val):,}  test={len(mbti_test):,}")
    print(f"  Essays train={len(ess_train):,}   val={len(ess_val):,}  test={len(ess_test):,}")

    # ── Final result ──────────────────────────────────────────
    print(f"\n{SEP}")
    total = len(passed) + len(failed)
    print(f"  {len(passed)}/{total} checks passed")
    if failed:
        print(f"\n  ❌  Failed checks:")
        for f in failed:
            print(f"      · {f}")
        print(f"\n  Fix the issues above before starting Phase 2.")
        sys.exit(1)
    else:
        print(f"""
  ✅  All checks passed. Phase 1 is complete.

  Clean files   →  data/clean/
  Manifest      →  data/clean/manifest.json

  ╔══════════════════════════════════════════╗
  ║  Ready to start Phase 2                  ║
  ║  Personality classifier  (RoBERTa)       ║
  ╚══════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()