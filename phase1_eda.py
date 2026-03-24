"""
phase1_eda.py  —  Exploratory Data Analysis
═════════════════════════════════════════════
Phase 1 · Step 1

Loads both datasets and prints a full health report:
  - Row counts, column names
  - MBTI class distribution + imbalance ratio
  - Big Five / HEXACO label distributions
  - Text length stats (words, chars)
  - Null / empty row counts
  - Duplicate detection
  - Sample rows

Run:
    python3 phase1_eda.py

Expected files:
    data/mbti_500.csv       (cols: type, posts)
    data/essays_hexaco.csv  (cols: text, O, C, E, A, N  or HEXACO dims)
"""

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
MBTI_PATH   = DATA_DIR / "mbti_500.csv"
ESSAYS_PATH = DATA_DIR / "essays_hexaco.csv"

MBTI_TYPES = [
    "INFJ","INFP","INTJ","INTP","ISFJ","ISFP","ISTJ","ISTP",
    "ENFJ","ENFP","ENTJ","ENTP","ESFJ","ESFP","ESTJ","ESTP",
]

SEP  = "═" * 62
SEP2 = "─" * 62


def header(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def section(title: str):
    print(f"\n  {SEP2}")
    print(f"  {title}")
    print(f"  {SEP2}")


def load_mbti() -> pd.DataFrame:
    if not MBTI_PATH.exists():
        print(f"\n❌  {MBTI_PATH} not found.")
        print("    Run:  python3 data_download.py")
        sys.exit(1)
    df = pd.read_csv(MBTI_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_essays() -> pd.DataFrame:
    if not ESSAYS_PATH.exists():
        print(f"\n❌  {ESSAYS_PATH} not found.")
        print("    Run:  python3 data_download.py")
        sys.exit(1)
    df = pd.read_csv(ESSAYS_PATH)
    return df


# ── MBTI EDA ─────────────────────────────────────────────────

def eda_mbti(df: pd.DataFrame):
    header("MBTI DATASET")

    # Basic shape
    section("Basic info")
    print(f"  Rows     : {len(df):,}")
    print(f"  Columns  : {df.columns.tolist()}")
    print(f"  Dtypes   : {df.dtypes.to_dict()}")

    # Nulls
    section("Null / empty values")
    for col in df.columns:
        nulls = df[col].isna().sum()
        empty = (df[col].astype(str).str.strip() == "").sum()
        print(f"  {col:<12}: {nulls} nulls  |  {empty} empty strings")

    # Duplicates
    section("Duplicates")
    dup_posts = df["posts"].duplicated().sum()
    dup_rows  = df.duplicated().sum()
    print(f"  Duplicate posts : {dup_posts:,}")
    print(f"  Duplicate rows  : {dup_rows:,}")

    # Class distribution
    section("Class distribution  (16 MBTI types)")
    counts = df["type"].value_counts()
    total  = len(df)
    print(f"  {'Type':<6} {'Count':>6}  {'%':>6}  {'Bar'}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*30}")
    for t in MBTI_TYPES:
        n   = counts.get(t, 0)
        pct = n / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {t:<6} {n:>6,}  {pct:>5.1f}%  {bar}")
    imbalance = counts.max() / max(counts.min(), 1)
    print(f"\n  Max / Min ratio  : {imbalance:.1f}×  ", end="")
    if imbalance > 20:
        print("⚠️  severe imbalance — class weights essential")
    elif imbalance > 5:
        print("⚠️  moderate imbalance — class weights recommended")
    else:
        print("✅  acceptable balance")

    # Dichotomy balance
    section("Dichotomy balance  (I/E · N/S · T/F · J/P)")
    axes = [("I","E"), ("N","S"), ("T","F"), ("J","P")]
    for a, b in axes:
        n_a = df["type"].str.contains(a).sum()
        n_b = df["type"].str.contains(b).sum()
        pct_a = n_a / total * 100
        print(f"  {a}: {n_a:>5,} ({pct_a:.1f}%)   {b}: {n_b:>5,} ({100-pct_a:.1f}%)")

    # Text length
    section("Post length  (words)")
    df["_wc"] = df["posts"].astype(str).str.split().str.len()
    wc = df["_wc"]
    print(f"  Min    : {wc.min():,}")
    print(f"  Median : {wc.median():,.0f}")
    print(f"  Mean   : {wc.mean():,.0f}")
    print(f"  Max    : {wc.max():,}")
    print(f"  Std    : {wc.std():,.0f}")
    # Percentiles
    for p in [10, 25, 75, 90, 99]:
        print(f"  p{p:<3}   : {np.percentile(wc, p):,.0f}")
    # Very short posts
    short = (wc < 20).sum()
    print(f"\n  Posts < 20 words : {short:,}  {'⚠️  consider filtering' if short > 0 else '✅'}")

    # Sample rows
    section("Sample rows  (3 random)")
    sample = df.sample(3, random_state=42)
    for _, row in sample.iterrows():
        preview = str(row["posts"])[:120].replace("\n", " ")
        print(f"\n  Type : {row['type']}")
        print(f"  Text : {textwrap.fill(preview, width=58, subsequent_indent='         ')}")

    df.drop(columns=["_wc"], inplace=True)
    return df


# ── ESSAYS / HEXACO EDA ───────────────────────────────────────

def eda_essays(df: pd.DataFrame):
    header("ESSAYS / HEXACO DATASET")

    section("Basic info")
    print(f"  Rows     : {len(df):,}")
    print(f"  Columns  : {df.columns.tolist()}")

    # Detect text column
    text_col = next(
        (c for c in df.columns if c.lower() in ("text","essay","posts","content")),
        df.columns[0]
    )
    print(f"  Text col : {text_col!r}")

    # Detect trait columns
    trait_candidates = [c for c in df.columns if c.lower() not in
                        ("text","essay","posts","content","#authid","authid","ptype","__index_level_0__")]
    print(f"  Trait cols: {trait_candidates}")

    # Nulls
    section("Null / empty values")
    for col in df.columns:
        nulls = df[col].isna().sum()
        print(f"  {col:<25}: {nulls} nulls")

    # Duplicates
    section("Duplicates")
    dup = df[text_col].duplicated().sum()
    print(f"  Duplicate texts : {dup:,}")

    # Trait distributions
    section("Trait label distributions")
    for col in trait_candidates:
        raw = df[col]
        if raw.dtype == object:
            vc = raw.str.lower().value_counts()
            print(f"\n  {col}  (categorical)")
            for val, cnt in vc.items():
                print(f"    {val!r:>5} : {cnt:>5,}  ({cnt/len(df)*100:.1f}%)")
        else:
            numeric = pd.to_numeric(raw, errors="coerce")
            pct_pos = (numeric >= 0.5).mean() * 100
            print(f"\n  {col}  (numeric)")
            print(f"    min={numeric.min():.3f}  max={numeric.max():.3f}  "
                  f"mean={numeric.mean():.3f}  std={numeric.std():.3f}")
            print(f"    >= 0.5  (positive) : {pct_pos:.1f}%")
            if numeric.std() < 0.05:
                print(f"    ⚠️  very low variance — labels may be degenerate")
            else:
                print(f"    ✅  has real variance")

    # Text length
    section("Essay length  (words)")
    wc = df[text_col].astype(str).str.split().str.len()
    print(f"  Min    : {wc.min():,}")
    print(f"  Median : {wc.median():,.0f}")
    print(f"  Mean   : {wc.mean():,.0f}")
    print(f"  Max    : {wc.max():,}")
    for p in [10, 25, 75, 90]:
        print(f"  p{p:<3}   : {np.percentile(wc, p):,.0f}")
    short = (wc < 30).sum()
    print(f"\n  Texts < 30 words : {short:,}  {'⚠️' if short > 0 else '✅'}")

    # Sample rows
    section("Sample rows  (3 random)")
    sample = df.sample(3, random_state=42)
    for _, row in sample.iterrows():
        preview = str(row[text_col])[:150].replace("\n", " ")
        trait_vals = {c: row[c] for c in trait_candidates if c in row.index}
        print(f"\n  Traits : {trait_vals}")
        print(f"  Text   : {textwrap.fill(preview, width=58, subsequent_indent='         ')}")

    return text_col, trait_candidates


# ── CROSS-DATASET SUMMARY ─────────────────────────────────────

def summary(mbti_df, essays_text_col, essays_trait_cols, essays_df):
    header("SUMMARY  —  Phase 1 readiness check")

    checks = []

    # MBTI checks
    mbti_ok   = len(mbti_df) >= 1000
    mbti_cols = {"type", "posts"}.issubset(set(mbti_df.columns))
    mbti_null = mbti_df[["type","posts"]].isna().sum().sum() == 0
    counts    = mbti_df["type"].value_counts()
    mbti_all16 = len(counts) == 16
    checks.append(("MBTI row count ≥ 1000",      mbti_ok))
    checks.append(("MBTI has type + posts cols",  mbti_cols))
    checks.append(("MBTI no nulls in key cols",   mbti_null))
    checks.append(("MBTI all 16 types present",   mbti_all16))

    # Essays checks
    essays_ok    = len(essays_df) >= 500
    essays_text  = essays_text_col is not None
    essays_traits = len(essays_trait_cols) >= 4
    any_variance = any(
        pd.to_numeric(essays_df[c], errors="coerce").std() > 0.05
        for c in essays_trait_cols
        if pd.to_numeric(essays_df[c], errors="coerce").std() is not None
    )
    checks.append(("Essays row count ≥ 500",       essays_ok))
    checks.append(("Essays has text column",        essays_text))
    checks.append(("Essays has ≥ 4 trait columns",  essays_traits))
    checks.append(("Essays traits have variance",   any_variance))

    all_pass = all(v for _, v in checks)
    for name, passed in checks:
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {name}")

    print()
    if all_pass:
        print("  ✅ Both datasets are ready for Phase 1 Step 2  (cleaning)")
        print("\n  Next:  python3 phase1_clean.py")
    else:
        print("  ❌ Fix the issues above before proceeding.")
        print("     Re-run:  python3 data_download.py")


# ── MAIN ─────────────────────────────────────────────────────

def main():
    print("\n" + SEP)
    print("  Phase 1 · Step 1  —  Exploratory Data Analysis")
    print(SEP)

    mbti_df = load_mbti()
    mbti_df = eda_mbti(mbti_df)

    essays_df = load_essays()
    text_col, trait_cols = eda_essays(essays_df)

    summary(mbti_df, text_col, trait_cols, essays_df)


if __name__ == "__main__":
    main()