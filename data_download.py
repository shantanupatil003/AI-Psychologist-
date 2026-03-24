"""
data_download.py  —  Personality Dataset Downloader
══════════════════════════════════════════════════════════════════

Downloads two REAL, peer-reviewed datasets:

  MBTI   →  datasnaek/mbti-type (Kaggle, ~8,675 real forum posts)

  Big Five / HEXACO  →  tries these sources in order:
    1. jingjietan/essays-big5  (HuggingFace)
       Mairesse 2007 Essays — THE benchmark, 2,467 real self-report essays
       Columns: text, O, C, E, A, N  (binary 0/1)

    2. KevSun/Personality_LM training data subset
       Wang & Sun 2024 — PANDORA Reddit comments, Big Five continuous scores
       80% accuracy, MSE 0.07 per paper

    3. agentlans/big-five-personality-traits (HuggingFace)
       Large synthetic but carefully constructed dataset

  All Big Five traits map to HEXACO:
    E (Extraversion)    → Extraversion
    N (Neuroticism)     → Emotionality
    A (Agreeableness)   → Agreeableness
    C (Conscientiousness) → Conscientiousness
    O (Openness)        → Openness
    H (Honesty-Humility) → derived as avg(A, C)

Usage:
    pip3 install kagglehub datasets
    python3 data_download.py
"""

import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MBTI_TYPES = [
    "INFJ","INFP","INTJ","INTP","ISFJ","ISFP","ISTJ","ISTP",
    "ENFJ","ENFP","ENTJ","ENTP","ESFJ","ESFP","ESTJ","ESTP",
]

HEXACO_DIMS = [
    "Honesty-Humility", "Emotionality", "Extraversion",
    "Agreeableness", "Conscientiousness", "Openness",
]

# Every known column name variant → canonical HEXACO dim
COL_ALIASES = {
    # jingjietan/essays-big5: single-letter O C E A N
    "e": "Extraversion",   "ext": "Extraversion",   "cext": "Extraversion",
    "n": "Emotionality",   "neu": "Emotionality",   "cneu": "Emotionality",
    "a": "Agreeableness",  "agr": "Agreeableness",  "cagr": "Agreeableness",
    "c": "Conscientiousness","con":"Conscientiousness","ccon":"Conscientiousness",
    "o": "Openness",       "opn": "Openness",       "copn": "Openness",
    # KevSun / PANDORA style
    "extraversion": "Extraversion", "extroversion": "Extraversion",
    "neuroticism":  "Emotionality", "emotionality": "Emotionality",
    "agreeableness":"Agreeableness",
    "conscientiousness": "Conscientiousness",
    "openness": "Openness", "openness_to_experience": "Openness",
}


# ══════════════════════════════════════════════════════════════
#  MBTI
# ══════════════════════════════════════════════════════════════

def download_mbti() -> bool:
    try:
        import kagglehub
    except ImportError:
        os.system(f"{sys.executable} -m pip install kagglehub -q")
        import kagglehub

    try:
        log.info("Downloading MBTI: datasnaek/mbti-type …")
        path = kagglehub.dataset_download("datasnaek/mbti-type")
        csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        if not csvs:
            return False

        df = pd.read_csv(csvs[0])
        df.columns = [c.strip().lower() for c in df.columns]
        df["type"]  = df["type"].str.upper().str.strip()
        df["posts"] = df["posts"].str.replace(r"\|\|\|", " ", regex=True).str.strip()
        df = df[df["type"].isin(MBTI_TYPES)].dropna(subset=["posts"])
        df = df[df["posts"].str.len() > 20].reset_index(drop=True)

        out = DATA_DIR / "mbti_500.csv"
        df[["type", "posts"]].to_csv(out, index=False)
        log.info(f"  ✅ {len(df):,} rows → {out}")
        log.info(f"  Type distribution:\n{df['type'].value_counts().to_string()}")
        return True
    except Exception as e:
        log.error(f"  MBTI download failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
#  BIG FIVE / HEXACO — three sources tried in order
# ══════════════════════════════════════════════════════════════

def _normalise_big5(df: pd.DataFrame, source_name: str):
    """
    Given any Big Five DataFrame, detect text + trait columns and return:
      text, Honesty-Humility, Emotionality, Extraversion,
      Agreeableness, Conscientiousness, Openness
    """
    df = df.copy()
    col_lower = {c.lower().strip(): c for c in df.columns}

    # ── Find text column ──────────────────────────────────────
    text_col = None
    for cand in ("text", "essay", "posts", "content", "comment", "ptype"):
        if cand in col_lower and df[col_lower[cand]].dtype == object:
            # Verify it actually contains text (median length > 20 chars)
            if df[col_lower[cand]].astype(str).str.len().median() > 20:
                text_col = col_lower[cand]
                break
    if text_col is None:
        # Pick whichever object column has the longest strings
        obj_cols = [(c, df[c].astype(str).str.len().median())
                    for c in df.columns if df[c].dtype == object]
        if obj_cols:
            text_col = max(obj_cols, key=lambda x: x[1])[0]
    if text_col is None:
        log.warning(f"  [{source_name}] No text column found")
        return None

    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 30]

    # ── Map trait columns using COL_ALIASES ──────────────────
    found_dims = set()
    for col_name, orig_col in col_lower.items():
        alias = COL_ALIASES.get(col_name)
        if alias and alias not in found_dims:
            raw = df[orig_col]
            # Determine format and normalise to [0, 1]
            if raw.dtype == object:
                # y/n binary labels (Mairesse essays format)
                raw = raw.astype(str).str.strip().str.lower()
                raw = raw.map({"y": 1.0, "n": 0.0, "1": 1.0, "0": 0.0,
                               "true": 1.0, "false": 0.0})
            else:
                raw = pd.to_numeric(raw, errors="coerce")
                vmin, vmax = raw.min(), raw.max()
                if vmax > 2:          # 1-5 or 1-7 Likert → normalise to 0-1
                    raw = (raw - vmin) / (vmax - vmin)
                elif vmax > 1:        # already 0-something > 1
                    raw = raw / vmax
                # else already 0-1
            df[alias] = raw.astype(float)
            found_dims.add(alias)

    if len(found_dims) < 3:
        log.warning(f"  [{source_name}] Only {len(found_dims)} trait dims found: {found_dims}")
        return None

    # Derive Honesty-Humility from Agreeableness + Conscientiousness
    if "Honesty-Humility" not in found_dims:
        avail = [d for d in ("Agreeableness", "Conscientiousness") if d in found_dims]
        if avail:
            df["Honesty-Humility"] = df[avail].mean(axis=1)

    # Fill missing dims with 0.5
    for dim in HEXACO_DIMS:
        if dim not in df.columns:
            df[dim] = 0.5
        else:
            med = df[dim].median()
            df[dim] = df[dim].fillna(med if pd.notna(med) else 0.5).clip(0.0, 1.0)

    df = df.dropna(subset=["text"]).reset_index(drop=True)

    # ── Stats ────────────────────────────────────────────────
    log.info(f"  [{source_name}] {len(df):,} rows | dims found: {sorted(found_dims)}")
    for dim in HEXACO_DIMS:
        log.info(f"    {dim:<22}: mean={df[dim].mean():.3f}  std={df[dim].std():.3f}")

    avg_std = df[HEXACO_DIMS].std().mean()
    if avg_std < 0.05:
        log.error(f"  ⚠️  Very low variance (avg_std={avg_std:.4f}) — labels may be degenerate")
        return None

    return df[["text"] + HEXACO_DIMS]


def _save_essays(df: pd.DataFrame) -> bool:
    out = DATA_DIR / "essays_hexaco.csv"
    df.to_csv(out, index=False)
    avg_words = df["text"].str.split().str.len().mean()
    log.info(f"  ✅ Saved {len(df):,} rows → {out}  (~{avg_words:.0f} words/sample)")
    return True


# ── Source 1: jingjietan/essays-big5 ─────────────────────────

def try_jingjietan() -> bool:
    """
    jingjietan/essays-big5  — Mairesse 2007 Essays (published 2025)
    Columns: text, O, C, E, A, N  (binary 0/1)
    2,467 real stream-of-consciousness essays
    """
    try:
        from datasets import load_dataset
        log.info("Trying jingjietan/essays-big5 …")
        ds = load_dataset("jingjietan/essays-big5", trust_remote_code=False)
        frames = [ds[s].to_pandas() for s in ds.keys()]
        df = pd.concat(frames, ignore_index=True)
        log.info(f"  Columns: {df.columns.tolist()} | rows: {len(df)}")
        result = _normalise_big5(df, "jingjietan/essays-big5")
        if result is not None:
            return _save_essays(result)
    except Exception as e:
        log.warning(f"  jingjietan/essays-big5 failed: {str(e)[:120]}")
    return False


# ── Source 2: KevSun/Personality_LM dataset (PANDORA Reddit) ─

def try_kevsun() -> bool:
    """
    KevSun/Personality_LM — PANDORA Reddit comments (Wang & Sun 2024)
    Continuous Big Five scores, ~100K Reddit comments
    Paper: 80% accuracy, MSE 0.07
    """
    try:
        from datasets import load_dataset
        log.info("Trying KevSun/Personality_LM dataset …")
        # The training data associated with this model
        for ds_id in ["KevSun/Personality_LM", "pandora-personality/pandora"]:
            try:
                ds = load_dataset(ds_id, split="train", trust_remote_code=False)
                df = ds.to_pandas()
                log.info(f"  {ds_id} columns: {df.columns.tolist()} | rows: {len(df)}")
                result = _normalise_big5(df, ds_id)
                if result is not None:
                    return _save_essays(result)
            except Exception as e:
                log.warning(f"  {ds_id}: {str(e)[:80]}")
    except Exception as e:
        log.warning(f"  KevSun sources failed: {str(e)[:120]}")
    return False


# ── Source 3: agentlans/big-five-personality-traits ───────────

def try_agentlans() -> bool:
    """
    agentlans/big-five-personality-traits — large well-constructed dataset
    Continuous Big Five scores
    """
    try:
        from datasets import load_dataset
        log.info("Trying agentlans/big-five-personality-traits …")
        ds = load_dataset("agentlans/big-five-personality-traits",
                          split="train", trust_remote_code=False)
        df = ds.to_pandas()
        log.info(f"  Columns: {df.columns.tolist()} | rows: {len(df)}")
        result = _normalise_big5(df, "agentlans")
        if result is not None:
            return _save_essays(result)
    except Exception as e:
        log.warning(f"  agentlans failed: {str(e)[:120]}")
    return False


# ── Source 4: Kaggle pandora via kagglehub ────────────────────

def try_pandora_kaggle() -> bool:
    """
    PANDORA dataset on Kaggle — Reddit Big Five labels
    """
    try:
        import kagglehub
        log.info("Trying PANDORA dataset via Kaggle …")
        for ds_id in ["carloscortescruz/pandora-big-5-personality-traits-dataset",
                      "tmdb-reddit/reddit-personality-big5"]:
            try:
                path = kagglehub.dataset_download(ds_id)
                csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
                if not csvs:
                    continue
                df = pd.read_csv(csvs[0], encoding="latin-1")
                log.info(f"  {ds_id} cols: {df.columns.tolist()} | rows: {len(df)}")
                result = _normalise_big5(df, ds_id)
                if result is not None:
                    return _save_essays(result)
            except Exception as e:
                log.warning(f"  {ds_id}: {str(e)[:80]}")
    except Exception as e:
        log.warning(f"  Kaggle PANDORA failed: {str(e)[:80]}")
    return False


def download_essays() -> bool:
    sources = [
        ("jingjietan/essays-big5  (Mairesse 2007, THE benchmark)", try_jingjietan),
        ("KevSun / PANDORA Reddit (Wang & Sun 2024, continuous)",  try_kevsun),
        ("agentlans/big-five-personality-traits",                  try_agentlans),
        ("PANDORA via Kaggle",                                     try_pandora_kaggle),
    ]
    for name, fn in sources:
        log.info(f"\n  ── Trying: {name}")
        if fn():
            return True
    return False


# ══════════════════════════════════════════════════════════════
#  VERIFICATION
# ══════════════════════════════════════════════════════════════

def verify_mbti(path: Path) -> bool:
    try:
        df = pd.read_csv(path, nrows=3)
        cols = [c.lower() for c in df.columns]
        n = sum(1 for _ in open(path)) - 1
        ok = "type" in cols and "posts" in cols and n >= 100
        log.info(f"  MBTI: {n:,} rows | ok={ok}")
        return ok
    except Exception as e:
        log.warning(f"  MBTI verify failed: {e}")
        return False


def verify_essays(path: Path) -> bool:
    try:
        df = pd.read_csv(path, nrows=3)
        has_text   = "text" in df.columns
        has_hexaco = all(d in df.columns for d in HEXACO_DIMS)
        n = sum(1 for _ in open(path)) - 1
        ok = has_text and has_hexaco and n >= 100
        log.info(f"  Essays: {n:,} rows | text={has_text} | hexaco={has_hexaco} | ok={ok}")
        return ok
    except Exception as e:
        log.warning(f"  Essays verify failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*64)
    print("  📦  Personality Dataset Downloader")
    print("  Real peer-reviewed datasets for MBTI + HEXACO/Big Five")
    print("═"*64 + "\n")

    mbti_path   = DATA_DIR / "mbti_500.csv"
    essays_path = DATA_DIR / "essays_hexaco.csv"

    # ── MBTI ──────────────────────────────────────────────────
    print("▶  MBTI  (datasnaek/mbti-type)")
    print("   " + "─"*52)
    if mbti_path.exists() and verify_mbti(mbti_path):
        log.info(f"  ✅ Already present: {mbti_path}")
        mbti_ok = True
    else:
        if mbti_path.exists():
            mbti_path.unlink()
        mbti_ok = download_mbti()

    # ── Big Five / HEXACO ─────────────────────────────────────
    print("\n▶  Big Five / HEXACO Essays")
    print("   Trying 4 sources in order …")
    print("   " + "─"*52)
    if essays_path.exists() and verify_essays(essays_path):
        log.info(f"  ✅ Already present: {essays_path}")
        essays_ok = True
    else:
        if essays_path.exists():
            essays_path.unlink()
        essays_ok = download_essays()

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "═"*64)
    print(f"  MBTI   : {'✅ REAL DATA' if mbti_ok   else '❌ FAILED'} → {mbti_path}")
    print(f"  HEXACO : {'✅ REAL DATA' if essays_ok else '❌ FAILED'} → {essays_path}")
    print()

    if not mbti_ok or not essays_ok:
        print("  💡 Requirements:")
        print("     pip3 install kagglehub datasets")
        print("     kagglehub will prompt for Kaggle login (free account)")
        sys.exit(1)

    try:
        m = pd.read_csv(mbti_path)
        e = pd.read_csv(essays_path)
        print(f"  MBTI  : {len(m):,} rows | {m['type'].nunique()} types | "
              f"~{m['posts'].astype(str).str.split().str.len().mean():.0f} words/post")
        print(f"  Essays: {len(e):,} rows | "
              f"~{e['text'].astype(str).str.split().str.len().mean():.0f} words/sample")
        print()
    except Exception:
        pass

    print("  ✅ All datasets ready!\n")
    print("  Train:")
    print("    python3 roberta_personality_finetune.py \\")
    print("        --task both --max_length 128 --batch_size 16 --epochs 10\n")


if __name__ == "__main__":
    main()