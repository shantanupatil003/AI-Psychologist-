"""
phase2_evaluate.py  —  Test Set Evaluation
═══════════════════════════════════════════
Phase 2 · Step 3

Loads the best checkpoint from outputs/phase2/best_model/
and evaluates on both test sets.

Reports:
  MBTI
    · Overall accuracy + macro F1 + weighted F1
    · Per-type precision / recall / F1
    · Confusion matrix (16×16)
    · Top-3 accuracy (is correct type in top 3 predictions?)

  HEXACO
    · Per-dimension accuracy, F1, precision, recall
    · Confusion matrix per dim (2×2)
    · Overall accuracy

  Saves:
    outputs/phase2/evaluation_results.json
    outputs/phase2/confusion_mbti.txt
    outputs/phase2/confusion_hexaco.txt

Run:
    python3 phase2_evaluate.py
"""

import json
import sys
import warnings
from pathlib import Path

import urllib3
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
warnings.filterwarnings("ignore", message=".*Some weights.*")
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

sys.path.insert(0, str(Path(__file__).parent))
from phase2_datasets import DatasetManifest, EssaysDataset, MBTIDataset, HEXACO_DIMS, MBTI_TYPES
from phase2_model import PersonalityClassifier

CKPT_DIR   = Path("outputs/phase2/best_model")
OUTPUT_DIR = Path("outputs/phase2")
SEP  = "═" * 62
SEP2 = "─" * 62


# ══════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════

def load_model(device):
    if not CKPT_DIR.exists():
        print(f"❌  Checkpoint not found at {CKPT_DIR}")
        print("   Run phase2_model.py first.")
        sys.exit(1)

    meta_path = CKPT_DIR / "checkpoint_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  Checkpoint val score : {meta['val_score']}")
    print(f"  Trained with args    : {meta['args']}")

    manifest  = DatasetManifest()
    tokenizer = RobertaTokenizer.from_pretrained(str(CKPT_DIR))
    model     = PersonalityClassifier(
        model_name         = str(CKPT_DIR),
        mbti_class_weights = manifest.mbti_class_weights.to(device),
        hexaco_dim_weights = manifest.hexaco_dim_weights.to(device),
    )
    state = torch.load(CKPT_DIR / "model_state.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  Model loaded  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model, tokenizer, meta


# ══════════════════════════════════════════════════════════════
#  MBTI EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_mbti(model, tokenizer, device, max_length, batch_size):
    print(f"\n  {SEP2}\n  MBTI Test Evaluation\n  {SEP2}")

    ds     = MBTIDataset("test", tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits = []
    all_labels = []

    for batch in tqdm(loader, desc="  MBTI test", dynamic_ncols=True, colour="cyan"):
        batch  = {k: v.to(device) for k, v in batch.items()}
        out    = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        all_logits.append(out["mbti_logits"].cpu().float().numpy())
        all_labels.append(batch["label"].cpu().numpy())

    logits = np.concatenate(all_logits,  axis=0)  # [N, 16]
    labels = np.concatenate(all_labels,  axis=0)  # [N]
    preds  = logits.argmax(axis=-1)               # [N]
    probs  = _softmax(logits)                     # [N, 16]

    # ── Metrics ───────────────────────────────────────────────
    acc        = accuracy_score(labels, preds)
    f1_macro   = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_weighted= f1_score(labels, preds, average="weighted", zero_division=0)

    # Top-3 accuracy
    top3_correct = sum(
        labels[i] in probs[i].argsort()[-3:]
        for i in range(len(labels))
    )
    top3_acc = top3_correct / len(labels)

    # Per-type report
    report = classification_report(
        labels, preds,
        target_names=MBTI_TYPES,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(16)))

    # Print
    print(f"\n  Overall")
    print(f"  {'─'*40}")
    print(f"  Accuracy          : {acc:.1%}")
    print(f"  Top-3 Accuracy    : {top3_acc:.1%}")
    print(f"  F1 macro          : {f1_macro:.4f}")
    print(f"  F1 weighted       : {f1_weighted:.4f}")
    print(f"  Random baseline   : 6.25%")
    print(f"  Weighted baseline : ~21%  (predict majority)")

    print(f"\n  Per-type  (sorted by F1 desc)")
    print(f"  {'─'*58}")
    print(f"  {'Type':<6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
    per_type = []
    for t in MBTI_TYPES:
        r = report[t]
        per_type.append((t, r["precision"], r["recall"], r["f1-score"], int(r["support"])))
    per_type.sort(key=lambda x: x[3], reverse=True)
    for t, p, r, f, s in per_type:
        bar = "█" * int(f * 10)
        print(f"  {t:<6} {p:>6.3f} {r:>6.3f} {f:>6.3f} {s:>8}  {bar}")

    return {
        "accuracy":   round(acc, 4),
        "top3_acc":   round(top3_acc, 4),
        "f1_macro":   round(f1_macro, 4),
        "f1_weighted":round(f1_weighted, 4),
        "per_type":   {t: {"precision": round(p,4), "recall": round(r,4),
                           "f1": round(f,4), "support": s}
                       for t,p,r,f,s in per_type},
        "confusion_matrix": cm.tolist(),
    }, cm


# ══════════════════════════════════════════════════════════════
#  HEXACO EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_hexaco(model, tokenizer, device, max_length, batch_size):
    print(f"\n  {SEP2}\n  HEXACO Test Evaluation\n  {SEP2}")

    ds     = EssaysDataset("test", tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds  = []
    all_labels = []

    for batch in tqdm(loader, desc="  HEXACO test", dynamic_ncols=True, colour="cyan"):
        batch = {k: v.to(device) for k, v in batch.items()}
        out   = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        all_preds.append(out["hexaco_scores"].cpu().float().numpy())
        all_labels.append(batch["labels"].cpu().float().numpy())

    preds_f  = np.concatenate(all_preds,  axis=0)  # [N, 6] float
    labels_f = np.concatenate(all_labels, axis=0)  # [N, 6] float

    # Binary threshold at 0.5
    preds_b  = (preds_f  >= 0.5).astype(int)
    labels_b = (labels_f >= 0.5).astype(int)

    results = {}
    print(f"\n  {'Dimension':<22} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}  {'Bar'}")
    print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*6} {'─'*6}  {'─'*20}")

    dim_accs = []
    for i, dim in enumerate(HEXACO_DIMS):
        y_true = labels_b[:, i]
        y_pred = preds_b[:, i]
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        cm   = confusion_matrix(y_true, y_pred, labels=[0,1])
        bar  = "█" * int(acc * 20)
        tag  = " [derived]" if dim == "Honesty-Humility" else ""
        print(f"  {dim:<22} {acc:>6.1%} {f1:>6.3f} {prec:>6.3f} {rec:>6.3f}  {bar}{tag}")
        dim_accs.append(acc)
        results[dim] = {
            "accuracy":  round(acc, 4),
            "f1":        round(f1,  4),
            "precision": round(prec,4),
            "recall":    round(rec, 4),
            "confusion_matrix": cm.tolist(),
        }

    avg_acc = float(np.mean(dim_accs))
    print(f"\n  Average accuracy  : {avg_acc:.1%}")
    print(f"  Random baseline   : 50.0%")
    results["avg_accuracy"] = round(avg_acc, 4)
    return results


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def save_confusion_matrix(cm: np.ndarray, path: Path):
    lines = ["MBTI Confusion Matrix (rows=true, cols=pred)\n"]
    header = "      " + "".join(f"{t:>5}" for t in MBTI_TYPES)
    lines.append(header)
    lines.append("─" * len(header))
    for i, t in enumerate(MBTI_TYPES):
        row = f"{t:<6}" + "".join(f"{cm[i,j]:>5}" for j in range(16))
        lines.append(row)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Confusion matrix → {path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print(f"\n{SEP}")
    print("  Phase 2 · Step 3  —  Test Set Evaluation")
    print(SEP)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    model, tokenizer, meta = load_model(device)
    args = meta["args"]
    max_length = args.get("max_length", 256)
    batch_size = args.get("batch_size", 16)

    mbti_results, cm = evaluate_mbti(model, tokenizer, device, max_length, batch_size)
    hexaco_results   = evaluate_hexaco(model, tokenizer, device, max_length, batch_size)

    # Save results
    results = {
        "checkpoint": str(CKPT_DIR),
        "val_score":  meta["val_score"],
        "mbti":       mbti_results,
        "hexaco":     hexaco_results,
    }
    out_path = OUTPUT_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {out_path}")

    save_confusion_matrix(cm, OUTPUT_DIR / "confusion_mbti.txt")

    # Final summary
    print(f"\n{SEP}")
    print("  Phase 2 complete  —  summary")
    print(SEP)
    print(f"  MBTI accuracy      : {mbti_results['accuracy']:.1%}   (random 6.25%,  majority 21%)")
    print(f"  MBTI top-3 acc     : {mbti_results['top3_acc']:.1%}")
    print(f"  MBTI F1 macro      : {mbti_results['f1_macro']:.4f}")
    print(f"  HEXACO avg acc     : {hexaco_results['avg_accuracy']:.1%}  (random 50%)")
    print()

    mbti_ok   = mbti_results["accuracy"]   > 0.25
    hexaco_ok = hexaco_results["avg_accuracy"] > 0.55

    print(f"  {'✅' if mbti_ok   else '⚠️ '} MBTI    : {'above' if mbti_ok   else 'below'} minimum threshold (25%)")
    print(f"  {'✅' if hexaco_ok else '⚠️ '} HEXACO  : {'above' if hexaco_ok else 'below'} minimum threshold (55%)")

    if mbti_ok and hexaco_ok:
        print(f"""
  ✅  Classifier is ready for Phase 3.

  ╔══════════════════════════════════════════╗
  ║  Ready to start Phase 3                  ║
  ║  Profile store + Bayesian updates        ║
  ╚══════════════════════════════════════════╝
""")
    else:
        print("\n  Consider more training epochs or data augmentation before Phase 3.")


if __name__ == "__main__":
    main()