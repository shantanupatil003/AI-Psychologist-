"""
phase2_model.py  —  Model + Training Loop
══════════════════════════════════════════
Phase 2 · Step 2

Defines:
  PersonalityClassifier   two-head RoBERTa model
    · MBTI head   → 16-class CrossEntropy (weighted)
    · HEXACO head → 6-dim BCE  (dim-weighted)

  Trainer                 training loop with:
    · MPS / CUDA / CPU auto-detection
    · bfloat16 autocast on MPS
    · progress bar per epoch
    · val loss + accuracy logged each epoch
    · early stopping (patience 3)
    · best checkpoint saved to outputs/phase2/best_model/

Run:
    python3 phase2_model.py

    # faster smoke-test (2 epochs, small batch):
    python3 phase2_model.py --epochs 2 --batch_size 8 --max_length 128
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import urllib3
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
warnings.filterwarnings("ignore", message=".*Some weights.*")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup

# Import our dataset classes from Step 1
sys.path.insert(0, str(Path(__file__).parent))
from phase2_datasets import DatasetManifest, EssaysDataset, MBTIDataset, HEXACO_DIMS, MBTI_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/phase2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEP  = "═" * 62
SEP2 = "─" * 62


# ══════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════

class PersonalityClassifier(nn.Module):
    """
    Shared RoBERTa encoder with two task heads.

    MBTI head:   Linear(768 → 256) → GELU → Dropout → Linear(256 → 16)
    HEXACO head: Linear(768 → 256) → GELU → Dropout → Linear(256 → 6) → Sigmoid

    Forward returns a dict with all outputs and the combined loss
    when labels are supplied.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.1,
        mbti_loss_weight: float = 0.5,
        hexaco_loss_weight: float = 0.5,
        mbti_class_weights: torch.Tensor = None,
        hexaco_dim_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.mbti_loss_weight   = mbti_loss_weight
        self.hexaco_loss_weight = hexaco_loss_weight

        # ── Encoder ───────────────────────────────────────────
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden = self.roberta.config.hidden_size  # 768

        # ── MBTI head ─────────────────────────────────────────
        self.mbti_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 16),
        )

        # ── HEXACO head ───────────────────────────────────────
        self.hexaco_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 6),
            nn.Sigmoid(),
        )

        # ── Loss functions ─────────────────────────────────────
        # CrossEntropy with class weights loaded from manifest
        self.mbti_loss_fn = nn.CrossEntropyLoss(
            weight=mbti_class_weights  # None = unweighted (overridden in Trainer)
        )
        # BCE: binary labels (0/1) per HEXACO dim
        # Dim weights applied manually in forward()
        self.hexaco_loss_fn   = nn.BCELoss(reduction="none")
        self.hexaco_dim_weights = hexaco_dim_weights  # [6] or None

    def encode(self, input_ids, attention_mask):
        """Return [CLS] token representation."""
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]   # [batch, 768]

    def forward(
        self,
        input_ids,
        attention_mask,
        mbti_label=None,     # LongTensor [batch]
        hexaco_labels=None,  # FloatTensor [batch, 6]
    ):
        cls = self.encode(input_ids, attention_mask)  # [batch, 768]

        results = {}
        # Always accumulate loss in float32 regardless of autocast dtype
        total_loss = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)

        # ── MBTI branch ───────────────────────────────────────
        mbti_logits = self.mbti_head(cls).float()   # cast to float32 for CrossEntropy
        results["mbti_logits"] = mbti_logits

        if mbti_label is not None:
            loss_mbti = self.mbti_loss_fn(mbti_logits, mbti_label)
            total_loss = total_loss + self.mbti_loss_weight * loss_mbti
            results["mbti_loss"] = loss_mbti

        # ── HEXACO branch ─────────────────────────────────────
        # Cast scores back to float32 before BCE — MPS bfloat16 autocast
        # produces bf16 activations but labels are float32; MPS requires
        # identical dtypes for all operands in arithmetic ops.
        hexaco_scores = self.hexaco_head(cls).float()   # [batch, 6] float32
        results["hexaco_scores"] = hexaco_scores

        if hexaco_labels is not None:
            labels_f32 = hexaco_labels.float()           # ensure float32
            bce = self.hexaco_loss_fn(hexaco_scores, labels_f32)
            if self.hexaco_dim_weights is not None:
                dw = self.hexaco_dim_weights.to(bce.device).float()
                bce = bce * dw.unsqueeze(0)
            loss_hexaco = bce.mean()
            total_loss  = total_loss + self.hexaco_loss_weight * loss_hexaco
            results["hexaco_loss"] = loss_hexaco

        results["loss"] = total_loss
        return results


# ══════════════════════════════════════════════════════════════
#  TRAINER
# ══════════════════════════════════════════════════════════════

class Trainer:

    def __init__(self, args):
        self.args = args
        self._setup_device()
        self._setup_tokenizer()
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()

    # ── Device ────────────────────────────────────────────────

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device    = torch.device("cuda")
            self.amp_dtype = torch.float16
            self.use_amp   = True
            self.use_scaler = True
        elif torch.backends.mps.is_available():
            self.device    = torch.device("mps")
            self.amp_dtype = torch.bfloat16
            self.use_amp   = True
            self.use_scaler = False
        else:
            self.device    = torch.device("cpu")
            self.amp_dtype = torch.bfloat16
            self.use_amp   = False
            self.use_scaler = False

        log.info(f"Device     : {self.device}")
        log.info(f"AMP        : {'on (' + str(self.amp_dtype) + ')' if self.use_amp else 'off'}")

    # ── Tokenizer ─────────────────────────────────────────────

    def _setup_tokenizer(self):
        log.info("Loading tokenizer …")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # ── Data ──────────────────────────────────────────────────

    def _setup_data(self):
        args = self.args
        log.info("Loading datasets …")
        self.manifest = DatasetManifest()

        pin = (self.device.type == "cuda")

        self.mbti_train_loader = DataLoader(
            MBTIDataset("train", self.tokenizer, args.max_length),
            batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=pin,
        )
        self.mbti_val_loader = DataLoader(
            MBTIDataset("val", self.tokenizer, args.max_length),
            batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=pin,
        )
        self.essays_train_loader = DataLoader(
            EssaysDataset("train", self.tokenizer, args.max_length),
            batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=pin,
        )
        self.essays_val_loader = DataLoader(
            EssaysDataset("val", self.tokenizer, args.max_length),
            batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=pin,
        )
        log.info(f"MBTI   batches/epoch: {len(self.mbti_train_loader):,}")
        log.info(f"Essays batches/epoch: {len(self.essays_train_loader):,}")

    # ── Model ─────────────────────────────────────────────────

    def _setup_model(self):
        log.info("Loading roberta-base …")
        self.model = PersonalityClassifier(
            model_name        = "roberta-base",
            dropout           = self.args.dropout,
            mbti_loss_weight  = self.args.mbti_weight,
            hexaco_loss_weight = self.args.hexaco_weight,
            mbti_class_weights = self.manifest.mbti_class_weights.to(self.device),
            hexaco_dim_weights = self.manifest.hexaco_dim_weights.to(self.device),
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(f"Trainable params: {n_params/1e6:.1f}M")

    # ── Optimizer + scheduler ─────────────────────────────────

    def _setup_optimizer(self):
        args   = self.args
        # Separate LR for encoder vs heads — heads can learn faster
        encoder_params = list(self.model.roberta.parameters())
        head_params    = (
            list(self.model.mbti_head.parameters()) +
            list(self.model.hexaco_head.parameters())
        )
        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": args.lr},
            {"params": head_params,    "lr": args.lr * 10},
        ], weight_decay=args.weight_decay)

        # Total steps across both tasks
        batches_per_epoch = len(self.mbti_train_loader) + len(self.essays_train_loader)
        total_steps  = batches_per_epoch * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        log.info(f"Total steps: {total_steps:,}  warmup: {warmup_steps:,}")

    # ── Training ──────────────────────────────────────────────

    def train(self):
        args = self.args
        best_val_score  = -np.inf
        patience_counter = 0
        history = []

        print(f"\n{SEP}")
        print(f"  Training  ·  {args.epochs} epochs  ·  batch {args.batch_size}  ·  lr {args.lr}")
        print(SEP)

        for epoch in range(args.epochs):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch()
            val_score     = self._combined_score(val_metrics)

            elapsed = time.time() - t0
            self._print_epoch(epoch, args.epochs, train_metrics, val_metrics, elapsed)

            # ── Checkpoint ────────────────────────────────────
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                self._save_checkpoint(val_score)
                print(f"  ✅  New best  (score={val_score:.4f})  saved.")
            else:
                patience_counter += 1
                print(f"  ⏭   No improvement  patience {patience_counter}/{args.patience}")
                if patience_counter >= args.patience:
                    print(f"  🛑  Early stopping.")
                    break

            history.append({
                "epoch": epoch + 1,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

        self._save_history(history)
        print(f"\n  ✅  Training complete.  Best val score: {best_val_score:.4f}")
        print(f"  Checkpoint → outputs/phase2/best_model/")
        print(f"\n  Next:  python3 phase2_evaluate.py   (Step 3)")

    # ── One training epoch ────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        args = self.args

        # Interleave MBTI and Essays batches
        mbti_iter   = iter(self.mbti_train_loader)
        essays_iter = iter(self.essays_train_loader)

        all_batches = (
            [("mbti",   b) for b in self.mbti_train_loader] +
            [("essays", b) for b in self.essays_train_loader]
        )
        # Reload iterators (consumed above)
        np.random.shuffle(all_batches)

        total_loss    = 0.0
        mbti_loss_sum = 0.0
        hex_loss_sum  = 0.0
        steps         = 0
        self.optimizer.zero_grad()

        pbar = tqdm(
            all_batches,
            desc=f"Epoch {epoch+1:>2}/{args.epochs} [train]",
            dynamic_ncols=True,
            smoothing=0.05,
            colour="cyan",
        )

        for task, batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            ctx = torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            )
            with ctx:
                if task == "mbti":
                    out = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        mbti_label=batch["label"],
                    )
                    mbti_loss_sum += out["mbti_loss"].item()
                else:
                    out = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        hexaco_labels=batch["labels"],
                    )
                    hex_loss_sum += out["hexaco_loss"].item()

                loss = out["loss"] / args.grad_accum

            self.scaler.scale(loss).backward()
            total_loss += loss.item() * args.grad_accum
            steps += 1

            if steps % args.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            n_mbti   = max(sum(1 for t,_ in all_batches[:steps] if t=="mbti"), 1)
            n_essays = max(sum(1 for t,_ in all_batches[:steps] if t=="essays"), 1)
            pbar.set_postfix({
                "loss":  f"{total_loss/steps:.4f}",
                "mbti":  f"{mbti_loss_sum/n_mbti:.4f}",
                "hex":   f"{hex_loss_sum/n_essays:.4f}",
                "lr":    f"{self.scheduler.get_last_lr()[0]:.1e}",
            }, refresh=True)

        return {
            "train_loss":       total_loss  / max(steps, 1),
            "train_mbti_loss":  mbti_loss_sum / max(len(self.mbti_train_loader), 1),
            "train_hex_loss":   hex_loss_sum  / max(len(self.essays_train_loader), 1),
        }

    # ── Validation epoch ──────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.model.eval()

        # MBTI val
        mbti_correct = 0
        mbti_total   = 0
        mbti_val_loss = 0.0

        pbar = tqdm(self.mbti_val_loader, desc="  Val [MBTI]  ",
                    dynamic_ncols=True, colour="yellow", leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out   = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                mbti_label=batch["label"],
            )
            mbti_val_loss += out["mbti_loss"].item()
            preds = out["mbti_logits"].argmax(dim=-1)
            mbti_correct += (preds == batch["label"]).sum().item()
            mbti_total   += len(batch["label"])

        mbti_acc = mbti_correct / max(mbti_total, 1)

        # HEXACO val
        all_preds  = []
        all_labels = []
        hex_val_loss = 0.0

        pbar = tqdm(self.essays_val_loader, desc="  Val [HEXACO]",
                    dynamic_ncols=True, colour="yellow", leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out   = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                hexaco_labels=batch["labels"],
            )
            hex_val_loss += out["hexaco_loss"].item()
            all_preds.append(out["hexaco_scores"].cpu().float().numpy())
            all_labels.append(batch["labels"].cpu().float().numpy())

        preds_np  = np.concatenate(all_preds,  axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        # Per-dim binary accuracy (threshold 0.5)
        binary_preds = (preds_np >= 0.5).astype(int)
        binary_labels = (labels_np >= 0.5).astype(int)
        dim_accs = (binary_preds == binary_labels).mean(axis=0)   # [6]
        hex_acc  = float(dim_accs.mean())

        n_mbti   = len(self.mbti_val_loader)
        n_essays = len(self.essays_val_loader)

        return {
            "val_mbti_acc":  mbti_acc,
            "val_hex_acc":   hex_acc,
            "val_mbti_loss": mbti_val_loss  / max(n_mbti, 1),
            "val_hex_loss":  hex_val_loss   / max(n_essays, 1),
            "val_hex_dim_accs": dim_accs.tolist(),
        }

    # ── Score + helpers ───────────────────────────────────────

    def _combined_score(self, val_metrics: dict) -> float:
        return (val_metrics["val_mbti_acc"] + val_metrics["val_hex_acc"]) / 2

    def _print_epoch(self, epoch, total_epochs, train_m, val_m, elapsed):
        print(f"\n  {SEP2}")
        print(f"  Epoch {epoch+1}/{total_epochs}   ({elapsed:.0f}s)")
        print(f"  {SEP2}")
        print(f"  {'Metric':<28} {'Train':>10}  {'Val':>10}")
        print(f"  {'─'*28} {'─'*10}  {'─'*10}")
        print(f"  {'Total loss':<28} {train_m['train_loss']:>10.4f}")
        print(f"  {'MBTI loss':<28} {train_m['train_mbti_loss']:>10.4f}  {val_m['val_mbti_loss']:>10.4f}")
        print(f"  {'HEXACO loss':<28} {train_m['train_hex_loss']:>10.4f}  {val_m['val_hex_loss']:>10.4f}")
        print(f"  {'MBTI accuracy':<28} {'':>10}  {val_m['val_mbti_acc']:>9.1%}")
        print(f"  {'HEXACO avg accuracy':<28} {'':>10}  {val_m['val_hex_acc']:>9.1%}")
        print(f"\n  HEXACO per-dim accuracy:")
        for dim, acc in zip(HEXACO_DIMS, val_m["val_hex_dim_accs"]):
            bar = "█" * int(acc * 20)
            print(f"    {dim:<22} {bar:<20} {acc:.1%}")

    def _save_checkpoint(self, score: float):
        ckpt_dir = OUTPUT_DIR / "best_model"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.roberta.save_pretrained(str(ckpt_dir))
        self.tokenizer.save_pretrained(str(ckpt_dir))
        torch.save(self.model.state_dict(), ckpt_dir / "model_state.pt")
        meta = {
            "val_score": round(score, 4),
            "args": vars(self.args),
        }
        with open(ckpt_dir / "checkpoint_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _save_history(self, history: list):
        path = OUTPUT_DIR / "training_history.json"
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        log.info(f"History → {path}")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 — Train personality classifier")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--max_length",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--grad_accum",   type=int,   default=2)
    p.add_argument("--patience",     type=int,   default=3)
    p.add_argument("--mbti_weight",  type=float, default=0.5)
    p.add_argument("--hexaco_weight",type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()