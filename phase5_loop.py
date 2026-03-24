"""
phase5_loop.py  —  Conversation Loop
══════════════════════════════════════
Phase 5

Wires together every component built so far:
  Phase 2  →  PersonalityClassifier   (scores answer text)
  Phase 3  →  ProfileManager          (tracks estimates)
  Phase 4  →  QuestionSelector        (picks next question)

Runs as an interactive CLI session.

Usage:
    python3 phase5_loop.py              # full interactive session
    python3 phase5_loop.py --demo       # auto-answers for testing
    python3 phase5_loop.py --max_turns 8

Session flow:
  1. Load classifier checkpoint
  2. Create fresh ProfileManager + QuestionSelector
  3. Loop:
       a. Select next question  (Phase 4)
       b. Show question to user
       c. Read user's answer
       d. Run classifier on answer text  (Phase 2)
       e. Update profile  (Phase 3)
       f. Show live profile snapshot
       g. Check convergence → stop if done
  4. Print final report
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import urllib3
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
warnings.filterwarnings("ignore", message=".*Some weights.*")
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)

import numpy as np
import torch
from transformers import RobertaTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from phase2_datasets  import HEXACO_DIMS, MBTI_TYPES
from phase2_model     import PersonalityClassifier
from phase3_profile   import ProfileManager
from phase4_questions import Question, QuestionSelector, QUESTION_BANK
from phase2_datasets  import DatasetManifest

CKPT_DIR   = Path("outputs/phase2/best_model")
OUTPUT_DIR = Path("outputs/phase5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEP  = "═" * 62
SEP2 = "─" * 62

# Demo answers for automated testing (one per dimension)
DEMO_ANSWERS = {
    "IE": "I really prefer spending time alone. Large social gatherings drain me completely and I need days to recover. I do my best thinking in quiet solitude.",
    "NS": "I'm always thinking about abstract possibilities and theoretical frameworks. I love exploring ideas that have no practical application whatsoever.",
    "TF": "I lead with logic and analysis. I find it uncomfortable when people let emotions override clear reasoning in decision making.",
    "JP": "I plan everything meticulously. My calendar is colour coded, I make detailed lists, and unexpected changes to plans stress me out a lot.",
    "Honesty-Humility": "I would never take credit for someone else's work. Honesty and fairness matter enormously to me even when dishonesty would benefit me.",
    "Emotionality": "I feel emotions very intensely. I worry a lot about things that might go wrong and I find it hard to hide how I'm feeling.",
    "Extraversion": "I'm fairly quiet and reserved. I don't enjoy being the centre of attention and I prefer one-on-one conversations to group settings.",
    "Agreeableness": "I try to forgive people quickly and assume the best of strangers. Holding grudges feels exhausting and pointless to me.",
    "Conscientiousness": "I'm very organised and reliable. I always meet deadlines, keep my space tidy, and follow through on everything I commit to.",
    "Openness": "I'm endlessly curious. I love learning about topics completely outside my expertise and I regularly change my opinions when I encounter good arguments.",
}


# ══════════════════════════════════════════════════════════════
#  CLASSIFIER WRAPPER
# ══════════════════════════════════════════════════════════════

class ClassifierWrapper:
    """
    Thin wrapper around PersonalityClassifier that:
      · handles device placement
      · tokenizes raw text
      · returns numpy arrays ready for ProfileManager.update()
    """

    def __init__(self, max_length: int = 256):
        self.max_length = max_length
        self._setup()

    def _setup(self):
        if not CKPT_DIR.exists():
            print(f"\n❌  Checkpoint not found: {CKPT_DIR}")
            print("    Run phase2_model.py first.")
            sys.exit(1)

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        manifest  = DatasetManifest()
        self.tokenizer = RobertaTokenizer.from_pretrained(str(CKPT_DIR))
        self.model = PersonalityClassifier(
            model_name         = str(CKPT_DIR),
            mbti_class_weights = manifest.mbti_class_weights.to(self.device),
            hexaco_dim_weights = manifest.hexaco_dim_weights.to(self.device),
        )
        state = torch.load(
            CKPT_DIR / "model_state.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, text: str):
        """
        Returns:
            mbti_logits   np.ndarray [16]
            hexaco_scores np.ndarray [6]
        """
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)

        logits = out["mbti_logits"].squeeze(0).cpu().float().numpy()
        scores = out["hexaco_scores"].squeeze(0).cpu().float().numpy()
        return logits, scores


# ══════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════

def print_header():
    print(f"\n{SEP}")
    print("  🧠  Personality Assessment  ·  Phase 5")
    print("  Powered by RoBERTa  ·  MBTI + HEXACO")
    print(SEP)
    print("""
  Answer each question in your own words.
  There are no right or wrong answers.
  The more naturally you write, the more accurate the result.
  Type 'quit' at any time to exit.
""")


def print_question(turn: int, max_turns: int, question: Question, target: dict):
    dim_label = question.dimension
    print(f"\n  {SEP2}")
    print(f"  Question {turn}/{max_turns}  ·  [{question.task.upper()}  {dim_label}]")
    print(f"  {SEP2}")
    print(f"\n  {question.text}\n")


def print_turn_result(logits: np.ndarray, scores: np.ndarray,
                      profile: ProfileManager, turn: int):
    """Show a compact live update after each answer."""
    # MBTI: top 3 predictions from this answer alone
    this_probs = _softmax(logits)
    top3_this  = [(MBTI_TYPES[i], float(this_probs[i]))
                  for i in this_probs.argsort()[::-1][:3]]

    # Profile state
    top3_profile = profile.top3_types
    dp           = profile.dichotomy_probs

    print(f"\n  {SEP2}")
    print(f"  Turn {turn} result")
    print(f"  {SEP2}")

    print(f"\n  This answer signals  : ", end="")
    print("  ".join(f"{t}({p:.0%})" for t, p in top3_this))

    print(f"  Profile (cumulative) : ", end="")
    print("  ".join(f"{t}({p:.0%})" for t, p in top3_profile))

    print(f"\n  Dichotomies:")
    axes = [("I","E"), ("N","S"), ("T","F"), ("J","P")]
    for a, b in axes:
        pa   = dp[a]
        pole = a if pa > 0.5 else b
        conf = max(pa, 1-pa)
        bar_a = "█" * int(pa * 16)
        bar_b = "█" * int((1-pa) * 16)
        print(f"    {a} {bar_a:<16}{bar_b:>16} {b}  →  {pole} ({conf:.0%})")

    print(f"\n  HEXACO snapshot:")
    hex_scores = profile.hexaco_scores
    hex_unc    = profile.hexaco_uncertainty
    for dim in HEXACO_DIMS:
        s   = hex_scores[dim]
        u   = hex_unc[dim]
        bar = "█" * int(s * 16)
        unc_bar = "?" * int(u * 4)
        print(f"    {dim:<22} {bar:<16} {s:.2f}  {unc_bar}")

    mu = profile.most_uncertain()
    print(f"\n  Most uncertain       : [{mu['task']}] {mu['dimension']}  ({mu['reason']})")
    print(f"  MBTI confidence      : {profile.mbti_confidence:.1%}")
    print(f"  Converged            : {profile.is_converged()}")


def print_final_report(profile: ProfileManager, turns_taken: int,
                       session_log: list, output_path: Path):
    d   = profile.to_dict()
    dp  = d["mbti"]["dichotomy_probs"]
    top3 = d["mbti"]["top3"]

    print(f"\n{SEP}")
    print("  🎯  Final Personality Report")
    print(SEP)

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  MBTI Type  :  {d['mbti']['predicted_type']}                         │")
    print(f"  │  Confidence :  {d['mbti']['confidence']:.1%}                         │")
    print(f"  └─────────────────────────────────────────┘")

    print(f"\n  Top candidates:")
    for t, p in top3:
        bar = "█" * int(p * 30)
        print(f"    {t:<6}  {bar:<30}  {p:.1%}")

    print(f"\n  Personality axes:")
    axes = [("I","E","Introversion","Extraversion"),
            ("N","S","Intuition","Sensing"),
            ("T","F","Thinking","Feeling"),
            ("J","P","Judging","Perceiving")]
    for a, b, la, lb in axes:
        pa   = dp[a]
        pole = la if pa > 0.5 else lb
        conf = max(pa, 1-pa)
        bar  = "█" * int(max(pa, 1-pa) * 20)
        print(f"    {pole:<12}  {bar:<20}  {conf:.0%}")

    print(f"\n  HEXACO six-factor scores:")
    hex_scores = d["hexaco"]["scores"]
    hex_var    = d["hexaco"]["variance"]
    for dim in HEXACO_DIMS:
        s   = hex_scores[dim]
        v   = hex_var[dim]
        bar = "█" * int(s * 24)
        tag = " [derived]" if dim == "Honesty-Humility" else ""
        print(f"    {dim:<22}  {bar:<24}  {s:.2f}  ±{v:.3f}{tag}")

    conv = d["convergence"]
    print(f"\n  Session stats:")
    print(f"    Turns taken    : {turns_taken}")
    print(f"    MBTI entropy   : {conv['mbti_entropy']:.3f} / 2.773 nats")
    print(f"    MBTI converged : {conv['mbti_converged']}")
    print(f"    HEXACO converged: {conv['hexaco_converged']}")

    # Save report
    report = {
        "profile":     d,
        "turns_taken": turns_taken,
        "session_log": session_log,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {output_path}")
    print(f"\n{SEP}\n")


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ══════════════════════════════════════════════════════════════
#  MAIN SESSION
# ══════════════════════════════════════════════════════════════

def run_session(args):
    print_header()

    # Load classifier
    print("  Loading classifier …")
    classifier = ClassifierWrapper(max_length=args.max_length)
    print(f"  ✅  Classifier ready  ({classifier.device})\n")

    # Initialise
    profile  = ProfileManager(
        tau_mbti=args.tau_mbti,
        tau_hex=args.tau_hex,
        decay=args.decay,
    )
    selector    = QuestionSelector()
    session_log = []
    turn        = 0

    while turn < args.max_turns:
        # ── Select question ───────────────────────────────────
        question = selector.select(profile)
        if question is None:
            print("  All questions exhausted.")
            break

        turn += 1
        target = profile.most_uncertain()
        print_question(turn, args.max_turns, question, target)

        # ── Get answer ────────────────────────────────────────
        if args.demo:
            # Use pre-written demo answer for this dimension
            answer = DEMO_ANSWERS.get(question.dimension,
                "I find this question interesting and have mixed feelings about it.")
            print(f"  [demo]  {answer[:80]}…")
        else:
            try:
                answer = input("  Your answer: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Session interrupted.")
                break

        if answer.lower() in ("quit", "exit", "q"):
            print("  Session ended by user.")
            break

        if len(answer.split()) < 3:
            print("  ⚠️   Please write at least a few words for a meaningful result.")
            turn -= 1
            continue

        # ── Score answer ──────────────────────────────────────
        logits, hexaco_scores = classifier.score(answer)

        # ── Update profile ────────────────────────────────────
        selector.mark_asked(question)
        profile.update(logits, hexaco_scores)

        # ── Show live result ──────────────────────────────────
        print_turn_result(logits, hexaco_scores, profile, turn)

        # ── Log ───────────────────────────────────────────────
        session_log.append({
            "turn":        turn,
            "question_id": question.id,
            "dimension":   question.dimension,
            "answer":      answer[:500],   # cap stored length
            "mbti_top1":   profile.predicted_type,
            "mbti_conf":   round(profile.mbti_confidence, 3),
            "hexaco":      {d: round(v, 3)
                            for d, v in profile.hexaco_scores.items()},
        })

        # ── Check convergence ─────────────────────────────────
        if profile.is_converged():
            print(f"\n  ✅  Profile converged after {turn} turns.")
            break

        if turn < args.max_turns:
            remaining = selector.remaining_count()
            print(f"\n  {remaining} questions remaining  ·  "
                  f"continuing to most uncertain dimension …")

    # ── Final report ──────────────────────────────────────────
    ts   = int(time.time())
    path = OUTPUT_DIR / f"session_{ts}.json"
    print_final_report(profile, turn, session_log, path)


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 — Personality conversation loop")
    p.add_argument("--max_turns",  type=int,   default=12,
                   help="Maximum questions to ask (default 12)")
    p.add_argument("--max_length", type=int,   default=256,
                   help="Tokenizer max length (must match training)")
    p.add_argument("--tau_mbti",   type=float, default=1.5,
                   help="MBTI entropy convergence threshold (nats)")
    p.add_argument("--tau_hex",    type=float, default=0.04,
                   help="HEXACO variance convergence threshold")
    p.add_argument("--decay",      type=float, default=0.7,
                   help="Bayesian update decay factor (0-1)")
    p.add_argument("--demo",       action="store_true",
                   help="Run with pre-written demo answers (no input needed)")
    p.add_argument("--output",     type=str,   default=None,
                   help="Path to save session JSON (default: auto-timestamped)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_session(args)