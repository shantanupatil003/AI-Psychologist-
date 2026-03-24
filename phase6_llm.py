"""
phase6_llm.py  —  Adaptive Questions + Final Report
═════════════════════════════════════════════════════
Phase 6

Part A:  AdaptiveQuestionGenerator
         Uses Claude API to generate a context-aware follow-up
         question grounded in the user's last answer and the
         most uncertain personality dimension.

Part B:  ReportGenerator
         Uses Claude API to write a full human-readable
         personality report from the final profile JSON.

Part C:  EnhancedSession
         Replaces phase5_loop.py's run_session() with a version
         that alternates between bank questions (Phase 4) and
         adaptive questions (Phase 6A), then ends with a full
         report (Phase 6B).

Run:
    python3 phase6_llm.py --demo          # automated demo
    python3 phase6_llm.py                 # interactive session
    python3 phase6_llm.py --report_only outputs/phase5/session_XYZ.json
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import urllib3
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
warnings.filterwarnings("ignore", message=".*Some weights.*")

sys.path.insert(0, str(Path(__file__).parent))
from phase2_datasets  import HEXACO_DIMS, MBTI_TYPES
from phase3_profile   import ProfileManager
from phase4_questions import Question, QuestionSelector, QUESTION_BANK
from phase5_loop      import (ClassifierWrapper, print_header,
                               print_turn_result, print_final_report,
                               DEMO_ANSWERS, _softmax)

import numpy as np

OUTPUT_DIR = Path("outputs/phase6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEP  = "═" * 62
SEP2 = "─" * 62

# MBTI type descriptions for the report
MBTI_DESCRIPTIONS = {
    "INTJ": ("Architect",    "Strategic, independent, driven by long-term vision"),
    "INTP": ("Logician",     "Analytical, inventive, loves abstract theories"),
    "ENTJ": ("Commander",    "Bold, decisive, natural leader"),
    "ENTP": ("Debater",      "Clever, curious, loves intellectual challenge"),
    "INFJ": ("Advocate",     "Insightful, principled, deeply empathetic"),
    "INFP": ("Mediator",     "Idealistic, empathetic, guided by strong values"),
    "ENFJ": ("Protagonist",  "Charismatic, inspiring, cares deeply about others"),
    "ENFP": ("Campaigner",   "Enthusiastic, creative, sees potential everywhere"),
    "ISTJ": ("Logistician",  "Reliable, detail-oriented, values duty and tradition"),
    "ISFJ": ("Defender",     "Caring, dependable, quietly dedicated to others"),
    "ESTJ": ("Executive",    "Organised, traditional, excellent at managing people"),
    "ESFJ": ("Consul",       "Warm, conscientious, values harmony and community"),
    "ISTP": ("Virtuoso",     "Practical, observant, thrives on hands-on problem-solving"),
    "ISFP": ("Adventurer",   "Gentle, flexible, lives fully in the present"),
    "ESTP": ("Entrepreneur", "Energetic, perceptive, bold and action-oriented"),
    "ESFP": ("Entertainer",  "Spontaneous, playful, loves making people smile"),
}


# ══════════════════════════════════════════════════════════════
#  PART A  —  ADAPTIVE QUESTION GENERATOR
# ══════════════════════════════════════════════════════════════

class AdaptiveQuestionGenerator:
    """
    Uses a local Ollama model to generate context-aware follow-up questions.

    Setup (one-time):
        brew install ollama
        ollama pull llama3.2
        ollama serve        # or it auto-starts on macOS

    Falls back silently to the question bank if Ollama is not running.
    """

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            import ollama
            ollama.list()   # ping the server
            return True
        except ImportError:
            return False
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self._available

    def generate(self, last_answer, profile, asked_questions, turn):
        if not self._available:
            return None

        uncertain  = profile.most_uncertain()
        dim        = uncertain["dimension"]
        task       = uncertain["task"]
        dp         = profile.dichotomy_probs
        scores     = profile.hexaco_scores

        if task == "mbti":
            a, b = dim[0], dim[1]
            dim_context = (
                f"Clarify whether this person is {a} or {b} on the {dim} axis. "
                f"Currently P({a})={dp[a]:.0%} vs P({b})={dp[b]:.0%}."
            )
        else:
            score = scores[dim]
            dim_context = f"Clarify their {dim} score (currently {score:.2f}/1.0)."

        asked_texts = [q if isinstance(q, str) else q.text for q in asked_questions[-4:]]
        asked_str   = "; ".join(asked_texts) if asked_texts else "nothing yet"

        prompt = (
            f"You are conducting a personality assessment.\n"
            f"The person just said: '{last_answer}'\n\n"
            f"Goal: {dim_context}\n\n"
            f"Already asked: {asked_str}\n\n"
            f"Write ONE short follow-up question (1 sentence). "
            f"Output only the question, no explanation, no preamble."
        )

        try:
            import ollama
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7, "num_predict": 80},
            )
            q = response["message"]["content"].strip()
            # Strip common preambles
            for pfx in ["Question:", "Q:", "Follow-up:", "Sure,", "Certainly,"]:
                if q.lower().startswith(pfx.lower()):
                    q = q[len(pfx):].strip()
            # Take only first sentence if model returned multiple
            if "?" in q:
                q = q[:q.index("?")+1]
            if len(q) > 10:
                return q if q.endswith("?") else q + "?"
        except Exception as e:
            print(f"  Warning: adaptive generator: {e}")

        return None





# ══════════════════════════════════════════════════════════════
#  PART B  —  REPORT GENERATOR
# ══════════════════════════════════════════════════════════════

class ReportGenerator:
    """
    Generates a full human-readable personality report
    from the final ProfileManager state.
    """

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            import ollama
            ollama.list()
            return True
        except ImportError:
            return False
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self._available

    def generate(self, profile: ProfileManager, session_log: list) -> str:
        """
        Returns the full report as a string.
        Falls back to a structured text report if API unavailable.
        """
        d = profile.to_dict()

        if self._available:
            return self._llm_report(d, session_log)
        else:
            return self._structured_report(d)

    def _llm_report(self, profile_dict: dict, session_log: list) -> str:
        mbti_type = profile_dict["mbti"]["predicted_type"]
        confidence = profile_dict["mbti"]["confidence"]
        top3       = profile_dict["mbti"]["top3"]
        dp         = profile_dict["mbti"]["dichotomy_probs"]
        hexaco     = profile_dict["hexaco"]["scores"]

        # Build the key answers summary (last 6 turns)
        answer_summary = "\n".join(
            f"Q: [{log['dimension']}] → A: {log['answer'][:100]}"
            for log in session_log[-6:]
        )

        type_name, type_desc = MBTI_DESCRIPTIONS.get(
            mbti_type, (mbti_type, "")
        )

        alt_type = top3[1][0] if len(top3) > 1 else "N/A"
        alt_prob = f"{top3[1][1]:.0%}" if len(top3) > 1 else ""

        prompt = f"""You are a thoughtful personality psychologist writing a personalised report.

ASSESSMENT DATA:
- MBTI predicted type: {mbti_type} ({type_name} — {type_desc})
- Confidence: {confidence:.0%}
- Alternative candidate: {alt_type} ({alt_prob})
- Introversion/Extraversion: {'Introversion' if dp['I'] > 0.5 else 'Extraversion'} ({max(dp['I'],dp['E']):.0%})
- Intuition/Sensing: {'Intuition' if dp['N'] > 0.5 else 'Sensing'} ({max(dp['N'],dp['S']):.0%})
- Thinking/Feeling: {'Thinking' if dp['T'] > 0.5 else 'Feeling'} ({max(dp['T'],dp['F']):.0%})
- Judging/Perceiving: {'Judging' if dp['J'] > 0.5 else 'Perceiving'} ({max(dp['J'],dp['P']):.0%})

HEXACO scores (0=low, 1=high):
- Honesty-Humility: {hexaco['Honesty-Humility']:.2f} [derived estimate]
- Emotionality:     {hexaco['Emotionality']:.2f}
- Extraversion:     {hexaco['Extraversion']:.2f}
- Agreeableness:    {hexaco['Agreeableness']:.2f}
- Conscientiousness:{hexaco['Conscientiousness']:.2f}
- Openness:         {hexaco['Openness']:.2f}

SAMPLE ANSWERS:
{answer_summary}

Write a personalised personality report with these sections:
1. **Your personality type** — 2-3 sentences describing who they are
2. **Your strengths** — 3-4 specific strengths based on their answers
3. **Your growth areas** — 2-3 honest but kind growth areas
4. **How you process the world** — based on HEXACO scores
5. **A note on uncertainty** — if confidence < 60%, acknowledge the alternative type

Keep the tone warm, direct, and specific to THEIR answers — not generic.
Total length: 300-400 words."""

        try:
            import ollama
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.6, "num_predict": 700},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"  ⚠️   Report generator error: {e}")
            return self._structured_report(profile_dict)

    def _structured_report(self, d: dict) -> str:
        """Fallback report when API is unavailable."""
        mbti_type  = d["mbti"]["predicted_type"]
        confidence = d["mbti"]["confidence"]
        top3       = d["mbti"]["top3"]
        dp         = d["mbti"]["dichotomy_probs"]
        hexaco     = d["hexaco"]["scores"]
        type_name, type_desc = MBTI_DESCRIPTIONS.get(mbti_type, (mbti_type, ""))

        axes = [
            ("I","E","Introversion","Extraversion"),
            ("N","S","Intuition","Sensing"),
            ("T","F","Thinking","Feeling"),
            ("J","P","Judging","Perceiving"),
        ]

        lines = [
            f"\n{'═'*62}",
            f"  PERSONALITY REPORT",
            f"{'═'*62}",
            f"\n  Type : {mbti_type}  —  {type_name}",
            f"  {type_desc}",
            f"  Confidence : {confidence:.0%}\n",
            "  Personality axes:",
        ]
        for a, b, la, lb in axes:
            pa   = dp[a]
            pole = la if pa > 0.5 else lb
            conf = max(pa, 1 - pa)
            bar  = "█" * int(conf * 20)
            lines.append(f"    {pole:<14} {bar:<20} {conf:.0%}")

        if len(top3) > 1 and confidence < 0.6:
            lines.append(
                f"\n  Note: {top3[1][0]} ({top3[1][1]:.0%}) is also a strong candidate."
            )

        lines += [
            "\n  HEXACO six-factor profile:",
        ]
        for dim in HEXACO_DIMS:
            s   = hexaco[dim]
            bar = "█" * int(s * 24)
            tag = " [derived]" if dim == "Honesty-Humility" else ""
            lines.append(f"    {dim:<22} {bar:<24} {s:.2f}{tag}")

        lines.append(f"\n{'═'*62}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  PART C  —  ENHANCED SESSION
# ══════════════════════════════════════════════════════════════

def run_enhanced_session(args):
    print_header()

    print("  Loading classifier …")
    classifier = ClassifierWrapper(max_length=args.max_length)
    print(f"  ✅  Classifier ready  ({classifier.device})")

    adaptive_gen = AdaptiveQuestionGenerator()
    report_gen   = ReportGenerator()

    if adaptive_gen.available:
        print(f"  ✅  Adaptive question generator ready  (Ollama: {adaptive_gen.model})")
    else:
        print("  ⚠️   Ollama not running — using question bank only")
        print("       Setup: brew install ollama && ollama pull llama3.2 && ollama serve\n")

    if report_gen.available:
        print(f"  ✅  Report generator ready  (Ollama: {report_gen.model})\n")
    else:
        print("  ⚠️   Report generator will use structured fallback\n")

    profile     = ProfileManager(
        tau_mbti=args.tau_mbti,
        tau_hex=args.tau_hex,
        decay=args.decay,
    )
    selector    = QuestionSelector()
    session_log = []
    asked_texts = []   # for adaptive generator context
    turn        = 0
    adaptive_turns = 0

    while turn < args.max_turns:
        # ── Decide: bank question or adaptive? ───────────────
        # Fire adaptive on turns 3, 6, 9 ... (every 3rd completed turn)
        use_adaptive = (
            adaptive_gen.available
            and turn > 0
            and turn % 3 == 0
            and len(session_log) >= turn
        )

        if use_adaptive:
            last_answer = session_log[-1]["answer"]
            q_text = adaptive_gen.generate(
                last_answer, profile, asked_texts, turn
            )
            if q_text:
                adaptive_turns += 1
                turn += 1
                dim    = profile.most_uncertain()["dimension"]
                task   = profile.most_uncertain()["task"]

                print(f"\n  {SEP2}")
                print(f"  Question {turn}/{args.max_turns}  ·  [{task.upper()}  {dim}]  [adaptive]")
                print(f"  {SEP2}")
                print(f"\n  {q_text}\n")
                question_id = f"adaptive_{turn}"
                dimension   = dim
                use_adaptive = True
            else:
                use_adaptive = False   # fall back to bank

        if not use_adaptive:
            question = selector.select(profile)
            if question is None:
                print("  All questions exhausted.")
                break

            turn += 1
            print(f"\n  {SEP2}")
            print(f"  Question {turn}/{args.max_turns}  ·  [{question.task.upper()}  {question.dimension}]")
            print(f"  {SEP2}")
            print(f"\n  {question.text}\n")
            q_text      = question.text
            question_id = question.id
            dimension   = question.dimension
            selector.mark_asked(question)

        # ── Get answer ────────────────────────────────────────
        if args.demo:
            answer = DEMO_ANSWERS.get(
                dimension,
                "I find this thought-provoking. I generally take a balanced approach."
            )
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
            print("  ⚠️   Please write a bit more for a meaningful result.")
            turn -= 1
            continue

        asked_texts.append(q_text)

        # ── Score + update profile ────────────────────────────
        logits, hexaco_scores = classifier.score(answer)
        profile.update(logits, hexaco_scores)

        # ── Show live result ──────────────────────────────────
        print_turn_result(logits, hexaco_scores, profile, turn)

        session_log.append({
            "turn":        turn,
            "question_id": question_id,
            "dimension":   dimension,
            "adaptive":    use_adaptive,
            "answer":      answer[:500],
            "mbti_top1":   profile.predicted_type,
            "mbti_conf":   round(profile.mbti_confidence, 3),
            "hexaco":      {d: round(v, 3)
                            for d, v in profile.hexaco_scores.items()},
        })

        # ── Check convergence ─────────────────────────────────
        if profile.is_converged():
            print(f"\n  ✅  Profile converged after {turn} turns.")
            break

        remaining = selector.remaining_count()
        print(f"\n  {remaining} bank questions remaining  ·  "
              f"adaptive used: {adaptive_turns}/{turn}")

    # ── Generate final report ─────────────────────────────────
    print(f"\n{SEP}")
    print("  Generating final report …")
    report = report_gen.generate(profile, session_log)
    print(f"\n{SEP}")
    print("  📋  PERSONALITY REPORT")
    print(SEP)
    print(report)

    # ── Save ──────────────────────────────────────────────────
    ts   = int(time.time())
    path = OUTPUT_DIR / f"session_{ts}.json"
    output = {
        "profile":      profile.to_dict(),
        "turns_taken":  turn,
        "adaptive_turns": adaptive_turns,
        "session_log":  session_log,
        "report":       report,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Session saved → {path}")
    print(f"{SEP}\n")


# ══════════════════════════════════════════════════════════════
#  REPORT ONLY MODE  (from existing phase5 session)
# ══════════════════════════════════════════════════════════════

def run_report_only(session_path: str):
    path = Path(session_path)
    if not path.exists():
        print(f"❌  Session file not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    profile_dict = data["profile"]
    session_log  = data.get("session_log", [])

    # Rebuild ProfileManager from saved state
    profile = ProfileManager()

    rg     = ReportGenerator()
    report = rg._llm_report(profile_dict, session_log) \
             if rg.available \
             else rg._structured_report(profile_dict)

    print(f"\n{SEP}")
    print("  📋  PERSONALITY REPORT")
    print(f"  (generated from {path.name})")
    print(SEP)
    print(report)

    # Save report alongside session
    report_path = path.parent / (path.stem + "_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved → {report_path}\n")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Phase 6 — Adaptive questions + report")
    p.add_argument("--max_turns",   type=int,   default=12)
    p.add_argument("--max_length",  type=int,   default=256)
    p.add_argument("--tau_mbti",    type=float, default=1.5)
    p.add_argument("--tau_hex",     type=float, default=0.04)
    p.add_argument("--decay",       type=float, default=0.7)
    p.add_argument("--demo",        action="store_true",
                   help="Run with demo answers")
    p.add_argument("--report_only", type=str,   default=None,
                   help="Path to existing phase5 session JSON — generate report only")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.report_only:
        run_report_only(args.report_only)
    else:
        run_enhanced_session(args)