"""
phase4_questions.py  —  Question Bank + QuestionSelector
══════════════════════════════════════════════════════════
Phase 4

Defines:
  QUESTION_BANK   list of Question objects, each tagged with:
                    · task       "mbti" | "hexaco"
                    · dimension  e.g. "IE", "Emotionality"
                    · depth      "open" | "followup"
                    · text       the question string

  QuestionSelector  reads ProfileManager.uncertain_dims_ranked()
                    and returns the best unasked question

Run self-test:
    python3 phase4_questions.py
"""

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent))
from phase3_profile import ProfileManager

SEP  = "═" * 62
SEP2 = "─" * 62


# ══════════════════════════════════════════════════════════════
#  QUESTION DATACLASS
# ══════════════════════════════════════════════════════════════

@dataclass
class Question:
    id:        str          # unique slug  e.g. "ie_01"
    task:      str          # "mbti" | "hexaco"
    dimension: str          # "IE" | "NS" | "TF" | "JP" | HEXACO dim name
    depth:     str          # "open" (any time) | "followup" (after open)
    text:      str          # the question shown to the user
    hint:      str = ""     # internal note — why this targets this dim


# ══════════════════════════════════════════════════════════════
#  QUESTION BANK
#  5 open + 3 followup questions per dimension = 10 dims × 8 = 80 questions
# ══════════════════════════════════════════════════════════════

QUESTION_BANK: List[Question] = [

    # ── I / E  ─────────────────────────────────────────────
    Question("ie_01","mbti","IE","open",
        "How do you typically recharge after a long or stressful week?",
        "Introverts describe solitude; extraverts describe social activity"),
    Question("ie_02","mbti","IE","open",
        "When you are working on something important, do you prefer silence and solitude or background energy and other people nearby?",
        "Environment preference strongly correlates with I/E"),
    Question("ie_03","mbti","IE","open",
        "Think about the last time you felt genuinely energised. What were you doing?",
        "Energy source is the core I/E marker"),
    Question("ie_04","mbti","IE","open",
        "How do you feel walking into a room full of people you have never met?",
        "Social comfort/anxiety is a reliable I/E signal"),
    Question("ie_05","mbti","IE","open",
        "How many close friends do you have, and how often do you see them?",
        "Introverts: fewer but deeper; extraverts: broader networks"),
    Question("ie_f1","mbti","IE","followup",
        "You mentioned spending time alone — do you find that draining or genuinely enjoyable?",
        "Distinguishes true introvert from socially anxious extravert"),
    Question("ie_f2","mbti","IE","followup",
        "When you have a problem to work through, do you talk it out with someone or think it through on your own first?",
        "Processing style maps to I/E"),
    Question("ie_f3","mbti","IE","followup",
        "At a gathering, are you more likely to be the one starting conversations or waiting for others to approach?",
        "Behavioural marker for E vs I in social settings"),

    # ── N / S  ─────────────────────────────────────────────
    Question("ns_01","mbti","NS","open",
        "When you are learning something new, do you prefer to understand the big picture first, or start with the concrete steps and details?",
        "N prefers concepts first; S prefers facts/steps first"),
    Question("ns_02","mbti","NS","open",
        "Do you find yourself thinking more about what could be, or about what currently is?",
        "Future-oriented = N; present/past-oriented = S"),
    Question("ns_03","mbti","NS","open",
        "Describe how you would approach planning a long trip — what do you focus on first?",
        "N focuses on possibilities/experiences; S focuses on logistics/practicalities"),
    Question("ns_04","mbti","NS","open",
        "Are you more drawn to theoretical ideas and abstract concepts, or to hands-on practical work?",
        "Direct N/S indicator"),
    Question("ns_05","mbti","NS","open",
        "When reading or listening, do you focus on the literal meaning or do you often look for deeper patterns and implications?",
        "Interpretation style is a reliable N/S marker"),
    Question("ns_f1","mbti","NS","followup",
        "Do you often find yourself getting lost in hypotheticals — imagining scenarios that probably will not happen?",
        "High N tendency"),
    Question("ns_f2","mbti","NS","followup",
        "How much do you trust gut instincts versus concrete evidence when making decisions?",
        "Intuition-trust correlates with N"),
    Question("ns_f3","mbti","NS","followup",
        "Would you describe yourself as someone who notices details others miss, or someone who sees the bigger picture others miss?",
        "Details=S, big picture=N"),

    # ── T / F  ─────────────────────────────────────────────
    Question("tf_01","mbti","TF","open",
        "When a friend comes to you with a problem, what is your instinct — to offer solutions or to listen and empathise?",
        "Core T/F distinction in interpersonal behaviour"),
    Question("tf_02","mbti","TF","open",
        "If you had to give someone difficult feedback that might upset them, how would you approach it?",
        "T prioritises honesty/clarity; F prioritises relationship/feelings"),
    Question("tf_03","mbti","TF","open",
        "Think of a time you had to make a tough decision. What factors carried the most weight?",
        "Logic/fairness = T; impact on people = F"),
    Question("tf_04","mbti","TF","open",
        "How do you feel when someone disagrees with you strongly — does it bother you or do you find it interesting?",
        "T engages intellectually; F takes it more personally"),
    Question("tf_05","mbti","TF","open",
        "Would you rather be known as someone who is fair or someone who is kind? Why?",
        "Classic T/F values question"),
    Question("tf_f1","mbti","TF","followup",
        "When a rule or policy seems unfair to a specific person, do you follow it anyway or look for exceptions?",
        "T applies rules consistently; F makes exceptions for people"),
    Question("tf_f2","mbti","TF","followup",
        "Do other people ever tell you that you come across as blunt or that you are too focused on feelings?",
        "External perception is a useful T/F calibration"),
    Question("tf_f3","mbti","TF","followup",
        "In a disagreement, is your goal to reach the correct answer or to reach an agreement that everyone is comfortable with?",
        "Truth-seeking=T, harmony-seeking=F"),

    # ── J / P  ─────────────────────────────────────────────
    Question("jp_01","mbti","JP","open",
        "How do you feel when plans change at the last minute?",
        "J dislikes disruption; P often welcomes flexibility"),
    Question("jp_02","mbti","JP","open",
        "Describe your typical approach to deadlines — do you work steadily toward them or tend to do your best work under pressure?",
        "Steady/early = J; pressure/last-minute = P"),
    Question("jp_03","mbti","JP","open",
        "How organised is your living or work space right now?",
        "Physical organisation is a reliable J/P marker"),
    Question("jp_04","mbti","JP","open",
        "Do you prefer having a clear plan for the day, or do you like leaving things open and deciding as you go?",
        "Structure preference is the core J/P marker"),
    Question("jp_05","mbti","JP","open",
        "How do you feel once a decision is made — relieved that it is settled, or do you sometimes wish you had kept your options open?",
        "Closure-seeking=J, options-keeping=P"),
    Question("jp_f1","mbti","JP","followup",
        "Do you make lists? If so, how often do you actually follow them?",
        "J makes and follows; P makes but improvises"),
    Question("jp_f2","mbti","JP","followup",
        "When starting a project, do you outline it first or dive straight in?",
        "Planning-first=J, dive-in=P"),
    Question("jp_f3","mbti","JP","followup",
        "How many unfinished projects do you have going at once?",
        "Multiple open loops are a classic P trait"),

    # ── Honesty-Humility ────────────────────────────────────
    Question("hh_01","hexaco","Honesty-Humility","open",
        "Have you ever bent the rules slightly to get something you wanted? How did that feel?",
        "Low H-H: comfortable bending rules; high H-H: discomfort"),
    Question("hh_02","hexaco","Honesty-Humility","open",
        "When you have done something wrong, do you tend to admit it quickly or look for ways to minimise your role?",
        "Self-accountability correlates with H-H"),
    Question("hh_03","hexaco","Honesty-Humility","open",
        "How important is status or being seen as successful to you?",
        "Status-seeking correlates with low H-H"),
    Question("hh_04","hexaco","Honesty-Humility","open",
        "If you discovered a wallet full of cash with an ID inside, what would you do?",
        "Scenario-based honesty probe"),
    Question("hh_05","hexaco","Honesty-Humility","open",
        "Do you ever exaggerate your achievements or accomplishments when talking to others?",
        "Self-enhancement=low H-H"),
    Question("hh_f1","hexaco","Honesty-Humility","followup",
        "How would you feel taking credit for a group achievement even if your contribution was small?",
        "Fairness/modesty marker"),
    Question("hh_f2","hexaco","Honesty-Humility","followup",
        "Are there situations where you think a little deception is completely justified?",
        "Low H-H accepts situational deception"),
    Question("hh_f3","hexaco","Honesty-Humility","followup",
        "How do you feel about people who use charm and flattery to get what they want?",
        "Attitude toward manipulation maps to H-H"),

    # ── Emotionality ────────────────────────────────────────
    Question("em_01","hexaco","Emotionality","open",
        "How do you typically respond when you are feeling anxious or worried?",
        "High Emotionality: expresses/dwells; low: dismisses/moves on"),
    Question("em_02","hexaco","Emotionality","open",
        "How strongly do you feel emotions compared to the people around you — more intensely, about the same, or less?",
        "Direct emotionality probe"),
    Question("em_03","hexaco","Emotionality","open",
        "When something sad happens — like a film or a news story — how does it affect you?",
        "Emotional reactivity to stimuli"),
    Question("em_04","hexaco","Emotionality","open",
        "Do you find yourself worrying about things that might go wrong, even when everything seems fine?",
        "Anxiety/worry correlates strongly with Emotionality"),
    Question("em_05","hexaco","Emotionality","open",
        "How comfortable are you asking for emotional support from others when you are struggling?",
        "Emotional neediness is a high-Emotionality marker"),
    Question("em_f1","hexaco","Emotionality","followup",
        "Would your friends describe you as someone who gets upset easily or takes things in stride?",
        "Social perception of emotional reactivity"),
    Question("em_f2","hexaco","Emotionality","followup",
        "When you have a conflict with someone important to you, how long does it take you to feel okay again?",
        "Recovery time maps to Emotionality"),
    Question("em_f3","hexaco","Emotionality","followup",
        "Do you tend to share your feelings openly, or keep them mostly to yourself?",
        "Emotional expression vs suppression"),

    # ── Extraversion (HEXACO version — liveliness/social confidence) ──
    Question("ex_01","hexaco","Extraversion","open",
        "How do you feel when you are the centre of attention — energised, nervous, or indifferent?",
        "Social confidence is core HEXACO Extraversion"),
    Question("ex_02","hexaco","Extraversion","open",
        "Do you consider yourself an optimistic person? What does that look like day-to-day?",
        "Positive affect is a key Extraversion component in HEXACO"),
    Question("ex_03","hexaco","Extraversion","open",
        "How much do you enjoy small talk with strangers?",
        "Social approach behaviour"),
    Question("ex_04","hexaco","Extraversion","open",
        "How often do you initiate plans or social activities with other people?",
        "Social initiative maps to Extraversion"),
    Question("ex_05","hexaco","Extraversion","open",
        "Would you describe yourself as a lively, talkative person or more reserved and quiet?",
        "Self-assessment of liveliness"),
    Question("ex_f1","hexaco","Extraversion","followup",
        "In a group setting, do you find yourself naturally taking the lead or following along?",
        "Leadership tendency correlates with Extraversion"),
    Question("ex_f2","hexaco","Extraversion","followup",
        "How often do you feel bored or restless when things are quiet?",
        "Stimulation-seeking maps to Extraversion"),
    Question("ex_f3","hexaco","Extraversion","followup",
        "Do you generally expect things to work out well for you?",
        "Positive expectation is a reliable Extraversion indicator"),

    # ── Agreeableness (HEXACO — patience/tolerance, not warmth) ──
    Question("ag_01","hexaco","Agreeableness","open",
        "When someone cuts in front of you in a queue, what is your typical reaction?",
        "HEXACO Agreeableness = patience/tolerance, not warmth"),
    Question("ag_02","hexaco","Agreeableness","open",
        "How do you handle it when someone is rude or dismissive to you?",
        "Forgiving vs critical/grudge-holding"),
    Question("ag_03","hexaco","Agreeableness","open",
        "Do you tend to assume the best of people you do not know well, or are you more cautious?",
        "Trust/charity toward others"),
    Question("ag_04","hexaco","Agreeableness","open",
        "How quickly do you forgive people who have let you down?",
        "Forgiveness is a direct Agreeableness marker"),
    Question("ag_05","hexaco","Agreeableness","open",
        "Are you the type of person who holds grudges, or do you tend to let things go?",
        "Grudge-holding = low Agreeableness"),
    Question("ag_f1","hexaco","Agreeableness","followup",
        "Do you find it easy or hard to stay calm when someone is being unreasonable with you?",
        "Patience under provocation"),
    Question("ag_f2","hexaco","Agreeableness","followup",
        "Have you ever stayed angry at someone for weeks or months? What happened?",
        "Duration of negative affect toward others"),
    Question("ag_f3","hexaco","Agreeableness","followup",
        "Do you find it easy to see situations from someone else's point of view, even when you disagree with them?",
        "Perspective-taking correlates with Agreeableness"),

    # ── Conscientiousness ────────────────────────────────────
    Question("co_01","hexaco","Conscientiousness","open",
        "How would you describe your relationship with organisation — are you naturally tidy or do you have to work at it?",
        "Orderliness is a core Conscientiousness facet"),
    Question("co_02","hexaco","Conscientiousness","open",
        "When you commit to something, how reliable are you at following through?",
        "Diligence and dependability"),
    Question("co_03","hexaco","Conscientiousness","open",
        "How do you feel when you make a mistake at work or in an important task?",
        "Perfectionism and self-criticism map to Conscientiousness"),
    Question("co_04","hexaco","Conscientiousness","open",
        "Describe how you prepare for something important — like a meeting, exam, or difficult conversation.",
        "Preparation behaviour is a direct C marker"),
    Question("co_05","hexaco","Conscientiousness","open",
        "How often do you start things you do not finish?",
        "Completion tendency = high C; scattered = low C"),
    Question("co_f1","hexaco","Conscientiousness","followup",
        "Would the people who know you best describe you as reliable and dependable?",
        "Social perception of conscientiousness"),
    Question("co_f2","hexaco","Conscientiousness","followup",
        "How do you feel when your environment is cluttered or disorganised?",
        "Discomfort with disorder = high C"),
    Question("co_f3","hexaco","Conscientiousness","followup",
        "Do you think careful planning is usually worth the effort, or does it often feel like wasted time?",
        "Attitude toward planning"),

    # ── Openness ────────────────────────────────────────────
    Question("op_01","hexaco","Openness","open",
        "How curious are you about topics or subjects outside your everyday life?",
        "Intellectual curiosity is the core Openness marker"),
    Question("op_02","hexaco","Openness","open",
        "Do you enjoy art, music, literature, or creative expression? How does it show up in your life?",
        "Aesthetic appreciation correlates with Openness"),
    Question("op_03","hexaco","Openness","open",
        "How do you feel about trying things that are completely unfamiliar to you?",
        "Openness to novel experience"),
    Question("op_04","hexaco","Openness","open",
        "When you encounter a complex idea, do you find yourself wanting to explore it or preferring a clear, simple answer?",
        "Tolerance for complexity and ambiguity"),
    Question("op_05","hexaco","Openness","open",
        "Have you ever changed a deeply held opinion or belief because of something you read or heard? What happened?",
        "Willingness to update beliefs = high Openness"),
    Question("op_f1","hexaco","Openness","followup",
        "Do you find yourself drawn to unusual or unconventional ideas and people?",
        "Attraction to novelty"),
    Question("op_f2","hexaco","Openness","followup",
        "How important is creativity in your work or daily life?",
        "Value placed on creativity"),
    Question("op_f3","hexaco","Openness","followup",
        "Would you rather visit a place you know you will love, or somewhere completely new and unpredictable?",
        "Novelty-seeking vs comfort-seeking"),
]


# ══════════════════════════════════════════════════════════════
#  QUESTION SELECTOR
# ══════════════════════════════════════════════════════════════

class QuestionSelector:
    """
    Reads ProfileManager.uncertain_dims_ranked() and returns
    the best unasked question for the most uncertain dimension.

    Strategy:
      1. Get ranked list of dimensions by uncertainty
      2. Find the top dimension that still has unasked questions
      3. Within that dimension, prefer "open" over "followup"
         (only ask followup if at least one open has been asked)
      4. Within depth, pick randomly to avoid feeling scripted

    Tracks which questions have been asked so it never repeats.
    """

    def __init__(self, bank: List[Question] = None):
        self.bank        = bank or QUESTION_BANK
        self.asked_ids   = set()          # question IDs already used
        self.asked_dims  = {}             # dim → count of questions asked

        # Index bank by dimension for fast lookup
        self._by_dim: Dict[str, List[Question]] = {}
        for q in self.bank:
            self._by_dim.setdefault(q.dimension, []).append(q)

    def select(self, profile: ProfileManager) -> Optional[Question]:
        """
        Return the best next question given the current profile,
        or None if all questions have been asked.
        """
        ranked = profile.uncertain_dims_ranked()

        for entry in ranked:
            dim       = entry["dimension"]
            available = self._available_for(dim)
            if not available:
                continue

            # Prefer open questions unless we already asked one open for this dim
            asked_open = self._asked_open_count(dim)
            opens   = [q for q in available if q.depth == "open"]
            followups = [q for q in available if q.depth == "followup"]

            if opens:
                chosen = random.choice(opens)
            elif followups and asked_open >= 1:
                chosen = random.choice(followups)
            elif followups:
                # No open left but haven't asked an open yet — skip this dim
                continue
            else:
                continue

            return chosen

        return None  # all questions exhausted

    def mark_asked(self, question: Question) -> None:
        """Call this after showing a question to the user."""
        self.asked_ids.add(question.id)
        self.asked_dims[question.dimension] = \
            self.asked_dims.get(question.dimension, 0) + 1

    def _available_for(self, dim: str) -> List[Question]:
        return [
            q for q in self._by_dim.get(dim, [])
            if q.id not in self.asked_ids
        ]

    def _asked_open_count(self, dim: str) -> int:
        return sum(
            1 for q in self.bank
            if q.dimension == dim
            and q.depth == "open"
            and q.id in self.asked_ids
        )

    def remaining_count(self) -> int:
        return sum(
            1 for q in self.bank
            if q.id not in self.asked_ids
        )

    def summary(self) -> Dict:
        dims = sorted(self._by_dim.keys())
        return {
            dim: {
                "total":     len(self._by_dim[dim]),
                "asked":     self.asked_dims.get(dim, 0),
                "remaining": len(self._available_for(dim)),
            }
            for dim in dims
        }


# ══════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════

def run_self_test():
    print(f"\n{SEP}")
    print("  Phase 4  —  Question bank + selector self-test")
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

    # ── 1. Bank structure ─────────────────────────────────────
    print(f"\n  {SEP2}\n  1 · Question bank structure\n  {SEP2}")

    dims_expected = {
        "mbti":   {"IE","NS","TF","JP"},
        "hexaco": {"Honesty-Humility","Emotionality","Extraversion",
                   "Agreeableness","Conscientiousness","Openness"},
    }
    all_dims = {q.dimension for q in QUESTION_BANK}
    expected_all = dims_expected["mbti"] | dims_expected["hexaco"]

    check("Bank has 80 questions",
          len(QUESTION_BANK) == 80,
          f"got {len(QUESTION_BANK)}")
    check("All 10 dimensions covered",
          all_dims == expected_all,
          f"missing: {expected_all - all_dims}")
    check("Each dim has 8 questions",
          all(
              sum(1 for q in QUESTION_BANK if q.dimension == d) == 8
              for d in expected_all
          ),
          str({d: sum(1 for q in QUESTION_BANK if q.dimension == d)
               for d in expected_all}))
    check("Each dim has 5 open + 3 followup",
          all(
              sum(1 for q in QUESTION_BANK if q.dimension == d and q.depth == "open") == 5
              and
              sum(1 for q in QUESTION_BANK if q.dimension == d and q.depth == "followup") == 3
              for d in expected_all
          ))

    ids = [q.id for q in QUESTION_BANK]
    check("All question IDs unique",
          len(ids) == len(set(ids)),
          f"duplicates: {[i for i in ids if ids.count(i) > 1]}")
    check("All questions have non-empty text",
          all(len(q.text.strip()) > 10 for q in QUESTION_BANK))

    # ── 2. Selector on fresh profile ──────────────────────────
    print(f"\n  {SEP2}\n  2 · Selector on fresh profile\n  {SEP2}")

    profile  = ProfileManager()
    selector = QuestionSelector()

    q1 = selector.select(profile)
    check("select() returns a Question on fresh profile",
          q1 is not None)
    check("First question is 'open' depth",
          q1 is not None and q1.depth == "open",
          f"got depth={q1.depth if q1 else None}")

    # ── 3. Selector tracks asked + no repeats ─────────────────
    print(f"\n  {SEP2}\n  3 · No repeated questions\n  {SEP2}")

    profile2  = ProfileManager()
    selector2 = QuestionSelector()
    asked     = []

    # Simulate 10 turns: ask question, fake a strong signal to shift profile
    for turn in range(10):
        q = selector2.select(profile2)
        if q is None:
            break
        check(f"Turn {turn+1}: question not repeated",
              q.id not in [x.id for x in asked],
              f"repeated: {q.id}")
        selector2.mark_asked(q)
        asked.append(q)

        # Inject a signal to move the profile (varies by turn to exercise different dims)
        import numpy as np
        logits = np.random.normal(0, 1, 16)
        hexaco = np.random.uniform(0.2, 0.8, 6)
        profile2.update(logits, hexaco)

    check("10 unique questions asked",
          len(set(q.id for q in asked)) == 10,
          f"unique={len(set(q.id for q in asked))}")

    # ── 4. Selector targets most uncertain dim ─────────────────
    print(f"\n  {SEP2}\n  4 · Selector targets uncertainty correctly\n  {SEP2}")

    import numpy as np

    # Create profile where IE is very certain (99% I) but TF is 50/50
    profile3  = ProfileManager()
    selector3 = QuestionSelector()

    # Push IE to be very certain (all I types)
    for _ in range(5):
        lg = np.full(16, -5.0)
        for t in ["INFJ","INFP","INTJ","INTP","ISTJ","ISTP","ISFJ","ISFP"]:
            from phase3_profile import MBTI2ID
            lg[MBTI2ID[t]] = 3.0
        profile3.update(lg, np.full(6, 0.5))

    q = selector3.select(profile3)
    check("Selector returns a question",
          q is not None)
    check("Selector question is not IE (IE already certain)",
          q is not None and q.dimension != "IE",
          f"got dimension={q.dimension if q else None}")

    # ── 5. Exhaustion handling ────────────────────────────────
    print(f"\n  {SEP2}\n  5 · Graceful exhaustion\n  {SEP2}")

    selector4 = QuestionSelector()
    # Mark every question as asked
    for q in QUESTION_BANK:
        selector4.mark_asked(q)
    result = selector4.select(ProfileManager())
    check("select() returns None when all questions asked",
          result is None)
    check("remaining_count() == 0",
          selector4.remaining_count() == 0)

    # ── 6. Summary printout ───────────────────────────────────
    print(f"\n  {SEP2}\n  6 · Bank summary\n  {SEP2}")
    sel_tmp = QuestionSelector()
    smry = sel_tmp.summary()
    print(f"  {'Dimension':<24} {'Total':>5} {'Asked':>5} {'Left':>5}")
    print(f"  {'─'*24} {'─'*5} {'─'*5} {'─'*5}")
    for dim, s in sorted(smry.items()):
        print(f"  {dim:<24} {s['total']:>5} {s['asked']:>5} {s['remaining']:>5}")
    check("Summary covers all 10 dims", len(smry) == 10)

    # ── Result ────────────────────────────────────────────────
    print(f"\n{SEP}")
    total = len(passed) + len(failed)
    print(f"  {len(passed)}/{total} checks passed")
    if failed:
        print(f"\n  ❌  Failed:")
        for f in failed:
            print(f"      · {f}")
    else:
        print("""
  ✅  Question bank and selector working correctly.

  Next:  python3 phase5_loop.py   (Phase 5 — conversation loop)
""")


if __name__ == "__main__":
    random.seed(42)
    run_self_test()