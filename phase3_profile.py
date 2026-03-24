"""
phase3_profile.py  —  ProfileManager
══════════════════════════════════════
Phase 3 · Step 1

The ProfileManager holds the live personality estimate for one
user session and updates it each time the classifier scores an
answer.

Design decisions (informed by Phase 2 results):
  · Tracks full 16-type probability distribution, not just top-1
    (justified by 68.8% top-3 accuracy from Phase 2)
  · Bayesian update: posterior ∝ prior × likelihood^decay
    decay = 1 / (1 + n_updates) so early answers count more
  · HEXACO uses online running mean + variance (Welford algorithm)
    stable, no need to store all past scores
  · most_uncertain() drives Phase 4 question selection:
      - For MBTI: returns highest-entropy dichotomy axis
      - For HEXACO: returns dim with highest variance
  · is_converged() gives Phase 6 the stopping signal

Run self-test:
    python3 phase3_profile.py
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MBTI_TYPES = [
    "INFJ","INFP","INTJ","INTP","ISFJ","ISFP","ISTJ","ISTP",
    "ENFJ","ENFP","ENTJ","ENTP","ESFJ","ESFP","ESTJ","ESTP",
]
MBTI2ID = {t: i for i, t in enumerate(MBTI_TYPES)}

# Each dichotomy axis: index → (char_A, char_B)
DICHOTOMIES = [
    ("I", "E"),   # axis 0
    ("N", "S"),   # axis 1
    ("T", "F"),   # axis 2
    ("J", "P"),   # axis 3
]

HEXACO_DIMS = [
    "Honesty-Humility",
    "Emotionality",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Openness",
]
# H-H is derived — trust it less
HEXACO_WEIGHTS = {
    "Honesty-Humility":  0.5,
    "Emotionality":      1.0,
    "Extraversion":      1.0,
    "Agreeableness":     1.0,
    "Conscientiousness": 1.0,
    "Openness":          1.0,
}

SEP  = "═" * 62
SEP2 = "─" * 62


# ══════════════════════════════════════════════════════════════
#  PROFILE MANAGER
# ══════════════════════════════════════════════════════════════

class ProfileManager:
    """
    Holds and updates the live personality estimate for one user.

    State after construction (before any answers):
      MBTI  — uniform distribution over 16 types (max uncertainty)
      HEXACO — all dims at 0.5 with max variance

    Each call to update() blends the new classifier output into
    the running estimate using a decayed Bayesian update so that
    later answers don't completely overwrite early ones.

    Usage:
        profile = ProfileManager()
        profile.update(mbti_logits, hexaco_scores)   # after each answer
        target = profile.most_uncertain()            # what to ask next
        if profile.is_converged():                   # when to stop
            report = profile.to_dict()
    """

    def __init__(
        self,
        tau_mbti: float = 1.5,    # entropy threshold (nats) for MBTI convergence
        tau_hex:  float = 0.04,   # variance threshold for HEXACO convergence
        decay:    float = 0.7,    # likelihood decay per update (0-1)
    ):
        self.tau_mbti = tau_mbti
        self.tau_hex  = tau_hex
        self.decay    = decay

        # ── MBTI state ────────────────────────────────────────
        # Start with uniform prior: equal probability for all 16 types
        self._type_logits = np.zeros(16, dtype=np.float64)  # log-space prior
        self.n_mbti_updates = 0

        # ── HEXACO state (Welford online mean + variance) ─────
        self._hex_mean = np.full(6, 0.5, dtype=np.float64)   # prior: 0.5
        self._hex_M2   = np.full(6, 0.25, dtype=np.float64)  # prior variance: 0.25
        self._hex_n    = np.zeros(6, dtype=np.int64)          # observations per dim

        # ── History ───────────────────────────────────────────
        self.history: List[Dict] = []   # one entry per update call

    # ── Core update ──────────────────────────────────────────

    def update(
        self,
        mbti_logits: np.ndarray,    # [16]  raw logits from classifier
        hexaco_scores: np.ndarray,  # [6]   scores in [0, 1] from classifier
    ) -> None:
        """
        Blend new classifier output into the running estimate.
        Call this after every user answer.
        """
        self._update_mbti(mbti_logits)
        self._update_hexaco(hexaco_scores)

        snap = self._snapshot()
        self.history.append(snap)

    # ── MBTI update  (log-space Bayesian blend) ───────────────

    def _update_mbti(self, logits: np.ndarray) -> None:
        """
        Posterior log-prob ∝ prior_log + decay^n × log_likelihood

        decay^n means the nth update has weight decay^(n-1) relative
        to the first — early strong signals dominate, later answers
        fine-tune rather than reset.
        """
        log_likelihood = self._log_softmax(np.array(logits, dtype=np.float64))
        weight = self.decay ** self.n_mbti_updates
        self._type_logits = self._type_logits + weight * log_likelihood
        self.n_mbti_updates += 1

    # ── HEXACO update  (Welford online algorithm) ─────────────

    def _update_hexaco(self, scores: np.ndarray) -> None:
        """
        Welford's algorithm: stable online mean and variance.
        Each dimension is updated independently.
        Uses HEXACO_WEIGHTS to downweight derived dimensions.
        """
        scores = np.clip(np.array(scores, dtype=np.float64), 0.0, 1.0)
        for i, dim in enumerate(HEXACO_DIMS):
            w = HEXACO_WEIGHTS[dim]
            if w < 1.0:
                # For derived dims: blend toward 0.5 with reduced weight
                # so it doesn't drift too far from the centre
                scores[i] = w * scores[i] + (1 - w) * 0.5

            self._hex_n[i]   += 1
            delta             = scores[i] - self._hex_mean[i]
            self._hex_mean[i] += delta / self._hex_n[i]
            delta2            = scores[i] - self._hex_mean[i]
            self._hex_M2[i]  += delta * delta2

    # ── Properties ───────────────────────────────────────────

    @property
    def type_probs(self) -> np.ndarray:
        """Probability distribution over 16 MBTI types. [16]"""
        return self._softmax(self._type_logits)

    @property
    def predicted_type(self) -> str:
        """Most probable MBTI type."""
        return MBTI_TYPES[int(self.type_probs.argmax())]

    @property
    def top3_types(self) -> List[Tuple[str, float]]:
        """Top 3 MBTI types with probabilities."""
        idx = self.type_probs.argsort()[::-1][:3]
        return [(MBTI_TYPES[i], float(self.type_probs[i])) for i in idx]

    @property
    def dichotomy_probs(self) -> Dict[str, float]:
        """
        P(first pole) for each dichotomy axis.
        E.g. {"I": 0.77, "N": 0.63, "T": 0.51, "J": 0.42}
        """
        probs = self.type_probs
        result = {}
        for i, (a, b) in enumerate(DICHOTOMIES):
            # Sum probabilities of all types containing pole A
            p_a = sum(
                probs[MBTI2ID[t]]
                for t in MBTI_TYPES
                if t[i] == a
            )
            result[a] = float(p_a)
            result[b] = float(1 - p_a)
        return result

    @property
    def mbti_entropy(self) -> float:
        """
        Shannon entropy of type distribution (nats).
        Max = ln(16) ≈ 2.77 (uniform), 0 = certain.
        """
        p = np.clip(self.type_probs, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    @property
    def mbti_confidence(self) -> float:
        """
        Confidence in [0, 1]:  1 = certain,  0 = uniform.
        Derived from entropy: 1 - entropy / max_entropy
        """
        max_entropy = math.log(16)
        return float(1.0 - self.mbti_entropy / max_entropy)

    @property
    def hexaco_scores(self) -> Dict[str, float]:
        """Current running mean per HEXACO dimension."""
        return {dim: float(self._hex_mean[i]) for i, dim in enumerate(HEXACO_DIMS)}

    @property
    def hexaco_variance(self) -> Dict[str, float]:
        """
        Current variance per HEXACO dimension.
        Uses Bessel-corrected sample variance where n > 1,
        falls back to prior variance 0.25 for n ≤ 1.
        """
        result = {}
        for i, dim in enumerate(HEXACO_DIMS):
            n = self._hex_n[i]
            if n > 1:
                var = float(self._hex_M2[i] / (n - 1))
            else:
                var = 0.25  # max uncertainty prior
            result[dim] = var
        return result

    @property
    def hexaco_uncertainty(self) -> Dict[str, float]:
        """Normalised uncertainty [0,1] per dim. 1=max uncertain."""
        var = self.hexaco_variance
        # Variance of Bernoulli is at most 0.25 → normalise to [0,1]
        return {dim: min(v / 0.25, 1.0) for dim, v in var.items()}

    # ── What to ask next ─────────────────────────────────────

    def most_uncertain(self) -> Dict:
        """
        Returns the dimension we know the least about.

        Used by Phase 4 (QuestionSelector) to pick the next question.

        Returns a dict with:
          task       : "mbti" | "hexaco"
          dimension  : specific dim name (e.g. "NS" or "Emotionality")
          uncertainty: float in [0, 1]
          reason     : human-readable explanation
        """
        # MBTI: find highest-entropy dichotomy axis
        dprobs = self.dichotomy_probs
        axis_uncertainties = []
        for a, b in DICHOTOMIES:
            p = dprobs[a]
            # Binary entropy: maximised at 0.5
            h = -(p * math.log(p + 1e-12) + (1-p) * math.log(1-p + 1e-12))
            h_norm = h / math.log(2)   # normalise to [0,1]
            axis_uncertainties.append((f"{a}{b}", h_norm, p))

        most_uncertain_axis, mbti_u, p = max(axis_uncertainties, key=lambda x: x[1])
        a_char = most_uncertain_axis[0]
        b_char = most_uncertain_axis[1]

        # HEXACO: highest variance dim
        hex_u = self.hexaco_uncertainty
        most_uncertain_hex = max(hex_u, key=hex_u.get)
        hex_u_val = hex_u[most_uncertain_hex]

        # Pick whichever is more uncertain
        # Weight MBTI a bit higher since it has more types to distinguish
        if mbti_u * 1.1 >= hex_u_val:
            return {
                "task":        "mbti",
                "dimension":   most_uncertain_axis,
                "uncertainty": round(mbti_u, 4),
                "reason":      f"P({a_char})={dprobs[a_char]:.2f} vs P({b_char})={dprobs[b_char]:.2f} — nearly tied",
            }
        else:
            return {
                "task":        "hexaco",
                "dimension":   most_uncertain_hex,
                "uncertainty": round(hex_u_val, 4),
                "reason":      f"variance={self.hexaco_variance[most_uncertain_hex]:.4f} after {self._hex_n[HEXACO_DIMS.index(most_uncertain_hex)]} obs",
            }

    def uncertain_dims_ranked(self) -> List[Dict]:
        """All dimensions ranked by uncertainty — for Phase 4."""
        result = []
        dprobs = self.dichotomy_probs
        for a, b in DICHOTOMIES:
            p = dprobs[a]
            h = -(p * math.log(p + 1e-12) + (1-p) * math.log(1-p + 1e-12))
            result.append({
                "task": "mbti", "dimension": f"{a}{b}",
                "uncertainty": h / math.log(2),
                "current": f"P({a})={p:.2f}",
            })
        hex_u = self.hexaco_uncertainty
        for dim in HEXACO_DIMS:
            result.append({
                "task": "hexaco", "dimension": dim,
                "uncertainty": hex_u[dim],
                "current": f"score={self._hex_mean[HEXACO_DIMS.index(dim)]:.2f}",
            })
        result.sort(key=lambda x: x["uncertainty"], reverse=True)
        return result

    # ── Convergence ───────────────────────────────────────────

    def is_converged(self) -> bool:
        """
        True when both MBTI and HEXACO have settled enough to stop.

        MBTI: entropy < tau_mbti  (we're confident in the type)
        HEXACO: all dims have variance < tau_hex  (scores are stable)
        """
        mbti_done  = self.mbti_entropy < self.tau_mbti
        hex_var    = self.hexaco_variance
        hexaco_done = all(
            v < self.tau_hex
            for dim, v in hex_var.items()
            if dim != "Honesty-Humility"   # skip derived dim
        )
        return mbti_done and hexaco_done

    def convergence_status(self) -> Dict:
        """Detailed convergence report for debugging."""
        hex_var = self.hexaco_variance
        return {
            "mbti_entropy":    round(self.mbti_entropy, 4),
            "mbti_threshold":  self.tau_mbti,
            "mbti_converged":  self.mbti_entropy < self.tau_mbti,
            "hexaco_variances": {d: round(v, 4) for d, v in hex_var.items()},
            "hexaco_threshold": self.tau_hex,
            "hexaco_converged": all(
                v < self.tau_hex for d, v in hex_var.items()
                if d != "Honesty-Humility"
            ),
            "overall_converged": self.is_converged(),
            "n_updates": self.n_mbti_updates,
        }

    # ── Serialisation ─────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Full profile as a plain dict — for storage or the final report."""
        return {
            "mbti": {
                "predicted_type":  self.predicted_type,
                "top3":            self.top3_types,
                "type_probs":      {MBTI_TYPES[i]: round(float(p), 4)
                                    for i, p in enumerate(self.type_probs)},
                "dichotomy_probs": {k: round(v, 4)
                                    for k, v in self.dichotomy_probs.items()},
                "entropy":         round(self.mbti_entropy, 4),
                "confidence":      round(self.mbti_confidence, 4),
                "n_updates":       self.n_mbti_updates,
            },
            "hexaco": {
                "scores":      {d: round(v, 4) for d, v in self.hexaco_scores.items()},
                "variance":    {d: round(v, 4) for d, v in self.hexaco_variance.items()},
                "uncertainty": {d: round(v, 4) for d, v in self.hexaco_uncertainty.items()},
                "n_obs":       {HEXACO_DIMS[i]: int(self._hex_n[i]) for i in range(6)},
            },
            "convergence": self.convergence_status(),
            "most_uncertain": self.most_uncertain(),
        }

    def _snapshot(self) -> Dict:
        """Compact snapshot stored in history after each update."""
        return {
            "update_n":         self.n_mbti_updates,
            "predicted_type":   self.predicted_type,
            "mbti_entropy":     round(self.mbti_entropy, 4),
            "mbti_confidence":  round(self.mbti_confidence, 4),
            "hexaco_scores":    {d: round(v, 4) for d, v in self.hexaco_scores.items()},
            "most_uncertain":   self.most_uncertain(),
        }

    def print_summary(self, title: str = "Profile snapshot"):
        """Pretty-print the current state — useful during development."""
        dp = self.dichotomy_probs
        print(f"\n  {SEP2}")
        print(f"  {title}")
        print(f"  {SEP2}")
        print(f"  MBTI prediction : {self.predicted_type}  "
              f"(confidence {self.mbti_confidence:.1%})")
        print(f"  Top-3           : {self.top3_types}")
        print(f"  Entropy         : {self.mbti_entropy:.3f} / {math.log(16):.3f} nats")
        print(f"\n  Dichotomies:")
        for a, b in DICHOTOMIES:
            pa = dp[a]
            bar_a = "█" * int(pa * 20)
            bar_b = "█" * int((1-pa) * 20)
            winner = a if pa > 0.5 else b
            print(f"    {a} {bar_a:<20} {bar_b:>20} {b}  → {winner} ({max(pa,1-pa):.0%})")
        print(f"\n  HEXACO scores:")
        scores = self.hexaco_scores
        variances = self.hexaco_variance
        for dim in HEXACO_DIMS:
            s = scores[dim]
            v = variances[dim]
            bar = "█" * int(s * 20)
            print(f"    {dim:<22} {bar:<20} {s:.3f}  ±{v:.4f}")
        mu = self.most_uncertain()
        print(f"\n  Most uncertain  : [{mu['task']}] {mu['dimension']}  "
              f"(uncertainty={mu['uncertainty']:.3f})")
        conv = self.convergence_status()
        print(f"  Converged       : {conv['overall_converged']}  "
              f"(MBTI:{conv['mbti_converged']}  HEXACO:{conv['hexaco_converged']})")

    # ── Static helpers ────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    @staticmethod
    def _log_softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        return x - np.log(np.sum(np.exp(x)))


# ══════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════

def run_self_test():
    print(f"\n{SEP}")
    print("  Phase 3  —  ProfileManager self-test")
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

    # ── 1. Fresh profile state ────────────────────────────────
    print(f"\n  {SEP2}\n  1 · Fresh profile (before any updates)\n  {SEP2}")
    p = ProfileManager()

    check("type_probs sums to 1",
          abs(p.type_probs.sum() - 1.0) < 1e-9,
          f"sum = {p.type_probs.sum()}")
    check("type_probs uniform (max entropy)",
          np.allclose(p.type_probs, 1/16, atol=1e-6),
          f"not uniform: {p.type_probs[:4]}")
    check("mbti_entropy == ln(16)",
          abs(p.mbti_entropy - math.log(16)) < 1e-6,
          f"{p.mbti_entropy:.4f} vs {math.log(16):.4f}")
    check("mbti_confidence == 0.0 when uniform",
          abs(p.mbti_confidence) < 1e-6)
    check("hexaco_scores all 0.5",
          all(abs(v - 0.5) < 1e-9 for v in p.hexaco_scores.values()))
    check("hexaco_variance all 0.25 (max prior)",
          all(abs(v - 0.25) < 1e-9 for v in p.hexaco_variance.values()))
    check("not converged when fresh",
          not p.is_converged())

    # ── 2. Single update ──────────────────────────────────────
    print(f"\n  {SEP2}\n  2 · After one update (strong INFP signal)\n  {SEP2}")

    # Simulate strong INFP logits: index 1 = INFP
    logits = np.full(16, -2.0)
    logits[1] = 5.0   # INFP strongly activated

    # Simulate HEXACO scores consistent with INFP (high O, high F tendency)
    hexaco = np.array([0.4, 0.7, 0.3, 0.6, 0.4, 0.8])

    p.update(logits, hexaco)
    p.print_summary("After 1 update — strong INFP signal")

    check("predicted_type == INFP after strong signal",
          p.predicted_type == "INFP",
          f"got {p.predicted_type}")
    check("INFP prob highest",
          p.type_probs[MBTI2ID["INFP"]] == p.type_probs.max())
    check("entropy decreased after update",
          p.mbti_entropy < math.log(16),
          f"{p.mbti_entropy:.4f}")
    check("confidence > 0 after update",
          p.mbti_confidence > 0)
    check("type_probs still sums to 1",
          abs(p.type_probs.sum() - 1.0) < 1e-9)
    check("hexaco_scores updated from 0.5",
          any(abs(v - 0.5) > 0.01 for v in p.hexaco_scores.values()))
    check("history has 1 entry",
          len(p.history) == 1)
    check("most_uncertain() returns valid dict",
          "task" in p.most_uncertain() and "dimension" in p.most_uncertain())

    # ── 3. Multiple updates — convergence ─────────────────────
    print(f"\n  {SEP2}\n  3 · 8 consistent updates — convergence\n  {SEP2}")

    p2 = ProfileManager(tau_mbti=1.0, tau_hex=0.04)
    for i in range(8):
        # Consistent INTJ signal each time (slight variation)
        lg = np.full(16, -3.0)
        lg[MBTI2ID["INTJ"]] = 4.0 + np.random.normal(0, 0.3)
        hx = np.array([0.3, 0.3, 0.4, 0.4, 0.8, 0.7]) + np.random.normal(0, 0.05, 6)
        hx = np.clip(hx, 0.0, 1.0)
        p2.update(lg, hx)

    p2.print_summary("After 8 consistent INTJ updates")

    check("predicted INTJ after 8 consistent updates",
          p2.predicted_type == "INTJ",
          f"got {p2.predicted_type}")
    check("confidence > 60% after 8 updates",
          p2.mbti_confidence > 0.6,
          f"{p2.mbti_confidence:.3f}")
    check("entropy < 1.5 after 8 consistent updates",
          p2.mbti_entropy < 1.5,
          f"{p2.mbti_entropy:.4f}")
    check("history has 8 entries",
          len(p2.history) == 8)

    # ── 4. Conflicting updates — stays uncertain ───────────────
    print(f"\n  {SEP2}\n  4 · Conflicting updates — stays uncertain\n  {SEP2}")

    p3 = ProfileManager()
    types_to_alternate = ["INFP", "INTJ", "ENFP", "INTP"]
    for i in range(8):
        t = types_to_alternate[i % 4]
        lg = np.full(16, -2.0)
        lg[MBTI2ID[t]] = 4.0
        hx = np.random.uniform(0.3, 0.7, 6)
        p3.update(lg, hx)

    p3.print_summary("After 8 conflicting updates")

    # With decay=0.7, the 1st update has weight 1.0, 2nd=0.7, 3rd=0.49...
    # so the first signal (INFP) dominates — the profile is NOT fully uniform.
    # What we verify instead:
    #   · entropy is higher than it would be after 8 CONSISTENT updates
    #   · the profile did NOT fully converge (HEXACO variance still high)
    #   · the first-seen type should rank in top-3 (decay gives it most weight)
    check("entropy higher than consistent-update case (decay effect)",
          p3.mbti_entropy > p2.mbti_entropy,
          f"conflicting={p3.mbti_entropy:.4f}  consistent={p2.mbti_entropy:.4f}")
    check("conflicting profile not fully converged",
          not p3.is_converged(),
          "expected not converged — HEXACO needs more obs")
    check("first-seen type (INFP) in top-3",
          any(t == "INFP" for t, _ in p3.top3_types),
          f"top3={p3.top3_types}")

    # ── 5. to_dict ────────────────────────────────────────────
    print(f"\n  {SEP2}\n  5 · Serialisation\n  {SEP2}")
    d = p.to_dict()
    check("to_dict has mbti key",    "mbti"        in d)
    check("to_dict has hexaco key",  "hexaco"      in d)
    check("to_dict has convergence", "convergence" in d)
    check("to_dict has most_uncertain","most_uncertain" in d)
    check("serialises to valid JSON",
          json.dumps(d) is not None)
    check("uncertain_dims_ranked returns 10 items",
          len(p.uncertain_dims_ranked()) == 10,
          f"got {len(p.uncertain_dims_ranked())}")

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
  ✅  ProfileManager is working correctly.

  Next:  python3 phase4_questions.py   (Phase 4 — question strategy)
""")


if __name__ == "__main__":
    np.random.seed(42)
    run_self_test()