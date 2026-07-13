"""
Correctness checking + k-dependent accuracy metrics: Avg@k, Pass@k, Potential@k.

Answer extraction/normalization is adapted from verl's
`verl/utils/reward_score/math_dapo.py` (itself adapted from the
lm-evaluation-harness MATH utils), vendored here so `eval/` doesn't need the
full `verl` package as an import-time dependency.

Pass@k uses the standard unbiased estimator from the Codex/HumanEval paper
(Chen et al., 2021):

    Pass@k(n, c) = 1 - C(n-c, k) / C(n, k)

where n = number of sampled completions, c = number of correct ones.

Potential@k measures how much of the *remaining* headroom above Pass@1 is
recovered by sampling k completions instead of 1 — i.e. "for problems pass@1
gets wrong, how often does pass@k rescue them":

    Potential@k = sum_i Pass@k(q_i) * (1 - Pass@1(q_i))
                  -----------------------------------------
                  sum_i (1 - Pass@1(q_i))
"""

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Answer extraction / normalization (vendored from verl math_dapo.py)
# ---------------------------------------------------------------------------

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    assert s[:len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left):-1]


SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","), ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km", "units",
    "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents", "degrees", "cm",
    "gm", "pounds", "meters", "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}",
    "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def extract_pred_answer(solution_str: str) -> Optional[str]:
    """Extract + normalize the last \\boxed{...} answer from a solution."""
    boxed = last_boxed_only_string(solution_str[-300:] if len(solution_str) > 300 else solution_str)
    if boxed is None:
        # boxed answers can be far from the tail on long CoT; retry on full text.
        boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None
    return normalize_final_answer(remove_boxed(boxed))


def is_correct(solution_str: str, ground_truth: str) -> bool:
    pred = extract_pred_answer(solution_str)
    if pred is None:
        return False
    return pred == normalize_final_answer(ground_truth)


def check_samples(samples: Sequence[str], ground_truth: str) -> List[bool]:
    """Correctness of each sampled response for one question."""
    return [is_correct(s, ground_truth) for s in samples]


# ---------------------------------------------------------------------------
# Pass@k (unbiased estimator, Chen et al. 2021 / OpenAI human-eval)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k samples drawn (without replacement)
    from n samples (c of which are correct) is correct."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


# ---------------------------------------------------------------------------
# Avg@k / Pass@k / Potential@k aggregated over a dataset
# ---------------------------------------------------------------------------

def avg_at_k(per_question_correct: Sequence[Sequence[bool]], k: int) -> float:
    """Mean accuracy over k samples per question, averaged over questions."""
    vals = []
    for correct in per_question_correct:
        assert len(correct) >= k, f"need >= {k} samples, got {len(correct)}"
        vals.append(sum(correct[:k]) / k)
    return sum(vals) / max(len(vals), 1)


def pass_at_k_dataset(per_question_correct: Sequence[Sequence[bool]], k: int) -> float:
    """Pass@k averaged over questions, using all available samples per question
    as the (n, c) pool for the unbiased estimator."""
    vals = [pass_at_k(len(c), sum(c), k) for c in per_question_correct]
    return sum(vals) / max(len(vals), 1)


def potential_at_k(per_question_correct: Sequence[Sequence[bool]], k: int) -> Optional[float]:
    """Potential@k := sum_i Pass@k(q_i)*(1-Pass@1(q_i)) / sum_i (1-Pass@1(q_i)).

    Weights each question by how much headroom Pass@1 left on the table, so
    questions already solved at Pass@1=1 don't dilute the score. Returns
    None if every question is already solved at Pass@1 (denominator is 0).
    """
    num = 0.0
    den = 0.0
    for correct in per_question_correct:
        n, c = len(correct), sum(correct)
        p1 = pass_at_k(n, c, 1)
        pk = pass_at_k(n, c, k)
        w = 1.0 - p1
        num += pk * w
        den += w
    if den == 0.0:
        return None
    return num / den


def accuracy_summary(
    per_question_correct: Sequence[Sequence[bool]],
    ks: Sequence[int],
) -> dict:
    """Bundle Avg@k / Pass@k / Potential@k for each k in `ks`."""
    out = {}
    for k in ks:
        out[k] = {
            "avg": avg_at_k(per_question_correct, k),
            "pass": pass_at_k_dataset(per_question_correct, k),
            "potential": potential_at_k(per_question_correct, k),
        }
    return out
