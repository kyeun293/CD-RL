"""Pass@k scoring for AIME rollouts.

Extracts each sample's boxed final answer and compares it against the ground-truth
answer using verl's DAPO math verifier in strict \\boxed{} mode -- this matches the
prompt format `sampling.VLLMSampler._format` uses ("put your final answer within
\\boxed{}"), unlike verl's own training-time default (strict_box_verify=False), which
expects the "Answer: X" format used by DAPO's step_instruction prompt instead.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np

_VERL_ROOT = Path(__file__).resolve().parent.parent.parent / "verl"
if str(_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERL_ROOT))

from verl.utils.reward_score.math_dapo import verify  # noqa: E402


def comb_estimator(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, Codex): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _ks_for(n: int) -> list[int]:
    """Doubling schedule of k's up to n (1, 2, 4, ..., n)."""
    ks = []
    k = 1
    while k < n:
        ks.append(k)
        k *= 2
    ks.append(n)
    return ks


def score_pass_at_k(problems: list[dict[str, Any]], rollouts: list[list[str]]) -> dict[str, Any]:
    """Compute pass@k over a set of problems, each with n sampled solutions.

    Args:
        problems: list of {"id", "problem", "answer"}.
        rollouts: list of list[str], shape [num_problems, n] (aligned with `problems`).

    Returns:
        {
            "per_question": [{"id", "n", "num_correct", "pass@k": {k: val, ...}}, ...],
            "pass@k": {k: mean over questions with >= k samples},
        }
    """
    per_question = []
    for prob, samples in zip(problems, rollouts):
        n = len(samples)
        num_correct = sum(bool(verify(s, prob["answer"], strict_box_verify=True)[0]) for s in samples)
        pass_at_k = {k: comb_estimator(n, num_correct, k) for k in _ks_for(n)} if n > 0 else {}
        per_question.append({
            "id": prob["id"],
            "n": n,
            "num_correct": num_correct,
            "pass@k": pass_at_k,
        })

    all_ks = sorted({k for q in per_question for k in q["pass@k"]})
    agg = {}
    for k in all_ks:
        vals = [q["pass@k"][k] for q in per_question if k in q["pass@k"]]
        if vals:
            agg[k] = float(np.mean(vals))

    return {"per_question": per_question, "pass@k": agg}
