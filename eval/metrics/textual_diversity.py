"""
BLEU-based textual diversity (Self-BLEU).

For each sample among k, compute BLEU-4 treating the other k-1 samples as
references; average over samples to get mean self-BLEU (Zhu et al., 2018,
"Texygen"). Low self-BLEU = high n-gram diversity. We report

    textual_diversity = 1 - mean_self_bleu

so higher is more diverse, matching the direction of the other diversity
metrics (RPD, Distinct Equations).

Dependency-free BLEU-4 implementation (word-level, add-1 smoothing per
Chen & Cherry 2014 method 1, so short/degenerate generations don't collapse
to a hard 0) — avoids pulling in nltk/sacrebleu just for this.
"""

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _sentence_bleu(candidate: Sequence[str], references: List[Sequence[str]], max_n: int = 4) -> float:
    if not candidate:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_counts = _ngram_counts(candidate, n)
        # NOTE: when the candidate is shorter than n, cand_counts is empty
        # and both clipped/total below are 0 — the add-1 smoothing formula
        # naturally gives precision 1.0 (nothing to penalize) rather than a
        # raw 0.0, which would blow up math.log() below.
        max_ref_counts = Counter()
        for ref in references:
            ref_counts = _ngram_counts(ref, n)
            for ngram, cnt in ref_counts.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], cnt)
        clipped = sum(min(cnt, max_ref_counts.get(ngram, 0)) for ngram, cnt in cand_counts.items())
        total = sum(cand_counts.values())
        # Add-1 smoothing (Chen & Cherry 2014, method 1) so zero-overlap
        # n-grams don't zero out the whole geometric mean.
        precisions.append((clipped + 1) / (total + 1))

    log_avg = sum(math.log(p) for p in precisions) / max_n
    geo_mean = math.exp(log_avg)

    cand_len = len(candidate)
    ref_len = min(references, key=lambda r: abs(len(r) - cand_len))
    ref_len = len(ref_len)
    if cand_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0

    return bp * geo_mean


def self_bleu(samples: Iterable[str], k: "int | None" = None) -> float:
    """Mean self-BLEU over `samples` (optionally truncated to the first k)."""
    texts = list(samples)
    if k is not None:
        texts = texts[:k]
    tokenized = [_tokenize(t) for t in texts]
    if len(tokenized) < 2:
        return 0.0

    scores = []
    for i, cand in enumerate(tokenized):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        scores.append(_sentence_bleu(cand, refs))
    return sum(scores) / len(scores)


def textual_diversity_score(samples: Iterable[str], k: "int | None" = None) -> float:
    """1 - mean self-BLEU: higher means more textually diverse."""
    return 1.0 - self_bleu(samples, k=k)
