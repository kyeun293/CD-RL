"""
Distinct Equations diversity metric.

From: "Diversity-Aware Policy Optimization for Large Language Model Reasoning"
      (arXiv:2505.23433)

Definition (per question i):
    DE_i = |U_i| / |A_i|
  where
    A_i = multiset of *all* equations extracted from k sampled responses
    U_i = set of *unique* equations (after canonicalization)

Aggregate:
    DE = (1/N) * sum_i DE_i

Equation extraction strategy (kept simple and dependency-light):
  1. Pull every LaTeX inline ($...$) and display ($$...$$, \\[...\\]) math span.
  2. Also catch bare lines that look like equations: contain '=' and at least
     one digit or variable token.
  3. Canonicalize:
       - strip whitespace, latex \\, \\left/\\right, \\,, \\;, \\!, \\quad, \\qquad
       - lowercase
       - normalize unicode minus / multiplication / division
       - drop trailing punctuation
  4. Keep only spans that contain '=' (an "equation", not a lone expression).

This matches the spirit of the paper: it is a coarse, fast, structure-level
diversity signal — not a perfect math parser. The whole point is that two
genuinely different solution paths produce different intermediate equations.
"""

import re
import unicodedata
from typing import Iterable, List, Set


_LATEX_INLINE = re.compile(r"\$(.+?)\$", re.DOTALL)
_LATEX_DISPLAY = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
_BARE_EQ_LINE = re.compile(r"^[^A-Za-z\n]*[A-Za-z0-9_\\^{}()+\-*/\s]*=.+$", re.MULTILINE)

_LATEX_NOISE = re.compile(
    r"\\left|\\right|\\,|\\;|\\!|\\quad|\\qquad|\\displaystyle|\\text\{[^}]*\}"
)
_WS = re.compile(r"\s+")


def _extract_eq_spans(text: str) -> List[str]:
    spans: List[str] = []

    # Display math first (so we don't double-match its inner $).
    cleaned = text
    for m in _LATEX_DISPLAY.finditer(text):
        s = m.group(1) or m.group(2) or ""
        spans.append(s)
        cleaned = cleaned.replace(m.group(0), " ")

    for m in _LATEX_INLINE.finditer(cleaned):
        spans.append(m.group(1))

    # Plain text lines that look like "x = 3" or "n^2 + 1 = 50".
    for m in _BARE_EQ_LINE.finditer(text):
        line = m.group(0)
        # Skip lines that are just prose with a stray '='.
        if line.count(" ") < 25:
            spans.append(line)

    return spans


def _canonicalize(eq: str) -> str:
    # Unicode normalize + replace fancy operators.
    eq = unicodedata.normalize("NFKC", eq)
    eq = (eq.replace("−", "-")
            .replace("×", "*")
            .replace("·", "*")
            .replace("÷", "/"))
    eq = _LATEX_NOISE.sub("", eq)
    eq = eq.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    eq = eq.replace("\\frac", "frac")  # don't try to fully parse, just normalize token
    eq = eq.replace("{", "").replace("}", "")
    eq = _WS.sub("", eq)
    eq = eq.lower().rstrip(".,;:")
    return eq


def extract_equations(response: str) -> List[str]:
    """Return canonicalized equation strings from a single response."""
    out = []
    for span in _extract_eq_spans(response):
        if "=" not in span:
            continue
        c = _canonicalize(span)
        if not c or len(c) < 3 or c.count("=") == 0:
            continue
        out.append(c)
    return out


def distinct_equations_score(samples: Iterable[str]) -> float:
    """
    DE_i = |U_i| / |A_i|  for one question's k sampled responses.

    If a question yields zero extractable equations across all samples,
    we return 1.0 (degenerate: nothing to repeat, treat as fully distinct)
    rather than 0/0. Flip to `0.0` if you'd rather penalize that case.
    """
    all_eqs: List[str] = []
    for r in samples:
        all_eqs.extend(extract_equations(r))
    if not all_eqs:
        return 1.0
    unique: Set[str] = set(all_eqs)
    return len(unique) / len(all_eqs)
    