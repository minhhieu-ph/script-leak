"""Leakage and accuracy metrics.

Leakage is measured only on utterances whose *reference* is purely in the
majority script, so every minority-script character in the hypothesis is an
error by construction.

  SLR-u  share of eligible utterances whose hypothesis has >= 1 minority char
  SLR-c  minority-script chars / all script chars in the hypotheses
  SLR-w  share of hypothesis words containing a minority char (needs a word
         tokenizer; whitespace by default, pass one for Thai or Chinese)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .languages import Language
from .scripts import Script


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance between two sequences."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def cer(refs: Sequence[str], hyps: Sequence[str], ignore_spaces: bool = False) -> float:
    errs = n = 0
    for r, h in zip(refs, hyps):
        if ignore_spaces:
            r, h = r.replace(" ", ""), h.replace(" ", "")
        errs += edit_distance(r, h)
        n += len(r)
    return errs / n if n else 0.0


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    errs = n = 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        errs += edit_distance(rw, hw)
        n += len(rw)
    return errs / n if n else 0.0


def minority_spans(text: str, minority: Script) -> List[str]:
    """Maximal runs of minority-script characters. Spaces and apostrophes
    inside a run do not break it, so "new york" is one span."""
    spans: List[str] = []
    cur: List[str] = []
    pending: List[str] = []
    for ch in text:
        if minority.contains(ch):
            if cur:
                cur.extend(pending)
            pending = []
            cur.append(ch)
        elif cur and (ch.isspace() or ch == "'"):
            pending.append(ch)
        else:
            if cur:
                spans.append("".join(cur))
            cur, pending = [], []
    if cur:
        spans.append("".join(cur))
    return spans


@dataclass
class LeakStats:
    utterances_total: int
    utterances_eligible: int
    utterances_leaked: int
    slr_u: float
    slr_c: float
    slr_w: float
    leak_spans: int
    mean_span_chars: float
    max_span_chars: int

    def to_dict(self) -> Dict:
        return asdict(self)


def leak_stats(
    refs: Sequence[str],
    hyps: Sequence[str],
    majority: Language,
    minority: Language,
    word_tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> LeakStats:
    """Leakage of `minority` script into utterances spoken in `majority`.

    Call with the two languages swapped to get the reverse direction.
    """
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have the same length")
    tokenize = word_tokenizer or str.split
    maj, mino = majority.script, minority.script

    eligible = leaked = 0
    min_chars = all_chars = 0
    words = leaked_words = 0
    span_lens: List[int] = []

    for ref, hyp in zip(refs, hyps):
        # Eligible: reference has majority-script text and no minority script.
        if mino.count(ref) > 0 or maj.count(ref) == 0:
            continue
        eligible += 1
        n_min = mino.count(hyp)
        min_chars += n_min
        all_chars += n_min + maj.count(hyp)
        if n_min:
            leaked += 1
            span_lens.extend(mino.count(s) for s in minority_spans(hyp, mino))
        for w in tokenize(hyp):
            words += 1
            if mino.count(w):
                leaked_words += 1

    return LeakStats(
        utterances_total=len(refs),
        utterances_eligible=eligible,
        utterances_leaked=leaked,
        slr_u=leaked / eligible if eligible else 0.0,
        slr_c=min_chars / all_chars if all_chars else 0.0,
        slr_w=leaked_words / words if words else 0.0,
        leak_spans=len(span_lens),
        mean_span_chars=sum(span_lens) / len(span_lens) if span_lens else 0.0,
        max_span_chars=max(span_lens) if span_lens else 0,
    )
