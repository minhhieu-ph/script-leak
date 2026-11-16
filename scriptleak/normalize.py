"""Strict spoken-form normalisation.

Principle: filter, do not guess. Changes that cannot alter what was spoken are
applied (Unicode normalisation, case, deleting punctuation). Anything whose
spoken form is ambiguous (digits, symbols, foreign-script words, repetition and
abbreviation marks) makes the utterance *rejected*, with the reason recorded.

Two entry points:
  normalize(text, lang)        strict, for building training/test manifests
  scoring_normalize(text, ...) lenient, never rejects, for scoring model output
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

from .languages import Language
from .scripts import script_of

# Symbols that are read aloud as words, so deleting them would break the
# audio/text match.
SPOKEN_SYMBOLS = frozenset("%&@#+=/$€£¥฿°×÷<>^~*_|\\§")

_APOSTROPHES = frozenset("'’‘ʼ`")


@dataclass(frozen=True)
class NormResult:
    text: str
    ok: bool
    reasons: Tuple[str, ...]


def _prepare(text: str) -> str:
    """Unicode steps that are safe for every language."""
    text = unicodedata.normalize("NFKC", text)
    # NFKC decomposes Thai SARA AM (U+0E33) into NIKHAHIT + SARA AA. Recompose
    # so Thai keeps its standard spelling.
    text = text.replace("ํา", "ำ")
    # Zero-width and other format characters (common in Thai web text).
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def normalize(text: str, lang: Language) -> NormResult:
    """Strict spoken-form normalisation of a reference transcript."""
    text = _prepare(text)
    if lang.lowercase:
        text = text.lower()

    out: List[str] = []
    reasons: List[str] = []

    def reject(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    for ch in text:
        if ch.isspace():
            out.append(" ")
            continue
        if ch in lang.reject_chars:
            reject(lang.reject_chars[ch])
            continue
        cat = unicodedata.category(ch)
        if cat[0] == "N":
            reject("digit")
            continue
        if ch in _APOSTROPHES:
            if lang.keep_apostrophe:
                out.append("'")
            else:
                out.append(" ")
            continue
        if lang.script.contains(ch):
            if lang.alphabet is not None and ch not in lang.alphabet:
                reject("out_of_alphabet")
            else:
                out.append(ch)
            continue
        other = script_of(ch)
        if other is not None:
            reject(f"foreign_script:{other}")
            continue
        if ch in SPOKEN_SYMBOLS or cat[0] == "S":
            reject("spoken_symbol")
            continue
        if cat[0] == "P":
            out.append(" ")
            continue
        if cat[0] == "L":
            reject("foreign_script:other")
            continue
        reject("unknown_char")

    norm = " ".join("".join(out).split())
    if lang.keep_apostrophe:
        # Drop apostrophes left dangling by removed quotes.
        norm = " ".join(w.strip("'") for w in norm.split())
        norm = " ".join(norm.split())
    if lang.remove_spaces:
        norm = norm.replace(" ", "")
    if not norm:
        reject("empty")
    return NormResult(text=norm, ok=not reasons, reasons=tuple(reasons))


def scoring_normalize(text: str, lowercase: bool = True) -> str:
    """Lenient normalisation applied identically to reference and hypothesis.

    Never rejects and never removes letters of any script, so leaked characters
    survive and can be counted.
    """
    text = _prepare(text)
    if lowercase:
        text = text.lower()
    out: List[str] = []
    for ch in text:
        if ch.isspace():
            out.append(" ")
        elif ch in _APOSTROPHES:
            out.append("'")
        elif unicodedata.category(ch)[0] == "P":
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())
