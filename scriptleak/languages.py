"""Per-language spoken-form rules.

Spoken form = the transcript contains only the letters of the language's own
script, written exactly as pronounced: no digits, no punctuation, no symbols,
no foreign-script words, no abbreviation or repetition marks.

To add a language, add one Language entry below. Nothing else in the package
is language-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional

from .scripts import CYRILLIC, HAN, HANGUL, LATIN, THAI, Script


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    script: Script
    # If set, only these characters of the script are accepted.
    alphabet: Optional[FrozenSet[str]] = None
    lowercase: bool = False
    keep_apostrophe: bool = False
    # Remove all spaces from the normalised text (Chinese).
    remove_spaces: bool = False
    # Compute CER with spaces removed (languages written without word spaces).
    cer_ignores_spaces: bool = False
    # Characters of the script that are not spoken form, with the reason.
    reject_chars: Dict[str, str] = field(default_factory=dict)


_EN_ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyz")
_RU_ALPHABET = frozenset("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

LANGUAGES: Dict[str, Language] = {
    "th": Language(
        code="th",
        name="Thai",
        script=THAI,
        # Thai is written without word spaces, and public corpora insert them
        # inconsistently, so they are removed.
        remove_spaces=True,
        cer_ignores_spaces=True,
        reject_chars={
            "ๆ": "repeat_mark",  # mai yamok: the word is spoken twice
            "ฯ": "abbrev_mark",  # paiyannoi: abbreviation, not as spoken
        },
    ),
    "en": Language(
        code="en",
        name="English",
        script=LATIN,
        alphabet=_EN_ALPHABET,
        lowercase=True,
        keep_apostrophe=True,
    ),
    "zh": Language(
        code="zh",
        name="Mandarin",
        script=HAN,
        remove_spaces=True,
        cer_ignores_spaces=True,
    ),
    "ru": Language(
        code="ru",
        name="Russian",
        script=CYRILLIC,
        alphabet=_RU_ALPHABET,
        lowercase=True,
    ),
    "ko": Language(code="ko", name="Korean", script=HANGUL),
}


def get_language(code: str) -> Language:
    try:
        return LANGUAGES[code]
    except KeyError:
        known = ", ".join(sorted(LANGUAGES))
        raise KeyError(f"Unknown language '{code}'. Known: {known}") from None
