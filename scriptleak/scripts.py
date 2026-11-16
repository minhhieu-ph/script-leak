"""Unicode script definitions.

A "script" here is a set of code-point ranges covering the *letters and
combining marks* of a writing system. Digits, punctuation and symbols are
deliberately excluded so that "contains a Latin character" means "contains a
Latin letter".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Script:
    name: str
    ranges: Tuple[Tuple[int, int], ...]

    def contains(self, ch: str) -> bool:
        cp = ord(ch)
        for lo, hi in self.ranges:
            if lo <= cp <= hi:
                return True
        return False

    def count(self, text: str) -> int:
        return sum(1 for ch in text if self.contains(ch))


# U+00D7 (multiplication sign) and U+00F7 (division sign) sit inside the
# Latin-1 letter block and are excluded on purpose.
LATIN = Script(
    "latin",
    ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0xD6), (0xD8, 0xF6), (0xF8, 0x24F)),
)
THAI = Script("thai", ((0x0E01, 0x0E3A), (0x0E40, 0x0E4E)))
HAN = Script(
    "han",
    ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF)),
)
CYRILLIC = Script("cyrillic", ((0x0400, 0x04FF), (0x0500, 0x052F)))
HANGUL = Script("hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7A3)))
KANA = Script("kana", ((0x3040, 0x309F), (0x30A0, 0x30FF)))
DEVANAGARI = Script("devanagari", ((0x0900, 0x097F),))
ARABIC = Script("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F)))

ALL_SCRIPTS = (LATIN, THAI, HAN, CYRILLIC, HANGUL, KANA, DEVANAGARI, ARABIC)
SCRIPTS_BY_NAME = {s.name: s for s in ALL_SCRIPTS}


def script_of(ch: str) -> Optional[str]:
    """Name of the script a character belongs to, or None."""
    for s in ALL_SCRIPTS:
        if s.contains(ch):
            return s.name
    return None
