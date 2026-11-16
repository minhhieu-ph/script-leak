"""Map output tokens to script groups.

Every token of the model's output layer is put in one of four groups. The
decoding methods and the frame analysis only ever look at these groups, so
they work with any tokenizer (shared or one per language).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .languages import Language

BLANK, MAJ, MIN, NEUTRAL = 0, 1, 2, 3
GROUP_NAMES = {BLANK: "blank", MAJ: "majority", MIN: "minority", NEUTRAL: "neutral"}


def token_groups(
    vocab: Sequence[str],
    majority: Language,
    minority: Language,
    blank_id: Optional[int] = None,
) -> np.ndarray:
    """Group id for every output index.

    `vocab` is the list of token strings in output order. NeMo puts the CTC
    blank at index len(vocab), which is the default here.
    Tokens with no letter of either script (a lone word-boundary marker, an
    apostrophe) are NEUTRAL and are never banned.
    """
    if blank_id is None:
        blank_id = len(vocab)
    size = max(len(vocab), blank_id + 1)
    groups = np.full(size, NEUTRAL, dtype=np.int8)
    for i, piece in enumerate(vocab):
        n_maj = majority.script.count(piece)
        n_min = minority.script.count(piece)
        if n_maj and not n_min:
            groups[i] = MAJ
        elif n_min and not n_maj:
            groups[i] = MIN
    groups[blank_id] = BLANK
    return groups


def pieces_to_text(pieces: List[str]) -> str:
    """Join SentencePiece-style pieces into text."""
    return " ".join("".join(pieces).replace("▁", " ").split())


def ids_to_text(ids: Sequence[int], vocab: Sequence[str]) -> str:
    return pieces_to_text([vocab[i] for i in ids if 0 <= i < len(vocab)])
