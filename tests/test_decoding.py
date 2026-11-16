import numpy as np
import pytest

from scriptleak import decoding as D
from scriptleak.languages import get_language
from scriptleak.tokenmap import BLANK, MAJ, MIN, NEUTRAL, ids_to_text, pieces_to_text, token_groups

TH, EN = get_language("th"), get_language("en")
#          0    1    2    3    4    5    6      (blank = 7)
VOCAB = ["▁", "ก", "ข", "ค", "a", "b", "▁he"]
GROUPS = token_groups(VOCAB, TH, EN)
B = 7


def frames(rows):
    """rows: list of dicts {token index: prob}; the rest goes to blank."""
    out = np.full((len(rows), len(VOCAB) + 1), 1e-8)
    for t, row in enumerate(rows):
        for i, p in row.items():
            out[t, i] = p
        out[t, B] = max(1e-8, 1.0 - sum(row.values()))
    return np.log(out / out.sum(axis=1, keepdims=True)).astype(np.float32)


THAI_OK = {1: 0.9, 4: 0.01}                    # clearly Thai
SPLIT = {1: 0.22, 2: 0.20, 3: 0.20, 4: 0.30}   # Thai mass 0.62 split in three, Latin 0.30 wins argmax
LATIN_STRONG = {1: 0.30, 4: 0.60}              # Latin mass really is larger
ENGLISH_OK = {4: 0.9, 1: 0.01}
BLANK_F = {}


def text(ids):
    return ids_to_text(ids, VOCAB)


def test_token_groups():
    assert GROUPS.tolist() == [NEUTRAL, MAJ, MAJ, MAJ, MIN, MIN, MIN, BLANK]


def test_token_groups_roles_swap():
    g = token_groups(VOCAB, EN, TH)
    assert g.tolist() == [NEUTRAL, MIN, MIN, MIN, MAJ, MAJ, MAJ, BLANK]


def test_pieces_to_text():
    assert pieces_to_text(["▁he", "▁", "a", "b"]) == "he ab"


def test_collapse():
    assert D.collapse(np.array([1, 1, B, 1, 2, 2, B]), B) == [1, 1, 2]


def test_group_logmass():
    lp = frames([SPLIT])
    assert np.exp(D.group_logmass(lp, GROUPS, MAJ))[0] == pytest.approx(0.62, abs=1e-3)
    assert np.exp(D.group_logmass(lp, GROUPS, MIN))[0] == pytest.approx(0.30, abs=1e-3)


def test_greedy_leaks_on_mass_split_frame():
    lp = frames([THAI_OK, BLANK_F, SPLIT, BLANK_F, THAI_OK])
    assert text(D.greedy(lp, GROUPS)) == "กaก"


def test_marginalized_fixes_mass_split_frame():
    lp = frames([THAI_OK, BLANK_F, SPLIT, BLANK_F, THAI_OK])
    assert text(D.script_marginalized(lp, GROUPS)) == "กกก"


def test_hard_mask_and_bias_fix_it_too():
    lp = frames([THAI_OK, BLANK_F, SPLIT, BLANK_F, THAI_OK])
    assert text(D.hard_mask(lp, GROUPS)) == "กกก"
    assert text(D.logit_bias(lp, GROUPS, bias=1.0)) == "กกก"
    assert text(D.logit_bias(lp, GROUPS, bias=0.0)) == "กaก"


def test_english_utterance_survives_marginalized_and_lock_but_not_mask():
    lp = frames([ENGLISH_OK, BLANK_F, ENGLISH_OK, BLANK_F, {5: 0.9}])
    assert text(D.greedy(lp, GROUPS)) == "aab"
    assert text(D.script_marginalized(lp, GROUPS)) == "aab"
    assert text(D.utterance_lock(lp, GROUPS)) == "aab"
    assert EN.script.count(text(D.hard_mask(lp, GROUPS))) == 0


def test_memoryless_marginalized_keeps_a_frame_where_latin_mass_is_larger():
    lp = frames([THAI_OK, BLANK_F, LATIN_STRONG, BLANK_F, THAI_OK])
    assert text(D.script_marginalized(lp, GROUPS)) == "กaก"


def test_hysteresis_holds_the_script_through_a_single_strong_latin_frame():
    lp = frames([THAI_OK] * 3 + [BLANK_F, LATIN_STRONG, BLANK_F, THAI_OK])
    out, track = D.script_marginalized(lp, GROUPS, tau=1.0, rho=0.9, return_track=True)
    assert EN.script.count(text(out)) == 0
    assert set(track.tolist()) == {MAJ}


def test_hysteresis_still_switches_on_sustained_evidence():
    lp = frames([THAI_OK] * 3 + [BLANK_F] + [ENGLISH_OK, BLANK_F] * 8)
    out, track = D.script_marginalized(lp, GROUPS, tau=1.0, rho=0.9, return_track=True)
    assert track[0] == MAJ and track[-1] == MIN
    assert EN.script.count(text(out)) > 0


def test_utterance_lock_full_and_prefix():
    lp = frames([THAI_OK, BLANK_F, THAI_OK, BLANK_F, LATIN_STRONG, BLANK_F, THAI_OK])
    assert EN.script.count(text(D.utterance_lock(lp, GROUPS))) == 0
    # decision after 3 frames, already Thai: later Latin frame is blocked
    assert EN.script.count(text(D.utterance_lock(lp, GROUPS, prefix_frames=3))) == 0
    # leak inside the prefix is not prevented
    early = frames([LATIN_STRONG, BLANK_F, THAI_OK, BLANK_F, THAI_OK, BLANK_F, THAI_OK])
    assert EN.script.count(text(D.utterance_lock(early, GROUPS, prefix_frames=6))) == 1
    # whole-utterance lock does prevent it
    assert EN.script.count(text(D.utterance_lock(early, GROUPS))) == 0


def test_utterance_lock_without_evidence_does_nothing():
    lp = frames([BLANK_F, BLANK_F])
    assert D.utterance_lock(lp, GROUPS) == []


def test_parse_and_dispatch():
    assert D.parse_method("marg:tau=0.5,rho=0.9") == ("marg", {"tau": 0.5, "rho": 0.9})
    lp = frames([THAI_OK, BLANK_F, SPLIT])
    for spec in D.DEFAULT_SWEEP:
        assert isinstance(D.decode(lp, GROUPS, spec), list)
    with pytest.raises(ValueError):
        D.decode(lp, GROUPS, "nope")


def test_empty_minority_group_does_not_crash():
    g = token_groups(["ก", "ข"], TH, EN)
    lp = np.log(np.array([[0.6, 0.3, 0.1]], dtype=np.float32))
    assert D.script_marginalized(lp, g) == [0]
    assert D.utterance_lock(lp, g) == [0]
