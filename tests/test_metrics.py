import pytest

from scriptleak.evaluate import score
from scriptleak.languages import get_language
from scriptleak.metrics import cer, edit_distance, leak_stats, minority_spans, wer
from scriptleak.scripts import LATIN

TH = get_language("th")
EN = get_language("en")


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0
    assert edit_distance(["a", "b"], ["a", "c", "b"]) == 1


def test_cer_and_wer():
    assert cer(["abcd"], ["abxd"]) == pytest.approx(0.25)
    assert cer(["ab cd"], ["abcd"], ignore_spaces=True) == 0.0
    assert wer(["the cat sat"], ["the cat"]) == pytest.approx(1 / 3)


def test_minority_spans_join_across_spaces():
    assert minority_spans("ไป new york นะ", LATIN) == ["new york"]
    assert minority_spans("ok ไป go", LATIN) == ["ok", "go"]
    assert minority_spans("ไปนะ", LATIN) == []


def test_no_leak():
    s = leak_stats(["สวัสดีครับ", "ขอบคุณ"], ["สวัสดีครับ", "ขอบคุน"], TH, EN)
    assert s.utterances_eligible == 2
    assert s.utterances_leaked == 0
    assert s.slr_u == 0.0 and s.slr_c == 0.0


def test_leak_counted_per_utterance_and_per_char():
    refs = ["สวัสดีครับ", "ขอบคุณ", "ไปไหน", "กินข้าว"]
    hyps = ["สวัสดี hello", "ขอบคุณ", "ไปไหน", "กินข้าว"]
    s = leak_stats(refs, hyps, TH, EN)
    assert s.utterances_eligible == 4
    assert s.utterances_leaked == 1
    assert s.slr_u == pytest.approx(0.25)
    assert s.leak_spans == 1 and s.max_span_chars == 5
    thai_chars = sum(TH.script.count(h) for h in hyps)
    assert s.slr_c == pytest.approx(5 / (5 + thai_chars))


def test_reference_with_minority_script_is_not_eligible():
    # A reference that itself contains Latin is not a leakage test case.
    s = leak_stats(["ใช้ facebook", "สวัสดี"], ["ใช้ facebook", "สวัสดี"], TH, EN)
    assert s.utterances_total == 2
    assert s.utterances_eligible == 1
    assert s.utterances_leaked == 0


def test_reverse_direction_uses_swapped_languages():
    s = leak_stats(["hello world"], ["hello สวัสดี"], EN, TH)
    assert s.utterances_eligible == 1 and s.utterances_leaked == 1


def test_word_level_rate_with_custom_tokenizer():
    refs = ["สวัสดีครับ"]
    hyps = ["สวัสดี hello ครับ"]
    s = leak_stats(refs, hyps, TH, EN)
    assert s.slr_w == pytest.approx(1 / 3)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        leak_stats(["a"], [], EN, TH)


def test_score_reports_cer_for_thai_and_wer_for_english():
    th = score(["สวัสดีครับ"], ["สวัสดี ครับ"], TH, EN)
    assert th["cer"] == 0.0 and "wer" not in th
    en = score(["Hello, world"], ["hello word"], EN, TH)
    assert en["wer"] == pytest.approx(0.5) and en["slr_u"] == 0.0
