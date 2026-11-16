from scriptleak.languages import get_language
from scriptleak.normalize import normalize, scoring_normalize

TH = get_language("th")
EN = get_language("en")
ZH = get_language("zh")
RU = get_language("ru")


def test_thai_clean_sentence_is_kept():
    r = normalize("สวัสดีครับ", TH)
    assert r.ok and r.text == "สวัสดีครับ"


def test_thai_punctuation_is_deleted_not_rejected():
    r = normalize("สวัสดี, ครับ!", TH)
    assert r.ok and r.text == "สวัสดีครับ"


def test_thai_word_spaces_removed():
    # the same sentence spaced two ways gives one transcript
    a = normalize("แจ้ง สลิป การ โอน เงิน ตรงไหน ครับ", TH)
    b = normalize("แจ้งสลิปการโอนเงินตรงไหน ครับ", TH)
    assert a.ok and a.text == b.text == "แจ้งสลิปการโอนเงินตรงไหนครับ"


def test_thai_arabic_digit_rejected():
    r = normalize("ราคา 5 บาท", TH)
    assert not r.ok and "digit" in r.reasons


def test_thai_native_digit_rejected():
    r = normalize("ราคา ๕ บาท", TH)
    assert not r.ok and "digit" in r.reasons


def test_thai_repeat_mark_rejected():
    r = normalize("ไปเร็วๆ", TH)
    assert not r.ok and r.reasons == ("repeat_mark",)


def test_thai_repetition_written_out_is_kept():
    r = normalize("ไปเร็วเร็ว", TH)
    assert r.ok and r.text == "ไปเร็วเร็ว"


def test_thai_abbreviation_mark_rejected():
    r = normalize("กรุงเทพฯ", TH)
    assert not r.ok and "abbrev_mark" in r.reasons


def test_thai_latin_word_rejected():
    r = normalize("ใช้ facebook ทุกวัน", TH)
    assert not r.ok and r.reasons == ("foreign_script:latin",)


def test_thai_currency_and_percent_rejected():
    assert "spoken_symbol" in normalize("ลด 50%", TH).reasons
    assert "spoken_symbol" in normalize("ราคา ฿", TH).reasons


def test_thai_zero_width_space_removed():
    r = normalize("สวัสดี​ครับ", TH)
    assert r.ok and r.text == "สวัสดีครับ"


def test_thai_sara_am_survives_unicode_normalisation():
    r = normalize("น้ำ", TH)
    assert r.ok and r.text == "น้ำ"
    # A decomposed input is brought to the same spelling.
    assert normalize("นํา", TH).text == "นำ"


def test_multiple_reasons_are_all_reported():
    r = normalize("iphone 15 ราคา", TH)
    assert set(r.reasons) == {"foreign_script:latin", "digit"}


def test_english_librispeech_style():
    r = normalize("HE DOESN'T KNOW", EN)
    assert r.ok and r.text == "he doesn't know"


def test_english_punctuation_and_hyphen():
    r = normalize("Well-known, isn't it?", EN)
    assert r.ok and r.text == "well known isn't it"


def test_english_quotes_do_not_leave_dangling_apostrophes():
    r = normalize("he said 'hello'", EN)
    assert r.ok and r.text == "he said hello"


def test_english_digit_and_accent_rejected():
    assert "digit" in normalize("room 42", EN).reasons
    assert "out_of_alphabet" in normalize("café", EN).reasons


def test_english_thai_word_rejected():
    assert normalize("hello สวัสดี", EN).reasons == ("foreign_script:thai",)


def test_chinese_spaces_removed_and_numerals_kept():
    r = normalize("今天 是 一月 一日", ZH)
    assert r.ok and r.text == "今天是一月一日"


def test_chinese_latin_and_fullwidth_digit_rejected():
    assert "foreign_script:latin" in normalize("我用iphone", ZH).reasons
    assert "digit" in normalize("我有３个", ZH).reasons


def test_chinese_fullwidth_punctuation_deleted():
    r = normalize("你好,世界!", ZH)
    assert r.ok and r.text == "你好世界"


def test_russian_lowercased():
    r = normalize("Привет, мир", RU)
    assert r.ok and r.text == "привет мир"


def test_empty_after_normalisation_rejected():
    assert normalize("...", TH).reasons == ("empty",)


def test_scoring_normalize_keeps_leaked_letters():
    assert scoring_normalize("สวัสดี, Hello!") == "สวัสดี hello"
    # digits are kept: scoring never hides what the model produced
    assert scoring_normalize("ราคา 5") == "ราคา 5"
