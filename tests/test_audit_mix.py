import json

import pytest

from scriptleak import audit, evaluate, mix
from scriptleak.manifest import read_manifest, write_manifest


def _entries(texts, dur=3600.0):
    return [{"audio_filepath": f"a{i}.wav", "duration": dur, "text": t} for i, t in enumerate(texts)]


def test_audit_counts_and_cleans():
    entries = _entries(["สวัสดีครับ", "ราคา 5 บาท", "ไปเร็วๆ", "ขอบคุณ, ครับ"])
    clean, report = audit.audit_entries(entries, "th")
    assert report["utterances_total"] == 4
    assert report["utterances_clean"] == 2
    assert report["hours_total"] == 4.0 and report["hours_clean"] == 2.0
    assert report["rejections"]["digit"]["utterances"] == 1
    assert report["rejections"]["repeat_mark"]["utterances"] == 1
    assert [e["text"] for e in clean] == ["สวัสดีครับ", "ขอบคุณครับ"]
    assert all(e["lang"] == "th" for e in clean)
    assert report["clean_utterances_modified_by_normalisation"] == 1


def test_audit_cli_roundtrip(tmp_path, capsys):
    src = tmp_path / "in.jsonl"
    out = tmp_path / "clean.jsonl"
    rep = tmp_path / "report.json"
    write_manifest(str(src), _entries(["HELLO WORLD", "ROOM 42"]))
    assert audit.main(["--manifest", str(src), "--lang", "en",
                       "--out-manifest", str(out), "--report", str(rep)]) == 0
    assert [e["text"] for e in read_manifest(str(out))] == ["hello world"]
    assert json.loads(rep.read_text(encoding="utf-8"))["utterances_rejected"] == 1
    assert "digit" in capsys.readouterr().out


def test_minority_hours_formula():
    assert mix.minority_hours_for(95.0, 0.05) == pytest.approx(5.0)
    assert mix.minority_hours_for(150.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        mix.minority_hours_for(10.0, 1.0)


def test_build_mix_hits_ratio_and_is_deterministic():
    maj = [{"audio_filepath": f"m{i}.wav", "duration": 360.0, "text": "ก", "lang": "th"} for i in range(200)]
    mino = [{"audio_filepath": f"n{i}.wav", "duration": 360.0, "text": "a", "lang": "en"} for i in range(200)]
    mixed, summary = mix.build_mix(maj, mino, majority_hours=9.0, minority_fraction=0.1, seed=1)
    assert summary["majority_hours"] == pytest.approx(9.0)
    assert summary["minority_hours"] == pytest.approx(1.0)
    assert summary["minority_fraction_of_hours"] == pytest.approx(0.1)
    again, _ = mix.build_mix(maj, mino, majority_hours=9.0, minority_fraction=0.1, seed=1)
    assert [e["audio_filepath"] for e in mixed] == [e["audio_filepath"] for e in again]
    other, _ = mix.build_mix(maj, mino, majority_hours=9.0, minority_fraction=0.1, seed=2)
    assert [e["audio_filepath"] for e in mixed] != [e["audio_filepath"] for e in other]


def test_build_mix_fails_when_not_enough_data():
    maj = [{"audio_filepath": "m.wav", "duration": 3600.0, "text": "ก"}]
    with pytest.raises(ValueError):
        mix.build_mix(maj, maj, majority_hours=5.0, minority_fraction=0.1)
    mixed, _ = mix.build_mix(maj, maj, majority_hours=5.0, minority_fraction=0.1, allow_short=True)
    assert len(mixed) == 2


def test_evaluate_cli_groups(tmp_path, capsys):
    src = tmp_path / "preds.jsonl"
    out = tmp_path / "res.json"
    write_manifest(str(src), [
        {"text": "สวัสดีครับ", "pred_text": "สวัสดีครับ", "stratum": "clean"},
        {"text": "ขอบคุณ", "pred_text": "ขอบคุณ", "stratum": "clean"},
        {"text": "ไปไหน", "pred_text": "ไป why", "stratum": "snr0"},
        {"text": "กินข้าว", "pred_text": "กินข้าว", "stratum": "snr0"},
    ])
    assert evaluate.main(["--manifest", str(src), "--majority", "th", "--minority", "en",
                          "--group-by", "stratum", "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["ALL"]["slr_u"] == pytest.approx(0.25)
    assert res["clean"]["slr_u"] == 0.0
    assert res["snr0"]["slr_u"] == pytest.approx(0.5)
    assert "snr0" in capsys.readouterr().out


def test_audit_cli_stamps_fields(tmp_path):
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_manifest(str(src), _entries(["สวัสดี"]))
    audit.main(["--manifest", str(src), "--lang", "th", "--out-manifest", str(out),
                "--set", "stratum=korat", "split=test"])
    e = list(read_manifest(str(out)))[0]
    assert e["stratum"] == "korat" and e["split"] == "test"


def test_split_by_speaker_is_disjoint_and_sized():
    from scriptleak.split import split_entries
    entries = [{"audio_filepath": f"s{s}_{i}.wav", "duration": 360.0, "text": "ก", "spk": s}
               for s in range(20) for i in range(10)]            # 20 speakers x 1 h
    train, dev, test = split_entries(entries, dev_hours=2, test_hours=3, speaker_field="spk", seed=0)
    spk = lambda part: {e["spk"] for e in part}
    assert len(spk(test)) == 3 and len(spk(dev)) == 2 and len(spk(train)) == 15
    assert not (spk(train) & spk(dev)) and not (spk(train) & spk(test)) and not (spk(dev) & spk(test))
    assert len(train) + len(dev) + len(test) == len(entries)
    again = split_entries(entries, 2, 3, "spk", seed=0)
    assert [e["audio_filepath"] for e in again[2]] == [e["audio_filepath"] for e in test]


def test_split_by_utterance():
    from scriptleak.split import split_entries
    entries = [{"audio_filepath": f"{i}.wav", "duration": 3600.0, "text": "ก"} for i in range(10)]
    train, dev, test = split_entries(entries, dev_hours=1, test_hours=2)
    assert (len(train), len(dev), len(test)) == (7, 1, 2)
