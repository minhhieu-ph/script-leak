"""Dump format, frame analysis, sweep, noise mixing and corpus converters."""

import os

import numpy as np
import pytest

from scriptleak import analysis, noise, prepare, sweep
from scriptleak.dumpio import DumpWriter, iter_dump, read_meta
from scriptleak.languages import get_language
from scriptleak.tokenmap import token_groups

TH, EN = get_language("th"), get_language("en")
VOCAB = ["▁", "ก", "ข", "ค", "a", "b", "▁he"]
B = 7
GROUPS = token_groups(VOCAB, TH, EN)


def frames(rows):
    out = np.full((len(rows), len(VOCAB) + 1), 1e-8)
    for t, row in enumerate(rows):
        for i, p in row.items():
            out[t, i] = p
        out[t, B] = max(1e-8, 1.0 - sum(row.values()))
    return np.log(out / out.sum(axis=1, keepdims=True)).astype(np.float32)


THAI_OK = {1: 0.9, 4: 0.01}
SPLIT = {1: 0.22, 2: 0.20, 3: 0.20, 4: 0.30}
LATIN_STRONG = {1: 0.30, 4: 0.60}
ENGLISH_OK = {4: 0.9, 1: 0.01}


def make_dump(path, utts, shard_size=2):
    with DumpWriter(str(path), VOCAB, B, shard_size=shard_size, extra_meta={"model": "toy"}) as w:
        for entry, rows in utts:
            w.add(entry, frames(rows))


def test_dump_roundtrip_across_shards(tmp_path):
    utts = [({"text": f"t{i}", "stratum": "clean"}, [THAI_OK] * (i + 1)) for i in range(5)]
    make_dump(tmp_path / "d", utts)
    meta = read_meta(str(tmp_path / "d"))
    assert meta["vocab"] == VOCAB and meta["blank_id"] == B and meta["model"] == "toy"
    got = list(iter_dump(str(tmp_path / "d")))
    assert [r["text"] for r, _ in got] == [f"t{i}" for i in range(5)]
    assert [lp.shape for _, lp in got] == [(i + 1, 8) for i in range(5)]
    assert sorted(f for f in os.listdir(tmp_path / "d") if f.endswith(".npz")) == [
        "logp_00000.npz", "logp_00001.npz", "logp_00002.npz"]
    # float16 storage keeps probabilities close enough
    np.testing.assert_allclose(np.exp(got[0][1]), np.exp(frames([THAI_OK])), atol=2e-3)


def test_frame_analysis_separates_split_from_strong(tmp_path):
    acc = analysis.LeakFrameAccumulator()
    acc.add(frames([THAI_OK, SPLIT, {}, LATIN_STRONG, THAI_OK]), GROUPS)
    s = acc.summary()
    assert s["leak"]["frames"] == 2
    assert s["normal"]["frames"] == 2
    # one of the two leak frames has more Thai mass than Latin mass
    assert s["leak"]["frac_majority_mass_exceeds_minority_mass"] == pytest.approx(0.5)
    assert s["leak"]["mean_entropy"] > s["normal"]["mean_entropy"]
    assert s["leak"]["mean_effective_majority_candidates"] > s["normal"]["mean_effective_majority_candidates"]
    assert s["leak_runs"]["count"] == 1 and s["leak_runs"]["max_frames"] == 2
    assert "H1" in analysis.format_summary(s)


def test_analyse_dump_skips_non_majority_references(tmp_path):
    make_dump(tmp_path / "d", [
        ({"text": "กข", "stratum": "clean"}, [THAI_OK, {}, THAI_OK]),
        ({"text": "กข", "stratum": "snr0"}, [THAI_OK, {}, SPLIT]),
        ({"text": "hello", "stratum": "clean"}, [ENGLISH_OK]),
    ])
    res = analysis.analyse_dump(str(tmp_path / "d"), "th", "en", group_by="stratum")
    assert res["ALL"]["utterances"] == 2
    assert res["clean"]["leak"]["frames"] == 0
    assert res["snr0"]["leak"]["frames"] == 1
    assert analysis.main(["--dump", str(tmp_path / "d"), "--majority", "th", "--minority", "en"]) == 0


def test_sweep_majority_and_minority_sides(tmp_path, capsys):
    make_dump(tmp_path / "th", [
        ({"text": "กก", "stratum": "clean"}, [THAI_OK, {}, THAI_OK]),
        ({"text": "กกก", "stratum": "snr0"}, [THAI_OK, {}, SPLIT, {}, THAI_OK]),
    ])
    make_dump(tmp_path / "en", [({"text": "a a"}, [ENGLISH_OK, {}, {0: 0.9}, {}, ENGLISH_OK])])
    methods = ["greedy", "mask", "marg"]
    th = sweep.run_sweep(str(tmp_path / "th"), "th", "en", "th", methods, group_by="stratum",
                         save_preds_dir=str(tmp_path / "preds"))
    assert th["greedy"]["ALL"]["slr_u"] == pytest.approx(0.5)
    assert th["greedy"]["snr0"]["slr_u"] == 1.0 and th["greedy"]["clean"]["slr_u"] == 0.0
    assert th["marg"]["ALL"]["slr_u"] == 0.0 and th["marg"]["ALL"]["cer"] == 0.0
    assert th["mask"]["ALL"]["slr_u"] == 0.0
    assert os.path.exists(tmp_path / "preds" / "preds_marg.jsonl")

    en = sweep.run_sweep(str(tmp_path / "en"), "th", "en", "en", methods)
    assert en["greedy"]["ALL"]["wer"] == 0.0
    assert en["marg"]["ALL"]["wer"] == 0.0       # proposed method keeps English
    assert en["mask"]["ALL"]["wer"] == 1.0       # hard mask destroys it

    assert sweep.main(["--dump", str(tmp_path / "th"), "--majority", "th", "--minority", "en",
                       "--methods", "greedy", "marg"]) == 0
    assert "SLR-u" in capsys.readouterr().out
    with pytest.raises(ValueError):
        sweep.run_sweep(str(tmp_path / "th"), "th", "en", "zh", methods)


def test_mix_at_snr_hits_target_and_keeps_length():
    rng = np.random.default_rng(0)
    speech = (0.1 * rng.standard_normal(16000)).astype(np.float32)
    noise_sig = (0.5 * rng.standard_normal(40000)).astype(np.float32)
    for snr in (10.0, 5.0, 0.0):
        mixed = noise.mix_at_snr(speech, noise_sig, snr, np.random.default_rng(1))
        assert mixed.shape == speech.shape
        added = mixed - speech
        got = 20 * np.log10(noise.rms(speech) / noise.rms(added))
        assert got == pytest.approx(snr, abs=0.05)


def test_mix_at_snr_loops_short_noise_and_is_deterministic():
    speech = np.sin(np.linspace(0, 100, 8000)).astype(np.float32) * 0.2
    short = np.random.default_rng(3).standard_normal(1000).astype(np.float32)
    a = noise.mix_at_snr(speech, short, 5.0, np.random.default_rng(7))
    b = noise.mix_at_snr(speech, short, 5.0, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert len(a) == len(speech) and np.max(np.abs(a)) <= 1.0


def test_mix_at_snr_silent_inputs():
    z = np.zeros(100, dtype=np.float32)
    x = np.ones(100, dtype=np.float32) * 0.1
    np.testing.assert_array_equal(noise.mix_at_snr(x, z, 0.0, np.random.default_rng(0)), x)
    np.testing.assert_array_equal(noise.mix_at_snr(z, x, 0.0, np.random.default_rng(0)), z)


def test_librispeech_converter(tmp_path):
    d = tmp_path / "LibriSpeech" / "train-clean-100" / "19" / "198"
    d.mkdir(parents=True)
    (d / "19-198.trans.txt").write_text(
        "19-198-0000 NORTHANGER ABBEY\n19-198-0001 THIS LITTLE WORK\n", encoding="utf-8")
    got = list(prepare.librispeech_entries(str(tmp_path / "LibriSpeech"), ["train-clean-100"],
                                           duration_fn=lambda p: 2.5))
    assert [e["text"] for e in got] == ["NORTHANGER ABBEY", "THIS LITTLE WORK"]
    assert got[0]["audio_filepath"].endswith(os.path.join("19", "198", "19-198-0000.flac"))
    assert got[0]["speaker"] == "19" and got[0]["duration"] == 2.5
    with pytest.raises(FileNotFoundError):
        list(prepare.librispeech_entries(str(tmp_path / "LibriSpeech"), ["dev-clean"],
                                         duration_fn=lambda p: 1.0))


def test_aishell1_converter(tmp_path):
    root = tmp_path / "data_aishell"
    (root / "transcript").mkdir(parents=True)
    (root / "transcript" / "aishell_transcript_v0.8.txt").write_text(
        "BAC009S0002W0122 而 对 楼市 成交\nBAC009S0002W0123 也 成为 地方\n", encoding="utf-8")
    wav = root / "wav" / "train" / "S0002"
    wav.mkdir(parents=True)
    for name in ("BAC009S0002W0122", "BAC009S0002W0123", "BAC009S0002W0999"):
        (wav / f"{name}.wav").write_bytes(b"")
    got = list(prepare.aishell1_entries(str(root), "train", duration_fn=lambda p: 4.0))
    assert [e["utt_id"] for e in got] == ["BAC009S0002W0122", "BAC009S0002W0123"]
    assert got[0]["text"] == "而 对 楼市 成交" and got[0]["speaker"] == "S0002"


def test_tsv_converter(tmp_path):
    tsv = tmp_path / "meta.tsv"
    tsv.write_text("path\ttext\tdialect\na.wav\tสวัสดี\tkorat\n", encoding="utf-8")
    got = list(prepare.tsv_entries(str(tsv), "/data", "path", "text", ["dialect"],
                                   duration_fn=lambda p: 1.0))
    assert got == [{"audio_filepath": os.path.join("/data", "a.wav"), "duration": 1.0,
                    "text": "สวัสดี", "dialect": "korat"}]


def test_resample_length_and_tone():
    from scriptleak.audio import resample, to_mono
    sr, target = 48000, 16000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    y = resample(x, sr, target)
    assert len(y) == target and y.dtype == np.float32
    peak = np.argmax(np.abs(np.fft.rfft(y)))
    assert peak == 440                      # 1 Hz per bin for a 1 s signal
    assert resample(x, sr, sr) is x
    assert to_mono(np.stack([x, x], axis=1)).shape == x.shape


def test_gigaspeech2_converter(tmp_path):
    audio = tmp_path / "test" / "0"
    audio.mkdir(parents=True)
    (audio / "0-1-2.wav").write_bytes(b"")
    tsv = tmp_path / "test.tsv"
    tsv.write_text("0-1-2\tสวัสดี\n0-1-3\tไม่มีไฟล์\n", encoding="utf-8")
    got = list(prepare.gigaspeech2_entries(str(tsv), str(tmp_path / "test"), duration_fn=lambda p: 3.0))
    assert len(got) == 1 and got[0]["utt_id"] == "0-1-2" and got[0]["text"] == "สวัสดี"
