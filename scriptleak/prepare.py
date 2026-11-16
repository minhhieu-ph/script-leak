"""Convert public corpora to raw NeMo-style manifests.

These manifests are *raw*: run `scriptleak.audit` on them afterwards to get
the spoken-form subset.

  python -m scriptleak.prepare librispeech --root /data/LibriSpeech \
      --subsets train-clean-100 train-clean-360 --out en_train.raw.jsonl

  python -m scriptleak.prepare aishell1 --root /data/data_aishell \
      --split train --out zh_train.raw.jsonl

  python -m scriptleak.prepare hf --dataset CMKL/Porjai-Thai-voice-dataset-central \
      --split train --text-col sentence --audio-out /data/porjai_central/wav \
      --out th_central.raw.jsonl --copy-cols speaker_id

  python -m scriptleak.prepare hf-columns --dataset CMKL/Porjai-Thai-voice-dataset-korat

  python -m scriptleak.prepare gigaspeech2 --tsv $D/gigaspeech2/data/th/test.tsv \
      --audio-dir $D/gigaspeech2/data/th/test --out th_gs2_test.raw.jsonl

  python -m scriptleak.prepare tsv --tsv meta.tsv --audio-root /data/wav \
      --path-col path --text-col text --out any.raw.jsonl

The hf converter writes 16 kHz mono wav files.
Needs soundfile to read durations (and `datasets` for the hf converter).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np

from .audio import resample, to_mono
from .manifest import write_manifest


def audio_duration(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def librispeech_entries(root: str, subsets: List[str],
                        duration_fn: Callable[[str], float] = audio_duration) -> Iterator[Dict]:
    """LibriSpeech layout: <root>/<subset>/<spk>/<chap>/<spk>-<chap>.trans.txt
    with lines "<utt-id> <TEXT>" and audio <utt-id>.flac beside it."""
    for subset in subsets:
        pattern = os.path.join(root, subset, "*", "*", "*.trans.txt")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No transcripts match {pattern}")
        for trans in files:
            folder = os.path.dirname(trans)
            with open(trans, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    utt, _, text = line.partition(" ")
                    audio = os.path.join(folder, utt + ".flac")
                    yield {
                        "audio_filepath": audio,
                        "duration": duration_fn(audio),
                        "text": text,
                        "utt_id": utt,
                        "speaker": utt.split("-")[0],
                        "source": f"librispeech/{subset}",
                    }


def aishell1_entries(root: str, split: str,
                     duration_fn: Callable[[str], float] = audio_duration) -> Iterator[Dict]:
    """AISHELL-1 layout: <root>/transcript/aishell_transcript_v0.8.txt with
    lines "<utt-id> <words...>", audio at <root>/wav/<split>/<spk>/<utt-id>.wav."""
    trans_path = os.path.join(root, "transcript", "aishell_transcript_v0.8.txt")
    text_by_id: Dict[str, str] = {}
    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                text_by_id[parts[0]] = parts[1]
    wavs = sorted(glob.glob(os.path.join(root, "wav", split, "*", "*.wav")))
    if not wavs:
        raise FileNotFoundError(f"No wav files under {os.path.join(root, 'wav', split)}")
    for wav in wavs:
        utt = os.path.splitext(os.path.basename(wav))[0]
        if utt not in text_by_id:
            continue  # a few AISHELL-1 files have no transcript
        yield {
            "audio_filepath": wav,
            "duration": duration_fn(wav),
            "text": text_by_id[utt],
            "utt_id": utt,
            "speaker": os.path.basename(os.path.dirname(wav)),
            "source": f"aishell1/{split}",
        }


def tsv_entries(tsv: str, audio_root: str, path_col: str, text_col: str,
                copy_cols: List[str], delimiter: str = "\t",
                duration_fn: Callable[[str], float] = audio_duration) -> Iterator[Dict]:
    with open(tsv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            audio = os.path.join(audio_root, row[path_col]) if audio_root else row[path_col]
            entry = {"audio_filepath": audio, "duration": duration_fn(audio), "text": row[text_col]}
            for c in copy_cols:
                entry[c] = row.get(c)
            yield entry


def gigaspeech2_entries(tsv: str, audio_dir: str,
                        duration_fn: Callable[[str], float] = audio_duration) -> Iterator[Dict]:
    """GigaSpeech 2 dev/test: a TSV of "<segment_id>\t<text>" and a tar of
    <segment_id>.wav files extracted somewhere under audio_dir."""
    by_id: Dict[str, str] = {}
    for wav in glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True):
        by_id[os.path.splitext(os.path.basename(wav))[0]] = wav
    if not by_id:
        raise FileNotFoundError(f"No wav files under {audio_dir}")
    missing = 0
    with open(tsv, "r", encoding="utf-8") as f:
        for line in f:
            seg, sep, text = line.rstrip("\n").partition("\t")
            if not sep:
                continue
            if seg not in by_id:
                missing += 1
                continue
            yield {"audio_filepath": by_id[seg], "duration": duration_fn(by_id[seg]),
                   "text": text, "utt_id": seg, "source": "gigaspeech2"}
    if missing:
        print(f"warning: {missing} segments in {tsv} have no audio file", file=sys.stderr)


def hf_columns(dataset: str, config: Optional[str], split: str) -> Dict:
    """Column names and the first row (audio omitted) without downloading the
    whole dataset."""
    from datasets import load_dataset

    args = (dataset, config) if config else (dataset,)
    ds = load_dataset(*args, split=split, streaming=True)
    row = next(iter(ds))
    return {k: (v if not isinstance(v, dict) else f"<{', '.join(v.keys())}>") for k, v in row.items()}


def hf_entries(dataset: str, config: Optional[str], split: str, audio_col: str, text_col: str,
               audio_out: str, copy_cols: List[str], limit: Optional[int] = None,
               target_sr: int = 16000) -> Iterator[Dict]:
    """Any Hugging Face audio dataset. Audio is written out as 16-bit wav so
    NeMo can read it from disk."""
    import soundfile as sf
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split) if config else load_dataset(dataset, split=split)
    if text_col not in ds.column_names:
        raise SystemExit(f"Column '{text_col}' not found. Columns: {ds.column_names}")
    os.makedirs(audio_out, exist_ok=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        audio = row[audio_col]
        wav = resample(to_mono(np.asarray(audio["array"], dtype=np.float32)),
                       audio["sampling_rate"], target_sr)
        path = os.path.join(audio_out, f"{split}_{i:08d}.wav")
        sf.write(path, wav, target_sr, subtype="PCM_16")
        entry = {
            "audio_filepath": path,
            "duration": len(wav) / float(target_sr),
            "text": row[text_col],
            "source": f"{dataset}/{split}",
        }
        for c in copy_cols:
            entry[c] = row.get(c)
        yield entry


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="corpus", required=True)

    a = sub.add_parser("librispeech")
    a.add_argument("--root", required=True)
    a.add_argument("--subsets", nargs="+", required=True)
    a.add_argument("--out", required=True)

    a = sub.add_parser("aishell1")
    a.add_argument("--root", required=True)
    a.add_argument("--split", required=True, choices=["train", "dev", "test"])
    a.add_argument("--out", required=True)

    a = sub.add_parser("tsv")
    a.add_argument("--tsv", required=True)
    a.add_argument("--audio-root", default="")
    a.add_argument("--path-col", required=True)
    a.add_argument("--text-col", required=True)
    a.add_argument("--copy-cols", nargs="*", default=[])
    a.add_argument("--delimiter", default="\t")
    a.add_argument("--out", required=True)

    a = sub.add_parser("gigaspeech2")
    a.add_argument("--tsv", required=True)
    a.add_argument("--audio-dir", required=True)
    a.add_argument("--out", required=True)

    a = sub.add_parser("hf-columns", help="print the columns and first row of a Hugging Face dataset")
    a.add_argument("--dataset", required=True)
    a.add_argument("--config")
    a.add_argument("--split", default="train")

    a = sub.add_parser("hf")
    a.add_argument("--dataset", required=True)
    a.add_argument("--config")
    a.add_argument("--split", required=True)
    a.add_argument("--audio-col", default="audio")
    a.add_argument("--text-col", required=True)
    a.add_argument("--audio-out", required=True)
    a.add_argument("--copy-cols", nargs="*", default=[])
    a.add_argument("--limit", type=int)
    a.add_argument("--target-sr", type=int, default=16000)
    a.add_argument("--out", required=True)

    args = p.parse_args(argv)
    if args.corpus == "hf-columns":
        for k, v in hf_columns(args.dataset, args.config, args.split).items():
            print(f"{k}: {v}")
        return 0
    if args.corpus == "gigaspeech2":
        n = write_manifest(args.out, gigaspeech2_entries(args.tsv, args.audio_dir))
        print(f"Wrote {n} entries to {args.out}")
        return 0
    if args.corpus == "librispeech":
        entries = librispeech_entries(args.root, args.subsets)
    elif args.corpus == "aishell1":
        entries = aishell1_entries(args.root, args.split)
    elif args.corpus == "tsv":
        entries = tsv_entries(args.tsv, args.audio_root, args.path_col, args.text_col,
                              args.copy_cols, args.delimiter)
    else:
        entries = hf_entries(args.dataset, args.config, args.split, args.audio_col,
                             args.text_col, args.audio_out, args.copy_cols, args.limit,
                             args.target_sr)
    n = write_manifest(args.out, entries)
    print(f"Wrote {n} entries to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
