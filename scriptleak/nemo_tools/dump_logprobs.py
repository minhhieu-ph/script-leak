"""Run a model over a manifest and save frame-level CTC log-probabilities.

  python -m scriptleak.nemo_tools.dump_logprobs --model exp/th150_en05/final.nemo \
      --manifest th_test.strata.jsonl --out dumps/th150_en05/th_test

The dump is what scriptleak.sweep and scriptleak.analysis read. With
--transducer the greedy output of the transducer head is also stored per
utterance (field `pred_transducer`), so leakage can be compared between the
two heads of the same encoder.

Audio at another sample rate is resampled on the fly.
Untested outside a NeMo install: run scriptleak.nemo_tools.check_env first.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np

from ..audio import resample, to_mono
from ..dumpio import DumpWriter
from ..manifest import read_manifest
from ..tokenmap import ids_to_text


def _texts(hyps) -> List[str]:
    """NeMo's transcribe() has returned strings, Hypothesis objects or a
    (best, all) tuple across versions."""
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    out = []
    for h in hyps:
        if isinstance(h, list):
            h = h[0]
        out.append(h if isinstance(h, str) else getattr(h, "text", str(h)))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help=".nemo checkpoint")
    p.add_argument("--pretrained", help="pretrained model name")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int)
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--transducer", action="store_true",
                   help="also store the greedy transducer transcript")
    args = p.parse_args(argv)

    import soundfile as sf
    import torch

    from .common import ctc_head, get_vocabulary, load_model

    model = load_model(args.model, args.pretrained, args.device)
    head = ctc_head(model)
    vocab = get_vocabulary(model)
    blank_id = len(vocab)
    sample_rate = int(model.cfg.preprocessor.sample_rate)

    entries: List[Dict] = list(read_manifest(args.manifest))
    if args.limit:
        entries = entries[: args.limit]
    # longest first: an out-of-memory error shows up immediately, and batches
    # are padded less
    order = sorted(range(len(entries)), key=lambda i: -float(entries[i].get("duration", 0.0)))

    results: Dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            wavs = []
            for i in idx:
                audio, sr = sf.read(entries[i]["audio_filepath"], dtype="float32", always_2d=False)
                audio = resample(to_mono(audio), sr, sample_rate)
                wavs.append(torch.from_numpy(np.ascontiguousarray(audio)))
            lengths = torch.tensor([len(w) for w in wavs], device=args.device)
            signal = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).to(args.device)
            processed, processed_len = model.preprocessor(input_signal=signal, length=lengths)
            encoded, encoded_len = model.encoder(audio_signal=processed, length=processed_len)
            logp = head(encoder_output=encoded).float().cpu().numpy()
            enc_len = encoded_len.cpu().numpy()
            for row, i in enumerate(idx):
                results[i] = logp[row, : int(enc_len[row])].astype(np.float16)
            done = min(start + args.batch_size, len(order))
            if (start // args.batch_size) % 20 == 0:
                print(f"{done}/{len(order)} utterances", flush=True)

    transducer_text: Dict[int, str] = {}
    if args.transducer:
        if hasattr(model, "change_decoding_strategy") and hasattr(model, "ctc_decoder"):
            model.change_decoding_strategy(decoder_type="rnnt")
        paths = [e["audio_filepath"] for e in entries]
        for i, t in enumerate(_texts(model.transcribe(paths, batch_size=args.batch_size))):
            transducer_text[i] = t

    extra = {"model": args.model or args.pretrained, "sample_rate": sample_rate,
             "manifest": args.manifest}
    with DumpWriter(args.out, vocab, blank_id, args.shard_size, extra) as w:
        for i, e in enumerate(entries):
            rec = dict(e)
            lp = results[i].astype(np.float32)
            rec["pred_ctc"] = ids_to_text(_collapse(lp.argmax(axis=1), blank_id), vocab)
            if i in transducer_text:
                rec["pred_transducer"] = transducer_text[i]
            w.add(rec, lp)
    print(f"Wrote dump of {len(entries)} utterances to {args.out}")
    return 0


def _collapse(frame_ids: np.ndarray, blank_id: int) -> List[int]:
    out, prev = [], -1
    for i in frame_ids.tolist():
        if i != prev and i != blank_id:
            out.append(i)
        prev = i
    return out


if __name__ == "__main__":
    sys.exit(main())
