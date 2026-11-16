"""On-disk format for frame-level CTC log-probabilities.

A dump directory holds
  meta.json        {"vocab": [...], "blank_id": int, "model": str, ...}
  utts.jsonl       one line per utterance: id, text, shard, key, frames, + any
                   manifest fields (stratum, lang, ...)
  logp_00000.npz   float16 arrays [T, V], keyed by utterance key

Only numpy is needed to read or write it.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


class DumpWriter:
    def __init__(self, out_dir: str, vocab: List[str], blank_id: int,
                 shard_size: int = 2000, extra_meta: Optional[Dict] = None):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_size = shard_size
        self._buf: Dict[str, np.ndarray] = {}
        self._shard = 0
        self._n = 0
        meta = {"vocab": list(vocab), "blank_id": int(blank_id)}
        meta.update(extra_meta or {})
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        self._utts = open(os.path.join(out_dir, "utts.jsonl"), "w", encoding="utf-8")

    def add(self, entry: Dict, logp: np.ndarray) -> None:
        key = f"u{self._n:08d}"
        self._buf[key] = logp.astype(np.float16)
        rec = dict(entry)
        rec.update({"key": key, "shard": self._shard, "frames": int(logp.shape[0])})
        self._utts.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._n += 1
        if len(self._buf) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if self._buf:
            np.savez(os.path.join(self.out_dir, f"logp_{self._shard:05d}.npz"), **self._buf)
            self._buf = {}
            self._shard += 1

    def close(self) -> None:
        self._flush()
        self._utts.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_meta(dump_dir: str) -> Dict:
    with open(os.path.join(dump_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def iter_dump(dump_dir: str) -> Iterator[Tuple[Dict, np.ndarray]]:
    """Yields (utterance record, logp float32 [T, V]) in order."""
    shard_id, shard = None, None
    with open(os.path.join(dump_dir, "utts.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["shard"] != shard_id:
                shard_id = rec["shard"]
                shard = np.load(os.path.join(dump_dir, f"logp_{shard_id:05d}.npz"))
            yield rec, shard[rec["key"]].astype(np.float32)
