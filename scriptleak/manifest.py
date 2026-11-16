"""NeMo-style JSON-lines manifests: one JSON object per line with at least
`audio_filepath`, `duration` (seconds) and `text`."""

from __future__ import annotations

import json
from typing import Dict, Iterable, Iterator, List


def read_manifest(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from None


def write_manifest(path: str, entries: Iterable[Dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n += 1
    return n


def total_hours(entries: List[Dict]) -> float:
    return sum(float(e.get("duration", 0.0)) for e in entries) / 3600.0
