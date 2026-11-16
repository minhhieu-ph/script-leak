"""Split a manifest into train / dev / test by hours.

  python -m scriptleak.split --manifest th_central.clean.jsonl \
      --dev-hours 3 --test-hours 10 --speaker-field speaker_id --out-prefix th_central

writes th_central.train.jsonl, th_central.dev.jsonl, th_central.test.jsonl.
With --speaker-field, whole speakers are assigned to one side, so no speaker
is shared between train and dev/test. Without it the split is by utterance.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from .manifest import read_manifest, total_hours, write_manifest


def split_entries(entries: List[Dict], dev_hours: float, test_hours: float,
                  speaker_field: Optional[str] = None, seed: int = 0
                  ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = random.Random(seed)
    if speaker_field:
        units: "OrderedDict[str, List[Dict]]" = OrderedDict()
        for e in entries:
            if speaker_field not in e:
                raise KeyError(f"entry without '{speaker_field}': {e.get('audio_filepath')}")
            units.setdefault(str(e[speaker_field]), []).append(e)
        groups = list(units.values())
    else:
        groups = [[e] for e in entries]
    rng.shuffle(groups)

    test: List[Dict] = []
    dev: List[Dict] = []
    train: List[Dict] = []
    t_sec = d_sec = 0.0
    for g in groups:
        dur = sum(float(e.get("duration", 0.0)) for e in g)
        if t_sec < test_hours * 3600.0:
            test.extend(g)
            t_sec += dur
        elif d_sec < dev_hours * 3600.0:
            dev.extend(g)
            d_sec += dur
        else:
            train.extend(g)
    return train, dev, test


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--dev-hours", type=float, required=True)
    p.add_argument("--test-hours", type=float, required=True)
    p.add_argument("--speaker-field")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-prefix", required=True)
    args = p.parse_args(argv)

    train, dev, test = split_entries(list(read_manifest(args.manifest)), args.dev_hours,
                                     args.test_hours, args.speaker_field, args.seed)
    for name, part in (("train", train), ("dev", dev), ("test", test)):
        path = f"{args.out_prefix}.{name}.jsonl"
        write_manifest(path, part)
        print(f"{name}: {len(part)} utterances, {total_hours(part):.2f} h -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
