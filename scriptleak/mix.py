"""Build a bilingual training manifest: N hours of the majority language plus
a minority language making up a given fraction of the total hours.

Example (150 h Thai, English = 5% of total hours):
  python -m scriptleak.mix --majority th.clean.jsonl --minority en.clean.jsonl \
      --majority-hours 150 --minority-fraction 0.05 --seed 0 --out train_th150_en05.jsonl

Role swap is the same command with the two manifests exchanged.
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Dict, List, Optional, Tuple

from .manifest import read_manifest, total_hours, write_manifest


def minority_hours_for(majority_hours: float, minority_fraction: float) -> float:
    """Hours of minority data so that it is `minority_fraction` of the total."""
    if not 0.0 <= minority_fraction < 1.0:
        raise ValueError("minority_fraction must be in [0, 1)")
    return majority_hours * minority_fraction / (1.0 - minority_fraction)


def take_hours(entries: List[Dict], hours: float, rng: random.Random) -> List[Dict]:
    """Random subset whose total duration just reaches `hours`."""
    if hours <= 0:
        return []
    pool = list(entries)
    rng.shuffle(pool)
    target = hours * 3600.0
    picked: List[Dict] = []
    acc = 0.0
    for e in pool:
        if acc >= target:
            break
        picked.append(e)
        acc += float(e.get("duration", 0.0))
    return picked


def build_mix(
    majority: List[Dict],
    minority: List[Dict],
    majority_hours: float,
    minority_fraction: float,
    seed: int = 0,
    allow_short: bool = False,
) -> Tuple[List[Dict], Dict]:
    rng = random.Random(seed)
    want_min = minority_hours_for(majority_hours, minority_fraction)
    have_maj, have_min = total_hours(majority), total_hours(minority)
    if not allow_short:
        if have_maj < majority_hours:
            raise ValueError(f"majority has {have_maj:.2f} h, need {majority_hours:.2f} h")
        if have_min < want_min:
            raise ValueError(f"minority has {have_min:.2f} h, need {want_min:.2f} h")
    maj = take_hours(majority, majority_hours, rng)
    mino = take_hours(minority, want_min, rng)
    mixed = maj + mino
    rng.shuffle(mixed)
    h_maj, h_min = total_hours(maj), total_hours(mino)
    total = h_maj + h_min
    summary = {
        "majority_hours": round(h_maj, 3),
        "minority_hours": round(h_min, 3),
        "minority_fraction_of_hours": round(h_min / total, 4) if total else 0.0,
        "majority_utterances": len(maj),
        "minority_utterances": len(mino),
        "seed": seed,
    }
    return mixed, summary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--majority", required=True)
    p.add_argument("--minority", required=True)
    p.add_argument("--majority-hours", type=float, required=True)
    p.add_argument("--minority-fraction", type=float, required=True,
                   help="minority share of total hours, e.g. 0.05")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-short", action="store_true",
                   help="do not fail if a manifest has fewer hours than requested")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    mixed, summary = build_mix(
        list(read_manifest(args.majority)),
        list(read_manifest(args.minority)),
        args.majority_hours,
        args.minority_fraction,
        args.seed,
        args.allow_short,
    )
    write_manifest(args.out, mixed)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Wrote {len(mixed)} utterances to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
