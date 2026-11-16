"""Run decoding methods over a log-probability dump and score each one.

The training roles (--majority / --minority) define which tokens a method may
restrict. --spoken says which language the dumped test set is in:

  spoken = majority  measures leakage of the minority script, and CER/WER
  spoken = minority  measures what each method costs the minority language

Running both on the same list of methods gives the trade-off curve.

Example:
  python -m scriptleak.sweep --dump dumps/th_test --majority th --minority en \
      --spoken th --group-by stratum --out sweep_th.json
  python -m scriptleak.sweep --dump dumps/en_test --majority th --minority en \
      --spoken en --out sweep_en.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from typing import Dict, List, Optional

from .decoding import DEFAULT_SWEEP, decode
from .dumpio import iter_dump, read_meta
from .evaluate import get_word_tokenizer, score
from .languages import get_language
from .manifest import write_manifest
from .tokenmap import ids_to_text, token_groups


def run_sweep(dump_dir: str, majority_code: str, minority_code: str, spoken_code: str,
              methods: List[str], group_by: Optional[str] = None,
              word_tokenizer=None, save_preds_dir: Optional[str] = None) -> "OrderedDict[str, Dict]":
    majority, minority = get_language(majority_code), get_language(minority_code)
    if spoken_code not in (majority_code, minority_code):
        raise ValueError("--spoken must be the majority or the minority language")
    spoken = majority if spoken_code == majority_code else minority
    other = minority if spoken is majority else majority

    meta = read_meta(dump_dir)
    vocab = meta["vocab"]
    groups = token_groups(vocab, majority, minority, meta["blank_id"])

    refs: List[str] = []
    strata: List[str] = []
    recs: List[Dict] = []
    hyps: Dict[str, List[str]] = {m: [] for m in methods}
    for rec, logp in iter_dump(dump_dir):
        refs.append(str(rec.get("text", "")))
        strata.append(str(rec.get(group_by, "UNKNOWN")) if group_by else "ALL")
        recs.append(rec)
        for m in methods:
            hyps[m].append(ids_to_text(decode(logp, groups, m), vocab))

    results: "OrderedDict[str, Dict]" = OrderedDict()
    names = ["ALL"] + (sorted(set(strata)) if group_by else [])
    for m in methods:
        results[m] = OrderedDict()
        for name in names:
            idx = range(len(refs)) if name == "ALL" else [i for i, s in enumerate(strata) if s == name]
            results[m][name] = score([refs[i] for i in idx], [hyps[m][i] for i in idx],
                                     spoken, other, word_tokenizer)
        if save_preds_dir:
            import os
            os.makedirs(save_preds_dir, exist_ok=True)
            safe = m.replace(":", "_").replace("=", "").replace(",", "_")
            write_manifest(os.path.join(save_preds_dir, f"preds_{safe}.jsonl"),
                           (dict(r, pred_text=h) for r, h in zip(recs, hyps[m])))
    return results


def format_sweep(results: "OrderedDict[str, Dict]", group: str = "ALL") -> str:
    width = max(len(m) for m in results)
    head = f"{'method':<{width}}  {'eligible':>8}  {'SLR-u':>8}  {'SLR-c':>8}  {'CER':>8}  {'WER':>8}"
    lines = [head, "-" * len(head)]
    for m, by_group in results.items():
        r = by_group[group]
        w = f"{r['wer']:.2%}" if "wer" in r else "-"
        lines.append(f"{m:<{width}}  {r['utterances_eligible']:>8}  {r['slr_u']:>8.2%}  "
                     f"{r['slr_c']:>8.2%}  {r['cer']:>8.2%}  {w:>8}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True)
    p.add_argument("--majority", required=True, help="majority language at training time")
    p.add_argument("--minority", required=True, help="minority language at training time")
    p.add_argument("--spoken", help="language of this test set (default: majority)")
    p.add_argument("--methods", nargs="*", help="method specs; default is a built-in sweep")
    p.add_argument("--group-by")
    p.add_argument("--word-tokenizer", default="space", choices=["space", "pythainlp"])
    p.add_argument("--save-preds", help="directory for per-method prediction manifests")
    p.add_argument("--out")
    args = p.parse_args(argv)

    methods = args.methods or list(DEFAULT_SWEEP)
    spoken = args.spoken or args.majority
    results = run_sweep(args.dump, args.majority, args.minority, spoken, methods,
                        args.group_by, get_word_tokenizer(args.word_tokenizer), args.save_preds)
    groups = list(next(iter(results.values())).keys())
    for g in groups:
        print(f"=== {g} (spoken: {spoken}) ===")
        print(format_sweep(results, g))
        print()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
