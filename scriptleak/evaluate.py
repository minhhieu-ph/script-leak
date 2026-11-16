"""Score a prediction manifest: leakage, CER/WER, optionally per stratum.

The manifest needs the reference in `text` and the model output in
`pred_text` (what NeMo's transcription scripts write).

Example:
  python -m scriptleak.evaluate --manifest preds.jsonl --majority th --minority en \
      --group-by stratum --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

from .languages import Language, get_language
from .manifest import read_manifest
from .metrics import cer, leak_stats, wer
from .normalize import scoring_normalize


def get_word_tokenizer(name: str) -> Optional[Callable[[str], List[str]]]:
    if name == "space":
        return None
    if name == "pythainlp":
        try:
            from pythainlp.tokenize import word_tokenize
        except ImportError:
            raise SystemExit("pythainlp is not installed: pip install pythainlp") from None

        def tok(text: str) -> List[str]:
            return [w for w in word_tokenize(text, engine="newmm") if w.strip()]

        return tok
    raise SystemExit(f"Unknown word tokenizer '{name}'")


def score(
    refs: List[str],
    hyps: List[str],
    majority: Language,
    minority: Language,
    word_tokenizer=None,
) -> Dict:
    refs = [scoring_normalize(r) for r in refs]
    hyps = [scoring_normalize(h) for h in hyps]
    stats = leak_stats(refs, hyps, majority, minority, word_tokenizer)
    # Accuracy on the same eligible utterances as the leakage rate.
    keep = [
        i for i, r in enumerate(refs)
        if minority.script.count(r) == 0 and majority.script.count(r) > 0
    ]
    r_keep = [refs[i] for i in keep]
    h_keep = [hyps[i] for i in keep]
    out = stats.to_dict()
    out["cer"] = cer(r_keep, h_keep, ignore_spaces=majority.cer_ignores_spaces)
    if not majority.cer_ignores_spaces:
        out["wer"] = wer(r_keep, h_keep)
    return out


def format_table(results: "OrderedDict[str, Dict]") -> str:
    cols = ["eligible", "SLR-u", "SLR-c", "SLR-w", "spans", "mean_span", "CER", "WER"]
    width = max(8, max(len(k) for k in results))
    head = f"{'group':<{width}}  " + "  ".join(f"{c:>9}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        w = f"{r['wer']:.2%}" if "wer" in r else "-"
        row = [
            str(r["utterances_eligible"]),
            f"{r['slr_u']:.2%}",
            f"{r['slr_c']:.2%}",
            f"{r['slr_w']:.2%}",
            str(r["leak_spans"]),
            f"{r['mean_span_chars']:.1f}",
            f"{r['cer']:.2%}",
            w,
        ]
        lines.append(f"{name:<{width}}  " + "  ".join(f"{c:>9}" for c in row))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--majority", required=True, help="language spoken in the test set")
    p.add_argument("--minority", required=True, help="language whose script must not appear")
    p.add_argument("--ref-field", default="text")
    p.add_argument("--hyp-field", default="pred_text")
    p.add_argument("--group-by", help="manifest field to stratify by, e.g. stratum, snr, dialect")
    p.add_argument("--word-tokenizer", default="space", choices=["space", "pythainlp"])
    p.add_argument("--out", help="write JSON results here")
    args = p.parse_args(argv)

    majority, minority = get_language(args.majority), get_language(args.minority)
    tokenizer = get_word_tokenizer(args.word_tokenizer)

    groups: "OrderedDict[str, Dict[str, List[str]]]" = OrderedDict()
    groups["ALL"] = {"refs": [], "hyps": []}
    for e in read_manifest(args.manifest):
        ref, hyp = str(e.get(args.ref_field, "")), str(e.get(args.hyp_field, ""))
        groups["ALL"]["refs"].append(ref)
        groups["ALL"]["hyps"].append(hyp)
        if args.group_by:
            g = groups.setdefault(str(e.get(args.group_by, "UNKNOWN")), {"refs": [], "hyps": []})
            g["refs"].append(ref)
            g["hyps"].append(hyp)

    results: "OrderedDict[str, Dict]" = OrderedDict()
    for name, g in groups.items():
        results[name] = score(g["refs"], g["hyps"], majority, minority, tokenizer)

    print(f"Leakage of {minority.name} script into {majority.name} speech")
    print(format_table(results))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
