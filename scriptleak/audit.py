"""Audit a manifest against the spoken-form rules and write the clean subset.

Example:
  python -m scriptleak.audit --manifest thai_central.jsonl --lang th \
      --out-manifest thai_central.clean.jsonl --report thai_central.audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from typing import Dict, List, Optional

from .languages import get_language
from .manifest import read_manifest, write_manifest
from .normalize import normalize


def audit_entries(entries, lang_code: str, text_field: str = "text", max_examples: int = 5):
    """Returns (clean_entries, report_dict)."""
    lang = get_language(lang_code)
    clean: List[Dict] = []
    n_total = 0
    sec_total = 0.0
    sec_clean = 0.0
    reason_utts: Counter = Counter()
    reason_secs: Counter = Counter()
    examples: Dict[str, List[str]] = {}
    changed = 0
    chars_before: Counter = Counter()
    chars_after: Counter = Counter()

    for entry in entries:
        n_total += 1
        dur = float(entry.get("duration", 0.0))
        sec_total += dur
        raw = str(entry.get(text_field, ""))
        chars_before.update(raw)
        res = normalize(raw, lang)
        if res.ok:
            if res.text != raw:
                changed += 1
            new_entry = dict(entry)
            new_entry[text_field] = res.text
            new_entry["lang"] = lang.code
            clean.append(new_entry)
            sec_clean += dur
            chars_after.update(res.text)
        else:
            for reason in res.reasons:
                reason_utts[reason] += 1
                reason_secs[reason] += dur
                ex = examples.setdefault(reason, [])
                if len(ex) < max_examples:
                    ex.append(raw)

    def describe(counter: Counter):
        rows = []
        for ch, n in counter.most_common():
            rows.append(
                {
                    "char": ch,
                    "codepoint": f"U+{ord(ch):04X}",
                    "name": unicodedata.name(ch, "UNKNOWN"),
                    "count": n,
                }
            )
        return rows

    report = {
        "language": lang.code,
        "utterances_total": n_total,
        "utterances_clean": len(clean),
        "utterances_rejected": n_total - len(clean),
        "hours_total": round(sec_total / 3600.0, 3),
        "hours_clean": round(sec_clean / 3600.0, 3),
        "clean_fraction_utterances": round(len(clean) / n_total, 4) if n_total else 0.0,
        "clean_fraction_hours": round(sec_clean / sec_total, 4) if sec_total else 0.0,
        "clean_utterances_modified_by_normalisation": changed,
        "rejections": {
            reason: {
                "utterances": reason_utts[reason],
                "fraction_utterances": round(reason_utts[reason] / n_total, 4),
                "hours": round(reason_secs[reason] / 3600.0, 3),
                "examples": examples.get(reason, []),
            }
            for reason, _ in reason_utts.most_common()
        },
        "clean_char_inventory_size": len(chars_after),
        "clean_char_inventory": describe(chars_after),
        "raw_non_script_chars": [
            row
            for row in describe(chars_before)
            if not lang.script.contains(row["char"]) and not row["char"].isspace()
        ],
    }
    return clean, report


def format_report(report: Dict) -> str:
    lines = [
        f"Language: {report['language']}",
        f"Utterances: {report['utterances_clean']} clean / {report['utterances_total']} total "
        f"({report['clean_fraction_utterances']:.1%})",
        f"Hours:      {report['hours_clean']:.2f} clean / {report['hours_total']:.2f} total "
        f"({report['clean_fraction_hours']:.1%})",
        f"Clean utterances changed by normalisation: {report['clean_utterances_modified_by_normalisation']}",
        f"Clean character inventory: {report['clean_char_inventory_size']} characters",
        "",
        "Rejections (an utterance can have several reasons):",
    ]
    if not report["rejections"]:
        lines.append("  none")
    for reason, info in report["rejections"].items():
        lines.append(
            f"  {reason:<28} {info['utterances']:>9} utt "
            f"({info['fraction_utterances']:.2%}), {info['hours']:.2f} h"
        )
        for ex in info["examples"][:2]:
            lines.append(f"      e.g. {ex}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="input JSON-lines manifest")
    p.add_argument("--lang", required=True, help="language code, e.g. th, en, zh, ru")
    p.add_argument("--out-manifest", help="write the clean, normalised subset here")
    p.add_argument("--report", help="write the full JSON report here")
    p.add_argument("--text-field", default="text")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="stamp fields on every clean entry, e.g. stratum=korat")
    args = p.parse_args(argv)

    clean, report = audit_entries(read_manifest(args.manifest), args.lang, args.text_field)
    stamps = dict(item.split("=", 1) for item in args.set)
    for entry in clean:
        entry.update(stamps)
    print(format_report(report))
    if args.out_manifest:
        write_manifest(args.out_manifest, clean)
        print(f"\nWrote {len(clean)} clean utterances to {args.out_manifest}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Wrote report to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
