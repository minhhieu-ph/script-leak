"""Frame-level analysis of leakage: why does the minority token win?

For utterances spoken in the majority language, a "leak frame" is a frame
whose top token belongs to the minority script. For each leak frame we ask:

  mass split   Is the summed probability of all majority tokens larger than
               the summed probability of all minority tokens? If yes, the
               majority script "lost" only because its mass was spread over
               several candidates. This is the direct test of hypothesis H1
               and the headroom for script-marginalized decoding.
  uncertainty  Is the frame more uncertain (higher entropy, more competing
               majority candidates) than normal majority frames?

Example:
  python -m scriptleak.analysis --dump dumps/th_snr0 --majority th --minority en
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

import numpy as np

from .decoding import group_logmass
from .dumpio import iter_dump, read_meta
from .languages import get_language
from .normalize import scoring_normalize
from .tokenmap import BLANK, MAJ, MIN, token_groups


def frame_table(logp: np.ndarray, groups: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-frame quantities used by the analysis."""
    p = np.exp(logp.astype(np.float64))
    top = logp.argmax(axis=1)
    p_maj = np.exp(group_logmass(logp, groups, MAJ))
    p_min = np.exp(group_logmass(logp, groups, MIN))
    entropy = -(p * np.where(p > 0, np.log(np.maximum(p, 1e-300)), 0.0)).sum(axis=1)
    # Effective number of majority candidates: exp(entropy of the majority
    # tokens renormalised among themselves).
    maj_cols = p[:, groups == MAJ]
    if maj_cols.shape[1]:
        q = maj_cols / np.maximum(maj_cols.sum(axis=1, keepdims=True), 1e-300)
        eff_maj = np.exp(-(q * np.where(q > 0, np.log(np.maximum(q, 1e-300)), 0.0)).sum(axis=1))
        top_maj = maj_cols.max(axis=1)
    else:
        eff_maj = np.zeros(len(logp))
        top_maj = np.zeros(len(logp))
    min_cols = p[:, groups == MIN]
    top_min = min_cols.max(axis=1) if min_cols.shape[1] else np.zeros(len(logp))
    return {
        "top_group": groups[top],
        "p_maj": p_maj,
        "p_min": p_min,
        "top_maj": top_maj,
        "top_min": top_min,
        "entropy": entropy,
        "eff_maj": eff_maj,
    }


class LeakFrameAccumulator:
    def __init__(self) -> None:
        self.rows: Dict[str, List[np.ndarray]] = {"leak": [], "normal": []}
        self.utts = 0
        self.utts_with_leak = 0
        self.run_lengths: List[int] = []

    def add(self, logp: np.ndarray, groups: np.ndarray) -> None:
        ft = frame_table(logp, groups)
        leak = ft["top_group"] == MIN
        normal = ft["top_group"] == MAJ
        self.utts += 1
        if leak.any():
            self.utts_with_leak += 1
            # lengths of runs of consecutive non-blank frames that are minority
            run = 0
            for g in ft["top_group"]:
                if g == MIN:
                    run += 1
                elif g == MAJ and run:
                    self.run_lengths.append(run)
                    run = 0
            if run:
                self.run_lengths.append(run)
        for name, sel in (("leak", leak), ("normal", normal)):
            if sel.any():
                self.rows[name].append(np.stack(
                    [ft["p_maj"][sel], ft["p_min"][sel], ft["top_maj"][sel],
                     ft["top_min"][sel], ft["entropy"][sel], ft["eff_maj"][sel]], axis=1))

    def summary(self) -> Dict:
        out: Dict = {
            "utterances": self.utts,
            "utterances_with_leak_frames": self.utts_with_leak,
        }
        for name in ("leak", "normal"):
            if not self.rows[name]:
                out[name] = {"frames": 0}
                continue
            a = np.concatenate(self.rows[name], axis=0)
            p_maj, p_min, top_maj, top_min, ent, eff = a.T
            out[name] = {
                "frames": int(len(a)),
                "mean_p_majority_total": float(p_maj.mean()),
                "mean_p_minority_total": float(p_min.mean()),
                "mean_top_majority_token": float(top_maj.mean()),
                "mean_top_minority_token": float(top_min.mean()),
                "mean_entropy": float(ent.mean()),
                "mean_effective_majority_candidates": float(eff.mean()),
            }
            if name == "leak":
                out[name]["frac_majority_mass_exceeds_minority_mass"] = float((p_maj > p_min).mean())
                out[name]["frac_majority_mass_above_0.3"] = float((p_maj > 0.3).mean())
        rl = self.run_lengths
        out["leak_runs"] = {
            "count": len(rl),
            "mean_frames": float(np.mean(rl)) if rl else 0.0,
            "max_frames": int(max(rl)) if rl else 0,
        }
        return out


def format_summary(s: Dict) -> str:
    lines = [
        f"Utterances analysed: {s['utterances']} "
        f"({s['utterances_with_leak_frames']} have at least one leak frame)",
    ]
    leak, normal = s["leak"], s["normal"]
    if not leak["frames"]:
        lines.append("No leak frames.")
        return "\n".join(lines)
    lines += [
        f"Leak frames: {leak['frames']}   normal majority frames: {normal['frames']}",
        "",
        f"H1 (mass splitting): majority total mass > minority total mass in "
        f"{leak['frac_majority_mass_exceeds_minority_mass']:.1%} of leak frames",
        f"  mean total mass at leak frames: majority {leak['mean_p_majority_total']:.3f}, "
        f"minority {leak['mean_p_minority_total']:.3f}",
        f"  mean top token at leak frames:  majority {leak['mean_top_majority_token']:.3f}, "
        f"minority {leak['mean_top_minority_token']:.3f}",
        "",
        "Uncertainty (leak frames vs normal majority frames):",
    ]
    if normal["frames"]:
        lines += [
            f"  entropy                        {leak['mean_entropy']:.3f} vs {normal['mean_entropy']:.3f}",
            f"  effective majority candidates  {leak['mean_effective_majority_candidates']:.2f} vs "
            f"{normal['mean_effective_majority_candidates']:.2f}",
        ]
    r = s["leak_runs"]
    lines += ["", f"Leak runs: {r['count']}, mean {r['mean_frames']:.1f} frames, max {r['max_frames']}"]
    return "\n".join(lines)


def analyse_dump(dump_dir: str, majority_code: str, minority_code: str,
                 group_by: Optional[str] = None) -> Dict[str, Dict]:
    majority, minority = get_language(majority_code), get_language(minority_code)
    meta = read_meta(dump_dir)
    groups = token_groups(meta["vocab"], majority, minority, meta["blank_id"])
    accs: Dict[str, LeakFrameAccumulator] = {"ALL": LeakFrameAccumulator()}
    for rec, logp in iter_dump(dump_dir):
        ref = scoring_normalize(str(rec.get("text", "")))
        # Only utterances whose reference is purely majority script.
        if minority.script.count(ref) or not majority.script.count(ref):
            continue
        accs["ALL"].add(logp, groups)
        if group_by:
            accs.setdefault(str(rec.get(group_by, "UNKNOWN")), LeakFrameAccumulator()).add(logp, groups)
    return {name: acc.summary() for name, acc in accs.items()}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True)
    p.add_argument("--majority", required=True)
    p.add_argument("--minority", required=True)
    p.add_argument("--group-by")
    p.add_argument("--out")
    args = p.parse_args(argv)
    res = analyse_dump(args.dump, args.majority, args.minority, args.group_by)
    for name, s in res.items():
        print(f"=== {name} ===")
        print(format_summary(s))
        print()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
