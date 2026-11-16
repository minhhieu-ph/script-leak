"""Create noisy copies of a test set at fixed SNRs (difficulty strata).

Example:
  python -m scriptleak.noise --manifest th_test.clean.jsonl \
      --noise-dir /data/musan/noise --snrs 10 5 0 \
      --out-dir /data/th_test_noisy --out-manifest th_test.strata.jsonl

The output manifest contains the clean utterances (stratum "clean") plus one
noisy copy per SNR (stratum "snr10", "snr5", "snr0"). Noise selection is
seeded per utterance, so reruns produce identical audio.
Needs: numpy, scipy, soundfile.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
from typing import List, Optional

import numpy as np

from .audio import resample, to_mono
from .manifest import read_manifest, write_manifest


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def fit_noise(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random crop, or loop, of the noise to exactly n samples."""
    if len(noise) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(noise) >= n:
        start = int(rng.integers(0, len(noise) - n + 1))
        return noise[start:start + n]
    reps = int(np.ceil(n / len(noise)))
    start = int(rng.integers(0, len(noise)))
    return np.roll(np.tile(noise, reps), -start)[:n]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: np.random.Generator) -> np.ndarray:
    """speech + scaled noise so that 20*log10(rms(speech)/rms(noise)) = snr_db.
    The mixture is scaled down only if it would clip."""
    speech = speech.astype(np.float32)
    noise = fit_noise(noise.astype(np.float32), len(speech), rng)
    rs, rn = rms(speech), rms(noise)
    if rs == 0.0 or rn == 0.0:
        return speech.copy()
    gain = rs / (rn * 10.0 ** (snr_db / 20.0))
    mixed = speech + gain * noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.999:
        mixed = mixed * (0.999 / peak)
    return mixed.astype(np.float32)


def _seed_for(text: str, base_seed: int) -> int:
    h = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--noise-dir", required=True, help="directory searched recursively for .wav noise files")
    p.add_argument("--snrs", type=float, nargs="+", default=[10.0, 5.0, 0.0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-manifest", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stratum-field", default="stratum")
    args = p.parse_args(argv)

    import soundfile as sf

    noise_files = sorted(glob.glob(os.path.join(args.noise_dir, "**", "*.wav"), recursive=True))
    if not noise_files:
        raise SystemExit(f"No .wav files under {args.noise_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    out_entries = []
    for i, e in enumerate(read_manifest(args.manifest)):
        clean = dict(e)
        clean.setdefault(args.stratum_field, "clean")
        out_entries.append(clean)
        speech, sr = sf.read(e["audio_filepath"], dtype="float32", always_2d=False)
        speech = to_mono(speech)
        rng = np.random.default_rng(_seed_for(e["audio_filepath"], args.seed))
        noise_path = noise_files[int(rng.integers(0, len(noise_files)))]
        noise, nsr = sf.read(noise_path, dtype="float32", always_2d=False)
        noise = resample(to_mono(noise), nsr, sr)
        for snr in args.snrs:
            tag = f"snr{snr:g}"
            out_path = os.path.join(args.out_dir, tag, f"{i:07d}.wav")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # same noise segment for every SNR of an utterance
            seg_rng = np.random.default_rng(_seed_for(e["audio_filepath"], args.seed + 1))
            sf.write(out_path, mix_at_snr(speech, noise, snr, seg_rng), sr, subtype="PCM_16")
            noisy = dict(e)
            noisy["audio_filepath"] = out_path
            noisy[args.stratum_field] = tag
            noisy["noise_file"] = noise_path
            out_entries.append(noisy)
        if (i + 1) % 1000 == 0:
            print(f"{i + 1} utterances processed", flush=True)

    write_manifest(args.out_manifest, out_entries)
    print(f"Wrote {len(out_entries)} entries to {args.out_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
