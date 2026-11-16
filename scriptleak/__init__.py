"""scriptleak: tools to measure minority-language (script) leakage in bilingual ASR.

Batch 1 (no NeMo / no GPU needed):
  - scripts.py    Unicode script definitions
  - languages.py  per-language spoken-form rules
  - normalize.py  strict spoken-form normalisation (filter, do not guess)
  - audit.py      audit + filter a manifest, report what was rejected and why
  - mix.py        build a majority/minority training manifest at a given ratio
  - metrics.py    leakage rates, CER, WER
  - evaluate.py   score a prediction manifest, optionally per stratum
"""

__version__ = "0.1.0"
