#!/usr/bin/env bash
# The two-week kill test, one stage at a time.
#
#   bash scripts/killtest.sh STAGE
#
# Stages, in order:
#   prepare    corpora -> raw manifests            (needs TEXT_COL)
#   audit      spoken-form filter; STOP and read the Thai hours
#   split      Thai train/dev/test, noisy strata, dialect test sets
#   mix        training manifests (English = 5% and 20% of hours)
#   tokenizer  one SentencePiece model per language
#   train05    train with 5% English                (needs BASE_CONFIG)
#   train20    train with 20% English
#   dump       frame-level log-probabilities for both models
#   results    leakage tables, frame analysis, method sweep
#
# Settings are environment variables, e.g.
#   D=/data/scriptleak TEXT_COL=sentence bash scripts/killtest.sh prepare
#
# NOT TESTED where it was written (no data, no GPU there).
set -euo pipefail

D="${D:-/data/scriptleak}"            # where download_data.sh put the corpora
W="${W:-$D/work}"                     # manifests, tokenizers, models, dumps
TEXT_COL="${TEXT_COL:-}"              # transcript column of the Thai datasets
SPEAKER_COL="${SPEAKER_COL:-}"        # speaker column, if the datasets have one
BASE_CONFIG="${BASE_CONFIG:-}"        # a NeMo FastConformer hybrid transducer-CTC BPE yaml
INIT="${INIT:-stt_en_fastconformer_hybrid_large_streaming_multi}"   # INIT=none trains from scratch
GPUS="${GPUS:-8}"
BATCH="${BATCH:-32}"                  # per GPU
EPOCHS="${EPOCHS:-10}"
TH_HOURS="${TH_HOURS:-300}"
DIALECT_TEST_HOURS="${DIALECT_TEST_HOURS:-5}"
DIALECTS="khummuang korat pattani"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p "$W"
need() { [ -n "${!1}" ] || { echo "Set $1 first. $2"; exit 1; }; }

stage_prepare() {
  need TEXT_COL "Find it with: python -m scriptleak.prepare hf-columns --dataset CMKL/Porjai-Thai-voice-dataset-korat"
  for part in train-clean-100:en_train dev-clean:en_dev test-clean:en_test; do
    python -m scriptleak.prepare librispeech --root "$D/LibriSpeech" \
        --subsets "${part%%:*}" --out "$W/${part##*:}.raw.jsonl"
  done
  local copy=()
  [ -n "$SPEAKER_COL" ] && copy=(--copy-cols "$SPEAKER_COL")
  for name in central $DIALECTS; do
    python -m scriptleak.prepare hf --dataset "CMKL/Porjai-Thai-voice-dataset-$name" \
        --split train --text-col "$TEXT_COL" "${copy[@]}" \
        --audio-out "$D/porjai/$name/wav" --out "$W/th_$name.raw.jsonl"
  done
}

stage_audit() {
  python -m scriptleak.audit --manifest "$W/th_central.raw.jsonl" --lang th \
      --out-manifest "$W/th_central.clean.jsonl" --report "$W/th_central.audit.json"
  for s in en_train en_dev en_test; do
    python -m scriptleak.audit --manifest "$W/$s.raw.jsonl" --lang en \
        --out-manifest "$W/$s.clean.jsonl" --report "$W/$s.audit.json"
  done
  for name in $DIALECTS; do
    python -m scriptleak.audit --manifest "$W/th_$name.raw.jsonl" --lang th \
        --out-manifest "$W/th_$name.clean.jsonl" --report "$W/th_$name.audit.json" \
        --set "stratum=$name"
  done
  echo
  echo "STOP: Thai-central needs at least $((TH_HOURS + 15)) clean hours (see above)."
}

stage_split() {
  local spk=()
  [ -n "$SPEAKER_COL" ] && spk=(--speaker-field "$SPEAKER_COL")
  python -m scriptleak.split --manifest "$W/th_central.clean.jsonl" \
      --dev-hours 3 --test-hours 10 "${spk[@]}" --out-prefix "$W/th_central"
  python -m scriptleak.noise --manifest "$W/th_central.test.jsonl" --noise-dir "$D/musan/noise" \
      --snrs 10 5 0 --out-dir "$D/th_test_noisy" --out-manifest "$W/th_test.strata.jsonl"
  for name in $DIALECTS; do
    python -m scriptleak.split --manifest "$W/th_$name.clean.jsonl" \
        --dev-hours 0 --test-hours "$DIALECT_TEST_HOURS" "${spk[@]}" --out-prefix "$W/th_$name"
    cat "$W/th_$name.test.jsonl" >> "$W/th_test.strata.jsonl"
  done
  echo "Thai test set with strata: $W/th_test.strata.jsonl"
}

stage_mix() {
  for pct in 05 20; do
    python -m scriptleak.mix --majority "$W/th_central.train.jsonl" --minority "$W/en_train.clean.jsonl" \
        --majority-hours "$TH_HOURS" --minority-fraction "0.$pct" --seed 0 \
        --out "$W/train_th${TH_HOURS}_en$pct.jsonl"
  done
}

stage_tokenizer() {
  python -m scriptleak.nemo_tools.build_tokenizer --manifest "$W/th_central.train.jsonl" \
      --out-dir "$W/tokenizers/th_bpe1024" --vocab-size 1024
  python -m scriptleak.nemo_tools.build_tokenizer --manifest "$W/en_train.clean.jsonl" \
      --out-dir "$W/tokenizers/en_bpe256" --vocab-size 256
}

train_one() {  # train_one PCT
  need BASE_CONFIG "Pick one of the yaml files printed by scripts/setup_env.sh (step 4)."
  local init=()
  [ "$INIT" != "none" ] && init=(--init-pretrained "$INIT")
  python -m scriptleak.nemo_tools.train --base-config "$BASE_CONFIG" \
      --train-manifest "$W/train_th${TH_HOURS}_en$1.jsonl" \
      --val-manifest "$W/th_central.dev.jsonl" "$W/en_dev.clean.jsonl" \
      --tokenizer "th=$W/tokenizers/th_bpe1024" "en=$W/tokenizers/en_bpe256" \
      "${init[@]}" --exp-dir "$W/exp/th${TH_HOURS}_en$1" \
      --max-epochs "$EPOCHS" --batch-size "$BATCH" --num-workers 8 --devices "$GPUS"
}

stage_dump() {
  for pct in 05 20; do
    local m="$W/exp/th${TH_HOURS}_en$pct"
    CUDA_VISIBLE_DEVICES=0 python -m scriptleak.nemo_tools.dump_logprobs --model "$m/final.nemo" \
        --manifest "$W/th_test.strata.jsonl" --out "$W/dumps/en$pct/th_test" --transducer
    CUDA_VISIBLE_DEVICES=0 python -m scriptleak.nemo_tools.dump_logprobs --model "$m/final.nemo" \
        --manifest "$W/en_test.clean.jsonl" --out "$W/dumps/en$pct/en_test" --transducer
  done
}

stage_results() {
  for pct in 05 20; do
    local dd="$W/dumps/en$pct" r="$W/results/en$pct"
    mkdir -p "$r"
    echo "################ English = $pct% of training hours ################"
    echo "--- 1. Is there leakage, and does it grow with difficulty? (CTC head)"
    python -m scriptleak.sweep --dump "$dd/th_test" --majority th --minority en --spoken th \
        --group-by stratum --methods greedy --out "$r/leak_ctc.json"
    echo "--- 2. Same question for the transducer head"
    python -m scriptleak.evaluate --manifest "$dd/th_test/utts.jsonl" --majority th --minority en \
        --hyp-field pred_transducer --group-by stratum --out "$r/leak_transducer.json"
    echo "--- 3. Why: frame analysis"
    python -m scriptleak.analysis --dump "$dd/th_test" --majority th --minority en \
        --group-by stratum --out "$r/frames.json"
    echo "--- 4. Methods on Thai speech"
    python -m scriptleak.sweep --dump "$dd/th_test" --majority th --minority en --spoken th \
        --out "$r/sweep_th.json"
    echo "--- 5. What each method costs English"
    python -m scriptleak.sweep --dump "$dd/en_test" --majority th --minority en --spoken en \
        --out "$r/sweep_en.json"
  done 2>&1 | tee "$W/results_summary.txt"
  echo "Summary saved to $W/results_summary.txt"
}

case "${1:-}" in
  prepare)   stage_prepare ;;
  audit)     stage_audit ;;
  split)     stage_split ;;
  mix)       stage_mix ;;
  tokenizer) stage_tokenizer ;;
  train05)   train_one 05 ;;
  train20)   train_one 20 ;;
  dump)      stage_dump ;;
  results)   stage_results ;;
  *) sed -n '2,22p' "$0"; exit 1 ;;
esac
