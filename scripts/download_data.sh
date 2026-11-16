#!/usr/bin/env bash
# Download the public corpora.
#
#   bash scripts/download_data.sh /data/scriptleak              # kill-test set
#   WITH_AISHELL=1 bash scripts/download_data.sh /data/scriptleak
#   WITH_LIBRI_360=1 bash scripts/download_data.sh /data/scriptleak
#
# Kill-test set and approximate download size:
#   LibriSpeech train-clean-100, dev-clean, test-clean    ~7 GB
#   MUSAN                                                 ~11 GB
#   Thai Dialect Corpus (4 Hugging Face datasets)         central ~7.5 GB + 3 dialects
# Optional:
#   AISHELL-1 (WITH_AISHELL=1)                            ~15 GB
#   LibriSpeech train-clean-360 (WITH_LIBRI_360=1)        ~23 GB
#   GigaSpeech 2 Thai test (gated, see the end)
#
# Downloads resume if interrupted (wget -c). The Thai audio is written as
# 16 kHz wav by the converter, roughly 115 MB per hour of speech.
#
# NOT TESTED where it was written (no network there).
set -euo pipefail

D="${1:?usage: download_data.sh DATA_ROOT}"
OPENSLR="${OPENSLR:-https://www.openslr.org/resources}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$D/downloads" "$D/manifests"

fetch() {  # fetch URL -> $D/downloads/<name>
  local url="$1" out="$D/downloads/$(basename "$1")"
  echo "-- $url"
  wget -c -q --show-progress -O "$out" "$url"
}

echo "== LibriSpeech"
LIBRI_PARTS="train-clean-100 dev-clean test-clean"
[ "${WITH_LIBRI_360:-0}" = "1" ] && LIBRI_PARTS="$LIBRI_PARTS train-clean-360"
for part in $LIBRI_PARTS; do
  if [ ! -d "$D/LibriSpeech/$part" ]; then
    fetch "$OPENSLR/12/$part.tar.gz"
    tar -xzf "$D/downloads/$part.tar.gz" -C "$D"
  else
    echo "have $part"
  fi
done

echo "== MUSAN"
if [ ! -d "$D/musan/noise" ]; then
  fetch "$OPENSLR/17/musan.tar.gz"
  tar -xzf "$D/downloads/musan.tar.gz" -C "$D"
else
  echo "have musan"
fi

if [ "${WITH_AISHELL:-0}" = "1" ]; then
  echo "== AISHELL-1"
  if [ ! -d "$D/data_aishell/wav/train" ]; then
    fetch "$OPENSLR/33/data_aishell.tgz"
    tar -xzf "$D/downloads/data_aishell.tgz" -C "$D"
    # the archive contains one tar.gz per speaker
    for f in "$D"/data_aishell/wav/*.tar.gz; do
      tar -xzf "$f" -C "$D/data_aishell/wav" && rm "$f"
    done
  else
    echo "have data_aishell"
  fi
fi

echo "== Thai Dialect Corpus (Hugging Face)"
cd "$HERE"
echo "Columns of the Korat set (use these to choose --text-col below):"
python -m scriptleak.prepare hf-columns --dataset CMKL/Porjai-Thai-voice-dataset-korat || true
cat <<EOF

The Thai sets are downloaded and converted in one step. Replace TEXT_COL with
the column that holds the transcript *as spoken* (for the three dialect sets
this is the dialect sentence, not its Central Thai translation):

  for name in central khummuang korat pattani; do
    python -m scriptleak.prepare hf --dataset CMKL/Porjai-Thai-voice-dataset-\$name \\
        --split train --text-col TEXT_COL \\
        --audio-out $D/porjai/\$name/wav --out $D/manifests/th_\$name.raw.jsonl
  done

EOF

cat <<EOF
== GigaSpeech 2 Thai test set (optional, gated)
1. Log in on huggingface.co, open datasets/speechcolab/gigaspeech2 and accept the terms.
2. huggingface-cli login
3. huggingface-cli download speechcolab/gigaspeech2 data/th/test.tar.gz data/th/test.tsv \\
       --repo-type dataset --local-dir $D/gigaspeech2
4. mkdir -p $D/gigaspeech2/data/th/test && tar -xzf $D/gigaspeech2/data/th/test.tar.gz -C $D/gigaspeech2/data/th/test
5. python -m scriptleak.prepare gigaspeech2 --tsv $D/gigaspeech2/data/th/test.tsv \\
       --audio-dir $D/gigaspeech2/data/th/test --out $D/manifests/th_gs2_test.raw.jsonl

Downloads finished. Disk used:
EOF
du -sh "$D"/* 2>/dev/null || true
