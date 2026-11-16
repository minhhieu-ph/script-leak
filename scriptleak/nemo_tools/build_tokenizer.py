"""Train one SentencePiece tokenizer per language, in the folder layout NeMo
expects (tokenizer.model, tokenizer.vocab, vocab.txt).

  python -m scriptleak.nemo_tools.build_tokenizer --manifest th.clean.jsonl \
      --out-dir tokenizers/th_bpe1024 --vocab-size 1024
  python -m scriptleak.nemo_tools.build_tokenizer --manifest en.clean.jsonl \
      --out-dir tokenizers/en_bpe256 --vocab-size 256

Text normalisation inside SentencePiece is switched off (`identity`): the
manifests are already normalised, and SentencePiece's default NFKC would split
the Thai vowel SARA AM into two characters.
Needs: sentencepiece.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import List, Optional

from ..manifest import read_manifest


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--model-type", default="bpe", choices=["bpe", "unigram", "char"])
    p.add_argument("--text-field", default="text")
    args = p.parse_args(argv)

    import sentencepiece as spm

    os.makedirs(args.out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        n = 0
        for m in args.manifest:
            for e in read_manifest(m):
                text = str(e.get(args.text_field, "")).strip()
                if text:
                    f.write(text + "\n")
                    n += 1
        text_path = f.name
    print(f"{n} sentences")

    prefix = os.path.join(args.out_dir, "tokenizer")
    spm.SentencePieceTrainer.train(
        input=text_path,
        model_prefix=prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=1.0,
        normalization_rule_name="identity",
        bos_id=-1,
        eos_id=-1,
        unk_id=0,
        hard_vocab_limit=False,
        shuffle_input_sentence=True,
    )
    os.unlink(text_path)

    # vocab.txt in the convention of NeMo's process_asr_text_tokenizer.py
    with open(prefix + ".vocab", "r", encoding="utf-8") as f, \
            open(os.path.join(args.out_dir, "vocab.txt"), "w", encoding="utf-8") as out:
        for i, line in enumerate(f):
            piece = line.split("\t")[0]
            if i == 0 and piece == "<unk>":
                continue
            if piece.startswith("\u2581"):
                out.write((piece[1:] if len(piece) > 1 else piece) + "\n")
            else:
                out.write("##" + piece + "\n")
    print(f"Tokenizer written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
