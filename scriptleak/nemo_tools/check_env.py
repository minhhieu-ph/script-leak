"""Check that the installed NeMo exposes what the glue code relies on.

  python -m scriptleak.nemo_tools.check_env [--model MODEL.nemo | --pretrained NAME]

Prints one line per check. Send the output back if anything says FAIL.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import List, Optional


def _try(label: str, fn) -> bool:
    try:
        detail = fn()
        print(f"OK    {label}" + (f": {detail}" if detail else ""))
        return True
    except Exception as e:  # noqa: BLE001 - this is a diagnostic tool
        print(f"FAIL  {label}: {type(e).__name__}: {e}")
        return False


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="path to a .nemo checkpoint to inspect")
    p.add_argument("--pretrained", help="name of a pretrained model to inspect")
    args = p.parse_args(argv)

    ok = True
    ok &= _try("python", lambda: sys.version.split()[0])
    for mod in ("numpy", "soundfile", "sentencepiece", "omegaconf", "torch"):
        ok &= _try(f"import {mod}", lambda m=mod: getattr(importlib.import_module(m), "__version__", ""))
    ok &= _try("torch cuda", lambda: __import__("torch").cuda.is_available()
               and f"{__import__('torch').cuda.device_count()} GPU(s), "
                   f"{__import__('torch').cuda.get_device_name(0)}")
    ok &= _try("import nemo", lambda: importlib.import_module("nemo").__version__)

    def lightning():
        try:
            import lightning.pytorch as pl
            return f"lightning.pytorch {pl.__version__}"
        except ImportError:
            import pytorch_lightning as pl
            return f"pytorch_lightning {pl.__version__}"

    ok &= _try("lightning", lightning)

    def asr_classes():
        import nemo.collections.asr as nemo_asr
        names = ["ASRModel", "EncDecHybridRNNTCTCBPEModel", "EncDecCTCModelBPE", "EncDecRNNTBPEModel"]
        missing = [n for n in names if not hasattr(nemo_asr.models, n)]
        if missing:
            raise AttributeError(f"missing {missing}")
        return ", ".join(names)

    ok &= _try("nemo.collections.asr.models", asr_classes)

    def agg_tokenizer():
        from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer  # noqa: F401
        return "AggregateTokenizer"

    ok &= _try("aggregate tokenizer", agg_tokenizer)

    if args.model or args.pretrained:
        def inspect_model():
            import nemo.collections.asr as nemo_asr
            if args.model:
                model = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cpu")
            else:
                model = nemo_asr.models.ASRModel.from_pretrained(args.pretrained, map_location="cpu")
            from .common import ctc_head, get_vocabulary
            head = ctc_head(model)
            vocab = get_vocabulary(model)
            latin = sum(1 for t in vocab if any("a" <= c.lower() <= "z" for c in t))
            thai = sum(1 for t in vocab if any("ก" <= c <= "๎" for c in t))
            return (f"{type(model).__name__}; CTC head {type(head).__name__}; vocab {len(vocab)} "
                    f"({latin} tokens with Latin letters, {thai} with Thai letters); "
                    f"sample rate {model.cfg.preprocessor.sample_rate}")

        ok &= _try("model", inspect_model)

    print("\nAll checks passed." if ok else "\nSome checks failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
