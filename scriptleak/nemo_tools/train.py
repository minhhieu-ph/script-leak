"""Train a bilingual FastConformer hybrid (transducer + CTC) model.

The architecture comes from one of NeMo's own config files, so nothing about
the network is re-specified here. This script only overrides data, tokenizer,
initialisation and trainer settings.

  python -m scriptleak.nemo_tools.train \
      --base-config $NEMO/examples/asr/conf/fastconformer/hybrid_cache_aware_streaming/fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml \
      --train-manifest train_th150_en05.jsonl \
      --val-manifest th_dev.jsonl en_dev.jsonl \
      --tokenizer th=tokenizers/th_bpe1024 en=tokenizers/en_bpe256 \
      --init-pretrained stt_en_fastconformer_hybrid_large_streaming_multi \
      --exp-dir exp/th150_en05 --max-epochs 30 --batch-size 64 --devices 1

One tokenizer per language builds NeMo's aggregate tokenizer: each language
gets its own block of output ids. A single --tokenizer shared=DIR gives a
plain shared tokenizer instead. Every manifest line must carry a "lang" field
(scriptleak.audit adds it).

Untested outside a NeMo install: run scriptleak.nemo_tools.check_env first.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-config", required=True, help="a NeMo hybrid transducer-CTC BPE yaml")
    p.add_argument("--train-manifest", nargs="+", required=True)
    p.add_argument("--val-manifest", nargs="+", required=True)
    p.add_argument("--tokenizer", nargs="+", required=True, metavar="LANG=DIR")
    p.add_argument("--tokenizer-type", default="bpe")
    p.add_argument("--init-pretrained", help="pretrained NeMo model name to initialise from")
    p.add_argument("--init-nemo", help=".nemo file to initialise from")
    p.add_argument("--init-include", nargs="*", default=["encoder"],
                   help="which sub-modules to copy from the initial model")
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--max-duration", type=float, default=20.0)
    p.add_argument("--lr", type=float)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="extra OmegaConf overrides, e.g. model.aux_ctc.ctc_loss_weight=0.3")
    args = p.parse_args(argv)

    from omegaconf import OmegaConf, open_dict

    import nemo.collections.asr as nemo_asr
    from nemo.utils.exp_manager import exp_manager

    from .common import import_lightning

    pl = import_lightning()
    pl.seed_everything(args.seed, workers=True)

    cfg = OmegaConf.load(args.base_config)
    tok = dict(item.split("=", 1) for item in args.tokenizer)
    with open_dict(cfg):
        cfg.model.train_ds.manifest_filepath = ",".join(args.train_manifest)
        cfg.model.train_ds.batch_size = args.batch_size
        cfg.model.train_ds.num_workers = args.num_workers
        cfg.model.train_ds.max_duration = args.max_duration
        cfg.model.train_ds.is_tarred = False
        cfg.model.train_ds.tarred_audio_filepaths = None
        cfg.model.validation_ds.manifest_filepath = list(args.val_manifest)
        cfg.model.validation_ds.batch_size = args.val_batch_size
        cfg.model.validation_ds.num_workers = args.num_workers
        if "test_ds" in cfg.model:
            cfg.model.test_ds.manifest_filepath = None

        if len(tok) == 1 and "shared" in tok:
            cfg.model.tokenizer = {"dir": tok["shared"], "type": args.tokenizer_type}
        else:
            cfg.model.tokenizer = {
                "type": "agg",
                "langs": {lang: {"dir": d, "type": args.tokenizer_type} for lang, d in tok.items()},
            }

        if args.lr is not None:
            cfg.model.optim.lr = args.lr

        cfg.trainer.max_epochs = args.max_epochs
        cfg.trainer.max_steps = -1
        cfg.trainer.devices = args.devices
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.strategy = "ddp" if args.devices > 1 else "auto"
        cfg.trainer.precision = args.precision

        cfg.exp_manager.exp_dir = args.exp_dir
        cfg.exp_manager.name = os.path.basename(os.path.normpath(args.exp_dir))
        cfg.exp_manager.create_wandb_logger = False

        if args.init_pretrained:
            cfg.init_from_pretrained_model = {
                "model0": {"name": args.init_pretrained, "include": list(args.init_include)}
            }
        if args.init_nemo:
            cfg.init_from_nemo_model = {
                "model0": {"path": args.init_nemo, "include": list(args.init_include)}
            }

    for item in args.set:
        key, value = item.split("=", 1)
        OmegaConf.update(cfg, key, _parse(value), force_add=True)

    os.makedirs(args.exp_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(args.exp_dir, "resolved_config.yaml"))

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg)
    trainer.fit(model)

    # With several GPUs every process reaches this point; only one may write.
    if trainer.is_global_zero:
        final = os.path.join(args.exp_dir, "final.nemo")
        model.save_to(final)
        print(f"Saved {final}")
    return 0


def _parse(value: str):
    """'0.3' -> 0.3, 'true' -> True, 'null' -> None, else the string."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


if __name__ == "__main__":
    sys.exit(main())
