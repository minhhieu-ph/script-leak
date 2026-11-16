"""Small helpers shared by the NeMo scripts. Untested outside a NeMo install."""

from __future__ import annotations

from typing import List


def import_lightning():
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl
    return pl


def load_model(model_path: str = None, pretrained: str = None, device: str = "cuda"):
    import nemo.collections.asr as nemo_asr

    if model_path:
        model = nemo_asr.models.ASRModel.restore_from(model_path, map_location=device)
    elif pretrained:
        model = nemo_asr.models.ASRModel.from_pretrained(pretrained, map_location=device)
    else:
        raise ValueError("give a model path or a pretrained name")
    return model.eval().to(device)


def ctc_head(model):
    """The module that maps encoder output to CTC log-probabilities.
    Hybrid transducer-CTC models expose it as `ctc_decoder`, pure CTC models
    as `decoder`."""
    if hasattr(model, "ctc_decoder"):
        return model.ctc_decoder
    name = type(model).__name__
    if "CTC" in name and hasattr(model, "decoder"):
        return model.decoder
    raise AttributeError(f"{name} has no CTC head (need a hybrid transducer-CTC or a CTC model)")


def get_vocabulary(model) -> List[str]:
    """Token strings in output order of the CTC head (blank excluded)."""
    head = ctc_head(model)
    vocab = getattr(head, "vocabulary", None)
    if vocab:
        return list(vocab)
    tok = getattr(model, "tokenizer", None)
    if tok is not None and getattr(tok, "vocab", None):
        return list(tok.vocab)
    raise AttributeError("could not find the model vocabulary")
