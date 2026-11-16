"""CTC decoding methods that act on frame-level log-probabilities.

All functions take `logp` of shape [T, V] (log-probabilities, blank included)
and `groups` from tokenmap.token_groups, and return the collapsed token ids.
Everything is causal except `utterance_lock` without a prefix.

Methods
  greedy              the model as it is
  hard_mask           ban the minority language outright (leak = 0, minority lost)
  logit_bias          subtract a constant from minority tokens
  utterance_lock      decide the script once for the utterance, then ban the other
  script_marginalized proposed: pick the script by summed probability mass,
                      then the best token inside that script
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .tokenmap import BLANK, MAJ, MIN

_NEG = -1e30
_LLR_CLIP = 50.0


def logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    if a.shape[axis] == 0:
        shape = list(a.shape)
        del shape[axis]
        return np.full(shape, -np.inf, dtype=np.float64)
    m = a.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def group_logmass(logp: np.ndarray, groups: np.ndarray, group: int) -> np.ndarray:
    """log of the total probability of one group, per frame. Shape [T]."""
    return logsumexp(logp[:, groups == group].astype(np.float64), axis=1)


def collapse(frame_ids: np.ndarray, blank_id: int) -> List[int]:
    """Standard CTC collapse: merge repeats, drop blanks."""
    out: List[int] = []
    prev = -1
    for i in frame_ids.tolist():
        if i != prev and i != blank_id:
            out.append(i)
        prev = i
    return out


def _blank_id(groups: np.ndarray) -> int:
    return int(np.flatnonzero(groups == BLANK)[0])


def _masked_argmax(logp: np.ndarray, banned: np.ndarray) -> np.ndarray:
    """Per-frame argmax with banned[T, V] or banned[V] columns excluded."""
    return np.where(banned, _NEG, logp).argmax(axis=1)


def greedy(logp: np.ndarray, groups: np.ndarray) -> List[int]:
    return collapse(logp.argmax(axis=1), _blank_id(groups))


def hard_mask(logp: np.ndarray, groups: np.ndarray, banned_group: int = MIN) -> List[int]:
    return collapse(_masked_argmax(logp, groups == banned_group), _blank_id(groups))


def logit_bias(logp: np.ndarray, groups: np.ndarray, bias: float, target: int = MIN) -> List[int]:
    shifted = logp - bias * (groups == target)
    return collapse(shifted.argmax(axis=1), _blank_id(groups))


def utterance_lock(
    logp: np.ndarray,
    groups: np.ndarray,
    prefix_frames: Optional[int] = None,
    min_evidence: float = 0.5,
) -> List[int]:
    """Decide the script once, then ban the other.

    prefix_frames=None  uses the whole utterance (not streaming).
    prefix_frames=k     decides from the first k frames; frames before the
                        decision are decoded without a lock, as a streaming
                        system would have to.
    If the script mass seen is below `min_evidence` (in expected frames), no
    lock is applied.
    """
    blank = _blank_id(groups)
    p_maj = np.exp(group_logmass(logp, groups, MAJ))
    p_min = np.exp(group_logmass(logp, groups, MIN))
    k = len(logp) if prefix_frames is None else min(prefix_frames, len(logp))
    s_maj, s_min = p_maj[:k].sum(), p_min[:k].sum()
    if s_maj + s_min < min_evidence:
        return collapse(logp.argmax(axis=1), blank)
    banned_group = MAJ if s_min > s_maj else MIN
    banned = groups == banned_group
    if prefix_frames is None:
        return collapse(_masked_argmax(logp, banned), blank)
    ids = logp.argmax(axis=1)
    ids[k:] = _masked_argmax(logp[k:], banned)
    return collapse(ids, blank)


def script_marginalized(
    logp: np.ndarray,
    groups: np.ndarray,
    tau: float = 0.0,
    rho: float = 0.0,
    return_track: bool = False,
):
    """Script-marginalized decoding (causal, no look-ahead).

    Per frame, sum the probability of all majority tokens and of all minority
    tokens. The log-ratio, weighted by how much non-blank script mass the frame
    carries, feeds an evidence accumulator e_t = rho * e_{t-1} + w_t * llr_t.
    The active script switches to minority when e_t > tau and to majority when
    e_t < -tau (hysteresis in between). The emitted token is the best token
    that is not in the inactive script.

    tau=0, rho=0 is the memoryless form: the script with more total mass wins
    each frame, which undoes "one minority token beats many majority tokens
    that split the mass".
    """
    blank = _blank_id(groups)
    lm_maj = group_logmass(logp, groups, MAJ)
    lm_min = group_logmass(logp, groups, MIN)
    llr = np.clip(np.nan_to_num(lm_min - lm_maj, nan=0.0, posinf=_LLR_CLIP, neginf=-_LLR_CLIP),
                  -_LLR_CLIP, _LLR_CLIP)
    weight = np.exp(lm_maj) + np.exp(lm_min)
    is_maj, is_min = groups == MAJ, groups == MIN

    ids = np.empty(len(logp), dtype=np.int64)
    track = np.empty(len(logp), dtype=np.int8)
    e, cur = 0.0, None
    for t in range(len(logp)):
        e = rho * e + weight[t] * llr[t]
        if e > tau:
            cur = MIN
        elif e < -tau:
            cur = MAJ
        if cur is None:
            ids[t] = logp[t].argmax()
            track[t] = BLANK
        else:
            banned = is_maj if cur == MIN else is_min
            ids[t] = np.where(banned, _NEG, logp[t]).argmax()
            track[t] = cur
    out = collapse(ids, blank)
    return (out, track) if return_track else out


def parse_method(spec: str) -> Tuple[str, dict]:
    """'marg:tau=0.5,rho=0.9' -> ('marg', {'tau': 0.5, 'rho': 0.9})"""
    name, _, rest = spec.partition(":")
    params = {}
    for item in filter(None, rest.split(",")):
        k, _, v = item.partition("=")
        params[k.strip()] = float(v)
    return name.strip(), params


def decode(logp: np.ndarray, groups: np.ndarray, spec: str) -> List[int]:
    """Decode with a method given as a spec string.

    greedy | mask | bias:b=4 | lock | lock:k=25 | marg | marg:tau=0.5,rho=0.9
    """
    name, p = parse_method(spec)
    if name == "greedy":
        return greedy(logp, groups)
    if name == "mask":
        return hard_mask(logp, groups)
    if name == "bias":
        return logit_bias(logp, groups, bias=p["b"])
    if name == "lock":
        k = int(p["k"]) if "k" in p else None
        return utterance_lock(logp, groups, prefix_frames=k)
    if name == "marg":
        return script_marginalized(logp, groups, tau=p.get("tau", 0.0), rho=p.get("rho", 0.0))
    raise ValueError(f"Unknown decoding method '{spec}'")


DEFAULT_SWEEP = (
    ["greedy", "mask", "lock"]
    + [f"lock:k={k}" for k in (6, 12, 25)]
    + [f"bias:b={b}" for b in (0.5, 1, 2, 4, 8)]
    + ["marg"]
    + [f"marg:tau={tau},rho={rho}" for rho in (0.5, 0.9) for tau in (0.25, 0.5, 1, 2)]
)
