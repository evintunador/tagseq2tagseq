"""Tier 2 — model-based end-to-end audit. Needs a trained cross_doc_link ckpt.

Three measurements on the SAME examples:
  * cross-doc NLL: score_completion_with_context_docs with the port's real
    aux docs (precise matching) — identical to the production eval path.
  * flat NLL: same completion tokens, no aux (paired baseline).
  * placebo NLL: aux docs REPLACED by aux from a different example (derangement
    over fired examples), with the ORIGINAL example's raw_identifiers so the
    grants still fire — the model attends to wrong-but-plausible code.

Gates:
  * n_cross_doc ≥ MIN_N and fire-rate ≥ MIN_FIRE_RATE (python/java ≈ 0.9)
  * Δnll_real = flat − cross_doc > 0 with bootstrap 95% CI excluding 0
  * placebo separation: Δnll_real − Δnll_placebo > 0 with CI excluding 0 —
    proves the benchmark rewards the RIGHT cross-file context, not just any
    extra in-language tokens. Placebo-condition NLLs are paired per example.

The placebo swap reuses each fired example's own identifiers on swapped
CONTENT, so fire-rate is preserved by construction and the two conditions
differ only in what the granted attention actually sees.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .schema import CrossDocExample, PortAdapter, encode_example

logger = logging.getLogger(__name__)

MIN_N = 200
MIN_FIRE_RATE = 0.5
BOOTSTRAP_RESAMPLES = 10_000
SEED = 42


def _bootstrap_ci(deltas: List[float], seed: int = SEED,
                  resamples: int = BOOTSTRAP_RESAMPLES) -> Tuple[float, float]:
    """95% percentile bootstrap CI of the mean of `deltas`."""
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas)
    means = rng.choice(arr, size=(resamples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _model_max_seq_len(model, default: int = 32768) -> int:
    """Rotary cos-table length = the model's hard positional cap. A pack longer
    than this trips the RoPE assertion in flex_self_attention, so oversized
    packs must be skipped, not scored (whole-file aux, e.g. Kotlin/ASE, can
    exceed it where RepoBench's small snippets never do)."""
    for name, buf in model.backbone.named_buffers():
        if name.endswith("rotary.cos"):
            return int(buf.size(0))
    return default


@dataclass
class Tier2Report:
    port: str
    checkpoint: str
    n_examples: int
    n_fired: int = 0
    n_oversized_skipped: int = 0
    mean_nll_cross: float = float("nan")
    mean_nll_flat: float = float("nan")
    mean_nll_placebo: float = float("nan")
    delta_real: float = float("nan")            # flat − cross (positive = benchmark works)
    delta_real_ci: Tuple[float, float] = (float("nan"), float("nan"))
    delta_placebo: float = float("nan")         # flat − placebo
    placebo_separation: float = float("nan")    # delta_real − delta_placebo
    placebo_separation_ci: Tuple[float, float] = (float("nan"), float("nan"))
    failures: List[str] = field(default_factory=list)

    @property
    def fire_rate(self) -> float:
        return self.n_fired / self.n_examples if self.n_examples else 0.0

    @property
    def passed(self) -> bool:
        return not self.failures


def run_tier2(
    port: PortAdapter,
    model,
    max_examples: Optional[int] = None,
    device: str = "cuda",
    seed: int = SEED,
) -> Tier2Report:
    from eval.scoring import (
        score_completion_with_context_docs,
        score_completions_independent_batched,
    )
    from eval.nlp_benchmarks import _make_encoder

    enc = _make_encoder(model.tokenizer)
    decode_fn = model.tokenizer.decode
    detector = port.detector_factory(decode_fn)

    examples = port.load(max_examples)
    rep = Tier2Report(port=port.name,
                      checkpoint=getattr(model, "checkpoint_path", "?"),
                      n_examples=len(examples))

    packed = [encode_example(ex, enc, port.identifier_fn) for ex in examples]

    # A pack = all aux tokens + context + completion. Skip packs over the RoPE
    # cap (they would abort the whole run on the flex_self_attention assertion).
    max_len = _model_max_seq_len(model)

    def _pack_len(p) -> int:
        return (sum(len(t) for t in p["aux_token_lists"])
                + len(p["context_tokens"]) + len(p["completion_tokens"]))

    # ── real cross-doc pass ──────────────────────────────────────────────
    cross_nlls: List[Optional[float]] = []
    for ex, p in zip(examples, packed):
        if _pack_len(p) > max_len:
            rep.n_oversized_skipped += 1
            cross_nlls.append(None)
            continue
        nll = score_completion_with_context_docs(
            model,
            aux_token_lists=p["aux_token_lists"],
            context_tokens=p["context_tokens"],
            completion_tokens=p["completion_tokens"],
            link_detector=detector,
            aux_raw_identifiers=p["aux_raw_identifiers"],
            source_file_path=p["source_file_path"],
            device=device,
        )
        cross_nlls.append(nll)

    fired = [i for i, nll in enumerate(cross_nlls) if nll is not None]
    rep.n_fired = len(fired)

    # ── paired flat pass (fired examples only) ───────────────────────────
    flat_pairs = [(packed[i]["context_tokens"], packed[i]["completion_tokens"])
                  for i in fired]
    flat_nlls = score_completions_independent_batched(model, flat_pairs, device=device)

    # ── placebo pass: swap aux CONTENT between fired examples ────────────
    # Derangement of fired indices; identifiers stay with the ORIGINAL example
    # so grants keep firing on identifier match, but attention sees another
    # example's code. Length mismatch between identifier list and swapped
    # content list is reconciled by cycling.
    rng = random.Random(seed)
    perm = fired[:]
    while True:
        rng.shuffle(perm)
        if len(fired) < 2 or all(a != b for a, b in zip(fired, perm)):
            break
    placebo_nlls: List[Optional[float]] = []
    for i, j in zip(fired, perm):
        own, donor = packed[i], packed[j]
        n_ids = len(own["aux_raw_identifiers"])
        donor_tok = donor["aux_token_lists"]
        swapped = [donor_tok[k % len(donor_tok)] for k in range(n_ids)]
        # Swapped-in aux may be larger than the original; guard the RoPE cap.
        if (sum(len(t) for t in swapped) + len(own["context_tokens"])
                + len(own["completion_tokens"])) > max_len:
            placebo_nlls.append(None)
            continue
        nll = score_completion_with_context_docs(
            model,
            aux_token_lists=swapped,
            context_tokens=own["context_tokens"],
            completion_tokens=own["completion_tokens"],
            link_detector=detector,
            aux_raw_identifiers=own["aux_raw_identifiers"],
            source_file_path=own["source_file_path"],
            device=device,
        )
        placebo_nlls.append(nll)

    # ── metrics on the triple-paired subset ──────────────────────────────
    triples = [(cross_nlls[i], f, pl)
               for i, f, pl in zip(fired, flat_nlls, placebo_nlls)
               if pl is not None]
    if triples:
        cross = [t[0] for t in triples]
        flat = [t[1] for t in triples]
        placebo = [t[2] for t in triples]
        rep.mean_nll_cross = sum(cross) / len(cross)
        rep.mean_nll_flat = sum(flat) / len(flat)
        rep.mean_nll_placebo = sum(placebo) / len(placebo)
        d_real = [f - c for c, f, _ in triples]
        d_sep = [p - c for c, _, p in triples]   # placebo − cross, per example
        rep.delta_real = sum(d_real) / len(d_real)
        rep.delta_real_ci = _bootstrap_ci(d_real, seed=seed)
        rep.delta_placebo = rep.mean_nll_flat - rep.mean_nll_placebo
        rep.placebo_separation = sum(d_sep) / len(d_sep)
        rep.placebo_separation_ci = _bootstrap_ci(d_sep, seed=seed)

    n_scored = len(triples)
    if n_scored < MIN_N:
        rep.failures.append(f"n_cross_doc {n_scored} < {MIN_N} (insufficient power)")
    if rep.fire_rate < MIN_FIRE_RATE:
        rep.failures.append(f"fire-rate {rep.fire_rate:.3f} < {MIN_FIRE_RATE}")
    if not (rep.delta_real_ci[0] > 0):
        rep.failures.append(
            f"Δnll_real {rep.delta_real:.4f} CI {rep.delta_real_ci} does not exclude 0")
    if not (rep.placebo_separation_ci[0] > 0):
        rep.failures.append(
            f"placebo separation {rep.placebo_separation:.4f} CI "
            f"{rep.placebo_separation_ci} does not exclude 0 — benchmark may "
            f"reward ANY extra context, not the imported code")

    return rep
