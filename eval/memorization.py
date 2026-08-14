"""eval/memorization.py — memorization probes for the epochs-to-degradation study.

Question being instrumented: does re-packed cross-doc training tolerate more
data-repetition than the classic "<=4 epochs is ~lossless" rule, because each
fresh-seed epoch re-packs every document with a different set of visible
neighbours (an augmentation the doc_causal baseline never gets)?

Two probes, each contrasting TRAINING documents (the model saw them) against
held-out documents (it did not). Memorization shows up as train << held-out and
a gap that widens with epoch count.

  train_val_gap(...)    Isolated-doc NLL / perplexity on a deterministic sample
                        of train docs vs held-out docs. ``gap_nll = val_nll -
                        train_nll`` grows as the model memorizes its training
                        docs. Thin wrapper over
                        ``eval.perplexity.run_held_out_perplexity`` (isolated-doc
                        scoring — see the note on masks below).

  verbatim_recall(...)  Greedy, free-running continuation from a ``prompt_len``
                        token prompt with NO link retrieval (weights only, a
                        single DocSpan), the Carlini-style extractable-memorization
                        probe. Reports how many continuation tokens are reproduced
                        verbatim before the first divergence, plus a cheap
                        teacher-forced greedy-accuracy add-on. Run on train vs
                        held-out docs.

Scoring is deliberately isolated-doc (``doc_causal``) regardless of the
checkpoint's training mask, so the probe is byte-identical across doc_causal /
cross_doc_link / doc_concat_link / doc_concatenated. It measures DOCUMENT-level
memory in the weights, not use of cross-doc context (that is the job of the
contrastive cross-doc benchmarks, not this file). "Retrieval off" is guaranteed
by construction: verbatim_recall never builds a corpus or runs a link detector,
so no auxiliary document can ever enter the context.

CLI:
    python -m eval.memorization \
        --checkpoint <run>/checkpoints/latest.pt \
        --train-dir  <ds>/splits/train \
        --val-dir    <ds>/splits/val_random \
        --mode both --max-docs 200 --prompt-len 256 --gen-len 128 \
        --out <run>/memorization.json
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from data.collate import DocSpan
from data.dataset import GraphIndex, PretokShardedBackend
from data.layout import DocLayoutInfo, DocLayoutPolicy
from eval.perplexity import run_held_out_perplexity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _select_doc_ids(graph: GraphIndex, split: str, max_docs: int, seed: int) -> List[int]:
    """Deterministic doc-id selection, mirroring run_held_out_perplexity.

    split == "all" → shuffle all ids with a fixed seed and take the first
    max_docs; otherwise take the split's ids in their stored order. Selection is
    deterministic so the same documents are probed at every checkpoint (trend
    tracking). max_docs is applied by the caller after length filtering.
    """
    if split == "all":
        ids = list(range(len(graph)))
        random.Random(seed).shuffle(ids)
        return ids
    ids = graph.get_split_ids(split)
    if not ids:
        logger.warning("No documents for split=%r; probe will be empty.", split)
    return ids


def _max_seq_len(model) -> Optional[int]:
    return getattr(getattr(model, "backbone", None), "max_seq_len", None)


@torch.no_grad()
def _forward_logits(model, seq: List[int], device: str,
                    normed_id: str = "", raw_id: str = "") -> torch.Tensor:
    """Single isolated-doc forward. Returns logits [T, V] (float, on device)."""
    tt = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
    span = DocSpan(
        doc_id=0, normed_identifier=normed_id, start=0, end=len(seq),
        truncated=False, outgoing_identifiers=[], raw_identifier=raw_id,
    )
    # mask_type="doc_causal": a single span is isolated anyway, but pinning it
    # documents intent and matches eval/scoring.score_doc.
    logits = model.forward_inference(tt, [span], mask_type="doc_causal")  # [1, T, V]
    return logits[0].float()


# ─────────────────────────────────────────────────────────────────────────────
# Probe 1 — train/held-out perplexity gap
# ─────────────────────────────────────────────────────────────────────────────

def train_val_gap(
    model,
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    layout_policy: Optional[DocLayoutPolicy] = None,
    max_docs: int = 500,
    device: str = "cuda",
    train_split: str = "all",
    val_split: str = "all",
) -> Dict[str, Any]:
    """Isolated-doc NLL on train docs vs held-out docs; the gap is the signal.

    gap_nll = val_nll - train_nll (>= 0 once the model fits its training docs;
    grows with memorization). gap_ppl_ratio = val_ppl / train_ppl (>= 1).
    Both sides use the identical isolated-doc scoring path, so the difference
    reflects only train-vs-heldout familiarity, not scoring asymmetry.
    """
    train_res = run_held_out_perplexity(
        model, train_dir, layout_policy=layout_policy,
        split=train_split, max_docs=max_docs, device=device,
    )
    val_res = run_held_out_perplexity(
        model, val_dir, layout_policy=layout_policy,
        split=val_split, max_docs=max_docs, device=device,
    )
    tr_nll, va_nll = train_res["mean_nll"], val_res["mean_nll"]
    tr_ppl, va_ppl = train_res["perplexity"], val_res["perplexity"]
    gap_nll = va_nll - tr_nll
    gap_ppl_ratio = (va_ppl / tr_ppl) if (tr_ppl and math.isfinite(tr_ppl) and tr_ppl > 0) else float("nan")
    logger.info(
        "train/val gap: train_nll=%.4f val_nll=%.4f  gap_nll=%.4f  ppl %.3f→%.3f (ratio %.3f)",
        tr_nll, va_nll, gap_nll, tr_ppl, va_ppl, gap_ppl_ratio,
    )
    return {
        "train": train_res,
        "val": val_res,
        "gap_nll": gap_nll,
        "gap_ppl_ratio": gap_ppl_ratio,
        "max_docs": max_docs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Probe 2 — verbatim (extractable) recall
# ─────────────────────────────────────────────────────────────────────────────

def _exact_prefix_len(gen: List[int], target: List[int]) -> int:
    """Number of leading tokens of ``gen`` that exactly match ``target``."""
    n = 0
    for a, b in zip(gen, target):
        if a == b:
            n += 1
        else:
            break
    return n


@torch.no_grad()
def _greedy_continuation(model, prefix: List[int], prompt: List[int],
                         gen_len: int, device: str,
                         normed_id: str = "", raw_id: str = "") -> List[int]:
    """Free-running greedy decode of ``gen_len`` tokens after ``prefix+prompt``.

    No link detection or corpus retrieval is ever invoked — the continuation
    comes purely from the weights. Re-runs a full forward each step (no KV
    cache); fine for a bounded probe. Stops early if the sequence would exceed
    the model's max_seq_len.
    """
    seq = list(prefix) + list(prompt)
    msl = _max_seq_len(model)
    gen: List[int] = []
    for _ in range(gen_len):
        if isinstance(msl, int) and len(seq) >= msl:
            break
        logits = _forward_logits(model, seq, device, normed_id, raw_id)  # [T, V]
        nxt = int(logits[-1].argmax().item())
        gen.append(nxt)
        seq.append(nxt)
    return gen


@torch.no_grad()
def _teacher_forced_greedy_acc(model, prefix: List[int], body: List[int],
                               prompt_len: int, gen_len: int, device: str,
                               normed_id: str = "", raw_id: str = "") -> float:
    """Greedy next-token accuracy over the target window under teacher forcing.

    One forward over ``prefix + body[:prompt_len+gen_len]``; for each target
    position we check whether the argmax of the preceding logit equals the true
    token. Unlike the free-running probe this always conditions on ground truth,
    so it never compounds errors — a complementary, cheaper memorization signal.
    """
    window = body[:prompt_len + gen_len]
    seq = list(prefix) + list(window)
    logits = _forward_logits(model, seq, device, normed_id, raw_id)  # [T, V]
    preds = logits.argmax(dim=-1)  # [T]
    plen = len(prefix)
    correct = 0
    total = 0
    # target token at full-seq position (plen + prompt_len + i) is predicted by
    # the logit at (plen + prompt_len + i - 1).
    for i in range(gen_len):
        tgt_pos = plen + prompt_len + i
        if tgt_pos >= len(seq):
            break
        if int(preds[tgt_pos - 1].item()) == seq[tgt_pos]:
            correct += 1
        total += 1
    return (correct / total) if total else float("nan")


@torch.no_grad()
def verbatim_recall(
    model,
    dataset_dir: Union[str, Path],
    split: str = "all",
    max_docs: int = 200,
    prompt_len: int = 256,
    gen_len: int = 128,
    layout_policy: Optional[DocLayoutPolicy] = None,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    """Greedy verbatim-continuation probe over a deterministic sample of docs.

    For each eligible doc (body length >= prompt_len + gen_len + 1): prompt the
    model with the layout prefix + the first ``prompt_len`` body tokens, greedily
    generate ``gen_len`` tokens with retrieval off, and measure how much of the
    true continuation is reproduced.

    Returns per-split aggregates: mean verbatim run length before divergence,
    mean fraction reproduced, fraction of docs reproduced in full, and mean
    teacher-forced greedy accuracy.
    """
    dataset_dir = Path(dataset_dir)
    graph = GraphIndex(dataset_dir)
    backend = PretokShardedBackend(graph)
    if layout_policy is None:
        layout_policy = model.active_layout_policy

    try:
        candidate_ids = _select_doc_ids(graph, split, max_docs, seed)
        min_len = prompt_len + gen_len + 1

        exact_lens: List[int] = []
        fracs: List[float] = []
        tf_accs: List[float] = []
        full = 0
        scored = 0
        skipped_short = 0

        for doc_id in candidate_ids:
            if scored >= max_docs:
                break
            arr = backend.get_tokens_by_id(doc_id)
            if arr is None or len(arr) < min_len:
                skipped_short += 1
                continue
            body = arr.tolist()
            normed = graph.get_normed_identifier(doc_id)
            raw = graph.get_raw_identifier(normed) or normed
            info = DocLayoutInfo(raw_identifier=raw, normed_identifier=normed, body_tokens=body)
            prefix = layout_policy.prefix_tokens(info)

            # Guard against prefix + prompt + gen overrunning max_seq_len.
            msl = _max_seq_len(model)
            if isinstance(msl, int) and len(prefix) + prompt_len + gen_len > msl:
                skipped_short += 1
                continue

            prompt = body[:prompt_len]
            target = body[prompt_len:prompt_len + gen_len]

            gen = _greedy_continuation(model, prefix, prompt, gen_len, device, normed, raw)
            epl = _exact_prefix_len(gen, target)
            tf = _teacher_forced_greedy_acc(model, prefix, body, prompt_len, gen_len, device, normed, raw)

            exact_lens.append(epl)
            fracs.append(epl / gen_len)
            tf_accs.append(tf)
            if epl >= gen_len:
                full += 1
            scored += 1

        if scored == 0:
            logger.warning("verbatim_recall: no eligible docs (split=%r, need len>=%d).",
                           split, min_len)
            return {
                "split": split, "num_docs": 0, "prompt_len": prompt_len, "gen_len": gen_len,
                "mean_exact_prefix_len": float("nan"), "mean_frac_reproduced": float("nan"),
                "frac_fully_reproduced": float("nan"), "mean_tf_greedy_acc": float("nan"),
                "skipped_short": skipped_short,
            }

        result = {
            "split": split,
            "num_docs": scored,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "mean_exact_prefix_len": float(np.mean(exact_lens)),
            "median_exact_prefix_len": float(np.median(exact_lens)),
            "mean_frac_reproduced": float(np.mean(fracs)),
            "frac_fully_reproduced": full / scored,
            "mean_tf_greedy_acc": float(np.nanmean(tf_accs)),
            "skipped_short": skipped_short,
        }
        logger.info(
            "verbatim_recall (%s, n=%d): mean_run=%.1f/%d  frac=%.3f  full=%.3f  tf_acc=%.3f",
            split, scored, result["mean_exact_prefix_len"], gen_len,
            result["mean_frac_reproduced"], result["frac_fully_reproduced"],
            result["mean_tf_greedy_acc"],
        )
        return result
    finally:
        backend.close()


def verbatim_recall_gap(
    model,
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    max_docs: int = 200,
    prompt_len: int = 256,
    gen_len: int = 128,
    layout_policy: Optional[DocLayoutPolicy] = None,
    device: str = "cuda",
    train_split: str = "all",
    val_split: str = "all",
    seed: int = 42,
) -> Dict[str, Any]:
    """verbatim_recall on train vs held-out, plus the train-minus-val deltas."""
    tr = verbatim_recall(model, train_dir, split=train_split, max_docs=max_docs,
                         prompt_len=prompt_len, gen_len=gen_len,
                         layout_policy=layout_policy, device=device, seed=seed)
    va = verbatim_recall(model, val_dir, split=val_split, max_docs=max_docs,
                        prompt_len=prompt_len, gen_len=gen_len,
                        layout_policy=layout_policy, device=device, seed=seed)
    return {
        "train": tr,
        "val": va,
        "delta_frac_reproduced": tr["mean_frac_reproduced"] - va["mean_frac_reproduced"],
        "delta_exact_prefix_len": tr["mean_exact_prefix_len"] - va["mean_exact_prefix_len"],
        "delta_tf_greedy_acc": tr["mean_tf_greedy_acc"] - va["mean_tf_greedy_acc"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(checkpoint: str, device: str, max_seq_len_override: Optional[int]):
    # Imported lazily so `import eval.memorization` does not pull in generate.py's
    # heavy deps unless the CLI is actually used.
    from generate import load_inference_model
    model, hp = load_inference_model(
        checkpoint_path=checkpoint,
        device=device,
        max_seq_len_override=max_seq_len_override,
    )
    return model, hp


def main():
    parser = argparse.ArgumentParser(
        description="Memorization probes (train-vs-heldout perplexity gap + verbatim recall).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a *.pt checkpoint.")
    parser.add_argument("--train-dir", required=True, help="Pretokenized TRAIN split dir.")
    parser.add_argument("--val-dir", required=True, help="Pretokenized held-out (val_random) split dir.")
    parser.add_argument("--mode", choices=["gap", "recall", "both"], default="both")
    parser.add_argument("--train-split", default="all")
    parser.add_argument("--val-split", default="all")
    parser.add_argument("--max-docs", type=int, default=200)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seq-len-override", type=int, default=None)
    parser.add_argument("--out", default=None, help="Write results JSON here (also printed).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Triton kernels launch on the *current* CUDA device; if the caller pins an
    # explicit index (e.g. --device cuda:1) set it as current so tensors and
    # kernel launches agree (otherwise: "Pointer ... cannot be accessed from
    # Triton"). Prefer CUDA_VISIBLE_DEVICES=N + --device cuda for isolation.
    if args.device.startswith("cuda") and ":" in args.device:
        torch.cuda.set_device(args.device)

    model, hp = _build_model(args.checkpoint, args.device, args.max_seq_len_override)

    out: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "mask_type": hp.get("model", {}).get("mask_type"),
        "train_dir": str(args.train_dir),
        "val_dir": str(args.val_dir),
    }

    if args.mode in ("gap", "both"):
        out["perplexity_gap"] = train_val_gap(
            model, args.train_dir, args.val_dir, max_docs=args.max_docs,
            device=args.device, train_split=args.train_split, val_split=args.val_split,
        )
    if args.mode in ("recall", "both"):
        out["verbatim_recall"] = verbatim_recall_gap(
            model, args.train_dir, args.val_dir, max_docs=args.max_docs,
            prompt_len=args.prompt_len, gen_len=args.gen_len, device=args.device,
            train_split=args.train_split, val_split=args.val_split, seed=args.seed,
        )

    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
