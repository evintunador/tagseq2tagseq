#!/usr/bin/env python
"""Phase-1 eval-time driver for the graph-sparsity scaling law.

Loads ONE trained cross_doc checkpoint and re-evaluates community_pack
perplexity across a grid of link keep-fractions, holding the packing fixed
(mask-time edge dropout — see eval.scoring.subsample_link_to_target and memory
[[graph-sparsity-scaling-law]]). Emits one JSON row per (keep_frac, mode) so a
downstream regression can fit "cross-doc Δ vs kept fraction" per dataset and,
pooled across datasets of differing inherent density, "Δ vs measured density".

The knob is applied ONLY to the cross_doc arm; the doc_causal baseline is
identical across the grid, so keep=0.0 reproduces the doc_causal number exactly
(sanity floor) and keep=1.0 reproduces the standard cross_doc eval.

Because the checkpoint is loaded once, an N-point grid costs ~N community_pack
evals of GPU time with a single (expensive) compile — not N loads.

Usage
-----
    python -m eval.sparsity_sweep \
        --checkpoint  runs/<run>/checkpoints/best_model.pt \
        --dataset     /fss-data/.../pretokenized_datasets/go \
        --split       val_community \
        --max-packs   500 \
        --keep-fracs  0,0.25,0.5,0.75,1.0 \
        --modes       edge,node \
        --seeds       0 \
        --output      runs/<run>/sparsity/go.json

Only meaningful on a cross_doc_link checkpoint (doc_causal has no grants).
"""

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("sparsity_sweep")


def _parse_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _parse_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _parse_strs(s: str):
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt (cross_doc).")
    p.add_argument("--dataset", required=True,
                   help="Pretokenized dataset dir (community_pack reads splits/<split> inside).")
    p.add_argument("--split", default="val_community",
                   help="Community split to score (val_community or test_community).")
    p.add_argument("--max-packs", type=int, default=500)
    p.add_argument("--keep-fracs", default="0,0.25,0.5,0.75,1.0",
                   help="Comma-separated keep fractions in [0,1].")
    p.add_argument("--modes", default="edge",
                   help="Comma-separated subsample modes: edge (density line), node (robustness).")
    p.add_argument("--seeds", default="0",
                   help="Comma-separated experiment seeds (multiple → subsample-noise band).")
    p.add_argument("--output", required=True, help="Output JSON path.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset-tag", default=None,
                   help="Label for this dataset in the output (defaults to dataset dir name).")
    p.add_argument("--inference-attention-backend", default="flex")
    p.add_argument("--cross-mask-type", default=None,
                   help="Force the cross arm's mask type (default: model's trained "
                        "mask). Pass 'cross_doc_link' to evaluate a DOC_CAUSAL-trained "
                        "checkpoint under a real cross-doc mask — the true train-keep=0 "
                        "point. Leave unset for cross_doc-trained checkpoints.")
    p.add_argument("--link-detector", default=None,
                   help="Link detector to build the cross-doc creator with when "
                        "--cross-mask-type promotes a doc_causal ckpt to cross_doc_link "
                        "(the doc_causal config has none). e.g. typescript, python, markdown.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    keep_fracs = _parse_floats(args.keep_fracs)
    modes = _parse_strs(args.modes)
    seeds = _parse_ints(args.seeds)
    dataset_tag = args.dataset_tag or Path(args.dataset).name

    # Load the checkpoint ONCE (single compile amortised across the whole grid).
    from generate import load_inference_model
    from eval.perplexity import run_community_pack_perplexity

    logger.info("Loading checkpoint: %s", args.checkpoint)
    # When forcing the cross arm to cross_doc_link on a doc_causal-trained ckpt,
    # build the model AS cross_doc_link so it registers a cross-doc creator (and
    # still a doc_causal one for the baseline arm). Weights load strictly — mask
    # type does not parameterize the model.
    load_kwargs = {}
    if args.cross_mask_type in ("cross_doc_link", "doc_concat_link"):
        load_kwargs["mask_type_override"] = args.cross_mask_type
        if args.link_detector:
            load_kwargs["link_detector_override"] = args.link_detector
    model, hp = load_inference_model(
        args.checkpoint, device=args.device,
        inference_attention_backend=args.inference_attention_backend,
        **load_kwargs,
    )
    model.eval()

    mask_type = hp.get("model", {}).get("mask_type", "?")
    if mask_type not in ("cross_doc_link", "doc_concat_link") and not args.cross_mask_type:
        logger.warning(
            "Checkpoint mask_type=%r has no cross-doc grants; the keep_frac grid "
            "will be flat. This sweep is intended for a cross_doc_link ckpt.",
            mask_type,
        )

    # node-mode at keep in {0,1} is identical to edge-mode (endpoints), so only
    # sweep node at the interior fractions to avoid redundant evals.
    rows = []
    t_start = time.time()
    for mode in modes:
        for keep in keep_fracs:
            # Endpoints are mode- and seed-independent (0.0→{}, 1.0→identity),
            # so score them once (under the first mode / first seed) and reuse.
            endpoint = (keep <= 0.0 or keep >= 1.0)
            if endpoint and (mode != modes[0]):
                continue
            grid_seeds = [seeds[0]] if endpoint else seeds
            for seed in grid_seeds:
                t0 = time.time()
                logger.info(
                    "→ %s | keep=%.3f mode=%s seed=%d", dataset_tag, keep, mode, seed)
                res = run_community_pack_perplexity(
                    model=model,
                    dataset_dir=args.dataset,
                    split=args.split,
                    max_packs=args.max_packs,
                    device=args.device,
                    keep_frac=keep,
                    keep_seed=seed,
                    keep_mode=mode,
                    cross_mask_type=args.cross_mask_type,
                )
                res.update({
                    "dataset": dataset_tag,
                    "dataset_dir": str(args.dataset),
                    "checkpoint": str(args.checkpoint),
                    "mask_type": mask_type,
                    "keep_frac": keep,
                    "keep_mode": mode,
                    "keep_seed": seed,
                    "elapsed_s": round(time.time() - t0, 1),
                })
                rows.append(res)
                logger.info(
                    "   n=%s delta=%.4f cross=%.4f base=%.4f (%.0fs)",
                    res.get("n_packs"), res.get("mean_delta", float("nan")),
                    res.get("mean_nll_cross_doc", float("nan")),
                    res.get("mean_nll_baseline", float("nan")),
                    res["elapsed_s"],
                )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_tag,
        "checkpoint": str(args.checkpoint),
        "mask_type": mask_type,
        "split": args.split,
        "max_packs": args.max_packs,
        "keep_fracs": keep_fracs,
        "modes": modes,
        "seeds": seeds,
        "total_elapsed_s": round(time.time() - t_start, 1),
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %d rows → %s (%.0fs total)",
                len(rows), out_path, payload["total_elapsed_s"])


if __name__ == "__main__":
    main()
