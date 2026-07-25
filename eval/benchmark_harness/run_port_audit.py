"""CLI — run the harness tiers on a registered port and print/save the report.

Usage (CPU tiers only):
    python -m eval.benchmark_harness.run_port_audit --port repobench_java \
        --tiers 0 1 --max-examples 500

With Tier 2 (needs a trained cross_doc_link checkpoint whose link detector
matches the port language):
    python -m eval.benchmark_harness.run_port_audit --port repobench_java \
        --tiers 0 1 2 --checkpoint runs/<id>/checkpoints/best_model.pt

Tier C (dedup) runs when --training-graph (tokenized_graph.jsonl for repo
names) and optionally --training-shards (raw shard dir for the hash pass)
are given.

Output: human-readable summary + JSON at --out (default
eval/benchmark_harness/reports/<port>.json).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("benchmark_harness")


def _to_jsonable(obj):
    if dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
        # include gate properties computed on the dataclass
        for prop in ("passed", "precision", "port_fire_rate",
                     "oracle_reachable_rate", "fire_rate"):
            if hasattr(obj, prop):
                d[prop] = getattr(obj, prop)
        return d
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--tiers", nargs="+", default=["0", "1"],
                    choices=["0", "1", "2", "C"])
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--checkpoint", default=None, help="best_model.pt for Tier 2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--training-graph", default=None,
                    help="tokenized_graph.jsonl of the training corpus (Tier C)")
    ap.add_argument("--training-shards", default=None,
                    help="raw shard dir with content for the hash pass (Tier C)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from eval.benchmark_harness.ports import get_port
    port = get_port(args.port)
    results = {"port": args.port, "language": port.language,
               "max_examples": args.max_examples}

    # Tokenizer-only encoder for CPU tiers (no model needed).
    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    enc = lambda text: tok.encode(text, disallowed_special=())
    decode_fn = tok.decode

    if "0" in args.tiers:
        from eval.benchmark_harness.tier0 import run_tier0
        r0 = run_tier0(port, enc, max_examples=args.max_examples)
        results["tier0"] = _to_jsonable(r0)
        logger.info("Tier 0 %s: %s", "PASS" if r0.passed else "FAIL",
                    r0.failures or "all invariants hold")

    if "1" in args.tiers:
        from eval.benchmark_harness.tier1 import run_tier1
        r1 = run_tier1(port, enc, decode_fn, max_examples=args.max_examples)
        results["tier1"] = _to_jsonable(r1)
        logger.info(
            "Tier 1 %s: precision=%.3f fire=%.3f oracle-reachable=%.3f %s",
            "PASS" if r1.passed else "FAIL", r1.precision,
            r1.port_fire_rate, r1.oracle_reachable_rate, r1.failures or "")

    if "C" in args.tiers:
        if not args.training_graph:
            ap.error("Tier C needs --training-graph")
        from eval.benchmark_harness.dedup import (
            run_dedup, load_training_repos_from_graph, iter_training_shards,
            file_hash)
        repos = load_training_repos_from_graph(Path(args.training_graph))
        hashes = None
        if args.training_shards:
            hashes = {file_hash(content) for _, content
                      in iter_training_shards(Path(args.training_shards))}
        examples = port.load(args.max_examples)
        _, rc = run_dedup(args.port, examples, repos, hashes)
        results["tierC"] = _to_jsonable(rc)
        logger.info(
            "Tier C: %d/%d survive (repo-overlap dropped %d, hash dropped %d)",
            rc.n_survivors, rc.n_examples,
            rc.n_repo_overlap_dropped, rc.n_hash_dropped)

    if "2" in args.tiers:
        if not args.checkpoint:
            ap.error("Tier 2 needs --checkpoint")
        from generate import load_inference_model
        from eval.benchmark_harness.tier2 import run_tier2
        model, _hp = load_inference_model(args.checkpoint, device=args.device)
        model.eval()
        r2 = run_tier2(port, model, max_examples=args.max_examples,
                       device=args.device)
        results["tier2"] = _to_jsonable(r2)
        logger.info(
            "Tier 2 %s: Δnll_real=%.4f CI=%s placebo_sep=%.4f CI=%s fire=%.3f %s",
            "PASS" if r2.passed else "FAIL", r2.delta_real, r2.delta_real_ci,
            r2.placebo_separation, r2.placebo_separation_ci, r2.fire_rate,
            r2.failures or "")

    out = Path(args.out) if args.out else (
        Path(__file__).parent / "reports" / f"{args.port}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Report written to %s", out)


if __name__ == "__main__":
    main()
