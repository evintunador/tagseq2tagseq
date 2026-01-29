import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import tiktoken

from data.dataset import GraphIndex, PretokShardedBackend
from data.layout import BOSEOSLayoutPolicy, NullLayoutPolicy
from data.packed_dataset import PackedSequenceDataset
from data.pack_sampler import PackBatchSampler
from data.traversal import (
    BFSStrategy,
    DFSStrategy,
    RandomSelectionStrategy,
    RandomWalkStrategy,
)


logger = logging.getLogger(__name__)


def main() -> None:
    """
    Demonstrate an end-to-end pipeline:

        GraphIndex -> PretokShardedBackend -> PackBatchSampler
        -> PackedSequenceDataset -> DataLoader -> single packed batch.
    """
    parser = argparse.ArgumentParser(
        description="Inspect a packed batch from the pre-tokenized DAGWiki dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to the pre-tokenized dataset run directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the pack sampler.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=2048,
        help="Global token budget per packed batch (including prefix/body/suffix).",
    )
    parser.add_argument(
        "--doc-budget",
        type=int,
        default=None,
        help="Optional per-document body token budget. Use None for no limit.",
    )
    parser.add_argument(
        "--use-bos-eos",
        action="store_true",
        help="If set, wrap each document with BOS/EOS tokens via BOSEOSLayoutPolicy.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=("random", "random_walk", "bfs", "dfs"),
        default="random",
        help=(
            "Traversal strategy to use when walking the document graph: "
            "'random' (uniform over docs), 'random_walk' (Markovian walk over edges), "
            "'bfs' (breadth-first), or 'dfs' (depth-first)."
        ),
    )
    parser.add_argument(
        "--order-mode",
        type=str,
        choices=("as_traversed", "prefer_targets_first"),
        default="prefer_targets_first",
        help=(
            "How to order documents within a pack after sampling: "
            "'as_traversed' keeps traversal order, 'prefer_targets_first' "
            "reorders so link targets tend to appear before linkers."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    if not args.dataset_dir.is_dir():
        logger.error("Dataset directory not found: %s", args.dataset_dir)
        return

    backend: PretokShardedBackend | None = None
    try:
        logger.info("Initializing GraphIndex from %s", args.dataset_dir)
        graph_index = GraphIndex(args.dataset_dir)
        logger.info("Loaded graph with %d nodes.", len(graph_index))

        logger.info("Initializing PretokShardedBackend...")
        backend = PretokShardedBackend(graph_index)

        # Choose a simple layout policy. For the demo we support either:
        #   - NullLayoutPolicy(): no decoration
        #   - BOSEOSLayoutPolicy(): add BOS/EOS around each document
        if args.use_bos_eos:
            # These token ids are experiment-specific; for a demo we use
            # placeholder values that can be wired up to a real tokenizer later.
            bos_id = 1
            eos_id = 2
            layout_policy = BOSEOSLayoutPolicy(bos_token_id=bos_id, eos_token_id=eos_id)
            logger.info(
                "Using BOSEOSLayoutPolicy with BOS id=%d and EOS id=%d.", bos_id, eos_id
            )
        else:
            layout_policy = NullLayoutPolicy()
            logger.info("Using NullLayoutPolicy (no per-doc decoration).")

        # Choose traversal strategy factory based on CLI flag.
        if args.strategy == "random":
            strategy_factory = lambda: RandomSelectionStrategy()
        elif args.strategy == "random_walk":
            # Simple defaults: outgoing edges, small restart probability.
            strategy_factory = lambda: RandomWalkStrategy(
                edge_mode="outgoing",
                restart_prob=0.05,
            )
        elif args.strategy == "bfs":
            strategy_factory = lambda: BFSStrategy(edge_mode="outgoing")
        elif args.strategy == "dfs":
            strategy_factory = lambda: DFSStrategy(edge_mode="outgoing")
        else:  # pragma: no cover - argparse choices should prevent this
            raise ValueError(f"Unknown strategy: {args.strategy!r}")

        pack_sampler = PackBatchSampler(
            graph=graph_index,
            strategy_factory=strategy_factory,
            token_budget=args.token_budget,
            doc_budget=args.doc_budget,
            overflow_policy="truncate",
            doc_level_trim_side="tail",
            pack_level_trim_side="head",
            max_candidates_per_component=1000,
            seed=args.seed,
            order_mode=args.order_mode,
            layout_policy=layout_policy,
        )

        dataset = PackedSequenceDataset(
            graph=graph_index,
            backend=backend,
            pack_sampler=pack_sampler,
            layout_policy=layout_policy,
            as_2d=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=None,  # Each item is already a full batch.
            num_workers=0,
        )

        logger.info("Fetching a single packed batch...")
        try:
            batch = next(iter(loader))
        except StopIteration:
            logger.warning("Sampler produced no batches under the current settings.")
            return

        tokens: torch.Tensor = batch["tokens"]
        doc_spans = batch["doc_spans"]

        print("\n--- Packed Batch Summary ---")
        print(f"tokens shape: {tuple(tokens.shape)}")
        flat = tokens.view(-1)
        print(f"Total tokens T: {flat.shape[0]}")
        print(f"Number of docs in batch: {len(doc_spans)}")

        print("\nDoc spans:")
        for span in doc_spans:
            length = span.end - span.start
            print(
                f"  doc_id={span.doc_id}, title={span.title!r}, "
                f"start={span.start}, end={span.end}, "
                f"length={length}, truncated={span.truncated}"
            )

        # Show connectivity information among the documents in this batch.
        doc_id_to_index = {span.doc_id: idx for idx, span in enumerate(doc_spans)}

        print("\nGraph connectivity within this batch:")
        for idx, span in enumerate(doc_spans):
            doc_id = span.doc_id
            out_neighbors = graph_index.neighbors_out(doc_id)
            in_neighbors = graph_index.neighbors_in(doc_id)

            out_in_batch = [
                (nbr, doc_id_to_index[nbr])
                for nbr in out_neighbors
                if nbr in doc_id_to_index
            ]
            in_in_batch = [
                (nbr, doc_id_to_index[nbr])
                for nbr in in_neighbors
                if nbr in doc_id_to_index
            ]

            print(f"  [{idx}] {span.title!r}")
            if out_in_batch:
                print(
                    "    outgoing ->",
                    ", ".join(
                        f"[{local_idx}] doc_id={nbr}"
                        for nbr, local_idx in out_in_batch
                    ),
                )
            else:
                print("    outgoing -> (none in-batch)")

            if in_in_batch:
                print(
                    "    incoming <-",
                    ", ".join(
                        f"[{local_idx}] doc_id={nbr}"
                        for nbr, local_idx in in_in_batch
                    ),
                )
            else:
                print("    incoming <- (none in-batch)")

        # Attempt to decode token ids back into text so that we can inspect
        # what the documents actually say and why they might be connected.
        tokenizer_name = getattr(graph_index, "metadata", {}).get("tokenizer")
        encoding = None
        if tokenizer_name is not None:
            try:
                encoding = tiktoken.get_encoding(tokenizer_name)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Unable to load tiktoken encoding %r; skipping text decoding.",
                    tokenizer_name,
                )

        if encoding is not None:
            print("\nDecoded text snippets (prefix + body + suffix) per doc:")
            for idx, span in enumerate(doc_spans):
                token_slice = flat[span.start : span.end].tolist()
                try:
                    text = encoding.decode(token_slice)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to decode tokens for doc_id=%s: %s", span.doc_id, exc
                    )
                    continue

                # Keep snippets reasonably small for terminal output.
                snippet = text.replace("\n", " ")[:400]
                print(f"  [{idx}] {span.title!r}:")
                print(f"    {snippet!r}")

        print("\nFirst 20 token ids:", flat[:20].tolist())
        print("\n----------------------------\n")

    except FileNotFoundError as exc:
        logger.error("A required dataset file was not found: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("An unexpected error occurred: %s", exc, exc_info=True)
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()

