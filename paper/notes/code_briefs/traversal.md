<!-- PROVENANCE
Written against commit 6134163 (main @ 2026-08-07). This brief describes the code as of
that commit; it is NOT authoritative if the source has changed since. Before trusting it,
check whether the covered sources have drifted:
    git diff --stat 6134163..HEAD -- data/traversal.py data/dataset.py data/pack_sampler.py
Empty output = brief still current. Non-empty = re-verify the changed parts against source.
Covered sources: data/traversal.py data/dataset.py data/pack_sampler.py
-->

# CODE BRIEF: graph traversal (agent aa492d3c)
Files: traversal.py, dataset.py (GraphIndex), pack_sampler.py, cross_doc_mask.py.

## Graph storage (dataset.py)
Dict keyed by normed_identifier from tokenized_graph.jsonl; doc_id = enumeration index (JSONL insertion order → determinism dependency). DIRECTED: neighbors_out reads outgoing, neighbors_in reads incoming (stored redundantly, not derived; no symmetry assumed). Neighbors not in index SILENTLY DROPPED (:179-181) → realized link density < raw graph. No adjacency dedup (multiplicity weights random walk).

## Strategies (traversal.py) — Protocol: reset_for_new_pack + propose_next; strategies DON'T enforce budget/dedup/stop (sampler does). edge_mode default "outgoing".
- Random: ignores structure, uniform randrange over ALL ids each step. "Random" pack = N uniform docs (seed still uniform).
- RandomWalk: edge_mode{in,out,both}, w_in/w_out, restart_prob. Single _current, NO visited (revisits ok). **Restart = teleport to UNIFORM-RANDOM, NOT restart-to-seed** (NOT standard RWR/personalized-PageRank — name precisely). both: pick DIRECTION by weight then uniform within side. First-order Markov, stationary ∝ out-degree, no PageRank normalization. Dead-end→teleport.
- BFS: deque frontier + visited set, FIFO, visited-on-enqueue, neighbors in file order (DETERMINISTIC). Frontier-empty restart draws uniform unvisited (can jump disconnected components).
- DFS: LIFO stack, **rng.shuffle(neighbors) before push** ("canonical DFS feel") → STOCHASTIC (consumes RNG, unlike BFS).
- Composite: NOT wired into main.py/epoch_precompute (experimental/dead).

## Budget/pack assembly (pack_sampler.py)
token_budget=batch_size*seq_len=max_seq_len, counts prefix+body+suffix. Per-doc body cap doc_budget, overflow_policy truncate/skip. **Seed = UNIFORM rejection sampling regardless of strategy** (strategy only governs GROWTH → "BFS packing" = uniform-seeded local BFS neighborhoods — report honestly). Fresh strategy instance per subwalk. Multiple subwalks per pack until budget/no-seed. reset_for_new_pack called per SUBWALK not per pack (contradicts docstring). Overshoot by ≤1 doc, then **exact-length truncation** (Triton seq-len is tl.constexpr → variable length = ~140s re-autotune + DDP desync); trims BODIES in pack_level_trim_side order (head default). order_mode: as_traversed | **prefer_targets_first (DEFAULT)** = per-component Kahn topo (targets before linkers), contiguous component blocks.

## Determinism
Single random.Random(seed) SHARED across seeding + all propose_next → stream-order-fragile (DFS shuffle, RW teleports shift stream). num_workers=0 preserves. rank_seed=base+rank. **Precompute uses restart_prob=0.0 vs live main.py restart_prob=0.05 — REAL INCONSISTENCY.** doc_id order depends on JSONL order.

## Mask interaction (THE key coupling)
Base doc-causal; DAG gate = target must start before link (cross_doc_mask.py:417-423). Causal q>=k + outgoing traversal ⇒ under as_traversed target lands AFTER linker → grant SILENTLY DROPPED. Hence default prefer_targets_first topo-sorts targets ahead. **outgoing-traversal ⇒ targets-first-ordering-required-to-realize-cross-doc-attention = single most important non-obvious design point.** Cycles: Kahn insertion-order fallback → some links violate gate, dropped. component_id = STRUCTURAL undirected union-find, NOT traversal order (frontier restarts pull unrelated docs; without structural components an exhausted-frontier restart merges unrelated repos). Ordinal run-index relabel (non-monotonic doc-ids) — see masks brief.

## Novel/publishable
- Graph-traversal-ordered packing w/ pluggable strategies + coordinated targets-first topo reorder so a CAUSAL cross-doc mask realizes link edges. traversal-order↔DAG-gate↔ordinal-relabel chain.
- Structural union-find components decoupled from traversal (robust to frontier restarts, contiguous for super-doc).
- Exact-length packing from kernel-constexpr constraint.

## Reviewer-attackable
Uniform seed independent of strategy; RW teleport-to-uniform (not RWR); RW restart_prob live0.05 vs precompute0.0; DFS consumes RNG BFS doesn't; grants silently dropped (DAG gate / max_grants / out-of-index neighbors) → realized density < raw, QUANTIFY; pack_level_trim head trims EARLIEST docs = targets under targets-first (targets lose tokens — intended?); determinism stream-fragile; Composite + RW edge_mode=both unwired (dead/experimental?).
Uncertain: runtime doc_concatenated/doc_causal kernel dispatch not traced; whether any config sets as_traversed.

## → LIT REVIEW IMPLICATIONS
- Graph traversal / node sampling: DeepWalk, node2vec (biased walk p/q), PageRank/personalized PageRank/RWR, GraphSAGE neighbor sampling → G6 + a walk-sampling mini-topic.
- Community detection / graph partitioning (Voronoi partition, METIS) for the worker sharding.
