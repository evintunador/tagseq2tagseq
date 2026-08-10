<!-- PROVENANCE
Written against commit 6134163 (main @ 2026-08-07). This brief describes the code as of
that commit; it is NOT authoritative if the source has changed since. Before trusting it,
check whether the covered sources have drifted:
    git diff --stat 6134163..HEAD -- data/packed_dataset.py data/bucketed_pack_dataset.py data/epoch_precompute.py data/pack_sampler.py data/collate.py data/layout.py precompute_epochs.py
Empty output = brief still current. Non-empty = re-verify the changed parts against source.
Covered sources: data/packed_dataset.py data/bucketed_pack_dataset.py data/epoch_precompute.py data/pack_sampler.py data/collate.py data/layout.py precompute_epochs.py
-->

# CODE BRIEF: packing / epoch precompute / density bucketing (agent a20eee62)

## kv_block_count density metric
Non-empty (q_block,kv_block) pairs in block-sparse mask @ 128-tok blocks = analytic proxy for FlexAttn backward cost.
- Method B (GPU/BlockMask, epoch_precompute.py:491-515): real BlockMask, sum kv_num_blocks+full_kv_num_blocks. ~36ms/pack sequential. --gpu-kv-pass verification only.
- **Method C (CPU analytical, DEFAULT, cross_doc_mask.py:169-230)**: per-worker parallel, ~1ms/pack. (a) intra-doc causal: per DocSpan add lower-tri (q_blk,kv_blk) in block range to a set; (b) cross-doc grant: per link_pos→target, query range link_pos(or span.start if whole_doc_grant=concat_link)→span.end × target block range, add rectangle to set; return len(set). **EXACT vs B when every target ends before link_end_pos (holds for standard packing order)**. 36× speedup.
- Method A brute-force [T,T] for tests only.

## Quantile bucketing (epoch_precompute.py:522-527)
Sort by kv_block_count, bucket_id=int(i/n*n_buckets), EQUAL-COUNT quantile buckets. Default **n_buckets=32**. Merged corpora re-run _assign_buckets over union (merge_packs.py:331 — "bucket B in one source ≠ density of bucket B in another").

## Bucket-shuffle schedule (bucketed_pack_dataset.py:51-64)
rng=Random(epoch_idx); n_repeats=1000× shuffle(buckets)+extend. Each bucket appears exactly n_repeats over the sequence, adjacent steps differ → gradient diversity + balanced visits.

## Per-step density match across DDP ranks (bucketed_pack_dataset.py:211-235) — CENTRAL CONTRIBUTION
bucket B = bucket_seq[global_accum_step % len] (same on every rank, seed=epoch_idx not rank). Rank r draws pack bucket_consumed[B]+r; consumed+=world_size. ALL RANKS SAME DENSITY BUCKET every accum step → removes FlexAttn backward variance / DDP straggler. world_size NOT baked at precompute (resume-flexible). Exhaustion: outward scan to nearest non-empty bucket (deterministic). Epoch tail = drop_last.

## Pack layout (layout.py, collate.py:84-199)
Per doc: [prefix identifier-card] + [body slice] + [EOS suffix]. Only BODY truncated, never decoration. Prefix = lang-valid card: markdown/py "# {title}", C-family "// {title}", LaTeX "% Title: {title}". Single EOS, no BOS (doc-causal already hard boundary). **Stochastic 50/50 per-(doc,epoch) prefix coin via md5(f"{id}:{epoch}")%2 (layout.py:233-237)** — deterministic across ranks/restarts (NOT python hash()); trains w/ and w/o card so not OOD on benchmark prompts. **layout_epoch** (PackRecord): epoch whose coin-flip a pack was budgeted under; merge stamps source-epoch; _materialize set_epoch(layout_epoch) so same docs get prefixes (else T≠budget → raises).
**prefer_targets_first** (pack_sampler.py:476-558, DEFAULT): Kahn topo sort per connected component so linked-TO docs precede linkers; components emitted as CONTIGUOUS blocks (hard req of doc_concatenated kernel); this makes the analytical-count "target ends before link_end_pos" assumption hold.

## Live vs precomputed
Live PackedSequenceDataset: traverse graph + link-detect + build mask EVERY STEP on GPU. Precomputed BucketedPackDataset: detect ONCE offline in workers, positions cached in PackRecord, reattached as batch["link_to_target"]; density scheduling; num_workers=0 synchronous (sub-ms load); BucketState resume (epoch_idx, global_accum_step, bucket_consumed). Eliminated cost = online link detection + block-mask build + traversal → offline parallel workers. Worker partitioning: TheStack repo-prefix; Wiki/ArXiv multi-source BFS Voronoi w/ 1.5× cap (else BFS hits shard boundary → degenerates to doc-causal).

## Novel/publishable
1. Analytic O(#blocks) density proxy from link positions, ~1ms vs ~36ms (36×), conditional exactness verifiable via --gpu-kv-pass.
2. Quantile-bucketed per-step density-matched DDP (world_size-decoupled).
3. Bucket-shuffle (diversity vs balance).
4. Deterministic md5 per-(doc,epoch) prefix coin.
5. Union-find component assignment decoupled from traversal for doc_concatenated.

## REVIEWER-ATTACKABLE (flag in paper)
1. Block-size mismatch: proxy+BlockMask use 128-tok blocks but custom Triton runs triton_block_size=64 (cross_doc_mask.py:264) → proxy is rank-preserving, not literal executed count.
2. max_grants ignored by analytical count but enforced at runtime (default 64, drops link_pos-sorted extras) → overcounts density for >64-grant packs.
3. Density balance degrades on epoch tail (outward-scan fallback draws different density; drop_last discards ≤world_size-1/bucket).
4. Within-bucket source clustering in merged corpora: pack_id source-sequential + rank takes consumed+r → one optimizer step's cross-rank batch can be source-homogeneous (shuffle is across steps + val-only within bucket).
5. Quantile ties split arbitrarily.
6. Method-C exactness conditional on target-before-link; cycles (Kahn insertion-order fallback) may diverge — unquantified.
7. Backward-compat defaults change semantics silently (missing component_ids → doc_concatenated degenerates to doc_causal; missing layout_epoch → loader epoch).

## → LIT REVIEW IMPLICATIONS
- P-slices: load-balancing / straggler mitigation in data-parallel training; curriculum by difficulty; sample packing.
- Analytic cost model for sparse attention → could cite sparse-attention FLOP/cost modeling.
- Component/topological ordering → DeepSeek-Coder topo (already have).
