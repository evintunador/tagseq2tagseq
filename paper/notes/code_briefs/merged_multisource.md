<!-- PROVENANCE
Written against commit 6134163 (main @ 2026-08-07). This brief describes the code as of
that commit; it is NOT authoritative if the source has changed since. Before trusting it,
check whether the covered sources have drifted:
    git diff --stat 6134163..HEAD -- data/merge_datasets.py data/merge_packs.py model/graph_traversal/composite_link_detector.py configs/
Empty output = brief still current. Non-empty = re-verify the changed parts against source.
Covered sources: data/merge_datasets.py data/merge_packs.py model/graph_traversal/composite_link_detector.py configs/
-->

# CODE BRIEF: merged_v2 multi-source path (agent a7f3021d)
One ~350M model (1024d/24L VE-off 32k muon_lr0.003/wd0.1 max_grants256) trained JOINTLY on 11 linked sources (wiki, arxiv, 9 code langs). Only variable vs specialist = corpus diversity. 3 rungs (3.9B/8B/16B), each cross_doc_link vs doc_causal FLOP-matched pair.

## Combining 11 heterogeneous sources — two-stage offline per-source-then-merge
Sources NEVER co-tokenized/co-traversed; each precomputed independently then stitched (EpochPrecomputer bakes exactly one link_detector+layout per run).
Stage A graph union (merge_datasets.py): shards intact, tok_shard_idx renumbered global (offsets shard-relative, no byte rewrite), hardlink default. **Collision-safe node union** keyed by normed_identifier highest-priority-source-first, later dup dropped+counted, each node stamped **source provenance** (linchpin for all downstream dispatch). Cross-source edges light up "free" (outgoing retained even dangling, incoming recomputed over union). Homogeneity gate: identical tokenizer+dtype or abort.
Stage B pack merge (merge_packs.py): each source's own within-source BFS schedule reused VERBATIM (byte-identical since packs never mix sources). 4 steps: (1) balance _select_balanced picks target #packs/source evenly across 32 density buckets (largest-remainder); (2) collision-safe id remap — pack doc_ids index SOURCE graph, rewritten to MERGED by normed_identifier, two drop cases: absent, and **collision HIJACK** (id present but winning merged node is DIFFERENT source — kotlin&java both bare FQN, java wins; naive lookup would silently resolve to wrong source's tokens); disjoint (wiki/arxiv) 100%, FQN-colliding ~0.06% loss; (3) concat + globally-unique pack_id, silently prune dead cross-doc grants; (4) re-bucket over union (bucket B ≠ density across sources).
Single batch multi-modality: packs within-source but EPOCH interleaves all 11. Each PackRecord carries own layout_name + layout_epoch → BucketedPackDataset resolves per-name layout (arxiv LaTeX card, code slash-comment). Mask consumes BAKED link_to_target, never calls config detector.

## Per-doc link dispatch by provenance — 3 times, 3 signals
- TRAIN: NO detector, baked link_to_target verbatim. config link_detector='markdown' = PLACEHOLDER.
- GRAPH-EDGE EVAL: no text detection, grants from graph edges (Option B).
- GENERATION: no graph shortcut, MUST detect from raw multi-source text → CompositeLinkDetector (SOLE place it runs). 11 sub-detectors, pick EXACTLY ONE per doc (avoid cross-fire). [detail in link_detectors brief]

## Token balancing
Equal-ish per-domain tokens. token_budget 32768/pack, 262144 tok/step.
- 3.9B = 355M/dom (~11809 packs/src, zig="all" 921 only).
- 8B = 727M/dom (~27256 packs/src, big sources hit, go/java/dart/zig give all).
- 16B = ~1.45B/dom ×2 variants: BALANCED (multi-epoch union, 48466 packs/src 1.59B, zig capped 4 distinct-seed epochs ~0.12B, shortfall redistributed; SAFE=4 epoch per Muennighoff 2-4-lossless) vs NATURAL (single-epoch, big sources absorb remainder; FLAG no dedicated build script found).
**Rungs INDEPENDENT samples NOT nested** → Δ-vs-budget could be resampling not scale (author flags disclose). **16B currently BROKEN** (LR too hot at length: WSD holds peak 0.003 flat ~36k ≈2× 8B exposure → back-half blowup, NOT resume corruption). So published curve = 3.9B + 8B ONLY.

## Cross-doc "ports" eval
(a) Held-out per-source ppl + community_pack Δ: Option B graph-edge grants (no text detector fires across all sources), span.start+1 key hack, vs doc_causal. Per-source val SEPARATE (per-link-type signal). Specialists win this base-LM axis (wrong axis).
(b) Discriminating PORTS: post-hoc on best_model.pt via frozen benchmark_harness. PortAdapter carries language + detector_factory constructing LANGUAGE-SPECIFIC detector (composite bypassed here, exact specialist detector per port). run_tier2 scores SAME examples 3 ways (cross real aux / flat no aux / placebo deranged-content-kept-identifiers), gates Δnll_real bootstrap CI>0 AND placebo separation CI>0. use_line scope. RoPE-cap oversized skipped.
HEADLINE: 8B merge beats specialists Δnll 1.7-11× every comparable port; FLAT 3.9B→8B (saturated at 355M tok/dom).

## Novel/publishable
1. Joint training over many heterogeneous link types strengthens the cross-doc-attention MECHANISM itself, OUTPACING base-LM gains (merge trails on raw ppl, exceeds on cross-doc Δ) — dissociates "more data→better LM" from "many link types→better cross-doc machinery." CORE.
2. Diversity-efficiency (beats specialist trained on ~3.9B own-lang while seeing ~727M).
3. Provenance-driven per-doc link dispatch (3 regimes).
4. Collision-hijack-safe id remap across FQN-colliding namespaces.

## REVIEWER-ATTACKABLE
1. Rungs independent not nested (resampling vs scale; noise floor not quantified beyond single-ckpt bootstrap CI).
2. Only 2 REAL rungs (3.9B, 8B); 16B broken → "does lead grow?" unanswered; FLAT claim rests on 2 points.
3. Specialist baselines HETEROGENEOUS (kotlin/ts external ports ASE/CrossCodeEval vs merge internal ports; go/rust/js no external baseline) → 2.8×/11× indicative not exact.
4. **Within-source packing → cross-source edges NEVER trained on.** Merged graph HAS cross-source edges but packs strictly within-source. "11 link types together" = interleaved within-source packs sharing params = MULTI-TASK MIXING not cross-modality attention within a pack. STATE explicitly to preempt.
5. doc_causal-arm control ports still TODO (should show ≈0 Δ); placebo-separation is current stand-in.
6. Composite single-lang-per-doc + generation content-sniff mis-class degrade silently (only qualitative gen).
7. wiki REGRESSED on ppl (6.64→6.79) lone source worse — unexplained negative transfer.
8. LR/WD never retuned for larger rungs; broken 16B = evidence recipe doesn't transfer across scale, undercuts "only variable is diversity" at scale.
FLAGS: epoch_16b_natural recipe (no script); cross-source co-packing (code implies never); exact 8B per-source pack counts.

## → LIT REVIEW IMPLICATIONS
- Multi-task / multi-domain LM training; data mixing / domain weighting (DoReMi, data mixture laws).
- Multilingual + multi-domain transfer; negative transfer / interference.
- Multi-epoch training / data-constrained scaling (Muennighoff 2-4 epochs) [have].
- Curriculum / balanced sampling across sources.
