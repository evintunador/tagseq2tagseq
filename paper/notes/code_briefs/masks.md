<!-- PROVENANCE
Written against commit 6134163 (main @ 2026-08-07). This brief describes the code as of
that commit; it is NOT authoritative if the source has changed since. Before trusting it,
check whether the covered sources have drifted:
    git diff --stat 6134163..HEAD -- model/graph_traversal/block_mask_creator.py model/graph_traversal/cross_doc_mask.py model/graph_traversal/link_detector.py
Empty output = brief still current. Non-empty = re-verify the changed parts against source.
Covered sources: model/graph_traversal/block_mask_creator.py model/graph_traversal/cross_doc_mask.py model/graph_traversal/link_detector.py
-->

# CODE BRIEF: cross-doc attention masks (agent ab9b73b9)
Files: block_mask_creator.py, cross_doc_mask.py, link_detector.py, model.py, attention.py. B==1 asserted (attention.py:149) → "batch"=1 packed seq of T tokens.

## Formal semantics (q,k in [0,T); doc(i)=doc_id or -1 for layout gaps; comp(i)=component_id or doc_id)
- doc_causal: M=（q>=k)&(doc(q)==doc(k)). -1 layout positions = ONE pseudo-doc (attend each other!).
- cross_doc_link: M=(q>=k)&(same_doc OR in_grant). Grant rect per link A→B: rows [link_end_pos, min(T,A.end)) × cols [B.start,B.end). "from link position onward"=link_end_pos (one past closing delimiter). ASYMMETRIC (A×B never transpose). DAG gate: skip if target_start>=link_end_pos (cross_doc_mask.py:417-423) → backward links only. causal never relaxed.
- doc_concat_link (compute control): grant_start=A.start (whole source doc) not link_end_pos; whole_doc_grant=True (:485). Strict FLOP-superset, same connectivity → isolates link-position gating.
- doc_concatenated (compute control): M=(q>=k)&(comp(q)==comp(k)). Component=undirected union-find over graph edges. Requires each component CONTIGUOUS (assert block_mask_creator.py:191-210). Most attention, no inference linking.
Ordering doc_causal<cross_doc_link<=doc_concat_link<=doc_concatenated follows by superset.

## Detected link→grant, max_grants, Option B
Pipeline __call__ (cross_doc_mask.py:916-1041): detect (per-doc _collect_links_per_doc offset by span.start for code import detectors, OR whole-seq detect_links) → _match_links_to_docs (index_doc_span(span) key, DAG check, multi-target per link_end_pos) → _build_grant_bitmasks.
**max_grants**: class default 64 BUT production wires **256** everywhere (model.py:68,149...). Grants consumed sorted by link_end_pos; >cap dropped w/ warning (POSITIONAL truncation, later links lose first). Cosine warmup max_grants_start→max_grants; _n_chunks sized for final (stable Triton shape).
**Bit-packed grants**: grant k = chunk k//64 bit k%64 (bit63=INT64_MIN). q_bitmasks[c][grant rows]|=bit; kv_bitmasks[c][target cols]|=bit. Membership PURE POINTWISE no reduction: in_grant=OR_c(q_bm[c][q]&kv_bm[c][k])!=0. ~KB-MB vs ~1GB dense. 256=4 chunks.
**Option B (baked graph-edge grants)**: __call__ accepts precomputed link_to_target → SKIP online detection. Training + graph-edge eval use it (epoch_precompute detects once, stores link_end_positions+link_target_doc_ids, bucketed_pack_dataset rehydrates batch["link_to_target"]); eliminates ~1.3s/step online PythonImportDetector@32k. GENERATION uses text detection (no graph shortcut). Same link_to_target dict semantics either way.

## Compilation: Flex BlockMask vs Triton BIM
- Flex (inference/eval default): closure cross_doc_link_mod compiled via create_block_mask, B=H=None; flex_attention torch.compile(dynamic=True); create_block_mask runs eager (@torch._dynamo.disable). **128×128 sparse blocks** (framework default; density metric counts 128-tok block pairs).
- Triton (TRAINING default per CLAUDE.md): BlockInteractionMask (BIM) = block-level CSR, computed ONCE per batch on CPU numpy, reused across ALL heads+layers (mask head/layer-independent). Predicates: same_doc (block overlap), grant ((q_union&kv_union)!=0), causal; interact=causal&(same_doc|grant). v12 @128 fwd; v17/v18 add @64 bwd (A100 SMEM/register). Block taxonomy: FULL (single-doc off-diag, no per-elem mask), PURE-CROSS (v13: single-doc diff-doc, only bitmask), BOUNDARY (straddles, full elem mask); CSR ordered [full,pure_cross,boundary,diagonal] for kernel branch. Auto-select cross_doc_link→triton_v18, doc_causal→varlen_bim_v2.

## Novel/publishable README omits
1. Bit-packed grant repr w/ pointwise membership (no seq reduction), O(T²)~1GB → O(T·n_chunks)~KB-MB. ENABLING trick.
2. **Ordinal run-index relabeling** (cross_doc_mask.py:636-669): block same_doc uses interval-overlap, only valid if per-pos doc labels monotonic; raw doc_ids NON-monotonic (traversal order); relabel ordinal[i]=cumsum(doc_id changes). Naive version caused thestack cross_doc_link training NaN (LSE collapse dQ≈5.7e4). Footnote-worthy correctness fix.
3. Full/pure-cross/boundary block taxonomy (kernel specialization).
4. Density-aware bucketing via kv_block_count (~6× live-block variance = DDP imbalance).
5. whole_doc_grant + doc_concatenated as MATCHED COMPUTE CONTROLS (attribute cross_doc win to linking bias not FLOPs).
6. Option B baked grants decoupling detection from masking + CompositeLinkDetector (11 syntaxes).

## Reviewer-attackable
1. **RoPE NOT reset per doc** — global positions [0,T); A reads B at relative offset = packing distance (arbitrary, traversal-order-dependent), not semantic. Confirm intended. (Rotary lives in tunalab.modules...flex_self_attention, not opened.)
2. -1 layout/EOS tokens attend each other (one pseudo-doc); BIM diagonal guard np.fill_diagonal(same_doc,True) to avoid OOB bwd.
3. link_end_pos containment half-open span.start<link_pos<=span.end; off-by-one sensitive; all 11 detectors must emit exclusive link_end_pos.
4. Multiple grants compose by UNION (OR), no weighting/precedence; max_grants truncation positional not importance → >256 links silently biased.
5. DAG/ordering dependence: only backward links granted; cycles get one direction only.
6. Flex vs Triton vs dense-viz = 3 mask reimplementations must agree; max_grants must match train/eval (else understates effect).
7. doc_concatenated contiguity assertion = hard packer-ordering coupling (crashes vs mis-masks).

FLAGS: 128 block size inferred (framework default, no explicit override found); RoPE no-per-doc-reset from attention.py, Rotary impl in tunalab not opened.

## → LIT REVIEW IMPLICATIONS
- A6 kernel slice: block-sparse attention kernels, bitmask/CSR attention, FlexAttention mask_mod.
- Positional encoding across packed docs / RoPE relative position → could need a mini-topic (position-id reset in packing, "attention mask reset + position reset" Megatron).
- Compute-control experimental design → causal-inference-style ablation framing.
