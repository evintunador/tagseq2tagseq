# CODE BRIEF: link-following generation / inference retrieval (agent a4aa95d5)
Files: generation_loop.py, document_context.py, document_corpus.py, generate.py, model.py.

## FLAG: CLI terminology corrections
- NO --allow-generation-fallback. Real flag = **--link-retrieval-mode** {corpus_only|generate_only|corpus_then_generate|link_but_skip|full_skip} (generation_config.py:36-43). "generation fallback"=corpus_then_generate.
- --max-link-depth EXISTS (generate.py:588, default 2).

## Generation loop (generation_loop.py _generate_doc:104-213), per token:
1. context.build_sequence() REBUILDS full packed seq from scratch (O(total_tokens) EVERY step).
2. model.forward_inference(tokens, doc_spans) → [1,T,V] FULL forward.
3. Locate active doc's last logit via span.end (aux doc can sit BEFORE active in packed order).
4. repetition penalty + sample_token (temp/top-k/top-p, allowed_vocab_size=50257 real GPT-2, avoid padded lm_head slots).
5. EOS/suffix handling (avoid double-EOS unseen in training).
6. Link detection on last max_recent_link_tokens=200 trailing tokens; fires only if link_end_pos==len(recent) (closed exactly at just-appended token); ≤1 link/token step.
7. Stop: EOS / max_new_tokens / max_tokens_per_document.
**Resolution+fetch+prepend (_handle_link:216-381)**: corpus.get_document(target) → optional head-truncate to max_corpus_doc_tokens (keep abstract+intro) → context.add_corpus_doc(before_entry=active_entry) INSERTS fetched doc IMMEDIATELY BEFORE linking doc (_docs.insert(idx)). Next build_sequence → target physically earlier → cross-doc mask grants linking positions attention to it. "Prepend into attention context" is LITERAL (mutates packed token list).
Corpus resolution (document_corpus.py): 3-tier _resolve_target: (1) exact raw_identifier→normed; (2) **detector-key index_doc_span(node)→normed = SAME key training uses**; (3) fuzzy HashNormTitleIndex (shared w/ eval annotators, only adds). Tokens from mmap PretokShardedBackend.

## max_link_depth + DocumentContext
Depth: sole recursion control (no allow_recursive_links). depth>=max_link_depth→no aux insertion. depth0=root, pulled doc=depth+1. GENUINE recursion: fetched corpus doc scanned for ITS links (_process_existing_doc_links) at depth+1; generated aux recurse via _generate_doc. Belt-and-suspenders double depth check (asymmetry: corpus docs bounded pre-scan vs generated docs unbounded per-token detection).
Window: _docs root-first but inserts go BEFORE linker (topological). Limits max_context_length + max_auxiliary_documents. Eviction drop_oldest|stop_new; make_room NON-MUTATING on failure (can_make_room simulates first). Active doc protected via exclude. **NO cached positions**; build_sequence recomputes offsets each call; pure RoPE so absolute indices shift freely (can run 32k from 8k ckpt). Re-eviction→restore_evicted (not re-fetch), depth rewritten.

## Generation fallback (corpus_then_generate, default when no dataset)
Corpus miss → add_generated_doc (empty, layout-prefix-seeded) before active → recursive _generate_doc depth+1 → synthesizes linked doc FROM SCRATCH → can itself spawn links to max_link_depth. Open-ended multi-doc synthesis.

## Train/inference mirror (STRONGEST claim, SHARED CODE not analogy)
Same LinkDetector protocol; same match key index_doc_span (document_corpus.py:18-20 "SAME match key training uses"); same grant geometry (link_end_pos onward, or whole_doc_grant); same DAG ordering (train: target starts before link; infer: prepend target so span.start<link). Same CrossDocLinkMaskCreator runs inside forward_inference. THESIS: link machinery (detector+match key+grant-from-link-position+DAG) = single impl in both regimes; inference retrieval = training cross-doc mask realized by materializing linked doc into packed seq.

## KV CACHE = NONE (full recompute every step)
grep kv_cache/past_key/use_cache = nothing. build_sequence()+forward_inference from scratch EVERY token; rebuilds block mask per forward; no state threaded. Fetched docs' KV recomputed every step. O(T²)×O(T)=~O(T³) over generation. Naive KV cache would be INCORRECT after insert/evict anyway (pure RoPE shifts positions). flex backend default (~40× faster than varlen for single-doc forward). → RAG/KV-cache contrast: unlike prompt-caching RAG, NO incremental decoding, NO KV reuse. Efficiency limitation to preempt.

## Novel/publishable
- Train/inference unification via shared link machinery (REAL in code).
- Retrieval-BY-INSERTION into packed seq (not context-string concat): fetched docs = first-class w/ DocSpan+layout+grants, masked exactly like training neighbor.
- Recursive bounded multi-hop retrieval (max_link_depth) unifying corpus-fetch / restore-evicted / generate-from-scratch.
- Detector-key resolution shared w/ eval annotators incl fuzzy cascade.
- Length-agnostic inference (32k from 8k via RoPE).
- Generation fallback = open-ended multi-doc synthesis (most RAG lacks).

## Reviewer-attackable
No KV caching/full O(T²) recompute; layout-prefix room under-estimation (uses max_tokens_per_document omitting prefix/suffix); silent link drops (over-long corpus doc no-ops if max_corpus_doc_tokens unset); ≤1 link/token + fire link_end_pos==len(recent) w/ 200-tok window (misses long-span / second link); depth-gating asymmetry corpus vs generated; restore rewrites depth (provenance confusion); determinism caveat (stochastic layout force-mapped deterministic at inference).
Uncertain: PythonImportDetector detect_links_for_doc (train) vs detect_links (infer loop) POTENTIAL train/infer divergence for code — not confirmed consistent; flex vs triton grant kernel numerical equivalence not diffed.

## → LIT REVIEW IMPLICATIONS
- R3/R4: contrast w/ KV-cache reuse RAG (we do NONE) — strengthen framing.
- Iterative/multi-hop retrieval (IRCoT, self-ask, RepoCoder iterative, FLARE active retrieval, self-RAG).
- Open-ended doc synthesis / recursive generation.
