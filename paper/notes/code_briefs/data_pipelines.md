# CODE BRIEF: data extraction pipelines (agent a0f6bc81)
Pipeline: extract→graph.jsonl(+content)→pretokenize→tokenized_graph.jsonl+shard_*.bin→split_graph. LINK POSITIONS NEVER STORED (re-detected from tokens at runtime).

## Edge extraction/resolution
- WIKI: 2-stage text-mediated. dump_extractor Cirrus JSON ns0 → .md per article; wikilinks [[t|label]]→markdown [label](target), target = RAW title spaces-preserved NO hashing. build_graph re-parses .md, regex ](...), URL-unquote + normalize_identifier vs node ids. Edge iff normalized target in-set; outgoing keeps ALL (danglers as latent cross-source); incoming only in-set. Scale: enwiki 275,199 nodes/2,321,731 edges avg deg 8.44; enwikisource 662,058/114,523 avg 0.17. **Link resolution FRACTION NOT LOGGED.**
- PYTHON: tree-sitter import extraction + stdlib/external denylist (~40 hardcoded). Module→file resolution per repo (dotting paths, __init__→parent); priority exact.py→module→prefix, tie-break _pick_from_candidates → **uniform random.choice (NON-DETERMINISTIC, unseeded)**. INTRA-REPO edges only; repos hash-partitioned 256 buckets. node id repo:path. Filter links_in_repo<2 → **only 28.7% files kept** (100M: 3.56M nodes/6.34M edges avg 1.78, ~34% zero-in ~25% zero-out).
- ARXIV \cite: unarXive 2024. Dual resolution: direct contained_arXiv_ids (~14.5%) + OpenAlex map (+~52%→~66%). Edge only if resolved id in-corpus; out-of-corpus {{cite}} REMOVED from text (no dangling noise, no edge); in-corpus rewritten \cite{title}. Title collisions +" (arxiv:id)" (~0.13%).

## Normalization ("canonical vs raw" hashing, normalization.py)
normed_identifier = {normalized_body}_{6-char-md5(RAW string)}. _norm_body: lowercase, spaces→_, non-[a-z0-9-_]→_, hyphens preserved. Flavors normalize_title(cap 193ch)/repo_name/package_name/arxiv(VERSION-STRIPPED first via canonical_arxiv_id then hash canonical → node+citations agree). Hash disambiguates "A+B" vs "A-B" both→a_b. ASYMMETRY: wiki edge targets in text = RAW titles-w-spaces, MarkdownLinkDetector.index_doc_span matches raw_identifier NOT hashed id; hashed = internal graph key. Stack node id = hashed repo : RAW file path. (Explains memory "stack graph stale normalization" — module is single source of truth, mismatch silently empties dataset.)

## Pretokenization
tiktoken gpt2 default (vocab 50257→uint16). Pool workers → single writer over mp.Queue (deadlock-avoidance comments). Shard .bin: 2GB target, 256×int32 (1024B) header magic 11041999. tokenized_graph.jsonl merges node + {tok_shard_idx,tok_offset_bytes,tok_len}. **Link positions NOT stored** — re-detected at runtime by *Detector (crux of detector-must-agree-with-extractor invariant).

## Split (split_graph.py) → splits/{train,val_community,val_random,test_community,test_random}
Each = self-contained GraphIndex dir, own tokenized_graph.jsonl w/ edges filtered to same-split nodes, shards SHARED (absolute paths not copied). Default 2.5%/split.
- val/test_random: uniform sample of non-community nodes (no structure).
- val/test_community = "community packs" = held-out linked subgraphs via BIDIRECTIONAL BFS (out+in adj) from high-degree seeds, size 50-5000, union shuffled split 50/50 val/test. Retain intact intra-community link structure → val loss reflects true cross-doc traversal.
- Edges filtered to same-split nodes (cross-split severed). Source stratification optional.

## Reviewer-attackable
1. TWO divergent split mechanisms (split_graph subdirs vs pretokenize splits.json inline field) — which canonical AMBIGUOUS.
2. Edge leakage: splits share same .bin bytes; prevented at node-membership level (node in exactly one split) — correct but note.
3. Non-deterministic import resolution (random.choice unseeded) → graph not reproducible run-to-run.
4. Dedup WEAK/sampling-only (Stack blake2b 1% sample REPORTED not filtered; the-stack-dedup upstream; wiki/arxiv NO dedup; no cross-split near-dup check).
5. links_in_repo>=2 discards 71% → over-represents package-heavy repos (top-degree = __init__.py/generated SDK).
6. Wiki resolution rate UNMEASURED; fix_mediawiki_links INVENTS md5[:6] link ids that WON'T match normalize_identifier → guaranteed danglers.
7. Import denylist hardcoded ~40; scipy/pydantic etc treated as intra-repo.
8. arXiv coverage depends on external OpenAlex map (14.5%→66%).

## → LIT REVIEW IMPLICATIONS
- Corpora D1/D2 (unarXive, S2ORC, WikiLinkGraphs, The Stack, KILT).
- Graph construction / entity+citation resolution / import resolution.
- Data dedup + decontamination (near-dup, MinHash, train-test leakage in LM pretraining).
- Tokenization (BPE, tiktoken, GPT-2).
