# CODE BRIEF: link detectors (agent ae8d37e9)
Root: model/graph_traversal/. LinkInfo(link_end_pos EXCLUSIVE, target_str DECODED string). Design: detector does ALL decoding, returns STRINGS not token spans → mask creator tokenizer-independent (link_detector.py:41-43).

## Protocol (link_detector.py:26-70)
- detect_links(input_ids)→[LinkInfo]: whole 1-D seq. Called: generation (rolling window, fires when link_end_pos==len(recent)) + training fallback (only if no detect_links_for_doc).
- detect_links_for_doc(span_tokens, raw_identifier) [NOT in protocol, hasattr]: single doc slice, span-local positions re-based by caller. PREFERRED training path (_collect_links_per_doc). Enables RELATIVE-IMPORT resolution. Impl by python/ts/js/rust/zig/dart/composite.
- index_doc_span(span)→str: LOOKUP KEY (target side); default raw_identifier. target_str = source side. Must land in same string space. Exact dict match. DAG: target starts strictly before link (cross_doc_mask.py:418).
Baked link_to_target (Option B) short-circuits detection → text detection = generation-time + grader-time concern.

## Per-modality (all on decode_fn output)
(a) MARKDOWN = token-ID matching, NO full decode: find ](  bigram GPT-2 token id 16151, backtrack ≤100 for [ (ids 58/685), growing forward window (50 tok) re-decode until ), target=before first ), link_end_pos=j+1.
(b) EVERYTHING ELSE = decode-once + blank comments/strings to equal-length spaces (preserve offsets) + regex + char→token remap via cumsum char-length + bisect_left. Python fast path decode_tokens_bytes ~2×.
- Python: 3 regexes (import / from-import / paren multi-line), strip alias, hand comment/string/docstring blanker. Emits CANDIDATE FILE PATHS not modules.
- LaTeX \cite: regex \cite[a-zA-Z]*(\[...\])*\{(...)\}, syntactic (bibkey→title at extraction), empty \cite{} SKIPPED, NO comment blanking.
- Go: single+grouped import, verbatim path no expansion, blanks //,/* */,strings/runes.
- Java: import [static] a.b.C[.*]; NO comment blanking (relies on ; + ^anchor), static→enclosing type, .* → nothing.
- Kotlin: import a.b.C [as X], wildcard dropped, blanks //,/* */,char,triple-quote.
- TS/JS: from"spec"/side-effect/dynamic import()/require() (guard (?<![.\w])), blanks comments + TEMPLATE-LITERAL backtick bodies but KEEPS quoted-string (specifier IS the string). TS sniffs TS-markers.
- Zig: @import("spec"), blanks // (no block comments),char,\\ multiline.
- Dart: import/export first URI, ANCHORED (?m)(?:^|;), part/part-of not matched.

## Identifier normalization (target_str ↔ index_doc_span same space, exact dict match)
Exact-identity keys: markdown (title w/ spaces), arxiv (title BYTE-IDENTICAL to extraction), go (full import path = node id), null.
Transforming keys: python bare-path (node repo:path, key after :, target side module_path_to_file_paths → foo/bar/baz.py + __init__.py candidates, submodule-vs-symbol ambiguity emit both; relative import only in detect_links_for_doc); java FQN dotification (strip repo:, drop .java, /→., SOURCE ROOT unknown = fragility); kotlin FQN (node id IS declared symbol FQN); rust module path (node owner/repo@crate::a::b, sep @ not : since paths have ::, use-tree expansion, self/super rewrite in per-doc); TS/JS ext-less (flat detect_links = specifier-space keys for GRADER only, detect_links_for_doc = resolved repo-rel keys training matches → flat alone renders EMPTY masks for these langs); zig/dart keep .zig/.dart.

## Composite (composite_link_detector.py) — merged model generation
11 members (COMPOSITE_MEMBERS), PICK EXACTLY ONE per doc never merge (avoid cross-fire). index_doc_span: identifier sniff (_sniff_by_identifier: :: → rust; strip repo prefix + extension map; go hostname heuristic) → sub-detector key; ambiguous (wiki/arxiv title) → identity. detect_links_for_doc: identifier sniff then CONTENT sniff. detect_links (generation, no identifier): content sniff only. Content sniff = Σ weight×match over signatures (\cite{ 3.0, @import( 3.0, use::  2.0, markdown 1.0 so stray ]( never outvotes); TS-vs-JS by TS_MARKERS; ties by priority (markdown last); None for prose. Graceful: mis-sniff rarely matches real key → no-ops. Deferred: mixed-syntax single doc (per-token routing Tier2).

## Novel/publishable
- Tokenizer-decoupled online detection (strings+exclusive positions → mask machinery tokenizer-agnostic).
- decode-once + offset-preserving comment/string blank + regex + bisect char→token = clean fast tree-sitter-parse approximation in training hot path. 3-WAY agreement design (token-space detector vs build-time tree-sitter extractor vs independent tree-sitter oracle).
- Dual-mode (flat detection-recall axis vs per-doc resolution axis).
- Offset-preserving blanking w/ per-lang lexers (nested block comments, raw/byte strings, lifetimes-vs-char, template/multiline bodies).
- Composite single-pick dispatch w/ scored content sniff + no-op-on-mismatch safety.

## Reviewer-attackable
- Markdown HARDCODED to GPT-2 token id 16151 (diff tokenizer → detects nothing); any ]( in prose/code fires; 50-tok window truncates; NO comment blanking.
- arXiv exact-title BYTE-IDENTICAL brittle (whitespace/Unicode/BPE drift breaks silently); no comment blanking (\cite in % comment detected).
- Java source-root ambiguity (can't verify builder keyed FQN-relative); no comment blanking.
- Python candidate over-generation; repo-root==path-root breaks for site-packages; dynamic/TYPE_CHECKING undetected.
- char→token off-by-one on multibyte tokens.
- Composite mis-sniff heuristic (no-op safety is soft not guaranteed); single-lang-per-doc fails on mixed generation.
- max_grants truncation (default 64) silently drops sorted-by-link_pos extras.

FLAGS: didn't read tree-sitter build-time extractors/oracles (agreement asserted in docstrings); didn't trace where Option-B link_to_target produced upstream; raw_identifier formats from docstrings not artifacts.

## → LIT REVIEW IMPLICATIONS
- Static analysis / import resolution / tree-sitter (parsing-based code structure).
- Entity linking / mention detection / wikification (link target resolution).
- Constrained decoding / trie (title index for generation).
