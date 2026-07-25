# Cross-doc benchmark survey for the 7 remaining languages (2026-07-24)

Goal: extend `run_repobench_cross_doc`-style evaluation (per example: primary file
with visible imports + short completion target + cross-file snippets keyed by
repo-relative path, resolvable by our `<Lang>ImportDetector`s) beyond python/java
to: **go, typescript, javascript, rust, kotlin, dart, zig**.

Produced by 6 parallel web-research agents (multilingual survey + per-language
deep dives + construction-pipeline survey). All dataset claims below were
verified by the agents fetching the actual pages/data unless marked UNVERIFIED.

## TL;DR per language

| Language   | Best external option | Fit | Fallback |
|------------|----------------------|-----|----------|
| Go         | **CoLT-132K** test split (Zenodo 15019938, CC-BY-4.0) — 3,000 Go test examples | dependency-based (SCIP) cross-file context WITH paths; post-2024-03 repos (Stack-clean) | self-built |
| TypeScript | **CrossCodeEval TS** (github.com/amazon-science/cceval, Apache-2.0) — 3,356 examples | paths per snippet; targets provably need cross-file symbols; BUT shipped context is retrieval chunks, not import-resolved files | re-clone repos, or self-built |
| Kotlin     | **ASE 2025 Context Collection** (JetBrains+Mistral, Zenodo 16964765, CC-BY-4.0) — 1,076 FIM points + FULL repo snapshots | we extract cross-file snippets ourselves by resolving imports against the snapshot — ideal for our pipeline | self-built |
| JavaScript | none released (verified) | — | **self-built** |
| Rust       | none released (verified) | — | **self-built** (mod→file mapping deterministic) |
| Dart       | none exists (verified) | — | **self-built** (URI imports resolve trivially) |
| Zig        | none exists (verified) | — | **self-built** (`@import("path.zig")` trivial) |

## Cross-cutting findings

1. **M2RC-Eval is vaporware.** The only benchmark covering most of our languages
   (18 langs incl. go/ts/js/rust/kotlin — NOT dart/zig) never released its data:
   GitHub repo is a 210-byte README, the Feb-2025 data-request issue is unanswered,
   nothing on HF/ModelScope (all independently verified by 4 agents). Even if
   released it's built from **The Stack v2** (contaminated vs our training data)
   and its cross-file context is Jaccard-retrieval, not import-based. Watch, don't wait.

2. **No benchmark ships its construction pipeline.** RepoBench, CrossCodeEval,
   M2RC-Eval, ExecRepoBench all release data + eval harness only. The
   methodologies are well documented, though:
   - RepoBench: tree-sitter import parse → "cross-file lines" = first usage of an
     imported module; contamination via repo-creation-date cutoff + Stack dedup.
   - CrossCodeEval: stub out intra-project imports → static-analysis errors
     pinpoint tokens that STRICTLY require cross-file context (necessity filter);
     plus a small-model filter dropping examples solvable in-file.
   The only public reusable pipeline piece is RepoCoder's retrieval scripts
   (language-agnostic sliding-window retrieval — not what we need).

3. **Heterogeneity problem.** The three adoptable externals differ in target
   shape (statement FIM vs next-line vs multi-line middle) and context selection
   (SCIP dependency vs retrieval chunks vs full snapshot). Cross-LANGUAGE
   comparisons on external benchmarks will therefore be confounded by benchmark
   construction. A uniform self-built benchmark over our own `test_community`
   splits (all 9 languages incl. python/java) is the only way to get an
   apples-to-apples cross-language table; the externals then serve as
   independent-provenance validation anchors for go/ts/kotlin (+ existing
   python/java RepoBench).

## Adoptable externals — detail + adaptation notes

### Go — CoLT-132K (aiXcoder-7B-v2, arXiv 2503.15301)
- Zenodo https://zenodo.org/records/15019938 (`CoLT-132K.zip`, 1.1 GB, CC-BY-4.0);
  code github.com/aixcoder-plugin/aixcoder-7b-v2. Not on HF.
- Schema (verified from the repo's `prompt/prompt_aixcoder_colt.py`): `prefix`,
  `suffix`, `middle`, `code_file_path`, `cross_file_dependency: [{code_file_path,
  abstraction}]` (SCIP dependency graph → files the current file imports),
  `similar_functions: [{code_file_path, code_block}]` (retrieval).
- Targets: cross-file API-invocation lines or similar-code spans (FIM `middle`).
  3,000 Go test samples (1,000 × 3 scenarios). Repos created after 2024-03 →
  clean vs The Stack (our training source).
- Adaptation: map `cross_file_dependency` → aux DocSpans keyed by
  `code_file_path` (GoImportDetector resolves package paths → dir paths);
  `prefix` → context; `middle` first line → completion. **Caveats to check on
  download**: (a) whether `prefix` starts at file top (imports visible) —
  UNVERIFIED; (b) `abstraction` is a signature skeleton, not the full file body —
  weaker aux content than RepoBench snippets; (c) Zenodo authorship listed as
  "Anonymous, Researcher".

### TypeScript — CrossCodeEval (AWS, NeurIPS 2023)
- Official tarball in github.com/amazon-science/cceval
  (`data/crosscodeeval_data.tar.xz`, Apache-2.0). Convenience parquet mirror
  `ZHENGRAN/cross_code_eval_typescript` (3,356 rows — matches paper; spot-check
  against tarball before trusting). `zijwang/CrossCodeEval` is an EMPTY placeholder.
- Schema: `prompt` (left context incl. imports — check
  `metadata.context_start_lineno == 0` to ensure import block not cropped),
  `groundtruth` (statement-level, avg 1.0–1.7 lines),
  `crossfile_context_retrieval{,wref}: [{filename, retrieved_chunk, score}]` —
  repo-relative paths.
- Strength: cursor positions were chosen by import-stub static analysis, so every
  groundtruth genuinely requires a cross-file symbol.
- Weakness: shipped snippets are retrieval-selected 10-line chunks, NOT the
  import-resolved files. Two adaptation paths: (i) accept chunks as aux DocSpans
  keyed by `filename` (TypeScriptImportDetector resolves `./relative` imports →
  those filenames — fragments may or may not contain the referenced symbol);
  (ii) re-clone the source repos (`metadata.repository`; collected 2023-03→09)
  and extract whole files. Contamination vs The Stack: repos created Mar–Jun 2023,
  possible overlap with Stack v2 crawl — undocumented; check against our corpus.
- Second sample for TS: **R2C2-Bench** (github.com/liujiaheng/R2C2-Coder `data/`,
  6,506 TS test rows, explicitly Stack-deduped, `// Path:` headers embedded in
  comment strings) — but NO license file; treat as comparison-only.

### Kotlin — ASE 2025 Context Collection Challenge (JetBrains + Mistral)
- Zenodo https://zenodo.org/records/16964765 (CC-BY-4.0); paper arXiv:2510.04349;
  starter kit github.com/JetBrains-Research/ase2025-starter-kit (MIT).
- Schema (verified by agent downloading `kotlin-practice.jsonl`): `id`, `repo`,
  `revision`, `path`, `modified`, `prefix`, `middle`, `suffix`, `archive` — plus a
  FULL repo snapshot zip per repo+revision (snapshot pre-dates the ground truth →
  no temporal leakage; Long-Code-Arena style).
- 1,076 Kotlin points (30 practice / 400 public / 646 private) from 50 repos;
  ~7.2 GB of snapshot zips. All 30 practice prefixes verified to start at file
  top (package/import block visible). `middle` is 2–11 lines — use first line
  as next-line target if desired.
- Adaptation: BEST fit of all three — no pre-baked context to fight; we resolve
  the file's import FQNs against snapshot paths exactly like the Java port
  (source-root strip already handles `src/main/kotlin/` AND `src/main/java/` —
  note Kotlin files DO appear under `java/` roots, e.g. the practice example).
  `modified` field lists commit-changed files (filter self-referential context).

## Self-built benchmark (JS, Rust, Dart, Zig + uniform cross-language track)

Already filed in TODOS.md ("Self-built cross-doc code benchmark from
test_community splits", 2026-07-23). The survey strengthens the case:

- Every language has `splits/test_community` + `test_random` on
  `/fss-data/.../pretokenized_datasets/<lang>/` (verified on disk today).
- Recipe = RepoBench targeting + CrossCodeEval necessity filtering, both fully
  documented (see item 2 above):
  1. Within a held-out community, take file B importing sibling file A
     (edges already in our graph — no annotator/injection).
  2. Target selection: first line in B that USES a symbol imported from A
     (RepoBench's "cross_file_first"; tree-sitter — we have per-lang grammars
     from graph_harness).
  3. Optional necessity filter (CrossCodeEval): drop targets whose token is
     predictable in-file (or use the paired flat-vs-crossdoc scoring we already
     report — the Δnll IS the necessity measure).
  4. Score with `score_completion_with_context_docs` (language-agnostic,
     takes any link_detector) — same dual-condition reporting as
     run_repobench_cross_doc (cross_doc_only vs flat paired baseline).
- Import→file resolution difficulty by language: zig (`@import("x.zig")`) ≈
  dart (URI) ≈ js/ts (relative path) < rust (`mod`/`use crate::` deterministic
  mapping) < go (package dir) < kotlin/java (FQN + source roots).
- Contamination: test_community is held out from training by construction —
  cleaner than any external benchmark.
- Known risk (why community_pack is near-noise): whole-body scoring dilutes the
  signal. The RepoBench-style token scoping (import-USING lines only) is the fix.

## Raw-material shortcuts noted by agents (if we want fresh external repos)
- `tianyang/repobench_raw_v1.1` (CC-BY-4.0): 6,362 post-cutoff repos, python/java
  only — selection criteria worth copying, data not reusable for our langs.
- `cartersusi/zig-llama` (HF, Apache-2.0): 141k Zig files from ~1,100 repos with
  repo-relative paths — external-provenance Zig repo source if we don't want to
  use our own thestack-derived corpus.
- DependEval (github.com/ink7-sudo/DependEval): 2,683 curated repos across 8
  langs incl. TS/JS — repo list usable as a source.
- SWE-smith-go / Multi-SWE-bench: curated modern Go repo lists (wrong task).

## Verified dead ends (don't re-investigate)
- M2RC-Eval (unreleased + Stack-v2-sourced + retrieval-based)
- ExecRepoBench, Codev-Bench, Long Code Arena completion, RepoEval/RepoCoder,
  CrossCodeLongEval: all Python-only (ExecRepoBench's `context_code =
  [[path, content]]` schema is a good template though)
- Stack-Repo/RepoFusion: Java-only AND Stack-derived
- CodeSearchNet: function-level, no imports/cross-file info
- McEval / MultiPL-E: function-level, no repo context (MultiPL-E added Zig
  2026-04, HumanEval-translation only)
- RepoMasterEval, R2C2-Bench-non-TS: no public release
- RustEvo2 / RustRepoTrans / CatCoder RustEval: wrong task shapes (API evolution /
  translation / function generation); RustRepoTrans + CatCoder crates usable as
  curated Rust repo sources at most
- Dart: `Hinno/flutter_code_dataset`, `KaQyn/flutter_stack` etc. — single-file
  or raw corpus only
