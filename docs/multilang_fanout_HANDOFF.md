# Handoff: full multi-language code-dataset fan-out

You are the orchestrator for extending the graph-structured code-dataset pipeline
to **every applicable language The Stack provides**. Two languages (Go, Java) are
already DONE end-to-end and serve as your proven templates. Your job: cover the
rest, dispatching one implementation sub-agent per language, each gated by an
existing frozen quality harness.

**READ FIRST (in order):**
1. `docs/multilang_code_datasets_DESIGN.md` — the full design. Esp. §2 (the two
   axes + anti-reward-hacking), §10–11 (the harness), §12–13 (Go/Java, what's built).
2. `CLAUDE.md` — project rules (compile always on, launch discipline, `/fss-data`
   for bulk I/O never `/fss`, commit directly on `main`, no branches).
3. This file.

---

## What already exists (your templates — do NOT rebuild)

**The harness** (`data/graph_harness/`) — FROZEN, do not modify its scoring:
- `run_detection.py` — grades a detector vs an independent **tree-sitter** oracle
  (precision AND recall). Gate: **P ≥ 0.95, R ≥ 0.90**.
- `run_audit.py` — checkpoint-free graph-quality report (dangling/self/isolated).
- `run_sample_dump.py` — the human visual gate (source → detected link → resolved
  target content).
- `specs/{python,go,java}_spec.py` + `spec.py` — per-language adapter. Two detection
  paths: SIMPLE (`oracle_query`+`canonical_import`) or RICH (`extract_keys`) when one
  import statement licenses several keys.
- `fixtures.py` + `fixtures_data/<lang>/` — resolution oracle vs hand-labeled edges.
- `go_nodes.py` / `java_nodes.py` — the node-model builders (package vs file).

**The two done languages (copy their structure):**
- **Go** (package-node): `model/graph_traversal/go_import_detector.py`,
  `data/go_graph_extractor/{download_go,build_go_graph}.py`, `data/pretokenize_go.py`,
  `configs/go_cross_doc.yaml`. Node = a package (dir of .go files); import path is
  globally unique; module path INFERRED (Stack has no go.mod).
- **Java** (file-node): `model/graph_traversal/java_import_detector.py`,
  `data/java_graph_extractor/{download_java,build_java_graph}.py`,
  `data/pretokenize_java.py`, `configs/java_cross_doc.yaml`. Node = a file keyed by
  FQN = `<package>.<ClassFromFilename>`.

**Datasets already built** (don't rebuild): `/fss-data/evin_t/tagseq2tagseq_artifacts/
pretokenized_datasets/{go,java}` (each with `splits/`).

---

## Step 0 — DISCOVER which languages The Stack actually offers (do this first)

You must NOT assume the language list. Enumerate the `data/<lang>` configs available
in `bigcode/the-stack-dedup` and decide applicability yourself. A language is
**applicable** iff it has a *static, resolvable, file-or-package-level import system*
where an import string maps deterministically to another source unit in the same
repo. Sketch of how to check per candidate language:
- Stream a few thousand files (`datasets.load_dataset(..., data_dir="data/<lang>",
  streaming=True)`), measure imports/file and whether imports look intra-repo-resolvable.
- Reject languages whose "imports" don't resolve to source files (e.g. C/C++
  `#include` needs header-search-path resolution; shell/HTML/CSS/Markdown/JSON have
  no code-import graph; dynamically-resolved imports like Ruby `require` with load
  paths are borderline — judge case by case and DOCUMENT the call).
- The strongest remaining candidates beyond Go/Java are likely: **TypeScript,
  JavaScript, Rust, Kotlin, Scala, C#, PHP, Python (already done — but see the
  migration note below), Swift**. VERIFY against the actual Stack listing rather
  than trusting this list.

Produce a short ranked applicability report (language → applicable? node model
file/package → resolution difficulty → decision) BEFORE dispatching implementers.
Ask the human to confirm the final language set if it's large or includes
borderline calls.

---

## Step 1 — Per-language implementation (one sub-agent each, in parallel)

Give each sub-agent its own git worktree (`isolation: worktree`) so they don't
collide on the two SHARED files (`model/graph_traversal/link_detector.py` registry
and `data/layout.py` map). Each sub-agent's deliverable, mirroring Go/Java:

1. A `LanguageSpec` in `data/graph_harness/specs/<lang>_spec.py` + register in
   `specs/__init__.py`. (This is the tree-sitter oracle — author it INDEPENDENTLY of
   the detector; the frozen query is the ground truth.)
2. A `<lang>_import_detector.py` (token-space `LinkDetector`) + register in
   `make_link_detector` + `LINK_DETECTOR_NAMES` (`link_detector.py`) + add a
   `_DETECTOR_INFERENCE_LAYOUT` entry (`data/layout.py`).
3. A graph extractor `data/<lang>_graph_extractor/build_<lang>_graph.py` (tree-sitter
   build engine) + `download_<lang>.py` + a node-model builder in
   `data/graph_harness/<lang>_nodes.py` + `pretokenize_<lang>.py` (reuse
   `ContentJsonlSource`).
4. A resolution fixture `data/graph_harness/fixtures_data/<lang>/<name>/` +
   `tests/harness/test_<lang>_resolution.py`, and detector unit tests
   `tests/test_<lang>_import_detector.py`.
5. Decide + document the **node model** (file vs package/module) — it's a property
   of the language's import system, decided empirically on real repos (see how Go
   became package-nodes and Java file-nodes in §12).

**ACCEPTANCE GATE (all must pass before a language's data is built):**
- `run_detection <lang> --files <real cloned repos>` → **P ≥ 0.95, R ≥ 0.90**.
- Resolution fixture test → high P/R (aim 1.0 on a hand-built fixture).
- Detector + builder unit tests pass; full suite stays green.
- These gates are what make the work un-gameable — do NOT relax thresholds to pass a
  language; if a language can't hit them, FLAG it and set it aside.

---

## Step 2 — Data pipeline per accepted language (the proven sequence)

For each language that passed the gate, run (see the Go/Java commits for exact
commands; all outputs to `/fss-data`, never `/fss`):

```
download_<lang>.py  → raw/<lang>/sample_<lang>.jsonl   (start 2M files; adjust)
build_<lang>_graph.py → graphs/<lang>/{graph,content}.jsonl  (+ graph_stats.json)
pretokenize_<lang>.py → pretokenized_datasets/<lang>_run  (symlink <lang> → <lang>_run)
split_graph.py       → splits/{train,val_*,test_*}
run_audit  <lang>    → MUST show ~0% dangling/self; sane degree dist
run_sample_dump <lang> → human-review artifact
configs/<lang>_cross_doc.yaml  (copy go/java; set link_detector + paths)
```

**STAGGER heavy jobs** — do not launch multiple 2M-file tree-sitter builds
simultaneously if they share a node (CLAUDE.md launch discipline). Downloads (~14
min each at ~2300 files/s) and builds (single-thread tree-sitter, ~15–25 min for 2M
files) are the long poles; sequence or space them.

---

## Known gotchas (learned the hard way — save yourself the debugging)

- **The Stack dedup has NO project files** (`go.mod`, `pom.xml`, `Cargo.toml`,
  `package.json`) — it's filtered to source-code extensions. So you CANNOT read a
  module/package root from a manifest; infer it from the code (Go infers the module
  prefix from the repo's own imports vs dir layout; Java reads `package` decls). Plan
  for this per language.
- **`run_sample_dump` OVERSTATES connectivity for global-namespace languages.** Its
  `PretokCorpus` resolver matches an emitted import against ALL nodes, so e.g. a Java
  `java.util.List` resolves to some other repo's stub. The STORED TRAINING graph is
  clean (edges are intra-repo by construction) — verify with `run_audit` + spot-check
  in-edges, don't trust the dump's resolution rate. See `TODOS.md` (Java framework-FQN
  entry) and design §13. If you add a `--restrict-to-repo` flag to run_sample_dump,
  future reviews get accurate — worth doing early.
- **Wildcard/on-demand imports** (`import a.b.*`) have no single target node — the
  detector should drop them; make the oracle agree (don't count them as recall misses).
- **`block_mask_creator` viz**: budget matters. Small token budgets pack 1–2 docs →
  sparse masks. Use ≥16k and sweep seeds for a well-connected batch (see how seed 33
  was chosen for Java).
- Interpreter-teardown `pyarrow`/GIL crash at process exit after `datasets` streaming
  is harmless (results already written); use `os._exit(0)` in throwaway probes.

---

## Also in scope / adjacent (do after the fan-out, or delegate)

- **Python → tree-sitter migration**: the harness proved the EXISTING regex Python
  detector has a real bug (unstripped `as` alias in `from x import y as z`; true
  P=0.985/R=0.958 at scale). Migrate it to tree-sitter to fix the bug + the O(tokens)
  per-token-decode slowness + make Python a peer. Concrete failing case + harness are
  ready. See design §10a.
- **The framework-FQN generation-resolution TODO** in `TODOS.md`.

## Definition of done for your whole run

Every applicable Stack language has: passing detection+resolution gates, a built +
audited + split dataset under `/fss-data/.../pretokenized_datasets/<lang>`, a
`configs/<lang>_cross_doc.yaml`, and a green test suite. Commit each language
separately on `main` (no branches per CLAUDE.md; the per-language WORKTREES are for
parallel isolation, merged back to main). Update design §13 with a progress log and
flag anything for human review — especially any language you set aside for failing
the gate, with the reason.
