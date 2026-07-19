# Multi-language code datasets — design doc (stage 1 of 4)

Status: **stage 3 (Go pilot) code COMPLETE; running data pipeline**. Author: Claude. Date: 2026-07-19.

Plan (agreed): **spec/design → harness → one pilot language → fan-out**, each stage
possibly handed to a fresh session.

- Stage 1 (spec): this doc.
- Stage 2 (harness): **COMPLETE — both axes built.** Detection (§10) + resolution
  (§11): fixtures runner, dataset auditor, sample-dump.
- Stage 3 (Go pilot): **code COMPLETE — see §12.** Detector, extractor, pretokenize,
  package-node model all built + validated on real Go. Data pipeline running.
  Still pending: Python→tree-sitter migration (§10a) as a fan-out task.
- Stage 4 (fan-out): in progress (§13).

## 12. Go pilot — code complete + validated (2026-07-19)
- `model/graph_traversal/go_import_detector.py` — `GoImportDetector`. Emits the
  raw import path as `target_str`, NO candidate expansion; `index_doc_span` returns
  the full import path. Strips comments (hand-scanner, string-literal-aware) so
  imports inside doc comments aren't matched. Registered in `make_link_detector`
  (`go`) + `LINK_DETECTOR_NAMES` + `_DETECTOR_INFERENCE_LAYOUT` (identifier_prefix_eos).
- `data/go_graph_extractor/build_go_graph.py` — tree-sitter package-graph builder.
  A node = a package (dir of non-test .go files, concatenated), keyed by full
  import path `<module>/<pkgdir>` (module from go.mod). Emits `graph.jsonl` +
  `content.jsonl`. Edges = exact-match imports between same-repo packages; dropped-
  package edges pruned (no dangling). Skips repos without go.mod (can't form unique
  paths).
- `data/document_sources.py::GoPackageContentSource` + `data/pretokenize_go.py` —
  thin content.jsonl reader + pretokenize wrapper (arxiv/fineweb pattern).
- `data/graph_harness/go_nodes.py::build_go_package_nodes` — the SAME package-node
  model, shared with the fixtures runner.
- **Node-unit decision RESOLVED empirically**: packages, not files (Go imports
  reference directories; same-dir files share a package w/o importing). See §Go pilot.

**Validated (2026-07-19):**
- Detection vs tree-sitter oracle on 70 real Go files (httprouter/logrus/mux):
  **P=1.0 / R=1.0**, 320 imports. (Harness caught + we fixed a real FP: `import (…)`
  inside a `/* */` doc comment.)
- Resolution on the `go/simple_module` fixture: **P=1.0 / R=1.0** in package-key space
  (store's 2 files correctly = 1 node; stdlib `fmt` correctly doesn't resolve).
- Full build→pretokenize→audit→sample-dump on real logrus: 6 package nodes, 5 edges,
  **0% dangling/self/isolated**; sample-dump shows intra-repo imports RESOLVE to the
  right package content, stdlib+external imports correctly unresolved.
- 120 tests pass (harness + go + python + corpus).

## 13. Fan-out (stage 4) — in progress
Running autonomously (user authorized overnight, 2026-07-19). Per §5, gated by the
frozen harness (detection P≥0.95/R≥0.90 + resolution fixture + human-style
sample-dump). Scope decision: DEPTH over breadth — fully complete + validate Go
(the pilot), then Java (cleanest remaining, richest imports 9.5/file), rather than
shipping 4 unvalidated datasets. Rust + TS/JS have messier resolution needing
careful fixtures + human review; specs/detectors will be stubbed but data NOT
shipped without validation.

**Progress log:**
- **Go data pipeline (task 3): COMPLETE — dataset training-ready (2026-07-19).**
  - Download: 2,000,000 Go files (9.2GB) → `/fss-data/.../raw/go/sample_go.jsonl`.
  - Graph build (full 2M, 326,796 repos): **358,812 package nodes, 388,608 edges**,
    avg out-degree 1.08, **0 dangling / 0 self-links**. → `/fss-data/.../graphs/go/`.
  - Pretokenize: **1.22B tokens**, 2 shards →
    `/fss-data/.../pretokenized_datasets/go_run` (symlinked `…/go`).
  - Audit: 0% dangling/self/isolated. Sample-dump (human review): intra/in-corpus
    imports RESOLVE to correct package content; stdlib+external correctly
    unresolved; even cross-repo resolution works (k8s.io/... via a vendoring repo).
  - Split: train 322,900 / val_community 8,986 / val_random 8,970 / test_* 8,986/8,970.
    Train split re-audited clean (0 dangling after edge filtering). Loads + tokens OK.
  - **Ready to train**: `configs/go_cross_doc.yaml` (single-GPU uses live dataset;
    precompute epoch_dirs for multi-GPU is optional, not yet run).
  - Precompute note: Go ids have no ':' → route to the Voronoi partitioner
    (correct; intra-repo edges = disconnected components Voronoi keeps together).
    No dispatcher fix needed for Go.
- **Java (fan-out #2):** detector + harness spec DONE, detection P=1.0/R=1.0 on 80
  real gson files (§12-adjacent). Extractor + source-root FQN keying + data run
  still TODO.
- **Rust, TS/JS:** grammars installed; not started (deferred — messier resolution).
- **Python→tree-sitter migration (§10a):** not started.

**The Stack has NO go.mod** (filtered to ext==go) → module path is INFERRED from
each repo's own imports vs. its directory layout (`build_go_graph.infer_module_path`).
Validated on 50k real records: real modules recovered (github.com/APTrust/exchange,
code.cloudfoundry.org/cli, ...), 0 dangling/self edges.

## 10. What's built (stage 2, DETECTION axis) — 2026-07-18
Package `data/graph_harness/` (tests in `tests/harness/`), tree-sitter + `tree_sitter_go`
+ `tree_sitter_python` installed. (Resolution axis in §11 — both axes now done.)

- `data/graph_harness/spec.py` — `LanguageSpec`: per-language adapter. Two detection
  paths, exactly one per spec: SIMPLE (`oracle_query` + `canonical_import`, one
  node→one key, used by Go) or RICH (`extract_keys` walker emitting the full set of
  legitimately-licensed keys per statement, used by Python because
  `from a.b import c` licenses both `a/b` and `a/b/c`).
- `data/graph_harness/oracle.py` — `TreeSitterOracle`: independent ground-truth import
  extractor. `import_keys(source)->set` on either path.
- `data/graph_harness/scoring.py` — FROZEN micro-averaged precision/recall scorer over
  canonical key sets, with concrete FP/FN examples. `canonical_target` collapses
  the detector's many candidates per import into the shared key space.
- `data/graph_harness/specs/{python_spec,go_spec}.py` + registry.
- `data/graph_harness/run_detection.py` — CLI: grade a registered `link_detector` on
  real files vs. the oracle, enforce `--min-precision/--min-recall`, print gaps.
- `tests/harness/test_oracle_and_scoring.py` — **6 tests pass**. Load-bearing:
  the trusted `PythonImportDetector` scores **P=1.0 R=1.0** vs. the oracle (harness
  validated against known-good); emit-nothing FAILS recall; emit-garbage FAILS
  precision (the reward hacks are provably caught).

**Known perf caveat:** the existing `PythonImportDetector._build_char_to_token_index`
decodes each token individually → O(tokens) decode calls per file, slow on large
files (a full `data/`+`model/` sweep did not finish in 120s). Fine for
sample-based grading (`--max-files`); a token-batch speedup is a possible follow-up
but is a property of the shipped detector, not the harness.

**Resolution axis: DONE in §11** (fixtures runner, dataset auditor, sample-dump).

### 10a. Harness caught a real Python-detector bug at scale (2026-07-18)
`P=1/R=1` held ONLY on 8 tiny curated files. Graded on **120 real Python files**
(project + venv third-party), the trusted detector scores **P=0.985 / R=0.958** —
passes the 0.95/0.90 gate but is NOT perfect. Two distinct issues surfaced:
- **Harness bug (fixed):** `from __future__ import annotations` parses to a
  distinct `future_import_statement` tree-sitter node the oracle walker missed →
  spurious FPs. Fixed in `python_spec.py`.
- **Detector bug (real):** `from x import y as z` — the detector does NOT strip the
  `as z` alias, emitting a `target_str` of literally `"x/y as z"`, and misses
  inline-aliased from-imports. Never caught by the 79 existing unit tests.
  Practical impact ≈ 0 (affected imports are external deps that never resolve to a
  co-packed node), but it's a genuine correctness gap.

**DECISION (2026-07-18): migrate Python extractor+detector to tree-sitter** — fixes
this bug AND the O(tokens) per-token-decode slowness AND makes Python a true peer of
the new languages. Scheduled as a fan-out task (own worktree + PR), stage-3-adjacent;
it now has a concrete failing case to fix + the harness to verify against.

## 11. Resolution-axis tooling (stage 2, built 2026-07-18)
Grades the axis tree-sitter cannot (§2): does a `target_str` resolve to the RIGHT
node? All model-free / checkpoint-free.

- `data/graph_harness/fixtures.py` — **fixtures runner**, the resolution oracle. Scores
  resolved edges (via the SAME `PretokCorpus._build_indexes`/`_resolve_target`
  training uses) against a hand-labeled `edges.json` on a tiny self-contained
  fixture repo. Fixture format: `<lang>/<name>/{files/, edges.json}`.
  `data/graph_harness/fixtures_data/python/simple_pkg/` is the worked example.
- `data/graph_harness/auditor.py` + `run_audit.py` — **checkpoint-free dataset
  auditor** (design §3b). Node/edge counts, degree dist, dangling/self-link/
  isolated/reciprocal rates + warnings, on ANY built dataset dir.
- `data/graph_harness/run_sample_dump.py` — **link sample-dump** (design §3c), the
  human-gate artifact: prints random (source, detected link, RESOLVES?, resolved
  target snippet) tuples from a built dataset.
- `tests/harness/test_resolution_and_audit.py` — 3 tests. Python detector +
  resolution scores P=1.0 R=1.0 on the fixture; auditor computes
  dangling/self/isolated/reciprocal correctly; edgeless graph flagged.

**Validated on real production data (2026-07-18):**
- `run_audit` on the full **thestack** graph: 3.56M nodes, 6.99M edges, 486k repos,
  out-deg mean 1.96 — matches the known baseline. Surfaced a real signal: **9.37%
  dangling-edge rate** (edges to files dropped by the `links_in_repo >= 2` build
  filter). Property of the existing Python pipeline, not introduced here — worth a
  look, since dangling edges are silently unattendable.
- `run_sample_dump` on a carved single-repo corpus: external deps (pytz, sqlalchemy)
  correctly show `unresolved`; the one intra-repo import RESOLVES and prints the
  correct target file content. This is the "visually spot-check batch content"
  capability the project brief asked for.

**Harness status: 9 harness tests + 103 existing related tests pass.** Nothing
committed yet.

---

## 0. Goal

Extend the graph-structured code corpus beyond the Python subset of The Stack to
additional programming languages, so we can train the cross-doc-link model on
richer, multi-language code graphs. The hard requirement: **the pipeline that
turns a new language's imports into graph edges must be un-gameable and produce
verifiably high-quality data**, confirmed ultimately by a human reading the actual
decoded content of packed batches.

---

## 1. How the existing Python pipeline works (shared background)

A "link" is an intra-repo, file→file dependency edge. `import` → directed edge
from the importing file-node to the imported file-node **within the same repo**.
Nodes are keyed `owner/repo:path/to/file.py`; edges live as `outgoing`/`incoming`
adjacency lists on each node in `tokenized_graph.jsonl`.

The edge becomes attention through **three stages that must agree on one string
key** — this agreement *is* the contract:

1. **Build time** — `data/github_graph_extractor/{build_graph_streaming.py,extract.py}`.
   Python-specific regexes extract imports; a hard-coded stdlib/3rd-party denylist
   skips external imports; `module_path → candidate file path` resolution (with
   `__init__.py` package semantics) picks the target file inside the repo.
2. **Train / inference time** — `model/graph_traversal/python_import_detector.py`,
   a `LinkDetector`. It **re-detects links from tokens** (not from the raw graph),
   emitting `LinkInfo(link_end_pos, target_str)`. `index_doc_span(span)` produces
   the corpus lookup key. The *only* thing that makes resolution work is that
   `target_str` and `index_doc_span` live in the **same string space** (Python
   strips the repo prefix so bare `utils/helpers.py` matches node
   `owner/repo:utils/helpers.py`).
3. **Mask** — `model/graph_traversal/cross_doc_mask.py`. Each resolved
   `(link_pos → target_doc_id)` co-packed in one sequence becomes a grant
   rectangle `[link_pos : source_end] × [target_start : target_end]`, OR-ed into
   the predicate `causal & (same_doc | in_grant)`.

**Single-repo constraint (Python):** because Python import strings say *what* not
*where* (`import utils` — which file?), and bare paths like `setup.py` recur across
repos, a Python corpus must contain **exactly one repo** or link targets are
ambiguous. `data/make_repo_corpus.py` carves one repo out of a pretokenized dataset.

### The `LinkDetector` contract a new language must satisfy
(from `model/graph_traversal/link_detector.py`)

```python
class LinkInfo(NamedTuple):
    link_end_pos: int   # EXCLUSIVE token pos just after the reference; grant starts here
    target_str: str     # decoded target identifier; matched against index_doc_span(span)

class LinkDetector(Protocol):
    def __init__(self, decode_fn: Callable[[List[int]], str]): ...
    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]: ...   # 1-D [seq_len]
    def index_doc_span(self, span: Any) -> str: ...                          # corpus lookup key
    # optional: detect_links_for_doc(span_tokens, raw_identifier) -> List[LinkInfo]
    #           (span-local positions; used by Python for relative imports)
```

### The ~5 touch-points to add a language `<lang>`
1. `data/document_sources.py` — a `DocumentSource` yielding `(normed_id, content)`.
2. `data/pretokenize_<lang>.py` — thin wrapper picking that source (mirrors
   `pretokenize_stack.py`).
3. Graph extraction — language-specific import extraction + module→file resolution
   (new module under `data/<lang>_graph_extractor/` or a parametrized extractor).
4. `model/graph_traversal/<lang>_..._detector.py` — a `LinkDetector`, plus one
   entry each in `make_link_detector` and `LINK_DETECTOR_NAMES`
   (`link_detector.py:75,95`).
5. `data/layout.py` — one entry in `_DETECTOR_INFERENCE_LAYOUT`
   (`inference_layout_for_detector` raises on unknown detectors, so this can't be
   silently skipped).

Reused **unchanged**, language-agnostic: the sharding/writer core
(`data/pretokenize.py`), `data.dataset.GraphIndex`/`PretokShardedBackend`,
`data/split_graph.py`, `data/make_repo_corpus.py`, all of `cross_doc_mask.py` and
the mask kernels.

---

## 2. The reward-hacking problem, precisely

Any single scalar we might optimize is gameable:

| Naive metric | The hack that scores perfectly | What it produces |
|---|---|---|
| link **resolution rate** (`resolved/detected`) | emit only links you're sure resolve; emit nothing → 0/0 | sparse or empty graph, "100%" |
| **edge count** | emit every plausible pair | dense garbage graph, hallucinated edges |
| detector==extractor **agreement** | same agent authors both → shared blind spot passes | subtly wrong, self-consistent |

The defense is **precision AND recall measured against an independent oracle the
implementing agent did not author**, plus structural invariants, plus a mandatory
human read of real batch content. No single number; a panel that closes each hack.

### Two axes: DETECTION vs. RESOLUTION (the key distinction)
Turning an import into an edge is two steps, validated by **different** oracles:

- **Detection** — find the import and extract its target *string* (`"github.com/x/y/pkg"`).
  "What does this file import?"
- **Resolution** — map that string to the *right* node in the corpus. "Which
  document does it point to?"

Tree-sitter answers **detection only** — it parses syntax and knows nothing about
the corpus, so it cannot say whether a target string resolves, or resolves
*correctly*. Resolution is the harder, more dangerous axis: a **mis-resolved** link
is a false edge in the training graph — arguably worse than a missing one. So each
axis needs its own oracle.

### Are the LLMs allowed to use tree-sitter? YES — and it's not circular
Language agents **should** use tree-sitter for the build-time extractor (hand-rolled
regex like the old Python code would be a worse wheel). The apparent circularity
(oracle = tree-sitter, impl = tree-sitter) is not a real hole:
1. The oracle grades on **random real Stack files the agent doesn't curate** — you
   can't teach-to-the-test when the test set is 50 unseen files.
2. **We author and freeze the oracle's tree-sitter query ourselves**, independently
   of the agent's; a query that misses an import form disagrees on real files.
3. The genuinely independent checks are **resolution** and the **token-space runtime
   detector** — neither is tree-sitter (the detector only has tokens, so it's a real
   second implementation).

### Layered oracle (strongest → broadest)
| Axis | Oracle | Coverage |
|---|---|---|
| Detection (find imports) | frozen tree-sitter query (ours) on random real files | strong, near-complete |
| Resolution (string → right node) | **language toolchain** (`go list`/`go/packages`, `cargo metadata`, `javac -Xlint`) on a *buildable* subset | gold, but narrow — Stack files rarely build standalone |
| Resolution (broad) | hand-labeled fixture repos + structural invariants (target exists, no dangling, no self-link, identifier round-trip self-consistent) | broad, shallow |
| Both, final | human sample-dump spot-check of `(import → resolved target)` pairs | judgment |

**Honest limitation:** resolution precision/recall cannot be fully auto-oracle'd on
arbitrary Stack code without the toolchain, and the toolchain needs buildable code.
So resolution leans on fixtures + invariants + the human gate more than detection
does. This is called out, not papered over.

Tree-sitter is a **new dependency** (`tree_sitter` + per-language grammar wheels;
none currently installed — verified). Where a grammar is weak, fall back to the
language toolchain or hand-labeled fixtures and say so explicitly.

---

## 3. The test harness (stage 2) — what gets built

A **frozen, language-agnostic conformance harness** that grades a language
implementation. "Frozen" = the harness is written and reviewed BEFORE any language
agent runs, lives behind tests, and an implementing agent may not modify it (only
add a small per-language adapter that declares fixtures + the tree-sitter grammar
name). This is what makes the gate un-gameable.

### 3a. Per-language conformance suite (automated, checkpoint-free)
Two scoring blocks, one per axis (§2), on a held sample of real files from that
language's Stack subset:

**DETECTION block** (oracle = frozen tree-sitter query):
1. **Extractor vs. tree-sitter** — precision/recall of the build-time extractor's
   import strings vs. tree-sitter's import set on the raw source text. Draft gate:
   recall ≥ 0.90, precision ≥ 0.95. Report exact false-pos / false-neg examples.
2. **Runtime detector vs. tree-sitter** — same, for `detect_links` on decoded
   tokens. This is the genuinely-independent second implementation (token-space, no
   tree-sitter). Both stages checked against the third-party oracle → closes the
   shared-blind-spot hole.

**RESOLUTION block** (oracle = toolchain on buildable subset + fixtures + invariants):
3. **Resolution correctness on fixtures** — hand-labeled fixture repos with known
   ground-truth edges; score precision/recall of `(target_str → resolved node)`
   against the labels. This is the axis tree-sitter CANNOT check.
4. **Toolchain cross-check (where buildable)** — on the subset of fixture/Stack code
   that actually builds, compare resolved edges to the language toolchain's own
   dependency output (`go list`/`go/packages`, etc.). Gold standard, narrow coverage.
5. **Structural invariants** (cheap, always on):
   - no self-links; every `outgoing` target exists as a node (no dangling edges);
   - `link_end_pos` lands in decoded text immediately after the real import
     statement (spot-verified by re-decoding around the position);
   - DAG guard honored; `index_doc_span(node)` and `detect_links` `target_str`
     occupy the same string space (round-trip: every real edge's target key is
     reachable from some emitted `target_str`).
6. **Resolvability audit** — fraction of emitted `target_str` that resolve via
   `PretokCorpus._resolve_target`, reported ALONGSIDE precision/recall (never
   alone), plus degree distribution, dangling rate, isolated-node fraction.

### 3b. General dataset auditor (fills a real gap; reusable for Python too)
A checkpoint-free CLI that points at any built `pretokenized_datasets/<name>` dir
and reports: node/edge counts, out/in-degree distribution, dangling-edge rate,
self-link rate, isolated-node fraction, and resolvability of a sampled set of
detected links. This does not exist today (extractor stats are build-time only and
per-dataset bespoke). It doubles as the Python-corpus sanity check.

### 3c. Link sample-dump tool (feeds the human gate)
A CLI that prints N random `(source doc, detected link surface text, resolved
target identifier, target snippet)` tuples from a built dataset — no model needed.
Complements `demo_traversal.py` (in-batch connectivity) and
`visualize_llm_input.py` (per-doc layout + detected links). This is the artifact a
human skims to confirm links point where they claim.

### 3d. Mandatory human visual gate (final, non-automatable)
For each language, before its data is accepted, a human runs
`visualize_llm_input.py` / `visualize_epoch.py` on real packed batches and reads
the decoded content: do the linked docs actually co-pack, does the link surface
text sit where the tool says, does the target doc make sense as the import target?
The harness produces a ready-to-review artifact bundle; a human signs off. **No
language ships on green automated metrics alone.**

---

## 4. Language plan

Ranked by how cleanly intra-repo file→file resolution maps onto the model.

### Pilot: **Go** (stage 3)
- Import strings are **globally unique by design**: modern Go (module mode,
  universal since ~2019) has **no relative imports** — even a same-repo sibling
  package is imported by its full module-qualified path
  (`import "github.com/owner/repo/internal/foo"`). The string is a namespace
  declared in `go.mod`; nothing hits the network at parse time. "Global" = the
  string is globally unique, not a live fetch.
- **Single-repo resolution** = strip the `go.mod` module prefix, remainder is the
  package dir. **Multi-repo** = the full string is already the unique key, no strip.
- **Enables multi-repo corpora** (cross-repo edges), since paths never collide.
  This removes Python's single-repo eval bottleneck and yields a much denser graph.
- **PILOT DECISION RESOLVED (2026-07-19): packages are the node unit for Go.**
  Settled empirically, not by preference. Inspecting real repos (httprouter,
  logrus): (1) files in the SAME directory share one `package` and do NOT import
  each other — Go has no file→file intra-package import; (2) every intra-repo
  import references a *directory* under the module path
  (`github.com/sirupsen/logrus/hooks/writer`), never a file. So a file-node model
  has no natural Go edge at all. A Go NODE = one package (all non-test `.go` files
  in a directory, concatenated); `raw_identifier = "<module>/<pkgdir>"` (the full
  import path). This is the one structural difference from Python's file-nodes, and
  it makes the detector trivially clean (next bullet).
- **Detector design (consequence):** because Go imports are unambiguous full paths,
  the detector emits the **raw import path as `target_str`** with NO candidate
  expansion (contrast Python's `foo/bar.py` + `foo/bar/__init__.py`).
  `index_doc_span(node)` returns the node's full import path. Match = exact string
  equality → also makes the multi-repo relaxation trivial later (no prefix strip
  needed; the full path is already globally unique).
- **Single vs. multi-repo:** start single-repo per the §4 decision (identifier is
  the full import path either way; single-repo just means one module per corpus).
- Best possible showcase and lowest-risk template for the fan-out.

### Then, in order
- **Java / Kotlin** — `package` == directory; imports are fully-qualified class
  names (`com.google.common.collect.ImmutableList`) mapping deterministically to a
  file path, near-globally-unique → also **multi-repo capable**.
- **Rust** — `mod`/`use` tree rooted at the crate; needs mod-declaration graph
  walking (not pure path convention). **Single-repo.**
- **TypeScript / JavaScript** (stretch) — relative imports (`./foo`) are clean and
  common, but full resolution (index files, extension inference, `package.json`
  exports, `node_modules`) is messy. **Single-repo**, relative imports only at
  first; high volume in The Stack.

### Corpus scope decision (per language) — RESOLVED
Approach: **start single-repo (identical to Python, lowest risk), make multi-repo
a configurable relaxation.** Confirmed that multi-repo is compatible with the
existing precompute partitioner (see §7), so the relaxation is safe where import
strings are globally unique.
- **Multi-repo capable** (relax after pilot): Go & Java/Kotlin — unique import
  strings make cross-repo edges unambiguous → denser graph, no single-repo carve.
- **Single-repo only**: Rust and TS/JS relative imports (path-relative → ambiguous
  across repos, same as Python).
- Multi-repo requires the language's `index_doc_span` to key on the **globally
  unique import string** (full module path), not a bare repo-relative path, AND the
  precompute dispatcher fix in §7.

---

## 5. Fan-out plan (stage 4)

Once harness + Go pilot are proven:
- One implementation sub-agent per approved language, **each in its own git
  worktree** (`isolation: worktree`), producing a **branch/PR** — not committing to
  a shared tree. Rationale: the *shared* files agents must edit —
  `link_detector.py`'s registry + `data/layout.py`'s map — are exactly where they'd
  collide; isolation + PR review avoids conflicts and gives a per-language human
  checkpoint that dovetails with the mandatory visual gate. (Per-language files —
  each detector, `pretokenize_<lang>.py`, extractor module — don't overlap.)
- Agent deliverable = the ~5 touch-points + a per-language harness adapter
  (fixtures + grammar name) — **not** any change to harness scoring code.
- Acceptance = harness green (detection + resolution precision/recall, invariants,
  resolvability report) **AND** human visual sign-off on real batches, reviewed at
  PR time.
- Launch/build discipline per CLAUDE.md (stagger jobs; artifacts to `/fss-data`,
  never `/fss` for bulk I/O).

---

## 6. Resolved decisions (from review 2026-07-17)
1. **Tree-sitter dependency — APPROVED.** Add `tree_sitter` + per-language grammar
   wheels to the env.
2. **Thresholds — APPROVED** as draft: recall ≥ 0.90 / precision ≥ 0.95, tunable
   per language. Recall denominator counts **only in-corpus targets** (external
   deps legitimately don't resolve and are excluded).
3. **Go data — none exists yet; downloader is trivially retargetable.** The Stack
   record schema is fully language-agnostic (`lang`, `ext`, `max_stars_repo_name`,
   `max_stars_repo_path`, `content`), organized by `data_dir="data/<lang>"`. The
   current `download_sample.py` hardcodes `data/python`; getting Go = parametrize
   that to `data/go`. `StackJSONLSource` barely changes. (Verified 2026-07-17.)
   TODO before pilot: confirm Go volume + `go.mod` presence in the downloaded slice.
4. **Extraction approach — RESOLVED: tree-sitter in three roles.** The two stages
   already consume different representations, so there's no tokenization conflict:
   - **Build-time extractor** consumes **raw source text** → use tree-sitter as the
     extraction engine (replaces hand regexes; strictly more correct).
   - **Runtime `LinkDetector`** consumes **tokens** (all it has at train time) →
     keeps its own lightweight token-space logic; can't run tree-sitter directly.
   - **Oracle** = tree-sitter, grading BOTH stages.
   The runtime detector stays simple *because* it is the checked thing, not the
   checker — testability preserved, extractor quality improved.
5. **Multi-repo — start single-repo, relax later (configurable).** See §4 and §7.

## 7. Precompute partitioner + multi-repo (investigated 2026-07-17)
Verdict: **the Voronoi partitioner is structurally repo-agnostic; multi-repo code
graphs are compatible with it.** Details, all in `data/epoch_precompute.py`:
- Two partitioners: `_partition_repos` (round-robins whole repos to workers,
  assumes edges stay in-repo — the single-repo path) and
  `_partition_graph_communities` (multi-source BFS **Voronoi**, used by wiki/arxiv,
  co-locates linked docs; **never inspects identifiers**, only `neighbors_out/in`).
- A multi-repo code graph with cross-repo edges would partition **identically to
  arxiv/wiki** under the Voronoi path — the worker view, component assignment,
  bucketing, and parallel driver are all repo-agnostic.
- **The one trap:** the dispatcher `_is_repo_partitioned` returns True whenever
  identifiers contain `:` (which `owner/repo:path` does), so it would *mis-route* a
  multi-repo corpus to `_partition_repos`, severing cross-repo edges and collapsing
  packs toward `doc_causal`. **Fix is one line** — prefer the Voronoi path when
  cross-repo edges exist (or key the heuristic on something better than `:`). No
  algorithmic work; the Voronoi code is unchanged.

## 8. Data-availability probe (run 2026-07-18)

Streamed ~3000 files/language from `bigcode/the-stack-dedup` (per-language
`data_dir`). **Python baseline** (from existing `graph_100M_stats.json`): 3.56M
nodes, 6.3M internal edges, ~7.3 files/repo, avg out-degree 1.78, ~4.0 imports/file.

| Lang | imports/file | % files w/imports | chars/file | local-ish imports |
|---|---|---|---|---|
| Go | ~7.6 | 92% | 7772 | (all module-qualified) |
| Java | ~9.5 | 88% | 5820 | (all FQN) |
| Rust | ~5.0 | 86% | 7500 | ~35% (`crate::`/`super::`/`mod`) |
| TypeScript | ~4.0 | 83% | 3878 | ~48% (`./`,`../`) |
| JavaScript | ~2.5 | 60% | 10019 | ~45% (`./`,`../`) |

**Takeaways:**
- **Go & Java have the richest import density** (7.6 / 9.5 imports/file, ~90% files
  with imports) — both beat Python's ~4.0. Strong graph potential. Go confirmed as
  the pilot.
- **Rust/TS/JS**: only 35–48% of imports are local (intra-repo); the rest are
  external deps that legitimately won't resolve — reinforces the §6.2 decision to
  count only in-corpus targets in the recall denominator.
- **METHODOLOGY CAVEAT:** streaming reads the dataset *head*, which is NOT
  repo-ordered, so a stream window shows ~1.1 files/repo — this is a **sampling
  artifact and does NOT measure real co-location.** The pipeline hash-partitions the
  *entire* dataset into repo buckets (Python's final graph averages ~7.3 files/repo).
  Actual multi-file-repo density can only be measured after a full
  `build_graph_streaming` pass. Import *density* and file *size* above ARE reliable
  from streaming; repo co-location is NOT.
- **`go.mod` files are sparse in a raw stream** (0 in first 4000) — expected, since
  they're 1 per repo. The extractor must read the module prefix from `go.mod` when
  present and fall back to a heuristic (longest common import prefix) when absent in
  the sampled slice. Flag for the Go pilot.

## 9. Companion task: Python re-implementation on the harness
Once the harness exists, one agent **rewrites the existing Python extractor +
detector to use tree-sitter** and pass the same conformance suite. Value: (a)
validates the harness against a known-good language, (b) removes the last
hand-regex extractor, (c) makes Python a peer of the new languages rather than a
special case. This is a fan-out task like any other (own worktree + PR), but should
run FIRST (or alongside Go) as the harness's reference conformance case.
