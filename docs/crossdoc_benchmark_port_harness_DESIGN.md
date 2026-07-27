# Benchmark-port verification harness — design (2026-07-24)

Goal: when sub-agents build the three external benchmark ports (Go←CoLT-132K,
TS←CrossCodeEval, Kotlin←ASE-2025), a FROZEN harness — authored before/independent
of the ports — judges whether each port is "as legit as" the existing
python/java `run_repobench_cross_doc`. Same philosophy as `data/graph_harness`:
the implementer authors only a thin adapter; all scoring/gating logic is frozen
and shared, and ground truth comes from tree-sitter, not from anything the
implementer wrote.

## What made python/java legit (the properties to enforce)

1. High **link fire-rate**: java 470/500 (94%) examples had ≥1 import resolve to
   an aux snippet. A port whose detector rarely matches silently degenerates to
   flat scoring and reports a meaningless "cross-doc" number.
2. **Precise grants**: each import grants attention only to the snippet it
   actually references (aux_raw_identifiers path), not blanket access.
3. **Discriminating paired signal**: Δnll = flat_linked_only − cross_doc_only
   positive on trained cross_doc_link ckpts (java final: +0.031..+0.079).
4. Fair pairing: flat and cross-doc conditions score the SAME completion tokens
   of the SAME examples.

## Canonical port output schema (frozen)

Every port adapter converts its upstream dataset into this one dataclass; the
harness and the runner consume ONLY this:

```python
@dataclass(frozen=True)
class CrossDocExample:
    repo: str
    file_path: str            # repo-relative path of the primary file
    context: str              # primary-file prefix INCLUDING import block
    target: str               # completion to score (next line / statement / FIM-middle first line)
    aux: List[AuxDoc]         # [{path: str (repo-relative), content: str}]
    meta: Dict[str, Any]      # upstream ids, provenance, target kind
```

The port also declares which `<Lang>ImportDetector` backs it. Scoring then goes
through the EXISTING language-agnostic `score_completion_with_context_docs` +
`_repobench_aux_identifier`-style per-language identifier shaping — the port's
only real logic is (a) upstream-schema mapping and (b) the aux-identifier shaping
function (the java source-root-strip analogue).

## Harness tiers (all frozen, in `eval/benchmark_harness/`)

### Tier 0 — schema + mechanical invariants (no GPU, runs in CI)
- Non-empty target; context non-empty; aux list non-empty; aux paths
  repo-relative (no leading `/`, no `..`); no duplicate aux paths per example.
- **Imports visible**: language's import construct appears in `context`
  (tree-sitter parse, not regex). For CCEval also assert
  `metadata.context_start_lineno == 0` on ported examples (else import block
  was cropped upstream — drop, don't guess).
- **Token-accounting parity**: the flat pair and the cross-doc pack encode the
  identical completion token ids (harness re-encodes and compares).
- Determinism: two adapter runs produce byte-identical example lists.

### Tier 1 — link-resolution oracle (CPU, reuses `data/graph_harness` specs)
Ground truth comes from the frozen per-language `LanguageSpec` (already exists
for all 9 languages, authored + validated during dataset builds):
- Oracle extracts canonical import keys from `context`.
- Harness independently computes which aux paths are *reachable*: project each
  aux path through `canonical_target` and intersect with oracle keys.
- Compare against what the port's detector+identifier-shaping actually matched
  (run the real `link_detector.detect_links` + index_doc_span path):
  - **match precision** ≥ 0.95: everything the port matched is oracle-licensed
    (a grant to a snippet the file does NOT import is benchmark corruption);
  - **fire-rate parity**: port fire-rate ≥ 0.9 × oracle reachable-rate
    (the port shouldn't lose many resolvable links to identifier-shaping bugs —
    this is exactly the class of bug the java source-root strip fixed).
- ~~Target-uses-aux identifier check~~ — REJECTED 2026-07-25: syntactic
  "identifier defined in granted snippet" is fragile; real dependencies can be
  conceptual (patterns, invariants, config values) with no in-line symbol
  match. Whether targets genuinely depend on aux content is instead validated
  end-to-end by the Tier-2 placebo control, plus a deferred LLM
  conceptual-dependency audit over ALL benchmarks incl. python/java (TODOS.md
  2026-07-25).

### Tier 2 — model-based end-to-end (GPU, one trained cross_doc_link ckpt/lang)
We have final sweep ckpts for go, typescript, kotlin (2026-07 sweeps).
- `n_cross_doc ≥ 200` and fire-rate ≥ 0.5 hard floor (python/java are ~0.9;
  report distance to that band).
- **Paired Δnll with bootstrap 95% CI** (per-example pairing, 10k resamples).
- **Placebo control (the key legitimacy test)**: re-score with aux snippets
  SHUFFLED across examples (wrong snippets, matched count/length distribution).
  Gate: `Δnll_real − Δnll_placebo > 0` with CI excluding 0. This proves the
  benchmark measures genuine cross-file dependence rather than "any extra
  in-language context helps" or a scorer artifact. (Run on python/java first to
  calibrate the expected placebo gap.)
- Determinism: same numbers on re-run (fixed seeds; compile cache warm).

### Tier C — dedup HARD GATE (CPU) [decision 2026-07-24: exclude, don't just report]
A repo may appear in the training dataset OR the benchmark, never both.
- **Repo-name intersection (hard)**: intersect benchmark repo names with our
  per-language raw-shard `max_stars_repo_name` values (shards on
  `/fss-data/.../raw/<lang>/shards/`). Benchmark examples from overlapping
  repos are EXCLUDED from the port (dropped at adapter build time; the harness
  re-verifies the shipped port has zero overlap).
- **File-hash pass (hard)**: for the surviving examples, hash-match
  (normalized-content SHA1) each primary + aux file against the training
  corpus to catch files copy-pasted BETWEEN repos (vendored deps, forks under
  different names). Any example whose primary file hash-matches training is
  dropped; an aux-only match drops just that aux doc (and the example too if it
  loses all import-licensed aux).
- Harness reports how many examples each pass removed and how many remain
  (gate: enough survivors for Tier-2 power, `n_cross_doc ≥ 200`).
- If exclusion guts a benchmark (too few survivors), the alternative is the
  inverse direction: blacklist the offending repos in the DATASET pipeline and
  rebuild/retrain — filed as a deferred TODO (TODOS.md 2026-07-24), since
  retraining all sweep ckpts is expensive. Until then exclusion happens on the
  benchmark side only, which keeps current ckpts usable.
- Rerun the intersection against Stack-v2 repo lists before any future v2
  retrain (CrossCodeEval's 2023 repos are inside v2's crawl window).

## Calibration results (2026-07-25, harness implemented in eval/benchmark_harness/)

500 examples/port; Tier 1 fire-rates exclude relative-import recovery (runtime-only,
excluded on both sides). Tier 2 ckpts: python = run_20260720_063128_690228 (bfs
cross_doc_link), java = run_20260722_191916_590119.

| port | T0 | T1 precision | T1 fire | T1 oracle-reach | T2 fire | Δnll_real (CI) | placebo sep (CI) |
|---|---|---|---|---|---|---|---|
| repobench_python | PASS | 1.000 | 0.618 | 0.660 | 0.870 | +0.094 (0.072..0.117) | +0.127 (0.105..0.150) |
| repobench_java   | PASS | 1.000 | 0.940 | 0.996 | 0.940 | +0.072 (0.053..0.091) | +0.086 (0.068..0.107) |

New ports (2026-07-25, built by sub-agents against the frozen harness):

| port | T0 | T1 prec | T1 fire/reach | T2 fire | Δnll_real (CI) | placebo sep (CI) | verdict |
|---|---|---|---|---|---|---|---|
| ase_kotlin | PASS | 1.000 | 0.996/1.000 | 0.979 | **−0.056 (−0.134..+0.011)** | −0.033 (−0.099..+0.019) | **T2 FAIL** — CI includes 0 |
| crosscodeeval_ts | PASS | 1.000 | 0.400/0.480 (static, advisory) | 0.400 | +0.013 (−0.003..+0.031) | **+0.023 (+0.007..+0.040)** | T2 near-miss (see below) |
| colt_go | — | — | — | — | — | — | BLOCKED (empty aux upstream) |

**Kotlin T2 negative (ckpt run_20260724_095209_785799, 242 ex, 237 fired, 4
oversized-skipped):** aux made next-line prediction slightly WORSE on average
and placebo separation is negative-CI-crosses-0. The harness worked exactly as
designed — it caught a port that does NOT demonstrate genuine cross-file
dependence on this checkpoint. Candidate causes (NOT yet disambiguated, do not
assume): (1) the Kotlin sweep ckpt may exploit cross-doc links weakly vs the
python/java ckpts; (2) ASE-2025 `middle`-first-line targets are arbitrary FIM
spans, NOT RepoBench's "first USE of an imported symbol" — so they may not be
import-dependent; (3) whole-file aux dilution / 32k truncation. This is a real
result to surface, not iterate away silently.

**TS T2 near-miss (ckpt run_20260722_003634_268441, 500 ex, 200 fired):**
Δnll_real +0.013 CI (−0.003..+0.031) barely includes 0, BUT placebo separation
+0.023 CI (+0.007..+0.040) EXCLUDES 0. Interpretation: the right retrieved
chunk beats a wrong one significantly (the discriminating placebo signal fires),
but the absolute cross-vs-flat gain is small and underpowered — expected,
because the shipped aux are retrieval CHUNKS that often lack the imported symbol
body, and only 200/500 fire under static grants. The placebo-positive result is
the encouraging half: it says the benchmark IS sensitive to context correctness.
Path to a clean pass: the v2 whole-file re-clone (from metadata.repository) +
runtime relative-import resolution to lift fire-rate from 0.40 → ~0.67, which
also raises n and should tighten Δnll_real CI above 0.

## Target-scope ablation (added 2026-07-25, `scopes.py`)

Motivated by the Kotlin miss + user input: the import LINE is uninteresting; the
signal lives at later USES of imported symbols. Tier 2 takes a `--scope`
(`native` | `use_line` | `use_block` | `rest_of_doc` | `all`) that re-anchors
scoring at the first line USING a symbol declared in a granted aux doc. Context
is rebuilt as the full-file prefix up to that use site and held IDENTICAL across
the three use-scopes, so only the scored width varies:
- `use_line` — the single logical statement at the use site (≈ RepoBench next_line).
- `use_block` — use site → end of enclosing syntactic block (tree-sitter). User's
  preferred boundary; the likely sweet spot vs single lines being "too small".
- `rest_of_doc` — use site → EOF (whole-doc-after-use; expect signal dilution).

"Uses an imported symbol" is decided WITHOUT import-syntax parsing: the aux docs
ARE the resolved imports, so we take the top-level names DECLARED in the aux
(tree-sitter) and match the first completion-region line referencing any — which
dissolves the `from x import *` problem (a star import's names = the aux's
declarations) and needs only a per-language top-level-declaration node set
(`_DECL_NODE_TYPES` in scopes.py), not an import grammar. RepoBench has no
post-hole file body (`all_code` is just the license header), so its use-scopes
collapse to use_line — a built-in validation (use_line should reproduce native
+0.094). The real multi-line ablation runs on ASE-2025/Kotlin (whole files).

Ports without full_file (CCEval ships only left-context+groundtruth) support
`native` only; scope_example returns None → dropped for use-scopes.

## Verdict summary (2026-07-25)
Harness + calibration COMPLETE and committed. Of the 3 external ports:
- **Kotlin/ASE-2025**: builds + passes CPU gates, but FAILS Tier 2 — no cross-doc
  benefit on the current ckpt. Needs cause diagnosis before it can be a headline
  benchmark (ckpt strength vs target import-dependence vs aux dilution).
- **TS/CrossCodeEval**: builds + passes CPU gates; Tier 2 near-miss with a
  POSITIVE placebo signal. Promising; needs the v2 whole-file upgrade for a clean
  pass.
- **Go/CoLT-132K**: BLOCKED upstream (empty aux in released data).
The harness itself is validated: it PASSED the two known-good references and
correctly withheld a pass from all three unproven ports. Next actions filed in
TODOS.md.

ALL GATES PASS on both reference ports. Notable: placebo separation EXCEEDS
Δnll_real on both — wrong-but-plausible aux actively hurts vs flat. The
benchmarks demonstrably reward the right cross-file context, not extra
in-language tokens. Legitimacy band for new ports: T1 precision ≈ 1.0,
T2 fire ≥ 0.87, Δnll_real CI > 0, placebo sep CI > 0.

## Calibration = run the harness on python/java FIRST

Before any new port is accepted, run all tiers on the existing python/java
RepoBench ports with their final ckpts. Their tier metrics define the
"legitimacy band" (fire-rate ~0.9+, placebo gap > 0, precision ~1.0). A new
port passes when every hard gate passes and its report is human-reviewed
against that band. This mirrors how graph_harness used python (the trusted,
hand-validated language) to sanity-check the framework itself.

## Sub-agent workflow (when we dispatch the builds)

Per language, the builder agent gets: the upstream dataset location, the
CrossDocExample schema, the existing detector, and the harness CLI
(`python -m eval.benchmark_harness.run_port_audit --language go --port ...`).
Acceptance = harness report all-green + calibration table. The agent iterates
against Tier 0/1 locally (fast); Tier 2 runs once on a GPU node at the end.
Builder agents may NOT touch `eval/benchmark_harness/` (same frozen-oracle rule
as graph_harness).

## Per-port notes (from the 2026-07-24 survey)

- **Go / CoLT-132K — BLOCKED (verified 2026-07-25)**: the released
  `CoLT-132K.zip` has EMPTY `cross_file_dependency` in all 3,000 Go test rows
  (Python ~6.8k entries/1k, Java ~128k/1k). The dependency records live in
  external `godata` JSONs (`dependency_file_path`) NOT shipped in the zip.
  `prefix` DOES include the import block (929/1000 parseable) — the survey's
  cropped-prefix risk was NOT the problem; the aux docs themselves are absent.
  Adapter `ports/colt_go.py` is written correctly and will work IF the
  dependency JSONs are recovered (email authors), else Go falls to the
  self-built path (like Kotlin/ASE: mine aux from repo snapshots). NOT
  registered/committed until unblocked.
- **TS / CrossCodeEval (built 2026-07-25, `ports/crosscodeeval_ts.py`)**: PASS
  Tier 0; precision 1.000; but Tier 1 STATIC fire-rate 0.400 < parity gate
  (0.9×0.480). Root cause (agent-verified, NOT a shaping bug): Tier 1 matches
  via stateless `detect_links` specifier keys, so relative imports through a
  subdir (`./sub/x`) or parent (`../x/y`) — 5.6% of examples reachable only
  that way — cannot resolve without the importing file's directory. Resolving
  specifiers against `ex.file_path` (what `score_completion_with_context_docs`
  does at RUNTIME via source_file_path) lifts fire-rate to 0.674 on the same
  500. So Tier 2 is the arbiter for TS; the static Tier-1 gate structurally
  under-fires on relative-import languages. Identifier shaping = ext-stripped
  basename + directory-index refinement (`src/foo/index.ts`→`foo/index`).
  0/3356 examples had cropped imports (context_start_lineno all 0). aux remain
  retrieval CHUNKS (v1); v2 = re-clone repos from `metadata.repository` for
  whole-file aux. Tier C: 500/500 survive (2023 repos disjoint from v1 by date).
- **Kotlin / ASE-2025 (built 2026-07-25, `ports/ase_kotlin.py`)**: PASS all CPU
  gates — Tier 0 PASS (242 examples of 430 datapoints; ~56% yield, rest resolve
  to zero cross-file siblings), Tier 1 precision 1.000 / fire 0.996 /
  oracle-reach 1.000 (dead in the reference band). Reused
  `_strip_java_source_root` unchanged (already had kotlin roots); file-path
  matching only (Java-style), NOT symbol→file — a file whose name ≠ imported
  symbol dotifies to an unemittable key, so including it would be dead context.
  target = first line of `middle` with NO prepended `\n` (prefix cuts mid-line,
  unlike RepoBench's fresh-line next_line). Tier C: 235/242 survive (0 repo
  overlap, 7 file-hash cross-repo copy-pastes). Data:
  /fss-data/.../raw/ase2025_kotlin/ (practice+public, 20 repos, ~2.4G; private
  split skipped — no public ground truth).

**Tier-1 gate limitation (noted 2026-07-25):** the static fire-rate-parity gate
structurally under-fires on RELATIVE-import languages (TS/JS/Dart/Zig) because
`detect_links` keys are stateless specifiers and subdir/parent relative imports
need the importing file's directory to resolve. FQN/absolute-import languages
(python/java/kotlin/go) are unaffected (kotlin fires 0.996). For relative-import
ports, treat Tier 1 fire-rate as ADVISORY and Tier 2 (runtime source_file_path
resolution) as authoritative. Precision stays a hard gate for all. Deferred
option: add a Tier-1 runtime-resolution mode mirroring
score_completion_with_context_docs' relative path handling.

## Dedup / contamination decision (settled 2026-07-24)

Training data today = bigcode/the-stack-dedup (v1; content cutoff ≈2022).
Possible future: retrain on Stack v2 (content cutoff ≈2023-09).

| Benchmark | vs v1 (current ckpts) | vs v2 (future retrain) |
|---|---|---|
| CoLT-132K (repos created ≥2024-03) | clean BY DATE | clean BY DATE |
| CrossCodeEval (repos created 2023-03..06) | clean BY DATE | Tier-C gate will bite |
| ASE-2025 Kotlin (repos can be OLD; only the target commit is recent) | Tier-C gate will bite | Tier-C gate will bite |

**Policy (user decision 2026-07-24): hard exclusion, not report-only.**
Repo-name intersection is a hard criterion (a repo lives in the dataset or the
benchmark, not both); file-hash matching then removes cross-repo copy-pastes
(vendored deps / renamed forks). Full substring/MinHash near-dedup remains out
of scope — date-cutoff + repo-intersection + exact-hash matches the field
standard (RepoBench, CrossCodeEval, CoLT). Exclusion is applied on the
BENCHMARK side for now so current ckpts stay valid; the inverse
(dataset-side blacklist + rebuild + retrain) is filed as a deferred TODO.
