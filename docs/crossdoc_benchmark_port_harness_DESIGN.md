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

- **Go / CoLT-132K**: verify on download that `prefix` starts at file top
  (imports visible) — UNVERIFIED in survey. Aux `abstraction` is a signature
  skeleton, not the full file: Tier-1 target-uses-aux must tolerate
  declaration-only aux bodies. 3 scenario types — port only the cross-file
  API-invocation scenario first (the dependency-based one).
- **TS / CrossCodeEval**: aux are retrieval CHUNKS keyed by filename; a chunk's
  path can be import-licensed while the chunk text lacks the symbol. Two-stage
  plan: v1 accepts chunks (Tier-1 path-level checks only), v2 re-clones repos
  from `metadata.repository` for whole-file aux. Gate v1 on placebo separation.
- **Kotlin / ASE-2025**: we mine aux ourselves from full snapshots (best fit).
  Reuse java FQN machinery incl. source-root strip; note Kotlin files live
  under BOTH `src/main/kotlin/` and `src/main/java/` roots. Use `middle` first
  line as target; `modified` field filters self-referential context files.

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
