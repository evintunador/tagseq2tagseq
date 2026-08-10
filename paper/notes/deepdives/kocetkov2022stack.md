# kocetkov2022stack — The Stack: 3 TB of permissively licensed source code

Kocetkov, Li, Ben Allal, Li, Mou, Muñoz Ferrandis, Jernite, Mitchell, Hughes, Wolf,
Bahdanau, von Werra, de Vries (BigCode, 2022). arXiv:2211.15533.

This is OUR code corpus. We do not train on The Stack directly — we download
`bigcode/the-stack-dedup` (the near-deduplicated subset; see
`data/stack_sharded_download.py:62`, `_list_parquet` hits repo
`"bigcode/the-stack-dedup"`), extract a Python (and multi-language) import-dependency
graph from it, then pretrain on that graph. So the paper is both our data provenance
document AND a source of the dedup/contamination pitfalls that hit us downstream.

## What the paper actually does

The Stack is a code pretraining corpus assembled by the BigCode project as the
data foundation for SantaCoder / StarCoder. It is a data + governance artifact, not a
modeling paper; the modeling experiments are small ablations that justify the design.

**Scale and collection.**
- Repository names harvested from **GHArchive** over **2015-01-01 → 2022-03-31**:
  **220.92M unique repo names**. Cloning succeeded for **137.36M repos** (>62% clone
  rate); clones ran Nov 2021 – Jun 2022.
- Across all clones there were **51.76B files**, but only **~5.28B unique (~10%)** —
  exact duplicates were dropped at collection time via git blob hashes. Uncompressed
  stored bytes: **92.36 TB**.
- Released permissive dataset: **3.1 TB (3135.95 GB)** in **30 programming languages**.
  The all-license internal version is **~29.6 TB**. The Stack is billed as >3× the size
  of CodeParrot, the next-largest released code dataset. (Dataset card v1.1 later expanded
  language coverage to ~358–370 languages; the paper describes the 30-language v1.0.)
- Selection filters: binary files excluded; files >1 MB dropped unless the extension is
  on an approved language list.

**Licensing / provenance (the "permissive" claim).**
- License detected with **go-license-detector**. GHArchive already carried a license for
  26.4M repos; the detector was run on the remaining ~110.9M.
- **No license detected for >80% of repositories** — these are excluded from the
  permissive release. Top permissive licenses: **MIT (9.6%)**, **Apache-2.0 (2.7%)**,
  plus BSD-3/BSD-2, CC0, Unlicense, ISC. GPL/copyleft intentionally excluded. Net: only
  ~10% of the collected corpus is in the permissive release.
- Known erratum acknowledged in the paper: **MPL/LGPL/EPL were mislabeled as permissive**
  when they are weak copyleft (<0.5% of Python).

**Data governance / opt-out.** An **"Am I in The Stack"** HF Space lets developers search
for their code; removal is a manual-verification form (GitHub username/email) applied in
the next release, with a planned 3-month refresh cadence. **PII (names, emails) is present
and NOT removed** in v1.0 (deferred to future work). The corpus was **not scanned for
malicious code**.

**Deduplication.** Two stages:
1. *Exact*: git-hash dedup at collection (51.76B → ~5.28B unique files).
2. *Near-dedup* (produces `the-stack-dedup`): tokenize on non-alphanumeric chars, drop
   files with <10 tokens, then **MinHash (256 permutations) + LSH**, cluster at
   **Jaccard > 0.85**. In the permissive set, near-dups were **38.6% of files / 53.7% of
   volume**; permissive shrinks **3.1 TB → 1.4 TB (1450.75 GB)**, >50% reduction. (Paper
   does not state the LSH band/row split.)

**Modeling ablations (350M decoder, 24L/1024d/16h, seq 2048, 300K steps, ~235.9B tokens,
Megatron-LM fork), pass@1 / pass@100:**

| Subset | HumanEval p@1 | p@100 | MBPP p@1 | p@100 |
|---|---|---|---|---|
| Permissive, no dedup | 10.99 | 27.21 | 11.60 | 44.99 |
| Permissive, near-dedup | 13.94 | 37.00 | 15.94 | 54.69 |
| All-license, no dedup | 13.11 | 36.67 | 17.41 | 53.59 |
| All-license, near-dedup | 16.60 | 44.00 | 22.99 | 61.00 |

Headline findings: **near-dedup boosts every configuration** (often +3–10 pass@100),
and permissive-only + near-dedup roughly matches Codex/CodeGen HumanEval pass@100 (~35–37)
— i.e. you can hit prior reported numbers using only permissive data. This is the whole
reason we take `the-stack-dedup` rather than raw The Stack.

## How WE build a graph over it (per data_pipelines.md + source)

We treat each source file as a node and each intra-repo import as an edge.
(`data/github_graph_extractor/build_graph_streaming.py`,
`data/github_graph_extractor/extract.py`.)

- **Field mapping**: we keep `max_stars_repo_name`, `max_stars_repo_path`, `content`,
  `ext`, `lang` from the parquet (`stack_sharded_download.py:35`), and per-language
  extension filters mirror the sequential downloaders (`.py` for Python, `.rs`, `.ts/.tsx`,
  `.kt` not `.kts`, `.go`+`go.mod`, etc.). Minified JS bundles (`*.min.js`) are excluded.
- **Import extraction**: tree-sitter parse → imported module strings, minus a hardcoded
  ~40-entry stdlib/external denylist.
- **Module → file resolution** is *per repo*: build `module_to_file` (dotted path,
  `__init__.py` → parent package), then resolve each import with priority
  exact-`.py` → exact module name → prefix match. On ties, `_pick_from_candidates`
  falls back to **`random.choice` (unseeded)** — non-deterministic run to run
  (`build_graph_streaming.py:208-209`).
- **Intra-repo edges only.** Repos are hash-partitioned into 256 buckets
  (`_det_hash32(repo_name) % 256`); cross-repo imports are structurally impossible to
  represent. Node id = `{normalized_repo}:{raw_file_path}` (repo name hashed/normalized,
  file path kept raw).
- **The `links_in_repo >= 2` filter.** Any node whose (outgoing + incoming) intra-repo
  degree is <2 is dropped (`build_graph_streaming.py:512`). On the 100M-file sample this
  keeps **28.7%** of files (`graph_100M_stats.json`: 12,420,478 → 3,560,799 nodes;
  `graph_kept_pct = 0.2867`), yielding 6,335,322 edges, avg degree 1.78, across 486,148
  repos. Even post-filter, **34.1% of kept nodes have in-degree 0 and 24.9% have
  out-degree 0** (they survive on the *other* direction's links).

## What The Stack's construction implies for our graph

**1. Coverage is gated twice, and both gates bias toward big, machine-generated repos.**
The Stack already keeps only ~10% permissive files and >1 MB files are dropped; then our
`links>=2` filter discards a further 71%. The survivors are files embedded in repos with
dense internal import webs. The actual top-degree nodes in our 100M graph confirm the
skew — they are almost entirely **auto-generated SDKs and package `__init__.py` hubs**:

```
in-degree  1244  homeassistant/__init__.py
            962  pytglib/api/utils/__init__.py     (Telegram TDLib bindings)
            912  spark_auto_mapper_fhir/...        (generated FHIR types)
            880  pulumi_google_native/__init__.py  (generated Pulumi SDK)
            880  pulumi_oci/__init__.py
out-degree  787  observations/r/__init__.py
            754  py_tdlib/constructors/__init__.py
```

These are exactly the "package-heavy / generated SDK" repos data_pipelines.md warns about
(reviewer-attackable #5). The link-prediction / traversal signal our model learns is
therefore disproportionately *the topology of generated code*, not hand-written human
dependency structure. Any claim about "learning cross-document structure" should be
hedged: much of the high-degree structure is boilerplate re-export graphs.

**2. Near-dedup is done UPSTREAM, at the file level, on the wrong granularity for us.**
`the-stack-dedup` removes near-duplicate *files* at Jaccard>0.85. But (a) it dedups
independently of repo membership, so a repo can lose some files to dedup and keep others —
which silently *breaks import edges* in our per-repo resolver (`_resolve_import_to_file`
returns None when the target file was deduped away, converting a real dependency into a
dangling/zero-degree node, and possibly pushing the source below the `links>=2` cutoff).
(b) It does **nothing** about *whole-repo* near-duplication: forks, vendored copies, and
templated SDK generators (Pulumi, TDLib, FHIR) produce many near-identical *repos* whose
files differ just enough (different package name) to survive file-level dedup. Those are
precisely our top-degree hubs. So the dedup that helped the paper's HumanEval numbers does
NOT protect us from the graph-topology duplication that matters for our method.

**3. Our own dedup is weaker than theirs and only diagnostic.**
data_pipelines.md flags this (reviewer-attackable #4): our
`_content_signature` (blake2b over a whitespace-normalized 4 KB prefix) is computed on a
**1% sample and only REPORTED as `dup_signature_sample_dupe_rate`, never used to filter**
(`build_graph_streaming.py:565,662`). We inherit `the-stack-dedup`'s file-level dedup and
add nothing. There is **no cross-split near-dup check** — our community/random splits
(`split_graph.py`) partition by node membership, but two near-identical generated-SDK repos
can land in train and val respectively, leaking structure and content. Given The Stack's
own finding that dedup swings pass@100 by ~10 points, this is a real threat to any
held-out val-loss / traversal metric we report.

**4. Contamination against HumanEval/MBPP is unmanaged.** The Stack paper reports near-dedup
but does **not** decontaminate against HumanEval/MBPP (that came later, in SantaCoder/
StarCoder, which strip benchmark solutions). Since we pull from `the-stack-dedup` without
adding benchmark decontamination, if we ever evaluate on HumanEval/MBPP-style code tasks
our train set may contain the solutions. We should either decontaminate or avoid those
benchmarks for headline claims.

**5. Provenance/PII and reproducibility caveats we inherit.** (a) PII (names, emails) is
present in `the-stack-dedup` v1.0 — our nodes carry it verbatim. (b) The MPL/LGPL/EPL
mislabel means a sliver of our corpus is weak-copyleft, not permissive. (c) Our own graph
is **non-reproducible** run-to-run because import tie-breaks use unseeded `random.choice`,
compounding The Stack's own non-determinism (v1.0 vs later dataset refreshes / opt-out
removals change the underlying files). Pinning the exact `the-stack-dedup` revision is
necessary for any reproducibility claim.

**6. Language coverage mismatch.** The Stack's v1.0 volume leaders are HTML/JS/Java/C
(>55% of bytes); Python is a minority. We built extractors for python, go, java, javascript,
typescript, kotlin, rust, dart, zig (`data/*_graph_extractor/`), but the 100M graph we
actually analyzed is **~100% Python** (`.py` = 3,560,254 of 3,560,799 nodes). So our
current code-graph results generalize to Python's import conventions (dotted modules,
`__init__.py` packages) and may not transfer to languages with different dependency
mechanics (Go modules, JS bundlers, Java packages) even though the download plumbing exists.

## Predictions & open questions for our method
- **Near-dedup should help our val-loss / traversal metric, not just downstream code tasks.**
  The paper shows dedup improving *every* config; because our held-out community packs
  reuse intra-repo link structure, undeduped whole-repo clones would let the model memorize
  a target file it "traverses" to. Expect our cross-doc attention benefit to look *larger*
  than it is unless we add whole-repo/near-dup decontamination between splits.
- **The linking inductive bias should be strongest on hand-written repos, weakest on the
  generated-SDK hubs that dominate our high-degree tail.** Generated `__init__.py` re-export
  graphs are trivially predictable (flat star topology, boilerplate), so a link-fetch that
  pulls the target into context buys little there. If we bucket eval by "generated vs
  organic" repo, we should see the concat-vs-link gap widen on organic code.
- **Scaling caveat:** The Stack's 350M ablations used only ~236B tokens at seq 2048. Our
  32k-context regime and larger models mean the paper's absolute numbers do not transfer;
  what transfers is the *direction* (dedup up, permissive-only viable).
- **Open question our design could resolve:** The Stack dedups files in isolation and
  admits it cannot handle whole-repo duplication. Our repo-partitioned graph gives a natural
  unit (the repo subgraph) on which to define and measure *structural* near-duplication —
  a cleaner dedup granularity than file-Jaccard for dependency-graph pretraining.

## Missed citations worth adding
Checked against `paper/bib/refs.bib` (present already: kocetkov2022stack, li2023starcoder,
lozhkov2024starcoder2, allal/benallal2023santacoder, guo2024deepseekcoder, nijkamp2023codegen,
fried2023incoder, xu2022polycoder, chen2021humaneval, austin2021mbpp). Genuinely missing and
relevant to our dedup/contamination and code-corpus story:

- **lee2022deduplicating** — Lee et al., "Deduplicating Training Data Makes Language Models
  Better," arXiv:2107.06499. The canonical result that near-dedup (SuffixArray / MinHash)
  cuts memorization and train-test overlap; directly justifies why our unhandled cross-split
  near-dup is a threat. The Stack's near-dedup design descends from this.
- **kandpal2022deduplicating** — Kandpal, Wallace, Raffel, "Deduplicating Training Data
  Mitigates Privacy Risks in Language Models," arXiv:2202.06539. Ties dup rate to
  memorization/extraction — relevant since we inherit PII-bearing, only-partly-deduped files.
- **carlini2023quantifying** — Carlini et al., "Quantifying Memorization Across Neural
  Language Models," arXiv:2202.07646. Memorization scales with duplication and model size;
  bears on whether our link-fetch mechanism just memorizes deduped-adjacent targets.
- **broder1997resemblance** — Broder, "On the resemblance and containment of documents"
  (MinHash origin). The primitive behind both The Stack's near-dedup and any dedup we add.
- **allamanis2019adverse** — Allamanis, "The Adverse Effects of Code Duplication in Machine
  Learning Models of Code," arXiv:1812.06469. Code-specific: duplication inflates reported
  metrics on code models — the most on-point warning for our generated-SDK-heavy graph.
  (Note: "Allamanis" appears in refs.bib only as a co-author string, not as this entry.)

## One-line takeaways for the write-up
- We use `the-stack-dedup` (near-dedup, Jaccard>0.85, 256-perm MinHash), NOT raw The Stack —
  cite that near-dedup gives the paper's +~10 pass@100 and is our provenance.
- The Stack's file-level dedup + our `links>=2` (28.7% kept) + generated-SDK hubs jointly
  bias our graph toward machine-generated re-export topology; caveat any "structure-learning"
  claim.
- Upstream file dedup can silently sever our import edges; whole-repo near-dup and
  train/val cross-dup are unhandled by us — real leakage risk for held-out metrics.
- Inherited gotchas: no HumanEval/MBPP decontamination, PII present, MPL/LGPL mislabel,
  and non-deterministic graph build — pin the dataset revision.

Confirmed against source: `data/stack_sharded_download.py`,
`data/github_graph_extractor/build_graph_streaming.py`,
`data/github_graph_extractor/graph_100M_stats.json`, and the code brief; paper numbers
confirmed from the ar5iv HTML of arXiv:2211.15533.
