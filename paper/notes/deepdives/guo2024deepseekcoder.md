<!-- Deep-dive for TAGSeq2TAGSeq lit review. Grounded in the DeepSeek-Coder paper
(arXiv 2401.14196, PDF read in full) and our code briefs merged_multisource.md,
data_pipelines.md, traversal.md, masks.md (commit 6134163). Paper facts are quoted
from the PDF; comparisons to our method cite the briefs / source. -->

# guo2024deepseekcoder — DeepSeek-Coder: When the Large Language Model Meets Programming — The Rise of Code Intelligence

## What the paper actually does

A family of open code LLMs (1.3B, 6.7B, 33B), decoder-only Transformers (RoPE, SwiGLU;
33B uses GQA group-8, FlashAttention-2), trained from scratch on **2T tokens** over
**87 programming languages**. The cleaned code corpus is **797.92 GB / 603.17M files**
(Table 1; Java 148.66 GB / 18.63% and Python 120.68 GB / 15.12% dominate). BPE tokenizer,
vocab **32,000**. AdamW (β 0.9/0.95), three-stage LR schedule with 2000 warmup and final
LR = 10% of peak; peak LRs 5.3e-4 / 4.2e-4 / 3.5e-4. Trained on HAI-LLM with TP + ZeRO-DP
+ PipeDream PP on A100/H800 clusters.

Two training objectives: **next-token prediction** and **Fill-in-the-Middle (FIM)** at
document level, applied *before* packing, at rate **0.5 in PSM mode** (chosen by a 1.3B
Python ablation: 100% FIM peaks on HumanEval-FIM but hurts left-to-right completion; 50%
PSM beat MSP — Figure 3). Long context is a *post-hoc* add-on: linear RoPE scaling factor
1→4, base 10000→100000, +1000 steps at seq-len 16K / batch 512; theoretically 64K but
"most reliable within 16K." Instruction tuning (Alpaca format, 2B tokens) gives the
`-Instruct` variants.

Headline numbers: DeepSeek-Coder-Base **33B** = **50.3% avg HumanEval / 66.0% MBPP**
(multilingual, Table 3); the 6.7B base already surpasses CodeLlama-Base-34B;
`-Instruct 33B` beats GPT-3.5-Turbo on HumanEval. DS-1000 base 33B avg 40.2%. Program-based
math (PAL) strong at 33B.

**The result that matters for us — Table 7, cross-file completion on CrossCodeEval**
(Python/Java/TS/C#, EM + edit-similarity, 2048 max seq, 512-token cross-file context via
BM25). DeepSeek-Coder-Base 6.7B leads every language, and crucially the ablation row
**"+ Retrieval w/o Repo Pre-training"** (same model but pretrained on *file-level* code
with no dependency ordering) **drops** on Java (17.72→16.64 EM), TypeScript (14.03→13.23)
and C# (16.23→14.48). Python is essentially flat (16.14→16.02). This is their sole direct
evidence that the topological-sort repo pretraining *itself* helps cross-file tasks — and
it's a small, language-dependent effect, present even when a BM25 retriever is also supplying
cross-file context at inference.

### §2.2 Dependency parsing (the part we care about)

They "only consider the invocation relationships between files and use **regular expressions**
to extract them" — `import` (Python), `using` (C#), `include` (C). Algorithm 1 is a
**topological sort**: build an adjacency list + in-degree dict over files in one repo;
`HasDependency(A,B)` adds edge B→A and increments in-degree(A); split into disconnected
subgraphs; then a **modified Kahn's algorithm that repeatedly pops the node of *minimal*
in-degree (not strictly zero)** so cycles are tolerated. Each subgraph's sorted file list is
**flat-concatenated into one training sample**, so "the context each file relies on is placed
before that file in the input sequence." A **comment giving the file path** is prepended to
each file. Repo-level near-dedup (§2.3) treats the whole concatenated repo as one unit.
n-gram decontamination (10-gram exact, ≥3-gram exact) against HumanEval/MBPP/GSM8K/MATH.

## Methodology: theirs vs. ours

**Same signal, opposite mechanism.** Both projects extract the import/dependency graph and
use it to co-locate related files, and both do a **topological sort so a dependency lands
before its dependent** under causal attention. That topo-sort-before-causal idea is exactly
our default `order_mode = prefer_targets_first` per-component Kahn ordering (traversal.md:24)
and its purpose is identical: our masks brief's single "most important non-obvious design
point" is that outgoing-traversal + a causal DAG-gated mask requires targets ordered *ahead*
of linkers or the grant is silently dropped (traversal.md:30, masks.md:15). DeepSeek-Coder
stops there — its signal is **expressed only as document order**, then plain causal attention
runs over the flat concatenation. It never adds an edge-keyed cross-file attention path.

That is precisely the gap we target. Our contribution is the **`cross_doc_link` mask**
(masks.md:15): for a detected/baked link A→B, rows `[link_end_pos, A.end) × cols [B.start,
B.end)` become attendable — a linking document gets *read-access into the target document
from the link position onward*, an asymmetric grant on top of doc-causal, realized as
bit-packed grants with pointwise membership (masks.md:23). DeepSeek-Coder, by contrast, lets
every token attend to every earlier token of the whole concatenated repo indiscriminately;
it has **no per-edge attention structure at all** — the graph exists only in the *ordering*.

The axis, sharply:
- **Train-on-structure, both.** Neither is retrieve-at-inference in pretraining. DeepSeek-Coder
does bolt on BM25 retrieval *at eval* (Table 7), which is orthogonal to their pretraining
recipe; our link mechanism is **used identically in pretraining and at inference** — a
generated link fetches the target doc into the attention context (brief/_DEEPDIVE_BRIEF).
So on the "attention edge vs cached-KV vs training-pair signal" axis, they are the pure
**document-order / training-pair** end (the edge influences *which tokens sit near which*,
nothing more), and we are the **attention-edge** end.
- **Their control ≈ our concat baseline.** Their "w/o Repo Pre-training" ablation is close in
spirit to our compute-matched `doc_concatenated` / `doc_concat_link` masks (masks.md:16-18)
that isolate the linking inductive bias from raw FLOPs. But note the asymmetry: their base
recipe **is** the ordered flat-concat, and the ablation *removes ordering entirely* (file-level
shuffle); it is not FLOP-matched and does not isolate "ordering vs. attention-edge." Our
`doc_concatenated` (full component attention, no inference linking) is the strict-superset
compute control that theirs lacks — and it's the exact thing that would tell us whether their
gain is the ordering or just more within-repo context.
- **Regex imports vs. resolved graph.** DeepSeek-Coder's edges are regex `import/using/include`
with an `O(n²)` all-pairs `HasDependency` and **no cross-repo edges** (subgraphs are
per-project). Our Python path is tree-sitter import extraction with module→file resolution,
stdlib denylist, and an explicit (non-deterministic, unseeded) tie-break (data_pipelines.md:15),
plus a `links_in_repo>=2` filter that keeps only ~28.7% of files. Both are intra-repo for code;
notably **our merged multi-source graph does have cross-source edges but packs strictly
within-source** (merged_multisource.md:46), which mirrors DeepSeek-Coder's per-project scoping
— a shared limitation to state, not a point of differentiation.
- **Cycle handling matches.** Their min-in-degree Kahn variant to tolerate import cycles is the
same problem our Kahn insertion-order fallback solves (traversal.md:30, masks.md:43); both
concede that in a cycle some edges get only one direction. Worth noting we can *represent* a
back-edge that they cannot use at all once ordering is fixed — but our causal DAG-gate also
drops the forward one, so neither realizes both directions.
- **FIM.** They use PSM-mode FIM at 0.5; we do not (our objective is next-token over packed
graph sequences, with an MTP auxiliary per training_loop/architecture notes). Their finding
that 100% FIM trades off against L2R completion is a caution if we ever add infilling.

Net: DeepSeek-Coder is the closest **code** prior art precisely because it independently
arrives at *import-graph topological ordering before causal packing* — the same pre-mask
ingredient we depend on — and then declines to take the next step (edge-keyed attention).
It is the natural "ordering-only, no cross-doc mask" reference point our ablations are built
to beat.

## Predictions & open questions for our method

- **Expect the cross-file effect to be real but modest and language-dependent.** Table 7 is the
best external calibration we have: ordered repo pretraining moved CrossCodeEval EM by roughly
1–2 points and *only on Java/TS/C#, not Python*, even with a retriever present. If mere
ordering buys ~1–2 EM, our attention-edge mechanism must clear a meaningfully higher bar to be
worth the kernel complexity. Our internal-ports Δnll headline (8B merge beats specialists
1.7–11×, merged_multisource.md:34) is on a different metric; DeepSeek-Coder warns that on a
*standard* cross-file completion benchmark the ceiling for structure signal is small.
- **Python may under-respond.** Their Python row barely moved. If our per-source cross-doc Δ is
also weakest on Python, that's consistent with theirs, not a bug — Python's flat import surface
and heavy `__init__.py`/generated-SDK top-degree nodes (data_pipelines.md:15) may already leak
enough via ordering that the extra attention edge adds little.
- **Ordering alone is a strong baseline, so the concat control is load-bearing.** DeepSeek-Coder
shows ordering-without-edge already helps. This predicts our `doc_concat_link` /
`doc_concatenated` controls will *not* be near-zero — they should capture much of the gain,
and the residual `cross_doc_link` − `doc_concat_link` gap (link-position gating) is the number
that actually isolates our novel bit. If that residual is small, DeepSeek-Coder is the
explanation: ordering + full within-component attention already delivers most of it.
- **Long-context is fragile and post-hoc for them; ours is native 32k.** They only trust 16K
despite claiming 64K, and got there with 1000 steps of RoPE-scaled continued training. Our
32k is trained natively — a genuine advantage — but their experience predicts **RoPE behavior
across packed docs is the risk area**, which dovetails with our own masks.md:39 flag that
**RoPE is not reset per doc** (A reads B at a packing-distance-dependent relative offset). Their
need to rescale RoPE base frequency to reach longer contexts suggests our arbitrary cross-doc
relative offsets could degrade link fetches at large packing distances; worth an ablation on
target-distance vs. grant efficacy.
- **Repo-level dedup as one unit** (§2.3) is a design our within-source packing could adopt to
avoid partial-repo leakage; predicts cleaner val if we dedup at community/pack granularity
rather than file.

Open question our design could resolve for them: DeepSeek-Coder cannot say whether the
cross-file gain is *ordering* or *access* because it only has ordering. Our compute-matched
mask ladder (`doc_causal < cross_doc_link ≤ doc_concat_link ≤ doc_concatenated`, masks.md:18)
is exactly the experiment that decomposes their effect. Conversely, their large-scale 87-lang
2T-token run is the scaling regime we can't reach — if our lead is real at 355M/8B it predicts
their recipe left cross-file performance on the table.

## Gotchas

- **Contamination via GitHub test-set presence.** They n-gram-filter (10-gram exact, 3–10-gram
exact) against HumanEval/MBPP/GSM8K/MATH, and pointedly note CrossCodeEval repos are dated
Mar–Jun 2023 vs. their pre-Feb-2023 cutoff to avoid leakage. Our data_pipelines brief flags
weak/sampling-only dedup and **no cross-split near-dup check** (data_pipelines.md:34); their
practice is the standard we'll be held to. Any cross-doc eval that reuses training repos will
be attacked.
- **Regex import extraction is lossy and they don't report resolution rate** — same blind spot
as our unlogged wiki/import resolution fraction (data_pipelines.md:14,36). If we cite them as
prior art we should not overclaim their graph fidelity; theirs is cruder than ours (regex,
no module resolution) yet still helped, which is a point in our favor but also means the bar
is "beat a crude-ordering baseline."
- **The cross-file ablation is one small table, not a scaling study.** ~1–2 EM, three of four
languages, single model size. Don't over-read it as "repo pretraining strongly helps"; it's
suggestive. Our own rungs-are-independent-not-nested caveat (merged_multisource.md:43) is the
same class of weakness — small effects need noise floors.
- **FIM/completion trade-off.** Their Figure 3 shows objective choice moves the benchmark you
optimize at the cost of another; if we add any auxiliary objective, expect a similar
tug-of-war and measure both axes.
- **RoPE rescaling needed for long context** — their empirical 16K-trust despite 64K-theory is
a direct warning about our no-per-doc-RoPE-reset design at 32k with far-apart packed targets.
- **Eval config sensitivity.** Table 7 fixes 2048 seq / 512 cross-file / 50 output tokens with
BM25 — a narrow, retriever-dependent setup. Cross-file numbers swing hard with retriever and
context budget; our eval must pin these or comparisons are meaningless.

## Missed citations worth adding

Checked against `paper/bib/refs.bib`. Already present: `kocetkov2022stack` (The Stack),
`li2023starcoder`, `roziere2023codellama`, `benallal2023santacoder`, `nijkamp2023codegen2`,
`bavarian2022fim` (FIM), `chen2023positioninterpolation` (2306.15595), `peng2023yarn`,
`zhu2024deepseekcoderv2`, `bi2024deepseekllm`, `ding2023crosscodeeval`. Genuinely missing and
relevant to us:

- **lee2022deduplicating** — Lee, Ippolito, Nystrom et al., "Deduplicating Training Data Makes
Language Models Better," ACL 2022 (arXiv **2107.06499**). Directly relevant: it's the
foundational near-dup result DeepSeek-Coder leans on for *repo-level* dedup, and it maps onto
our own reviewer-attackable dedup gap (data_pipelines.md:34 — sampling-only Stack dedup, no
wiki/arxiv dedup, no cross-split near-dup). We should cite it in the data section. (Not to be
confused with `kocetkov2022stack`, which we have.)

- **du2022glm** — Du, Qian, Liu et al., "GLM: General Language Model Pretraining with
Autoregressive Blank Infilling," ACL 2022 (arXiv **2103.10360**). Lower priority: it's an
attention-mask design (bidirectional within masked spans + autoregressive across) that is a
structural cousin to our custom cross-doc mask, and DeepSeek-Coder cites it in the FIM lineage.
Worth it only if we expand the "structured attention masks / blank-infilling" framing;
tangential to our graph-edge focus.

(Everything else in their reference list — CodeGen, CodeT5, MultiPL-E, DS-1000, PAL, the eval
suites GSM8K/MATH/HellaSwag/ARC/WinoGrande/BBH/MMLU, LLaMA-2, GPT-4, FlashAttention-2, ZeRO,
PipeDream — is either already in refs.bib or not relevant to our graph-attention thesis.)

---
Confirmation: analysis grounded in the full DeepSeek-Coder PDF (2401.14196) and code briefs merged_multisource.md, data_pipelines.md, traversal.md, masks.md; §2.2 topological-sort details verified against the paper's Algorithm 1, and the two suggested citations were checked absent from paper/bib/refs.bib (verify before adding).
