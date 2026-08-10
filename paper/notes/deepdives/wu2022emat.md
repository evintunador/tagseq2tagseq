# wu2022emat — An Efficient Memory-Augmented Transformer for Knowledge-Intensive NLP Tasks

Wu, Zhao, Hu, Minervini, Stenetorp, Riedel. EMNLP 2022 (main). arXiv:2210.16773.
Sources read: arXiv PDF (full text, all 13 pages incl. appendices A–B and Tables 1–6);
abstract/metadata cross-checked against aclanthology.org/2022.emnlp-main.346. Result
numbers below are transcribed from the paper's own tables. Our-side claims are grounded
in `paper/notes/code_briefs/masks.md`, `generation_retrieval.md`, and
`related_work_notes.md` (EMAT already appears there in the "frozen precomputed corpus-KV"
trio alongside wu2022memorizing and dejong2022tome).

---

## What the paper actually does

EMAT augments a **T5-base** seq2seq model (225M params vs. 221M for vanilla T5-base) with
an **external key-value memory** that is encoded once, frozen, and queried by **MIPS**.
The thesis is efficiency: get retrieval-augmented accuracy at (nearly) parametric-model
throughput, by making the "retrieval" a single dense inner-product lookup that overlaps
with the forward pass rather than a retrieve-encode-read pipeline.

**Knowledge source and what keys/values are.** The memory is built from **PAQ** (Lewis et
al. 2021), a corpus of ~65M *machine-generated* question-answer pairs mined from Wikipedia
(pre-training uses PAQ-L1, a 14M-pair subset). Each memory slot is one QA pair: **the key
is the question, the value is the answer**. This is the crucial framing point for us —
EMAT's "corpus" is not raw documents but a precomputed QA index; the model never sees
Wikipedia passage text at inference, only dense key/value vectors.

**How keys and values are encoded (the offline pass).** EMAT's own encoder produces both.
A learned **PREFIX of length P=2** is prepended to the question; hidden states at encoder
layer `l_k` are passed through a **1-D convolutional layer**, and the *prefix positions* of
that conv output become the key `k ∈ R^{P×h}`. The value is the *prefix positions* of the
answer's hidden states at a later layer `l_v` (no conv), `v ∈ R^{P×h}`. So each slot is a
handful (P=2) of dense vectors of width h, not text. The full memory over PAQ is ~**300GB
of CPU RAM** (a stated limitation).

**Retrieval (MIPS).** The query is the *same* prefix-conv representation computed from the
input at layer `l_k`; query and keys are flattened by averaging over the P prefix positions
(`q̄ = (1/P)Σ q_j`), and similarity is the inner product `⟨q̄, k̄⟩`. Top-k pairs are pulled
by MIPS — **FAISS with an HNSW index**, run **on CPU**, "millions of vectors in
milliseconds." Because the query is read off an *early* layer `l_k` and consumed at a
*later* layer `l_c`, the MIPS lookup **runs concurrently with layers l_k+1…l_c−1** — this
overlap is the whole efficiency trick, and it needs only **one forward pass** (contrast
QAMAT, which needs two encoder passes; EMAT reports being ~4.2–5× faster than QAMAT).

**Integration.** At encoder layer `l_c` the retrieved keys are concatenated into
`K' ∈ R^{Pk×h}`, given relative positional encodings, and **prepended to the layer's hidden
states**; at layer `l_v` the corresponding values `V'` are *added* onto those same prepended
positions. The rest of the encoder runs normally; the decoder generates conditioned on the
memory-augmented encoder output. Two latency regimes are studied: **FKSV** (fast-key
slow-value: `l_k=3, l_c=3, l_v=7`) and **SKSV** (`l_k=3, l_c=10, l_v=11`).

**Training pipeline.** Initialized from T5-base; prefix embeddings + key conv trained from
scratch. Three-part **multi-task pre-training** objective on PAQ-L1:
- **KAE** (key auto-encoding): recover the question from the key embedding.
- **VAE** (value auto-encoding): recover the answer from the value embedding.
- **Gen**: for each PAQ pair, RePAQ retrieves 10 related QA pairs; the model must generate
  the answer given the question + those 10 retrieved key/value embeddings — this teaches the
  *integration strategy* over multiple slots. Loss = KAE+VAE+Gen (weights 0.5/0.5/1.0).

**Fine-tuning** adds a **weakly-supervised retrieval loss** `L_Ret` (contrastive: a
retrieved pair is a positive if its answer lexically matches / is contained in the target;
sample one positive, m negatives, InfoNCE-style) plus the generation loss `L_Gen`. A
**memory-caching** trick freezes M within an epoch (retrieve top-n once, n=384) and
re-encodes the whole memory only at epoch end, since updating 14M+ entries per step is
infeasible.

**Concrete results (from the paper's tables):**
- **ODQA Exact Match (Table 1).** EMAT-FKSV: **NQ 44.3, TQA 44.4, WQ 36.7** at **~1000 Q/s**;
  EMAT-SKSV 43.3 / 43.7 / 33.2 at ~1200 Q/s. Vanilla **T5-base: 25.8 / 24.4 / 26.6** at 1600
  Q/s — so +18.5 / +20.0 / +10.1 EM over the same backbone. Beats RePAQ-large (41.2 NQ) and
  matches RAG (44.5 NQ). **FiD-base (48.2 NQ) and FiD-large (51.4) are more accurate** but at
  **3.7 and 1.4 Q/s** — EMAT is ~2 orders of magnitude faster. DPR reader 41.5 NQ @ 2.7 Q/s.
- **Wizard-of-Wikipedia / KILT (Table 2).** EMAT-FKSV **F1 15.78, R-L 14.73 @ 141 U/s**, beating
  T5-base (13.53/12.40) *and* RAG (13.11/11.57 @ 3.4 U/s) and BART+DPR (15.19/13.23 @ 0.7 U/s).
- **ELI5 / LFQA (Table 3).** EMAT-SKSV **F1 19.03, R-L 20.91 @ 71 Q/s**, beating T5-base
  (16.01/19.08) and RAG (14.51/14.05 @ 0.4 Q/s) — "160× faster than RAG."
- **Ablation (Table 4).** Removing pre-training pieces is catastrophic: −generation task drops
  TQA 44.4→24.7; −auto-encoding drops WQ 36.7→12.9; −all pre-training NQ 44.3→27.1 (worse than
  T5-base). Without *any* fine-tuning, pre-trained EMAT already beats fine-tuned T5-large on
  NQ/TQA.
- **Scaling (App. A).** EM rises monotonically with #retrieved pairs (2→10) and with PAQ
  memory size (14M→65M) — more memory and more neighbors both help; the model tolerates noisy
  retrieved slots.

A notable qualitative claim (Table 5/6): EMAT does **not** just copy the top-1 value. On WoW,
retrieving PAQ QA pairs directly ("RePAQ w/ EMAT key encoder") scores F1 **1.84** — near
zero — yet EMAT reaches 15.78. The decoder *reranks and recombines* dense key/value slots into
novel text; the dense embeddings carry information the surface answers do not.

---

## Methodology: theirs vs. ours

The one-line axis: **EMAT trains a model to *consume* a frozen, precomputed, MIPS-retrieved
key-value store; we train the cross-document *edge itself* into attention and re-run the same
edge at inference by materializing the target document verbatim into the packed sequence.**
Concretely:

**1. Store granularity and content — dense QA-slot vs. verbatim node tokens.** EMAT's memory
slot is a QA pair compressed to P=2 dense vectors per field; the raw text is discarded and
never re-enters the model. Our "retrieval" fetches the **actual token stream** of the target
node and inserts it into the sequence (`document_corpus.get_document` → tokens from an mmap
`PretokShardedBackend`; `generation_retrieval.md` §_handle_link). We attend to real document
tokens under the trained mask, not to a distilled 2-vector summary. This is the same contrast
`related_work_notes.md` draws for the whole "frozen precomputed corpus-KV" trio
(wu2022memorizing / dejong2022tome / wu2022emat): they *precompute and freeze* the store and
train the model only to read it; we keep no precomputed store at all.

**2. What the "edge" is — MIPS similarity vs. deterministic graph/identifier resolution.**
EMAT's neighbor selection is a **learned dense inner-product** search (query encoder trained
by a weak-supervision contrastive loss), approximate (HNSW), with no exact-match guarantee.
Ours is a **deterministic link resolution**: a detected link string is resolved by the *same
match key training uses* (`index_doc_span`, a 3-tier exact→detector-key→fuzzy cascade,
`generation_retrieval.md`; `masks.md` §"Detected link→grant"). There is no embedding, no ANN
index, no approximation error, and no learned retriever — the "who to attend to" is dictated
by the graph, not by cosine geometry. `related_work_notes.md` explicitly frames FAISS/HNSW as
"unnecessary when target resolution is an exact hashmap lookup."

**3. Where structure lives — cross-attention over external slots vs. an in-sequence
block-sparse mask.** EMAT prepends retrieved key/value vectors at *encoder layers l_c/l_v*
and lets standard full attention mix them in; it is an encoder-decoder with an external memory
side-channel. We have **no external memory and no cross-attention module**: the target
document becomes ordinary tokens in the *one* 32k decoder sequence, and a
**CrossDocLinkMaskCreator** grant rectangle (rows `[link_end_pos, A.end) ×` cols
`[B.start, B.end)`, `masks.md`) lets the linking positions read it via the *same self-attention*
used everywhere. The edge is a **hard binary mask entry**, asymmetric and DAG-gated (backward
links only), not a soft attention over a memory bank.

**4. Train/inference symmetry — asymmetric vs. identical.** EMAT is asymmetric by design:
memory is *populated and frozen offline*, then *queried* at inference; pre-training uses
RePAQ-retrieved neighbors, fine-tuning uses the model's own MIPS. Our strongest claim is the
opposite — the **same link machinery (detector + match key + grant-from-link-position + DAG
ordering) runs in both regimes as a single implementation** (`generation_retrieval.md`
§"Train/inference mirror"). At training the target already sits earlier in the packed sequence;
at inference we *insert* it earlier so the identical mask geometry applies. EMAT cannot make
this claim: there is a hard train/serve split between "encode the memory" and "read the memory."

**5. Efficiency posture — inverted.** EMAT's entire selling point is throughput: one forward
pass, MIPS overlapped with layers l_k→l_c, ~1000 Q/s. **We deliberately forgo all of this.**
Our generation loop does a **full O(T²) recompute every token, no KV cache** — inserting a
fetched node shifts RoPE positions, which makes paged/prefix KV reuse *incorrect*, so we pay
~O(T³) over a generation to preserve train/inference symmetry (`generation_retrieval.md`
§"KV CACHE = NONE"). EMAT is the efficiency pole; we are the fidelity pole. This is the sharpest
single contrast and worth stating plainly in R3/R4.

**6. Compute-controls.** EMAT has no matched-compute concatenation baseline — it compares to
T5-base (no memory), RePAQ (retrieval-only), RAG/FiD (different compute entirely). Our design
isolates the *linking inductive bias* from raw FLOPs via `doc_concat_link` (whole-source grant,
strict FLOP-superset) and `doc_concatenated` (component-contiguous), `masks.md` §compute-control.
That causal-inference framing is something EMAT lacks and a reviewer will notice.

**Shared DNA.** Both are "a generated/emitted key fetches knowledge into the model's context in
a single pass," both descend from key-value memory networks (miller2016kvmemnn, in our refs),
both exploit that the fetch can overlap or be cheap. And notably EMAT shares *authors and
lineage* with much of our RAG/ODQA cluster (Riedel, Stenetorp, Minervini; PAQ, RePAQ, KILT,
DPR, FiD) — it is squarely in the family we position against.

---

## Predictions & open questions for our method

- **Memory helps most where the answer is a discrete fact covered by the store; margins shrink
  as outputs get long/open-ended.** EMAT's EM gains over T5-base are huge on ODQA (+18–20 EM)
  but compress to F1 +2–3 on WoW/ELI5. Prediction for us: the cross-doc link edge should give
  its largest Δnll on **fact-carrying multi-hop QA** (HotpotQA/2WikiMultiHop, our target evals
  in `related_work_notes.md`) and a **smaller but positive** effect on open-ended continuation.
  Single-doc controls (HellaSwag/ARC/etc.) should be unaffected — EMAT's ablation-to-baseline
  collapse when the memory is noise is the mirror of that expected null.

- **More neighbors / bigger corpus monotonically help and the model tolerates retrieval noise.**
  EMAT's App. A shows EM rising with both #retrieved pairs (2→10) and PAQ size (14M→65M). Our
  analogue is **max_grants** (production 256, cosine-warmed) and corpus coverage. Prediction:
  raising the grant cap and packing more/closer graph neighbors should help monotonically until
  a noise floor — but note our **positional truncation** of grants past the cap (`masks.md`:
  later links silently dropped) is a *biased* form of "too many neighbors," unlike EMAT's
  score-ranked top-k. If we see degradation at high link density, suspect the positional-drop
  bias before concluding the edge saturates.

- **The value of the fetched content exceeds its surface answer.** EMAT's RePAQ-w/-EMAT-key
  ablation (WoW F1 1.84 → EMAT 15.78) shows the model extracts far more from dense slots than
  from copying top-1. For us this predicts that **attending to full target-document tokens**
  (not a summary) should beat any concat-of-titles or copy baseline by a wide margin — and it
  motivates our verbatim-token design over compression. Good ammunition for "why not just
  retrieve summaries."

- **Pre-training the integration mechanism is do-or-die.** EMAT's ablation: removing the
  generation pre-training task drops TQA 44.4→24.7; removing all pre-training makes it *worse
  than no memory*. Direct prediction: a model that sees the cross-doc mask **only at inference**
  (or only briefly) will underperform or be actively harmed by materialized documents it never
  learned to read. This is precisely why our mask is applied **identically in pretraining** —
  EMAT is empirical evidence that the read-strategy must be trained, and that a bolt-on
  inference-time edge would fail. Our train/inference-mirror thesis inherits strong support here.

- **Open question EMAT poses that our design resolves.** EMAT's stated limitations are (a) the
  retriever needs *task-specific weak supervision* (different lexical-match heuristics per task),
  and (b) the 300GB frozen store. Our deterministic graph/identifier resolution needs **no
  learned retriever and no weak-supervision labels at all** — the edge is given by the corpus
  structure. If our results hold, we answer their open "learn the retriever end-to-end" wish by
  sidestepping the retriever entirely. Conversely, EMAT resolves a question *for* us: it shows a
  frozen precomputed store *can* match RAG accuracy at 1000× the speed — a reviewer may ask why
  we accept O(T³) recompute instead of caching. Our honest answer (RoPE-shift correctness +
  train/serve symmetry) should be pre-stated; EMAT is the exhibit that caching is viable when
  you *don't* insist on position-faithful materialization.

---

## Gotchas

- **QA-generated store ≠ document store — beware conflating them.** EMAT's "external knowledge"
  is 65M *synthetic QA pairs*, and PAQ's generator was trained on NQ+TQA, giving it **high
  coverage of exactly the ODQA test sets it's evaluated on**. This is a soft form of
  train-eval alignment (they flag it: "PAQ has high coverage for these two datasets"). If we
  ever benchmark against EMAT numbers, remember its NQ/TQA gains are partly a corpus-coverage
  artifact, not a pure architecture win — its own WoW/ELI5 (where "it is not clear how PAQ can
  be used") are the fairer generalization signal. Do not cite EMAT's 44.3 NQ as an
  apples-to-apples architecture comparison.

- **Weak-supervision label leakage / metric fragility.** EMAT's retrieval loss selects
  positives by *lexical match of the retrieved answer to the target*. For short ODQA this is
  clean; for long outputs (WoW/ELI5) it's a "value contained in normalized target" heuristic —
  brittle and task-specific. Our eval discipline (paired Δnll, bootstrap CIs, token parity per
  `related_work_notes.md`) is a better footing, but the lesson is: **any weak-match signal that
  touches the target risks inflating apparent retrieval quality.** Keep our weak-supervision
  (if any) out of the eval-scored path.

- **Compression bottleneck at P=2.** EMAT squeezes a whole question/answer into 2 dense
  vectors; it works because slots are short QA pairs. The naive lesson "you can compress
  knowledge to a few vectors" **does not transfer** to our setting where a node is a full
  document — this is exactly why we materialize verbatim tokens. Don't be tempted by a
  "compress the fetched doc to k vectors" shortcut; EMAT succeeds *because* its unit is already
  tiny.

- **Throughput comparisons are hardware-entangled.** EMAT's Q/s are measured on one A100 with
  CPU-side FAISS; QAMAT's on 32 TPU-v3. Speed claims across these systems are not comparable
  without care. If we report any efficiency contrast, fix the hardware and the measurement
  (they even footnote this discrepancy in Table 1).

- **300GB CPU RAM store.** Their frozen memory is huge even for compressed slots; a
  raw-document store at our granularity would be far larger, reinforcing that our
  fetch-from-mmap-on-demand (no resident dense index) is the right call — but also a reminder
  that "precompute everything" doesn't scale to document-token granularity.

---

## Missed citations worth adding

I grepped `paper/bib/refs.bib` (533 entries) for each below; none are present (verified by
cite-key, author, title-phrase, and arXiv-id searches). arXiv/CoRR ids marked "from paper" are
transcribed from EMAT's own reference list; others are my best recollection and must be
verified before adding.

- **wu2022emat itself is already in refs.bib** and in `related_work_notes.md` (§"Precomputed
  corpus memory") — no action needed for the focal paper; the items below are its *cited works*
  missing from our set.

- **lewis2021paq** — "PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them"
  (Lewis, Wu, Liu, Minervini, Karpukhin, Goyal, Küttler, Stenetorp, Riedel), TACL 2021,
  arXiv **2102.07033** (verify). The QA-pair corpus that *is* EMAT's/RePAQ's memory; the
  cleanest example of "precompute a queryable QA index from a corpus" — a direct foil to our
  document-graph store and relevant to our retrieval/ODQA cluster. **RePAQ** is the same paper.

- **chen2022qamat** — "Augmenting Pre-trained Language Models with QA-Memory for Open-Domain
  Question Answering" (Wenhu Chen, Verga, de Jong, Wieting, Cohen), CoRR **abs/2204.04581**
  (from paper). The concurrent QA-key-value-memory transformer EMAT benchmarks against; a
  two-encoder-pass precomputed-KV model — belongs directly in our frozen-corpus-KV trio
  discussion and sharpens the "single vs. multi pass" efficiency axis.

- **geva2021ffnkeyvalue** — "Transformer Feed-Forward Layers Are Key-Value Memories" (Geva,
  Schuster, Berant, Levy), EMNLP 2021, arXiv **2012.14913** (verify). The result that FFN layers
  *already* act as key-value memories over the training corpus; foundational for any "memory in
  a transformer" framing and for contrasting parametric vs. external memory. (Note: geva2021*
  keys in our refs are StrategyQA/Break/SCROLLS — this specific paper is absent.)

- **yao2022kformer** — "Kformer: Knowledge Injection in Transformer Feed-Forward Layers" (Yao,
  Huang, Zhang, Dong, Wei, Chen), CoRR **abs/2201.05742** (from paper). Injects retrieved
  knowledge by *extending FFN weights* rather than the attention stream — an alternative
  injection locus to our attention-mask edge; a clean mechanistic contrast for the graph-aware
  / KG-enhanced section.

- **lee2021densephrases** — "Learning Dense Representations of Phrases at Scale" (Lee, Sung,
  Kang, Chen), ACL 2021, arXiv **2012.12624** (verify). Phrase-level dense retrieval that
  returns answers directly by MIPS over phrase embeddings (a strong EMAT/RePAQ baseline);
  relevant to our "generate-an-identifier / retrieve-a-span" cluster (bevilacqua2022seal,
  min2023npm) as the phrase-indexing pole.

- **paranjape2021hindsight** — "Hindsight: Posterior-Guided Training of Retrievers for Improved
  Open-Ended Generation," CoRR **abs/2110.07752** (from paper). EMAT names this as the
  end-to-end retriever-training technique it *didn't* use; relevant to our contrast on whether
  the retriever is learned end-to-end vs. deterministic — a pointer for the "we need no learned
  retriever" argument.

(Lower priority / likely out of scope: roberts2020closedbook "How Much Knowledge Can You Pack…"
EMNLP 2020, arXiv 1910.10683-adjacent — the closed-book-QA parametric-memory reference; and
wu2020/wu2021 "adaptive computation for ODQA" — same authors, efficiency-of-readers angle.
Include only if we expand the parametric-vs-retrieval framing.)

---

Confirmation: deep-dive complete and written to
`/fss/evin_t/tagseq2tagseq/paper/notes/deepdives/wu2022emat.md`, grounded in the full EMAT PDF
and verified against our masks.md / generation_retrieval.md briefs and refs.bib.
