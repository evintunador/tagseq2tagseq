# Related-work notes

Annotated bibliography for the TAGSeq2TAGSeq paper: for each cited work, what it
does and how it relates to this project. The organizing question throughout is
whether a work **trains on** link structure or only **retrieves at inference**, and
whether it uses an attention edge, a GNN message-passing edge, a cached-KV store, or
a training-pair signal. Entries are keyed by their BibTeX cite key in
`bib/refs.bib`; the bibliography there is the source of truth for the entries
themselves. ~506 distinct works across 6 themes.

## Contents
1. [Retrieval-augmented and memory-augmented language models](#retrieval-augmented-and-memory-augmented-language-models)
2. [Graph-aware pretraining, text-attributed graphs, and graph transformers](#graph-aware-pretraining-text-attributed-graphs-and-graph-transformers)
3. [Sequence packing, boundary masking, and efficient/sparse attention](#sequence-packing-boundary-masking-and-efficientsparse-attention)
4. [Optimizer, Learning-Rate Schedule, Architecture, and Training Systems](#optimizer-learning-rate-schedule-architecture-and-training-systems)
5. [Code language models, repository-level context, and cross-file benchmarks](#code-language-models-repository-level-context-and-cross-file-benchmarks)
6. [Multi-hop QA, text-attributed-graph corpora, and evaluation methodology](#multi-hop-qa-text-attributed-graph-corpora-and-evaluation-methodology)

> Not-yet-collected: a data-selection/curation cluster (SemDeDup, D4, DoReMi,
> QuRating, MeCo, Quest, DsDm, Skill-it, MinHash/near-dup dedup) was surveyed but its
> BibTeX was not retained; these are uncited. Re-collect if a data-curation paragraph
> is added.

---
## Retrieval-augmented and memory-augmented language models

This body of work augments a language model with an external store of text or
representations, either retrieved at inference or folded into training. It is the
natural point of comparison for TAGSeq2TAGSeq, whose "retrieval" is a deterministic
graph-edge/identifier resolution rather than a learned similarity search, and whose
memory is verbatim document tokens brought into a trained cross-document attention span
in a single forward pass, rather than compressed summaries, interpolated kNN
probabilities, or reused KV. The groupings below trace that contrast from end-to-end RAG
through to KV-cache serving systems.

### Retrieval-augmented generation and retrieval-in-pretraining

**lewis2020rag** — the canonical RAG formulation: a DPR retriever feeds concatenated
passage text to a BART generator, with generator and query encoder trained and the index
frozen; retrieval at both train and inference, no cached KV. **guu2020realm** bakes a
learned retriever into masked-LM pre-training and backpropagates through retrieval,
making it the reference point for retrieval during pretraining. **izacard2021fid**
(Fusion-in-Decoder) encodes passages independently and lets the decoder attend over their
concatenation; it is the backbone reused by later memory-attention work. **izacard2023atlas**
jointly trains a Contriever retriever with a T5 FiD reader for few-shot knowledge tasks.
**borgeaud2022retro** is the closest train-time comparator: it retrieves kNN chunks from a
trillion-token datastore and integrates them by chunked cross-attention, trained with
retrieval from scratch against a frozen retriever with precomputed neighbor encodings —
but the granularity is text chunks, not graph links, and integration is cross-attention
into a separate encoder rather than an in-sequence attention edge.
**wang2023shallwepretrain** (RETRO++) is the most direct study of
retrieval-integrated autoregressive pretraining, again with a frozen retriever and
chunked cross-attention rather than an in-sequence self-attention graph edge.
**huang2023raven**, **sachan2021emdr2** (end-to-end reader-retriever training),
**lan2023copyisallyouneed** (generation as copying spans from a corpus), and
**rubin2023retrievalpretrained** (RPT, long-range self-retrieval) all bake retrieval into
pretraining but each learns its own retriever or fusion module.
**lin2024radit** (RA-DIT) instead retrofits retrieval onto an existing model post hoc via
dual instruction tuning.

A second cluster keeps the base model frozen and retrieves purely at inference.
**shi2024replug** prepends and ensembles retrieved documents around a black-box LM;
**ram2023incontextralm** prepends retrieved documents to a frozen LM with an optional
trained reranker — together these mark the inference-only, no-training pole.
**asai2024selfrag** trains the model to decide when to retrieve and to emit self-reflection
critique tokens. A related line makes retrieval active and iterative:
**jiang2023flare** triggers retrieval per-token on generation confidence (the closest
analogue to per-token link-detection fetching), while **trivedi2023ircot** (IRCoT),
**shao2023iterretgen** (Iter-RetGen), and **khattab2023dsp** (Demonstrate-Search-Predict)
interleave retrieval with multi-step reasoning — a contrast to recursive,
link-following generation over a document graph.

### kNN datastore and semiparametric language models

**khandelwal2020knnlm** is the exemplar of the KV-cache-as-datastore idea and the sharpest
contrast to this project: a frozen LM plus a datastore mapping hidden states to next
tokens, combined by interpolating a kNN distribution with the LM distribution, entirely at
inference with no retrieval training. **khandelwal2021knnmt** ports the same interpolation to
machine translation, and **zheng2021adaptiveknnmt** adds a trained meta-k gate over the
neighbors. **he2021efficientknnlm** reduces datastore cost, and **alon2022retomaton**
(RetoMaton) follows datastore pointers along an automaton to amortize lookups.
**min2023npm** (NPM) reframes prediction itself as nonparametric retrieval over a phrase
datastore. **yogatama2021spalm** (SPALM) and **zhong2022trime** (TRIME) train the memory
gate or the memory mechanism, moving partway toward training on structure rather than
interpolating at inference. A set of critiques motivates moving from interpolation to
attention: **drozdov2022neighbors** examines when to trust the retrieval component,
**xu2023whyknnlm** analyzes why kNN-LM works at all, and **wang2023knnlmgeneration** shows it
fails to help open-ended generation.

### Precomputed corpus memory and the cached-KV frozen-store family

This is the family whose frozen, precomputed per-document or per-entity representations are
attended at inference — the closest prior art to caching document representations for reuse.
**wu2022memorizing** (Memorizing Transformers) is the clearest instance: it caches literal
(key, value) attention states into a non-differentiable frozen external memory, retrieved by
kNN into one designated attention layer, with the memory populated from document corpora.
**dejong2022tome** (Mention Memory / TOME) and **wu2022emat** (EMAT) extend the same idea to a
precomputed, swappable per-corpus or per-entity memory table — encoded once offline, frozen,
and MIPS-retrieved and attended at inference. This trio is the frozen precomputed-corpus-KV
family that TAGSeq2TAGSeq departs from: those models precompute and freeze the store and train
the model only to consume it, whereas here cached KV lives at document/node granularity and the
graph-structured cross-document attention mask is applied in both training and inference, with a
generated link deterministically fetching its target node.

The Google memory-attention line continues with **dejong2023lumen** (LUMEN, precomputed vs.
on-the-fly hybrid encoding), **dejong2023glimmer** (GLIMMER late-interaction memory reranker;
preprint), and **dejong2023fido** (FiDO), which optimizes FiD memory bandwidth and stands as a
no-cache contrast. **zemlyanskiy2021readtwice** (ReadTwice) is a summary-memory ancestor of this
line. Earlier neural memory foundations sit here too: **sukhbaatar2015memn2n** (end-to-end memory
networks) and **miller2016kvmemnn** (key-value memory networks for reading documents).
**lample2019productkey** and **berges2024memorylayers** are trained product-key memory layers that
add parametric capacity rather than an external corpus. **fan2021kif** composes a frozen kNN memory
for dialog. **verga2020factsasexperts** and **dai2022neuralknowledgebank** attach editable symbolic
knowledge stores, and **das2024larimar** (Larimar) adds episodic memory control. A model-editing
subfamily locates knowledge in the weights rather than an external, re-fetchable store:
**dai2022knowledgeneurons** localizes factual associations to neurons, **meng2022rome** edits them by
rank-one weight updates, and **mitchell2022mend** learns fast editors — the opposite of keeping
knowledge external and re-fetchable.

### Memory-augmented and recurrent-memory transformers

These architectures extend context with a trained, attended memory, but almost all compress the
past into sequential summary states — the design choice TAGSeq2TAGSeq rejects in favor of verbatim,
graph-selected document tokens. **dai2019transformerxl** caches previous-segment hidden states for
recurrence; **rae2020compressive** (Compressive Transformer) and **martins2022inftyformer**
(∞-former) compress older activations into a bounded or continuous memory.
**bulatov2022rmt** (Recurrent Memory Transformer) and its scaling follow-up
**bulatov2023scalingrmt** pass memory tokens between segments, as do
**burtsev2020memorytransformer** (Memory Transformer) and **wu2020memformer** (Memformer).
**hutchins2022blockrecurrent** (Block-Recurrent Transformers) and **hwang2024transformerfam**
(TransformerFAM, feedback attention as working memory) similarly recur a compressed state.
**wang2023longmem** (LongMem) and **klett2024extendedmind** (Extended Mind) instead bolt a frozen
kNN token-KV memory onto the model — the KV-reuse contrast to full recomputation.
**packer2023memgpt** (MemGPT) manages context by inference-only prompt paging with no trained memory
structure, an analog to document-context eviction. Across all of these the memory is either
compressed or sequential; document-granularity verbatim tokens inside a trained attention span is
the distinguishing element here.

### Dense retrieval backbones and ANN infrastructure

TAGSeq2TAGSeq uses no learned retriever, so this learned-similarity paradigm is the one it departs
from. **karpukhin2020dpr** (DPR) is the canonical learned dual-encoder;
**reimers2019sentencebert** (Sentence-BERT) is the foundational bi-encoder ancestor.
**khattab2020colbert** and **santhanam2022colbertv2** perform token-level late-interaction MaxSim
matching — matching, not attention, with no gradient across the passage boundary, versus an
attention edge computed in one forward pass. **izacard2022contriever** (Contriever) trains a dense
retriever contrastively without labels (a midpoint: unlabeled like this work, but a learned
similarity proxy rather than a true structural edge). **ni2022gtr** (GTR) scales the retriever;
**xiong2021ance** (ANCE) mines hard negatives from the model's own ANN index;
**formal2021splade** (SPLADE) learns sparse lexical expansion; **wang2022e5** (E5) is a widely used
weakly-supervised embedding baseline. The ANN indexing infrastructure that RAG systems depend on —
**johnson2019faiss** (FAISS), **guo2020scann** (ScaNN anisotropic quantization), and
**malkov2020hnsw** (HNSW) — is unnecessary when target resolution is an exact hashmap lookup with no
approximation error; HNSW is itself a graph built over embeddings, a useful juxtaposition against a
semantic document graph.

### KV-cache reuse and serving systems

These systems make attention cheaper by reusing or reorganizing the KV cache — precisely the
optimization TAGSeq2TAGSeq deliberately forgoes, since inserting a fetched node shifts RoPE positions
and makes paged/prefix KV reuse incorrect, forcing full O(T²) recompute for train/inference symmetry.
**kwon2023pagedattention** (PagedAttention/vLLM) is the canonical paged-KV system;
**zheng2024sglang** (RadixAttention) reuses exact shared prefixes via a radix tree, the strongest
reuse contrast; **ye2024chunkattention** (ChunkAttention) organizes prefix KV in a trie; and
**juravsky2024hydragen** (Hydragen) batches attention over shared prefixes. **hu2024epic** (EPIC)
is the closest analogue to the position-shift problem here: its position-independent caching exists
precisely because naive KV reuse is incorrect once positions move, as in insertion-based retrieval.
**gim2024promptcache** (Prompt Cache) precomputes KV for reusable prompt segments and adjusts
positions on splice-in; **chan2025cag** (Cache-Augmented Generation) preloads an entire knowledge
base into one monolithic corpus KV cache; **jin2024ragcache** (RAGCache) caches the KV of retrieved
RAG documents; **yao2025cacheblend** (CacheBlend) precomputes per-chunk KV and selectively recomputes
cross-chunk tokens — the closest inference analogue to fetching a target document into attention, but
it approximates cross-document interaction rather than training it. **ma2025blockattention**
(Block-Attention) is the most structurally similar: it applies a custom block-structured mask and
fine-tunes the model to it at both training and inference, re-indexing positions — the difference
being flat retrieved blocks versus a graph/link-structured mask where a generated link fetches a
specific target node. **liu2024cachegen** (CacheGen) compresses and streams KV for transport;
**yang2025ape** (APE) enables reuse via adaptive parallel encoding that approximates cross-chunk
attention; **yu2024pensieve** (Pensieve) keeps KV stateful across multi-turn serving; and
**ye2025flashinfer** (FlashInfer) is a customizable attention engine, a contrast to bespoke
mask-specific kernels.

---

## Graph-aware pretraining, text-attributed graphs, and graph transformers

This body of work shares our raw signal — documents connected by links, citations,
hyperlinks, entity mentions, or knowledge-graph edges — but almost universally injects
that structure through a different mechanism than ours. The recurring axis of
comparison is **how the edge enters the model**: as a training-time attention edge
(our approach: a hard, binary, block-sparse, direction-gated mask over a single 32k-token
autoregressive sequence, with no learned edge bias and no message passing), versus a
GNN message-passing edge, versus a soft attention bias / positional encoding, versus a
KV-cache or retrieval hop, versus a contrastive relevance-pair label. Keeping that
distinction sharp is what separates our thesis from every antecedent below.

### Pretraining directly on link structure (sharpest antecedents)

**yasunaga2022linkbert** — pretrains an encoder by placing two hyperlinked documents in
the same context and adding a document-relation prediction objective; the canonical
"train on links as context" antecedent. Uses links to *select* what co-occurs and a
symmetric encoder MLM objective, versus our decoder-causal, direction-gated attention
grant that makes the link itself the compute primitive.

**caciularu2021cdlm** — the sharpest cross-document pretraining comparison: pretrains a
Longformer over clusters of related (cross-document-coreference-linked) documents so that
cross-doc reading happens through global attention. Contrast is mechanistic — encoder MLM
with *symmetric* global attention versus our decoder, causal, asymmetric, directed,
link-gated attention.

**xiao2022primera** — LED pretraining over document clusters with a pyramid masked-sentence
objective; closest to our concatenated-cluster compute-control setup, but its clustering
is topical rather than an explicit per-link attention edge, and the objective is
summarization-oriented.

**yasunaga2022dragon** — self-supervised *joint* LM+KG pretraining (MLM plus KG
link-prediction) over aligned text–subgraph pairs; the sharpest knowledge-graph analog to
our objective. Differs in that the graph is a symbolic KG consumed via a GNN over retrieved
subgraphs, whereas our graph is the document-hyperlink structure consumed via an attention
mask.

### Knowledge-graph-enhanced language models

These inject KG signal into pretraining through auxiliary losses, GNN sub-modules, entity
embeddings, or verbalization, rather than through the attention pattern itself.

**liu2020kbert** — the closest attention-mask analog: injects KG triples into the input and
uses a visible/soft-position mask so injected knowledge attends only locally. Same
"structure as attention mask" spirit, but for inserted triples inside one document, not
cross-document links.
**wang2021kepler** — jointly optimizes a knowledge-embedding (TransE-style) objective and MLM
on entity-description text.
**yu2022jaket** — joint pretraining of a KG module and an LM with information flowing both ways.
**sun2020colake** — builds a word-knowledge graph and pretrains over the joint structure.
**qin2021erica** — contrastive entity/relation pretraining objectives.
**he2020kgplm** — generative + discriminative knowledge-guided pretraining objectives.
**lu2022kelm** — message passing over hierarchical relational graphs layered onto an LM.
**agarwal2021kelm** — verbalizes the KG into synthetic text for pretraining; the opposite of
keeping the graph explicit as structure.
**zhang2022greaselm** and **yasunaga2021qagnn** — fuse an LM with a GNN over a retrieved KG
subgraph at QA time (inference-time fusion, not pretraining).
**yamada2020luke** — entity-aware self-attention with entity embeddings as first-class tokens;
treats entities as typed nodes but stays within a single document.

### Text-attributed graph representation learning

All of these produce node embeddings or node labels — via a GNN, an LLM adapter, or
prompt-serialization of neighbors — rather than performing autoregressive generation with a
cross-document link as the pretraining attention edge. Sharpest delineations noted.

**zou2023thlm** — the nearest "pretrain an LM on a text-attributed heterogeneous graph," but
via a contrastive pretext task that yields embeddings, not generation.
**ye2024instructglm** — generative (an LLM emits labels) but neighbors are flattened into the
prompt via concatenation rather than an attention edge.
**chen2024llaga** — sequence-izes the local graph neighborhood into node-embedding tokens
(not raw neighbor text).
**chien2021giant** — graph-aware self-supervised text encoding via multi-scale neighborhood
prediction.
**duan2023simteg** — a simple two-stage LM-then-GNN pipeline for textual graphs.
**zhao2023glem** — an EM-style loop alternating between an LM and a GNN.
**he2024tape** — uses LLM-generated explanations as augmented node features.
**liu2024oneforall** — one graph model spanning classification tasks via a unified prompt.
**zhu2024engine** — efficient tuning/inference for LLMs on textual graphs.
**tang2024graphgpt** and **zhao2023graphtext** — graph instruction tuning / graph reasoning
in text space.
**wang2023nlgraph**, **fatemi2024talklikeagraph** — probe whether LLMs can solve graph
problems stated in natural language / study graph-to-text encodings.
**chen2024exploringllmgraphs** — survey-style exploration of LLMs on graphs.

Related text-attributed-graph transformers that use graph structure as an architectural
component rather than an attention-mask training signal:
**yang2021graphformers**, **jin2023patton**, **jin2023heterformer**, **jin2023edgeformers** —
interleave GNN message passing with text-transformer layers over node text.

### Graph transformers (structure as soft bias / PE / message passing)

Here nodes are tokens and structure enters as a soft attention bias, a positional encoding,
or a GNN sub-module over small graphs — versus our hard binary block-sparse mask over a
32k-token sequence with no learned edge bias and no message passing.

**dwivedi2021graphtransformer** — restricts attention to graph neighbors; the nearest ancestor
of "the graph defines who attends," but symmetric adjacency at the node level.
**shirzad2023exphormer** — the closest sparse-mask graph transformer, using expander-graph
edges to sparsify attention (versus our document-link edges).
**diao2023relationalattention** — generalizes attention to typed relational edges.
**rampasek2022graphgps** — a modular recipe combining message passing with global attention
plus positional/structural encodings.
**kim2022tokengt** — treats nodes and edges as plain tokens fed to a standard transformer.
**kreuzer2021san** — spectral (Laplacian eigenvector) attention.
**chen2023nagphormer** — tokenizes multi-hop neighborhoods for node classification.
**chen2022sat** — structure-aware attention via subgraph representations.
**wu2021graphtrans** — a GNN followed by a global-attention module for long-range context.
**zhang2020graphbert** — attention over sampled subgraphs with positional encodings only, no
explicit edges.
**ma2023grit** — graph inductive biases (random-walk structural encodings) without message
passing.
**park2022grpe** — relative positional encoding of graph distances.
**ying2021graphormer** — degree/spatial/edge encodings as attention bias terms.

### Hyperlink / anchor / markup pretraining

Two families: intra-document markup structure, and hyperlinks used as a *relevance-pair
label* for a retriever (same raw signal as ours, but a contrastive dual-encoder mechanism
rather than an in-context attention grant).

**wang2022webformer** — structure *as attention pattern*, but over the intra-page DOM tree
rather than cross-document hyperlinks; the closest markup analog to our masking idea.
**aghajanyan2021htlm** — trains on simplified HTML, treating markup as tokens to denoise
(markup is content, not an edge).
**li2022markuplm**, **deng2022domlm** — encode intra-document DOM structure as positional
features.
**gur2023understandinghtml** — LLMs consume raw HTML directly.
**chang2020pretrainingretrieval** — introduces a Wikipedia link-prediction pretraining task
for embedding-based retrieval.
**ma2021harp**, **zhou2022hlp** — hyperlinks as weak-supervision relevance pairs for
passage/ad-hoc retrieval (contrastive dual-encoder).
**zhang2020reinfoselect** — selective weak supervision from anchor text for neural IR.
**zhou2023master** — multi-task bottlenecked masked-autoencoder pretraining for dense retrieval.
**liu2023anchorprediction** — source→target span localization ("which span does this link land
on"); closest to "the linking document reads into the target," but framed as an evaluation
task, not a pretraining objective.

### Entity linking, wikification, and autoregressive-identifier retrieval

Our trie-constrained title index has its clearest antecedents here — models that
autoregressively *generate* an identifier under a prefix constraint — but they stop at the
identifier, whereas we fetch the target and attend to its content.

**decao2021genre** — the most relevant: autoregressively generates an entity title under a
prefix-trie constraint, exactly our TrieTitleIndex mechanism; but the generated title *is* the
answer, whereas for us the title is a fetch-key that pulls content into attention.
**bevilacqua2022seal** — generates substrings as document identifiers via an FM-index.
**tay2022dsi** — a transformer memorizes the corpus and generates document ids ("differentiable
search index").
**wang2022nci** — trie-constrained beam search over hierarchical doc ids.
(All four "generate-an-identifier-to-retrieve" — none then attend to the fetched content.)
**wu2020blink** — dense (bi-encoder) zero-shot entity linking, versus our deterministic
index-doc-span match.
**li2020elq**, **kolitsas2018endtoend** — joint mention detection and linking, versus our
regex/offset-based link detector.
**logeswaran2019zeroshotel** — zero-shot linking by reading entity descriptions.
**letitov2018latentrelations**, **hoffart2011aida** — global/coherence-based entity
disambiguation across mentions.
**ratinov2011wikification**, **mihalcea2007wikify** — the origin of wikification: *predict*
links into an encyclopedia, whereas we *consume* existing links.

### Cross-document coreference and multi-document discourse

**cattan2021crossdoc**, **barhom2019crossdoc** — cross-document entity/event coreference; here
links are the model *output*, versus our use of links as an input attention bias.
**caciularu2022longcontext** — supervised contrastive learning for long-context (cross-doc) QA.
**liu2019hiersumm**, **li2020graphsum** — task-specific hierarchical / graph attention for
multi-document summarization (not pretraining).
**fabbri2019multinews** — a multi-document summarization dataset with a generic
concatenation-based hierarchical baseline.
**huang2021efficient** — efficient attention variants for long-document summarization.
**barzilay2008entitygrid**, **grosz1995centering** — classical discourse-coherence theory
(entity grids, centering) that motivates ordering and targets-first document layout.

### Graph-learning and random-walk foundations

The traversal-packing method draws on classical node-sampling and random-walk theory
rather than on GNN message passing. DeepWalk (**perozzi2014deepwalk**) established that
a random walk over a graph can be treated as a training sequence — the premise behind
packing a traversal into a token stream, though DeepWalk feeds a shallow skip-gram over
node ids and discards node content. node2vec (**grover2016node2vec**) adds biased
second-order walks interpolating between breadth- and depth-first exploration, the
direct antecedent of the BFS/DFS/random-walk strategy axis (our walks are first-order
and distinct algorithms rather than a smooth p/q reweighting). GraphSAGE
(**hamilton2017graphsage**) introduced fixed-size neighbor sampling to build a node's
computation graph — the analogue of neighborhood-limited pack growth, but it aggregates
neighbor features into an embedding rather than concatenating documents into a causal
sequence. The random-walk-with-restart family — PageRank (**page1999pagerank**) and
Random Walk with Restart (**tong2006rwr**) — is the reference point for the restart
mechanism; note the packer restarts to a uniform-random node (PageRank-style
teleportation), not to the seed as in personalized PageRank / RWR.

---

## Sequence packing, boundary masking, and efficient/sparse attention

This body of work sits directly upstream of our design: how documents are concatenated into training sequences, whether attention is allowed to cross document boundaries, and what kernel and sparsity machinery makes long packed sequences tractable. The organizing contrast is that prior packing either forbids cross-document attention (block-diagonal masking) or arranges related documents adjacently but still reads them under standard causal attention, whereas we grant cross-document attention *selectively along explicit link edges*, and prior sparsity is either a fixed geometric pattern or a learned-from-content template, whereas ours is *data-defined by the link graph*, causal, and applied at both train and generation time.

### Packing and sequence composition

**shi2024incontext** — In-Context Pretraining is the most important prior art: it packs related documents into a sequence via nearest-neighbor retrieval plus TSP-style ordering, but reads them under standard causal attention with no cross-document attention mechanism. We differ by defining explicit graph edges and a mask that *grants* cross-document attention along them.

**staniszewski2025structured** — SPLiCe collates retrieval-linked documents into long training examples; its SPLiCe-Repo variant uses repository/directory structure, the closest existing analog to edge traversal, but again adds no attention along the traversed links.

**zhao2024analysing** — Finds cross-document attention in packed sequences is *harmful* and masks it out entirely (intra-document causal masking). This is the key baseline motivating block-diagonal isolation, and precisely the finding we revisit: we selectively re-enable cross-document attention only along known link edges rather than banning it wholesale.

**ding2024fewer** — Best-fit bin-packing to reduce truncation and preserve document integrity; orthogonal to relatedness and to attention structure, but a relevant packing-mechanics reference.

**krell2021packing** — Origin of the "cross-contamination" framing and the demonstration that masked packing is equivalent to unpacked training; it is the strongest framing foil, since our cross-document link is exactly the *controlled violation* of the isolation it enforces.

**wang2024packinganalysis** — Quantifies contamination harm across 8B–70B models in supervised fine-tuning, motivating our matched-compute controls.

**fu2024data128k** — Data engineering for scaling to 128K context; informs the long-context data-composition regime we operate in.

**brown2020gpt3**, **touvron2023llama** — Canonical concat-with-delimiter packing recipes (GPT-3, LLaMA); standard-practice contrast points. **raffel2020t5**, **tay2022ul2** — span handling and the principle of separating the training objective from architecture (T5, UL2).

### Boundary masking and position handling

Our block-isolation mask and our choice *not* to reset RoPE positions per document within a packed sequence are the concerns here.

**kundu2024packingflash** — Position-id reset with FlashAttention in the HuggingFace stack, the direct analog of our varlen boundary-isolation kernel path. **shoeybi2019megatron** — Megatron's reset-attention-mask / reset-position-ids at document boundaries, the doc-boundary masking primitive we generalize.

On not resetting positions: **kazemnejad2023positional** and **ruoss2023randomized** defend the choice — NoPE generalizes better in length and positions are largely interchangeable under sufficient variety — while **chen2023positioninterpolation** (Positional Interpolation) marks the risk, since large untrained relative RoPE offsets can destabilize. Cite both sides for the "why no position reset" reviewer concern.

### Curriculum, data ordering, and density scheduling

**bengio2009curriculum** — The definitional curriculum-learning anchor. **kocmi2017curriculum** — Minibatch bucketing for length/difficulty homogeneity. **pouransari2024datasetdecomposition** — Dataset Decomposition buckets sequences by length/cost and samples a variable-length curriculum; it is both the one-document-per-sequence alternative to packing and the strongest match to our density bucketing, which reuses difficulty-bucket machinery for *compute balancing* rather than learnability, adding per-step cross-rank synchronization as a novel axis.

### FlashAttention and programmable-attention machinery we build on

**dao2022flashattention**, **dao2023flashattention2**, **shah2024flashattention3** — The IO-aware exact-attention kernel lineage (FA1/2/3) that makes 32K sparse training tractable. **flexattention_blog2024**, **flexattention_paper2024** — The FlexAttention programming model we build on: a `mask_mod` predicate compiled to a BlockMask that skips fully-masked KV blocks. Its worked example is block-diagonal document masking; we generalize to a block-sparse *graph* mask that lights off-diagonal blocks along link edges.

**rabe2021selfattention** (O(n²)-memory-free attention) and **milakov2018onlinesoftmax** (online softmax normalizer) are the basis for our per-KV-block accumulation and log-sum-exp combine, including the sentinel-LSE guard for fully-masked rows.

**pagliardini2023sparseflash** — The closest precedent for a flash kernel consuming a *runtime* sparse block pattern, but content/hash-driven rather than graph-link-driven. **wang2024flashmask** — Column-wise sparse masking, a competitor to FlexAttention's `mask_mod`; its contiguous-interval masks cannot express our arbitrary A→B block rectangles, which motivates our bit-packed attention grants.

**lefaudeux2022xformers** — Memory-efficient attention library (infrastructure lineage). **flashdecoding2023**, **hong2023flashdecodingpp**, **ye2025flashinfer** — Long-context and serving-side decode kernels relevant to the inference regime of link-fetch decoding.

### Kernel and systems foundation

**tillet2019triton** — Triton, the compiler underlying our entire custom-kernel stack. **gray2017blocksparse** — Block-sparse GPU kernels, the direct ancestor of our block-isolation mask; they skip empty blocks but sparsify *weights* statically, whereas we sparsify *scores* data-dependently per batch with per-block-class dispatch. **markidis2018tensorcore** — Tensor-core precision (bf16 multiply, fp32 accumulate). **ivanov2021datamovement** — Data-movement analysis of transformers, the systems justification for our no-permute (THD) layout. **hsu2024liger** — Liger Triton kernels, the closest analog to and basis for our fused MLP / fused cross-entropy. **thakkar2023cutlass** — CUTLASS tensor-core GEMM lineage (context).

### Fixed-pattern and learned sparse attention

Fixed geometric sparsity: **child2019sparsetransformers** (strided/local Sparse Transformers), **beltagy2020longformer** (sliding-window + global tokens), **zaheer2020bigbird** (random + window + global), **ainslie2020etc** and **guo2022longt5** (local + global template for structured/long text). Learned-from-content sparsity: **kitaev2020reformer** (LSH buckets), **roy2021routing** (k-means routing), **tay2020sinkhorn** (differentiable sorting), **sukhbaatar2019adaptivespan** (learned window length), **gao2024seerattention** (learned mask predictor). All define sparsity by geometry or content similarity, not by document identity or link structure.

**yuan2025nsa** (DeepSeek Native Sparse Attention) and **lu2025moba** (Kimi Mixture of Block Attention) are the sharpest recent contrasts: natively *trained* block sparsity like ours (not an inference-time bolt-on), but with learned importance or a router over the same sequence, versus our graph-dictated key ranges. **tay2022efficientsurvey** — The efficient-transformers taxonomy, which positions our relational/structure-defined sparsity as a distinct category. **ainslie2023colt5** — Conditional-computation axis (we do not use it). **ying2021graphormer** — Attention over graphs via soft shortest-path/edge *bias* terms on small dense graphs; we instead impose a *hard* block-sparse mask over long token sequences.

### Retrieval into attention (contrast)

These augment attention with external memory at inference rather than training cross-document attention as a pretraining objective, the archetypal contrast to our approach. **wu2022memorizing** — Non-differentiable frozen kNN KV memory. **borgeaud2022retro** — RETRO's frozen datastore with embedding-similarity neighbors and chunked cross-attention. **bertsch2023unlimiformer** — Inference-time kNN over encoder states (encoder-decoder). **tworkowski2023focused** (LongLLaMA) — Trains to improve memory attention, but context remains an external kNN memory at inference. **mohtashami2023landmark** — Learned block retrieval at inference within a single sequence, not pretraining over an external document graph.

### SSM and linear-time models (positioning-only)

These compress history into a fixed recurrent state or global-kernel summary, which forecloses content-addressed cross-document attention: a linking token must read a *specific* target's retained per-token KV, which requires softmax attention plus sparsity rather than a summarized state. None train on graph structure. **gu2022s4** (canonical SSM), **gu2023mamba** (selective/content-gated), **dao2024mamba2** (state-space duality), **peng2023rwkv** and **katharopoulos2020linear** (linear-attention RNNs discarding individual KV), **choromanski2021performer** (random-feature softmax approximation), **sun2023retnet** (decay retention), **poli2023hyena** (long convolution), **arora2024based** (recall-throughput tradeoff, relevant to our long-range recall argument), **de2024griffin** (gated linear recurrence with local attention only).

### Distributed and sequence-parallel attention (orthogonal scaling axis)

We use plain DDP plus density bucketing at the data-scheduling layer; these are orthogonal scaling mechanisms and useful contrasts. **liu2023blockwise** (Blockwise Parallel Transformer) and **liu2023ringattention** (Ring Attention) — blockwise/distributed near-infinite context. **korthikanti2022sequence** (Megatron sequence parallelism), **jacobs2023ulysses** (DeepSpeed-Ulysses), **li2021sequenceparallel**, **fang2024usp** (unified sequence parallelism). **brandon2023striped** (Striped Attention) and **li2023distflashattn** (DistFlashAttn/LightSeq) are the closest analogs to our balancing concern: they fix ring-causal load imbalance via token-level load balancing, versus our per-rank *density* balance. **narayanan2021megatron** (PTD-P), **huang2019gpipe** (pipeline parallelism), and **rajbhandari2020zero** (ZeRO optimizer-state sharding) round out the parallelism landscape; ZeRO is the precedent for our bespoke Muon round-robin shard with replicated AdamW and portable name-keyed state.

---

## Optimizer, Learning-Rate Schedule, Architecture, and Training Systems

The methods below are prior art that this project adopts, adapts, or positions against; none is claimed as a contribution here. The Muon lineage and the modded-nanoGPT speedrun recipes supply the recipe backbone, while a broad optimizer, scheduling, architecture, and systems literature situates the specific engineering choices.

### Muon and orthogonalization-based optimizers

**jordan2024muon** is Muon itself — orthogonalized momentum updates for 2D hidden-layer weights, the optimizer this project builds on. **liu2025muonscalable** demonstrates Muon scaling to LLM-scale training and motivates its use beyond speedrun toys. **amsel2025polarexpress** (arXiv:2505.16932) is the orthogonalization backend actually used (`kernels/polar_express.py`, cited in `muon.py`): it computes per-iteration, non-stationary quintic coefficients via a minimax matrix-sign construction, is bf16-targeted, and replaces the classic Newton–Schulz-5 iteration. **higham2008functions** is the classical reference for Newton–Schulz iteration, the matrix sign function, and the polar decomposition underpinning all of these schemes.

**li2025normuon** (NorMuon, arXiv:2510.05491) is a published paper — not an in-repo coinage — proposing neuron-wise normalized Muon: a per-neuron second moment applied as a renormalization after orthogonalization, integrated with FSDP2. This project's optimizer is best described as "NorMuon-style," using a rank-1 Adafactor-style second-moment estimate with a Frobenius renormalization as an adaptation of that idea.

**bernstein2024oldoptimizer** (Old Optimizer, New Norm) and **bernstein2024modularduality** (Modular Duality) provide the norm/duality theory behind per-parameter-group norm choices and the rectangular Newton–Schulz update that is the direct ancestor of Muon. **large2024modularnorm** (Scalable Optimization in the Modular Norm) supplies the shape-scaling rationale, i.e. the `max(1, M/N)^0.5` factor. **pethick2025scion** (norm-constrained LMOs / Scion), **ahn2025dion** (Dion, distributed orthonormalized updates), and **riabinin2025gluon** (Gluon) are the LMO/orthogonalization sibling family; Dion in particular is a direct competitor to this project's round-robin sharded distributed-Muon implementation.

### Preconditioned and second-order optimizers

**gupta2018shampoo** (Shampoo) and **anil2020distributedshampoo** (Scalable Second Order Optimization for Deep Learning) define the full-matrix preconditioning family, and **vyas2024soap** (SOAP) stabilizes Shampoo by running Adam in its eigenbasis. **liu2023sophia** is a scalable second-order (diagonal Hessian) optimizer for LM pre-training. These bound the "how much curvature information" design axis around Muon's orthogonalization.

### Cautious, sign-based, and adaptive updates

**liang2024cautious** (Cautious Optimizers, C-Optim) introduces the sign-aligned gating that masks updates disagreeing with the gradient — the lineage behind this project's cautious update logic (`muon.py:237-257`). **chen2025cautiouswd** (Cautious Weight Decay) is the direct source for the cautious weight-decay variant used here: it applies decay only where it aligns with the update sign, and is demonstrated on Muon. **chen2023lion** (Lion, symbolic optimizer discovery) and **bernstein2018signsgd** (signSGD) are the sign-based update relatives, and **you2020lamb** (LAMB) is the layerwise-adaptive large-batch relative.

### Memory-efficient optimizer state

**shazeer2018adafactor** (Adafactor) is the rank-1 factored second-moment estimator that this project's NorMuon-style renormalization descends from. **dettmers2022eightbit** (8-bit optimizers via block-wise quantization), **zhao2024galore** (gradient low-rank projection), **zhang2024adammini** (fewer learning rates), and **luo2023came** (confidence-guided memory-efficient optimization) span the broader optimizer-state-compression landscape.

### Low-precision training and stochastic rounding

**gupta2015limited** (deep learning with limited numerical precision; stochastic rounding) and **zamirai2020bfloat16** (Revisiting BFloat16 Training; Kahan summation) are direct prior art for this project's bf16 mantissa-tracking uint16 shadow state — all three address the same failure mode where small updates vanish in low-precision accumulation. **micikevicius2018mixed** (Mixed Precision Training) is the loss-scaling / master-weight foundation for mixed-precision training generally.

### Baseline optimizers

**kingma2015adam** (Adam) and **loshchilov2019adamw** (decoupled weight decay, AdamW) are the baselines; AdamW is retained as the auxiliary optimizer for parameters Muon does not cover (embeddings, scalars, 1D tensors).

### Learning-rate schedules, warmup, and weight-decay scaling

The scheduler here is warmup–stable–decay (WSD): a short linear warmup (~300 steps), a stable plateau, then a linear decay to a floor (`min_lr_ratio` 0.1) over a cooldown fraction (~0.4). **hu2024minicpm** introduced WSD and observed loss dropping sharply in the decay phase; it favors exponential decay, in contrast to the linear decay used here. **hagele2024wsd** establishes WSD scaling laws by branching cooldowns at multiple horizons, justifying decoupling the maximum optimizer-step count from the cooldown length. **bi2024deepseekllm** uses a constant-then-multistep decay and motivates continual reuse of checkpoints, which underlies the absolute-step WSD-clock resume behavior; **ibrahim2024continual** provides the rewarm-then-redecay continual-pretraining substrate for that resume.

**loshchilov2017sgdr** (SGDR cosine) and **smith2017cyclical** (cyclical LR) are the cosine/cyclical baselines that WSD argues against. **goyal2017imagenet1hour** is the origin of linear warmup plus the linear LR-scaling rule for large batches, and **wortsman2023smallproxies** explains why warmup and related stability tricks work via small-scale instability proxies. **kosson2024warmup** shows that normalized-update optimizers need less warmup — a live tension with using Muon alongside a 300-step warmup, worth a methods footnote. **wang2025adamwwd** analyzes AdamW's weight-decay-times-learning-rate coupling as an EMA timescale, relevant to the auxiliary AdamW settings and the interaction of learning rate with cautious weight decay.

### Architecture backbone

**vaswani2017attention** (the Transformer), **radford2019gpt2** and **brown2020gpt3** (decoder-only LM scaling) are the standard backbone. **su2021roformer** (RoPE) and **zhang2019rmsnorm** (RMSNorm) are the positional-encoding and normalization primitives; **so2021primer** contributes the squared-ReLU MLP and **shazeer2020glu** the SwiGLU/GLU variants. **gemmateam2024gemma2** is the source for logit soft-capping. **press2017tying** and **inan2017tying** justify tying input and output embeddings. **ainslie2023gqa** (grouped-query attention) is the KV-head-sharing option in the attention design space.

### Attention normalization, RoPE base, and depth stability

**henry2020qknorm** is the origin of query-key normalization (RMSNorm applied to q and k here), and **liu2022swinv2** (scaled-cosine attention), **dehghani2023vit22b** (QK-norm for stability at scale), and **chameleon2024** show why QK-norm became standard in the speedrun lineage. **peng2023yarn** (YaRN) is the citeable vehicle for NTK-aware RoPE extension; this project instead trains from scratch at a small base (θ=1024, half-dimension), and **liu2024ropescaling** together with **men2024ropebase** form the two-sided RoPE-base theory that both defends and caveats that small-base choice. **loshchilov2024ngpt** (nGPT hypersphere normalization) is the radical normalization endpoint. **touvron2021cait** (CaiT LayerScale) anchors the learned residual-scaling coefficients, while **xiong2020layernorm** (pre- vs post-norm) and **wang2022deepnet** (scaling to 1000 layers) provide the residual/norm depth-stability stack.

### Residual, initialization, and speedrun tricks (context only)

These situate architectural micro-choices whose exact forms have no formal paper and derive from the modded-nanoGPT repo (**jordan2024moddednanogpt**, with **karpathy2022nanogpt** as the baseline it speedruns); they are context rather than adopted methods with clean citations. **zhou2024valueresidual** is the best formalization of value-residual / shared-value-embedding ideas, adjacent to the token-conditioned value-embedding banks used here (the per-token bank plus zero-gate itself traces to the modded-nanoGPT repo). **pagliardini2024denseformer** (depth-weighted averaging including the embedding) is the closest published analogue to the x0 highway, and **ronneberger2015unet** with **nawrot2021hourglass** motivate the U-Net-style skip topology. **bachlechner2020rezero**, **zhang2019fixup**, and **de2020skipinit** are the zero-initialized residual-scaling references behind the residual lambdas, x0 gate, and zero-initialized output projections. **yang2022tensorprogramsv** (muP), **lingle2024mup** (large-scale µ-transfer), and **blake2024umup** (unit-scaled µP) are the maximal-update-parametrization basis for calling the zero-init scheme "muP-friendly." **roy2022ngrammer** (latent n-grams) and **svenstrup2017hashembeddings** (hash embeddings) underpin the bigram-hash embedding. **geiping2023cramming** and **portes2023mosaicbert** capture the efficient-pretraining-recipe ethos.

### Multi-token prediction

**gloeckle2024mtp** is the origin of multi-token prediction, the basis for the offset-2/3 auxiliary loss (reusing the same head rather than separate heads). **deepseekv3** deploys MTP as a training objective at scale, and **cai2024medusa** is the inference-time speculative-decoding counterpart with multiple decoding heads.

### Scaling laws and token budgets

**kaplan2020scaling** and **hoffmann2022chinchilla** set the compute-optimal scaling and token-budget framing for model and dataset sizing.

### Training systems, distributed execution, and fault tolerance

**hsu2024liger** (Liger Kernel) is used directly — the fused-linear cross-entropy loss wraps `LigerFusedLinearCE` (with a dynamo-disable workaround). **paszke2019pytorch**, **li2020pytorchddp** (DDP), and **ansel2024pytorch2** (torch.compile) are the framework and compilation substrate (ansel2024pytorch2 venue/DOI unverified); **submitit** is the SLURM launch tool. **patarasuk2009allreduce** is the ring all-reduce algorithm underlying NCCL and the DDP plus distributed-Muon all-gather collectives.

**zhao2023fsdp** (PyTorch FSDP) is the sharding contrast point: this project shards only the optimizer state via a bespoke round-robin distributed Muon while replicating AdamW, rather than sharding parameters. **chen2016checkpointing** is activation (gradient) checkpointing; **ren2021zerooffload** and **rajbhandari2021zeroinfinity** are the offload foils this project deliberately avoids by keeping state on-device (fighting a host-RAM spike at checkpoint-save time instead). **micikevicius2022fp8formats** and **peng2023fp8lm** mark the FP8 precision frontier that this project stops short of (staying in bf16).

**zhang2022opt** and **bigscience2022bloom** are large-run reliability chronicles supporting the observed host-OOM barrier and compile-warmup findings. **mohan2021checkfreq** (frequent fine-grained checkpointing), **wang2023gemini** (in-memory checkpoints for fast recovery), and **athlur2022varuna** (elastic low-cost training) cover the checkpointing and fault-tolerance space; Varuna is the closest analogue to this project's world-size-portable resume.

---

## Code language models, repository-level context, and cross-file benchmarks

The central axis for positioning this project runs through the code-LM literature: cross-file signal can enter a model either at **pretraining time** (how documents are packed, ordered, or wired into attention) or at **inference time** (retrieval, prompt assembly, KV augmentation). This project trains an import-graph edge directly into attention over a cross-document corpus. The two closest precedents are **guo2024deepseekcoder** (import-topological file ordering in pretraining — same signal, but expressed only as concatenation order) and **guo2021graphcodebert** (dependency edges as graph-guided attention masks — same mechanism, but intra-file dataflow in an encoder). Everything else falls between or beside these two poles.

### Code-LM pretraining: single-file and function-level

These models pretrain on code as flat streams of files or functions, discarding the cross-file dependency graph; they are the compute-matched "concat-only" baseline class.

- **nijkamp2023codegen** — autoregressive code LM (CodeGen) trained on function/file-level program synthesis; single-file, no cross-file structure.
- **nijkamp2023codegen2** — follow-up studying FIM objectives and data mixtures; file-level lessons, no dependency structure.
- **zheng2023codegeex** — multilingual code LM with the HumanEval-X benchmark; single-file generation.
- **roziere2023codellama** — long-context code models that ingest multiple files as a concatenated stream; the long-context-as-concat setting is exactly this project's concat compute-control (no edge signal).
- **xu2022polycoder** — systematic evaluation of code LMs; function-level.
- **li2022alphacode** — competition-level generation via massive sampling; problem-level, not repository context.
- **gunasekar2023phi1** — "textbooks are all you need"; data-quality contrast point (curation vs. structural signal).
- **codegemma2024** — open code models on the Gemma base; single-file.
- **luo2024wizardcoder** — Evol-Instruct instruction tuning for code; post-training, not structural pretraining.
- **allal2023santacoder** — small code LM and precursor to StarCoder, notable for dedup and filtering ablations on The Stack; flat concat.

The large open corpora and their scaled models likewise pack files without graph structure:

- **kocetkov2022stack** — The Stack, 3TB permissively licensed source; the dominant code-pretraining corpus, stored as independent files.
- **li2023starcoder** / **lozhkov2024starcoder2** — StarCoder and StarCoder2 (the latter with The Stack v2); repo-aware data assembly but flat concatenation with plain causal attention, no edge-keyed cross-file attention.
- **hui2024qwen25coder** — Qwen2.5-Coder, with an explicit repo-level training stage that is still flat concatenation of repository files.
- **zhu2024deepseekcoderv2** — scaled successor to DeepSeek-Coder; inherits the import-topological-sort concat recipe (ordering, not attention mask).

Function-level corpora that explicitly drop the import graph:

- **manh2023vault** — large multilingual function/docstring dataset.
- **husain2019codesearchnet** — function-plus-docstring pairs for semantic code search.

### Repository structure injected during pretraining

The most relevant prior work: models that let cross-file or syntactic structure shape pretraining itself.

- **guo2024deepseekcoder** — the closest code prior art. Topologically sorts a repository's files by their import graph before packing, then does flat concatenation under plain causal attention. It uses the same signal this project uses (import edges) but expresses it only as document order, with no edge-keyed cross-file attention — precisely the gap this project targets.
- **guo2021graphcodebert** — the key mechanistic comparison. Encodes data-flow edges as a graph-guided mask over attention during pretraining. This project can be framed as lifting GraphCodeBERT's graph-attention from intra-function variable dataflow (encoder, MLM, short context) to a cross-document corpus-level import graph (decoder, generative, long context).
- **guo2023longcoder** — sparse-attention decoder for long code (sliding window plus bridge and memory tokens); reaches distant context by position heuristics, not by dependency-graph fetch.
- **tipirneni2022structcoder** — structure-aware transformer conditioning generation on AST and dataflow; intra-file structure.
- **jiang2021treebert** — tree-based pretraining over AST paths; intra-file syntax.
- **wang2021syncobert** — syntax-guided multi-modal contrastive pretraining; intra-file.
- **wang2022codemvp** — multi-view (source, AST, CFG) contrastive pretraining; intra-file views.
- **guo2022unixcoder** — flattens the AST into the token stream (serialize rather than mask); unified cross-modal representation.
- **wang2021codet5** — identifier-aware encoder-decoder pretraining; token/identifier structure, single-file.
- **wang2023codet5plus** — open encoder-decoder code LMs scaling the CodeT5 line; single-file.
- **guo2021grammformer** — completion with grammar-derived sketches; syntactic scaffolding, not cross-file.
- **zhang2024codesage** — representation learning at scale emphasizing dedup and data volume; structure-lite, useful as a scale/compute-control framing.

### Inference-time cross-file context: retrieval, graphs, and prompt assembly

This is the contrasting line: cross-file information is supplied per query at inference (retrieved snippets, graph indices, KV augmentation, or assembled prompts) rather than trained into the weights.

- **zhang2023repocoder** — iterative retrieve-then-generate for repository completion; the headline inference-retrieval method and canonical foil.
- **lu2022reacc** — retrieval-augmented completion (retrieve-then-complete) baseline.
- **tang2023knmlm** — code kNN-LM doing logit interpolation against a decoupled domain datastore (KV/datastore augmentation, not trained edges).
- **cheng2024draco** — DraCo, dataflow-guided retrieval augmentation; the closest inference-time analog to this project's import graph, but injects a dataflow context graph into the prompt per query rather than training an attention edge.
- **liu2024graphcoder** — code context graph used as a retrieval index at inference.
- **ouyang2024repograph** — repository-level code graph as an inference-time tool/index for AI software engineering.
- **phan2024repohyper** — search-expand-refine over semantic graphs for repo completion; inference-time graph reasoning.
- **ding2024cocomic** — jointly models in-file and retrieved cross-file context at inference (co-modeling, still retrieval-fed).
- **shrivastava2023repofusion** — trains a model to fuse multiple retrieved repo contexts; retrieval-conditioned training, but fuses retrieved chunks rather than wiring the dependency graph into attention.
- **shrivastava2023rlpg** — learns which repo context to place in the prompt (repo-level prompt generation).
- **liao2023a3codgen** — three-channel prompt assembly (local, global, third-party-library aware).
- **wu2024repoformer** — selective retrieval that learns *when* to retrieve; a midpoint on the train-vs-inference axis and a mirror of this project's firing-conditioned concern. Also the source of the CrossCodeLongEval benchmark. (This project verified its release is Python-only.)
- **pei2023bettercontext** — studies non-local context for function-call argument completion at both training and inference; the closest train-vs-inference precedent, but supplies context as concatenated tokens, not an attention edge.
- **agrawal2023monitor** — monitor-guided decoding that queries static-analysis monitors during generation to enforce global correctness; inference-time global context.
- **bairi2024codeplan** — repository-level coding as an LLM-plus-planning agent loop; inference-time orchestration.
- **jain2024reporift** — LLM agents for semantic code *search* (retrieval quality), adjacent to but not code completion.

### Fill-in-the-middle and infilling objectives

- **bavarian2022fim** — the FIM training objective (sequence permutation so a causal LM infills within a document); this project generalizes the permutation across *linked* documents plus an attention grant.
- **fried2023incoder** — generative code infilling and synthesis at scale (InCoder); FIM over flat-concatenated Stack code, no attention edge.

(SantaCoder, above, also trains FIM at scale on The Stack under flat concatenation.)

### Grammar-, type-, and static-analysis-constrained decoding, plus tooling

Constrained-decoding methods that restrict the output space to valid programs; this project's title/identifier trie is a domain instance of the same idea.

- **scholak2021picard** — PICARD, incremental parsing to reject invalid continuations during autoregressive decoding.
- **geng2023gcd** — grammar-constrained decoding for structured tasks without finetuning.
- **willard2023outlines** — efficient guided generation via finite-automaton-indexed regex/grammar masks.
- **ugare2024syncode** — grammar-augmented generation with incremental parser feedback.
- **poesia2022synchromesh** — combines retrieval with constrained decoding (target-language grammar plus example retrieval) for reliable code generation.
- **park2024grammaraligned** — grammar-aligned decoding; a caveat that naive grammar constraints distort the model's sampling distribution.
- **mundler2025typeconstrained** — type-constrained generation using static analysis to prune type-incorrect continuations (semantic, not just syntactic constraint).
- **brunsfeld2018treesitter** — tree-sitter incremental parser; a direct tooling dependency here, used for scope resolution, import extraction, and the Tier-1 oracle.

### Cross-file and repository-level benchmarks

The evaluation landscape. RepoBench is the headline benchmark for repository-level completion; most others evaluate with Pass@k or execution rather than this project's paired per-token Δnll.

- **liu2024repobench** — RepoBench, the headline benchmark for repository-level code auto-completion across retrieval, completion, and pipeline settings.
- **ding2023crosscodeeval** — CrossCodeEval, diverse multilingual cross-file completion requiring genuine cross-file dependency use.
- **deng2024r2c2coder** — R2C2-Coder, real-world repository-level completion benchmark and enhancement method; the source of this project's TypeScript-comparison sample.
- **li2024evocodebench** — EvoCodeBench, an evolving benchmark aligned to real repositories (mitigates contamination).
- **li2024deveval** — DevEval, manually annotated repository-aligned generation.
- **yu2024codereval** — CoderEval, pragmatic (non-standalone) function generation needing project context.
- **du2023classeval** — ClassEval, class-level generation.
- **bogomolov2024longcodearena** — Long Code Arena, a suite of long-context code benchmarks (project-level completion, module summarization, etc.).
- **liu2024repoqa** — RepoQA, long-context code *understanding* (needle-style function retrieval).
- **jimenez2024swebench** — SWE-bench, execution-based resolution of real GitHub issues; the extreme opposite end (agentic, whole-repo, patch-and-test).
- **wang2024coderagbench** — CodeRAG-Bench, the sharpest benchmark-side contrast: it measures the benefit of retrieval *at inference*, directly against this project's trained-edge alternative.
- **zhuo2024bigcodebench** — BigCodeBench, diverse function calls and complex instructions; function-level.
- **wang2023odex** — ODEX, execution-based open-domain generation; function-level.
- **chen2021humaneval** — HumanEval (the Codex evaluation set); the canonical single-file function-synthesis benchmark.
- **austin2021mbpp** — MBPP, basic single-file Python programming problems.
- **muennighoff2024octopack** — OctoPack / HumanEvalPack, instruction tuning and multilingual code-editing evaluation.
- **lu2021codexglue** — CodeXGLUE, the broad multi-task code understanding/generation benchmark suite.

**Project port sources.** Two benchmarks are the direct data sources for this project's non-Python ports:

- **ustalov2025contextcollection** — the JetBrains/Mistral ASE-2025 Context Collection Challenge (Python and Kotlin); the source for this project's Kotlin port (Zenodo 16964765, CC-BY-4.0).
- **li2025aixcoder** — aiXcoder-7B-v2 on long-context repository completion (ASE 2025); the intended Go port source. The associated dataset appears as **COLA-132K** in the paper but **CoLT-132K** in the released code and Zenodo record — the same corpus under two names. The Go port was ultimately removed (empty cross-file dependencies in the release).

---

## Multi-hop QA, text-attributed-graph corpora, and evaluation methodology

This section covers the benchmarks, corpora, and evaluation practices that frame TS2TS. The recurring distinction throughout is whether prior work uses link structure at *inference* (retrieve-then-read pipelines, external GNN readers, PPR walks, prompt-time decomposition) versus baking graph proximity into *pretraining* as a learned cross-document attention edge, which is what TS2TS does.

### Multi-hop QA benchmarks

**yang2018hotpotqa** — HotpotQA is the headline benchmark: 2-hop questions over two Wikipedia paragraphs with sentence-level supporting-fact annotations. It supplies the cross-document setting we construct (answer paragraph plus its linked supporting document), letting us test cross-doc attention over a graph edge against flat concatenation. **welbl2018wikihop** — WikiHop/MedHop, the canonical "reasoning across linked documents" formulation. **ho2020twowiki** — 2WikiMultiHopQA, harder and evidence-path annotated, mitigating HotpotQA reasoning shortcuts. **trivedi2022musique** — MuSiQue builds 2–4-hop questions by single-hop composition, stress-testing beyond two hops. **jiang2020hover** — HoVer extends the many-hop setting from QA to claim verification. **geva2021strategyqa** — StrategyQA probes implicit multi-step reasoning strategies. **press2023bamboogle** — Bamboogle, a clean 2-hop compositionality probe (source of the compositionality-gap framing).

Broader multi-hop and complex-QA datasets round out the benchmark landscape: **talmor2018complexwebquestions** (ComplexWebQuestions, composed SPARQL over KB traversal); **saha2018csqa** (complex sequential QA over a knowledge graph); **khot2020qasc** (two-fact sentence composition, multiple-choice-scorable like ARC/OpenBookQA); **chen2020hybridqa** (multi-hop over table cells linked to hyperlinked passages — itself a text-attributed-graph structure); and **wolfson2020break** (Break/QDMR, which formalizes what a "hop" is via question-decomposition meaning representations). Recent multi-doc retrieval benchmarks form a deliberate contrast, since they assume retrieval at inference rather than a trained structural edge: **tang2024multihoprag** (MultiHop-RAG), **zhu2024fanoutqa** (FanOutQA, multi-hop multi-document), and **krishna2025frames** (FRAMES, unified retrieval-augmented evaluation). Two newer benchmarks directly answer contamination-related reviewer concerns: **schnitzler2024morehopqa** (MoreHopQA adds a final hop that cannot be shortcut, certifying genuinely multi-hop reasoning) and **gong2025phantomwiki** (PhantomWiki generates synthetic contamination-free linked-wiki datasets on demand, addressing HotpotQA-2017 leakage).

### Single-document controls

These commonsense/knowledge multiple-choice benchmarks are single-document by construction and should be unaffected by a cross-document attention edge; they serve as controls that isolate the multi-hop effect. **zellers2019hellaswag** (HellaSwag), **clark2018arc** (ARC), **bisk2020piqa** (PIQA), **sakaguchi2020winogrande** (WinoGrande), **mihaylov2018openbookqa** (OpenBookQA), **clark2019boolq** (BoolQ).

### Multi-hop reasoning methods

*Graph-reader models (external GNN over an explicit evidence graph, at inference).* **decao2019entitygcn** — Entity-GCN builds a graph over mention nodes (same-doc / string-match / coref edges) across candidate documents and runs an R-GCN to propagate for answer selection; the purest "read across documents by message-passing" precursor, but the graph is a separate GNN module at task time with coref/match edges and no LM pretraining on structure. **song2018mhqagrn** — Coref-GRN/MHQA-GRN, same-era Graph Recurrent Network over a coreference-linked evidence graph; identical contrast to Entity-GCN. **ding2019cognitivegraph**, **tu2019hde**, **qiu2019dfgn** — Cognitive Graph, Heterogeneous Document-Entity graph, and Dynamically Fused Graph Network are the canonical HotpotQA-era graph readers; TS2TS replaces the bolt-on GNN reader with native attention across TAG edges inside a decoder-only LM. **kundu2019pathnet** — PathNet extracts and scores explicit reasoning paths (entity chains through intermediate passages); architecturally analogous to link-following, but realized as external path enumeration plus neural scoring rather than a learned attention grant.

*Question decomposition and prompting (inference-time, no structural training).* **min2019decomprc** — DecompRC decomposes a multi-hop question into single-hop sub-questions answered by an off-the-shelf RC model; the canonical decomposition baseline. **wei2022cot** — chain-of-thought elicits intermediate reasoning from a frozen LM by few-shot exemplars, with multi-hop capability emergent from scale rather than retrieval or graph. **zhou2023leasttomost** — least-to-most prompting reduces a problem to an ordered sequence of easier subproblems. **khot2023decomp** — Decomposed Prompting, a modular controller-LM framework dispatching sub-tasks to handler prompts/tools. **trivedi2023ircot** — IRCoT interleaves chain-of-thought with retrieval. All frame TS2TS's alternative: supply the right documents into context via a trained link edge rather than verbalized decomposition of a frozen model.

*Learned and graph-based retrieval (external index/graph at query time).* **xiong2021mdr** — Multi-hop Dense Retrieval iteratively re-encodes question-plus-retrieved-passage to fetch the next hop; the closest learned iterative-retrieval analogue, but hops are separate MIPS lookups against an external index, whereas TS2TS resolves the next hop in-sequence via the attention grant. **zhang2024beamretrieval** — end-to-end beam retrieval keeps a beam of passage hypotheses across hops and trains the encoder jointly, still an external retrieve-then-read pipeline. **edge2024graphrag** — GraphRAG has an LLM extract an entity-relation KG plus community summaries offline, then answers by map-reduce over communities; key contrast is derived-graph-at-inference versus TS2TS's native-graph-in-pretraining. **gutierrez2024hipporag** — HippoRAG runs Personalized PageRank over an open KG seeded by query entities for single-step multi-hop retrieval; the strongest "graph-structure-as-memory" baseline, but the graph and PPR are external non-parametric memory queried at inference. **yu2024chainofnote** — Chain-of-Note generates per-document reading notes before answering, improving robustness to noisy retrieval; relevant to cross-doc reading reliability rather than the link mechanism itself.

### Training-corpus graph sources: Wikipedia and web hyperlink graphs

**consonni2019wikilinkgraphs** — WikiLinkGraphs, the complete longitudinal Wikipedia link network underlying our wiki corpus, and the actual source of the hyperlink edges we train on. **yamada2020wikipedia2vec** — Wikipedia2Vec trains on the wikilink graph; the closest "learn from wikilinks" prior, but produces shallow skip-gram entity embeddings rather than link-aware attention in a 32k-context decoder. **merity2017pointer** — WikiText, the conventional wiki LM benchmark (node text with edges discarded). **mahoney2011ltcb** (enwik8/enwik9/text8) and **simplewiki** (Simple English Wikipedia dumps) are further wiki node-text resources with edges stripped — exactly the structure TS2TS restores. **petroni2021kilt** — KILT, a Wikipedia snapshot with provenance, used as a retrieval/eval target rather than as trained link structure. Typed knowledge bases sit alongside the free-text wikilink graph but encode curated relations rather than raw hyperlinks: **auer2007dbpedia** and **lehmann2015dbpedia** (DBpedia), **vrandecic2014wikidata** (Wikidata). Web-scale hyperlink-graph analyses characterize the same edge topology at crawl scale, though without usable node text: **meusel2014graphrevisited** and **meusel2015graphstructure** (web graph structure studies), plus the crawl/anchor-text resources **clueweb12** (ClueWeb12) and **commoncrawl** (Common Crawl), where hyperlink edges are usually stripped.

### Training-corpus graph sources: scientific citation corpora

**saier2020unarxive** and **saier2023unarxive** — unarXive and unarXive 2022 provide arXiv full text with annotated in-text citations and the citation network (papers as nodes, `\cite` edges); this is our scientific training corpus. **lo2020s2orc** — S2ORC, the larger citation-graph alternative. **soldaini2023pes2o** — peS2o, S2ORC cleaned for pretraining but *without* edges, serving as the flat (edgeless) corpus baseline against our citation graph. **hu2020ogb** — Open Graph Benchmark (ogbn-arxiv) is the canonical text-attributed citation graph for GNN comparability. **sen2008collective** — the Cora/CiteSeer/PubMed origin of the docs-as-nodes / citations-as-edges TAG paradigm we generalize to LM pretraining (message-passing edge versus our attention edge).

**taylor2022galactica** is a key contrast: Galactica models citation prediction as a task, autoregressively generating cited references via `[START_REF]` — the closest prior to "a generated link fetches the target" — but the reference strings are memorized in parameters, with no retrieval of or attention into the cited document, unlike our read-access edge. Citation-informed document representations use citation edges for embeddings rather than generative cross-doc attention: **beltagy2019scibert** (SciBERT, scientific text but citation-graph-agnostic), **cohan2020specter** (SPECTER, citation-informed document embeddings), **ostendorff2022scincl** (SciNCL, neighborhood-contrastive citation embeddings). Citation-recommendation and citation-intent work treats citations as training-pair/ranking signal at inference: **bhagavatula2018citation** (content-based citation recommendation), **he2010contextaware** (context-aware citation recommendation), **cohan2019scicite** (typed citation-intent classification).

Provenance and citation-graph-construction resources underpin our reference-to-node resolution: **priem2022openalex** (OpenAlex, used to resolve arXiv `\cite`s — lifting resolution from ~14.5% to ~66% via the open-access map), **sinha2015mag** (Microsoft Academic Graph), **zhang2019oag** (Open Academic Graph entity linking), **ammar2018literaturegraph** (the Semantic Scholar literature graph). Further scholarly corpora and tasks: **clement2019arxiv** (arXiv-as-dataset provenance), **cachola2020scitldr** (SciTLDR extreme summarization), **bird2008aclarc** (ACL Anthology Reference Corpus), **knoth2012core** (CORE open-access aggregation).

### Long-context evaluation and utilization

**liu2024lostmiddle** — "Lost in the Middle" documents U-shaped positional utilization, motivating a direct cross-doc edge over information buried in a long packed context. **vodrahalli2024michelangelo** — Michelangelo evaluates dispersed relational reasoning via latent-structure queries; the most aligned with our cross-doc thesis against single-fact needle probes. **kamradt2023niah** — the Needle-in-a-Haystack pressure test (repository resource). **hsieh2024ruler** (RULER) and **yen2025helmet** (HELMET) show effective context is far below claimed length and argue NIAH is insufficient, validating our 8k→32k RoPE-inference claims and best-practice evaluation. **levy2024sametask** — "Same Task, More Tokens" isolates input length as a causal variable in reasoning degradation, paralleling our concatenation compute-controls. **xu2024retrievalmeetslong** — retrieval-versus-long-context comparison, the central contrast we address by fusing structure at training time. **an2024film** — FILM fixes lost-in-the-middle via data, contrasting with our architectural edge. Realistic long-context suites: **bai2024longbench** and **bai2025longbenchv2** (LongBench, LongBench v2), **shaham2022scrolls** and **shaham2023zeroscrolls** (SCROLLS, ZeroSCROLLS), **zhang2024infinitebench** (∞Bench beyond 100K tokens), **an2024leval** (L-Eval), and **fu2024data128k** (data engineering for long-context scaling).

### Evaluation-methodology backbone

**gao2021lmeval** — the LM Evaluation Harness defines the loglikelihood/multiple-choice primitive (acc, byte-length-normalized acc_norm) that is our teacher-forced NLL measurement. **biderman2024lessons** — "Lessons from the Trenches" documents why likelihood evaluation is fragile, supporting our token-parity discipline and preference for continuous Δnll over accuracy. **schaeffer2023mirage** — "Are Emergent Abilities a Mirage?" shows metric-choice artifacts, justifying continuous Δnll over discretized multiple-choice accuracy and answering the "metric-manufactured effect" concern. **liang2023helm** — HELM's multi-metric framing informs our condition matrix. **fourrier2023leaderboard** — documents cross-harness multiple-choice-normalization discrepancies (the byte-length-normalization debate), with **brown2020gpt3** as the origin of the few-shot MC-scoring convention.

*Statistical significance.* **koehn2004statsig** (bootstrap resampling for MT/NLP significance) and **dror2018hitchhiker** (paired significance testing guide) justify our bootstrap-CI-excludes-zero gate together with paired cross-versus-flat comparison and the derangement placebo.

*Contamination detection and mitigation.* **sainz2023contamination** (the case for per-benchmark contamination measurement), **golchin2024timetravel** (Time Travel tracing of contamination), **oren2024contamination** (proving test-set contamination in black-box models), **jacovi2023stopuploading** (practical mitigation strategies), and **yang2023rephrased** (rephrased samples evade n-gram dedup — a stated limit on our SHA1/n-gram dedup). Together these support our HotpotQA-2017-leakage caveat and dedup pipeline.
