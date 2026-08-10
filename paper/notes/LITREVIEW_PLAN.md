# Expanded literature-review taxonomy (round 2 — "dozens of slices")

Round 1 (7 broad slices, ~118 entries) is in bib/refs.bib. Round 2 splits those
into NARROW, DEEP slices + adds subfields the docs-only pass missed. Each round-2
agent gets: (a) the project summary, (b) the relevant code brief(s) from
notes/code_briefs/, (c) the list of already-collected keys in its area so it goes
DEEPER / ADDS rather than repeating, (d) instruction to return 5-15 verified
bibtex + notes, drawing the train-on-structure vs retrieve-at-inference contrast.

## Slices (target ~36)
### Retrieval / memory (6)
R1  RAG core + retrieval-in-pretraining (RAG, FiD, Atlas, REALM, RETRO, RAVEN, Retro++)
R2  kNN / datastore-augmented LMs (kNN-LM, SPALM, TRIME, He-efficient-kNN, adaptive kNN-LM)
R3  KV-cache reuse & prompt caching systems (PromptCache, CacheBlend, Block-Attention, CAG, RAGCache, RadixAttention/SGLang, vLLM PagedAttention)
R4  Memory-augmented / recurrent-memory transformers (Transformer-XL, Compressive T., Memformer, inf-former, RMT, Memorizing T., Focused/LongLLaMA)
R5  Late-interaction & dense retrieval backbones (DPR, ColBERT/v2, Contriever, GTR, SPLADE)
R6  Long-term/episodic memory & editable knowledge stores (Mention Memory/TOME, LUMEN, EMAT, kNN-memory editing)

### Graph & structure (7)
G1  Text-attributed-graph representation learning (Patton, GLEM, TAPE, GIANT, GraphFormers, Heterformer, Edgeformers, survey)
G2  Graph transformers (Graphormer, GraphGPS, SAN, TokenGT, NAGphormer, Exphormer)
G3  Knowledge-graph-enhanced LMs (ERNIE-THU, KnowBERT, KEPLER, K-BERT, JAKET, K-Adapter, GreaseLM)
G4  Hyperlink/anchor/markup pretraining (LinkBERT, WebFormer, MarkupLM, HTLM, anchor-text, DocT5)
G5  Entity-aware LMs & entity memory (EaE, Mention Memory, LUKE, entity-conflicts)
G6  GNNs for text / node classification foundations (GCN, GraphSAGE, GAT, node2vec, DeepWalk)
G7  Retrieval/attention over document/passage graphs & multi-document discourse

### Packing / masking (3)
P1  Document packing & sequence composition (ICP, SPLiCe, Zhao composition, best-fit, dataset decontamination)
P2  Intra-doc / boundary attention masking (Megatron reset-attn, GPT-3, OLMo, Llama packing)
P3  Curriculum / data ordering / example selection for pretraining

### Efficient attention / long context (6)
A1  IO-aware exact attention & programmable masks (FlashAttention 1/2/3, FlexAttention, xFormers, Blocksparse)
A2  Fixed-sparse-pattern attention (Sparse Transformers, Longformer, BigBird, LongT5, ETC)
A3  Linear / kernelized / SSM sequence models (Performer, Linear Transformers, RWKV, Mamba, S4, Based)
A4  Conditional computation & routing for long context (CoLT5, Mixture-of-Depths, MoE-attn)
A5  Distributed / blockwise / sequence-parallel attention (Ring, Blockwise, Megatron-SP, DeepSpeed-Ulysses)
A6  Triton & GPU kernel programming for attention (Triton, block-sparse kernels, FlashAttn kernel design, bitmask attention)

### Optimizer / training recipe (7)
O1  Muon & orthogonalized/spectral-norm optimizers (Muon, Muon-scalable, spectral descent, Newton-Schulz/polar-express, Bernstein modular norm)
O2  Preconditioned / second-order optimizers (Shampoo, SOAP, distributed Shampoo, K-FAC, Adafactor, Sophia)
O3  LR schedules & WSD (WSD/MiniCPM, cosine, cooldown/annealing, cyclical, scaling-law-optimal schedules)
O4  Scaling laws & compute-optimal / data-constrained (Kaplan, Chinchilla, data-constrained scaling, over-training)
O5  Efficient-pretraining recipes & speedruns (modded-nanoGPT, nanoGPT, Cramming, MosaicBERT, value-embeddings/skip-connections speedrun tricks)  <-- refine w/ arch code brief (VE banks!)
O6  Architecture components (RoPE, RMSNorm, squared-ReLU/Primer, SwiGLU, weight tying, logit soft-cap, QK-norm)
O7  Mixed precision, DDP/ZeRO, torch.compile / systems (bf16, PyTorch, DDP, FSDP/ZeRO, TorchInductor, submitit, Megatron)

### Code models (5)
C1  Code LLM pretraining & corpora (Codex, CodeGen, InCoder, StarCoder/2, DeepSeek-Coder, CodeLlama, The Stack v1/v2, CodeT5+)
C2  Repo-level via inference-time retrieval (RepoCoder, RepoFusion, CoCoMIC, RepoHyper, R2C2, kNM-LM, ReAcc)
C3  Repo-level via pretraining structure / topological ordering (DeepSeek-Coder topo, StarCoder repo-concat, LongCoder, RepoFormer)
C4  Cross-file & repo code benchmarks (RepoBench, CrossCodeEval, RepoEval, EvoCodeBench, R2C2-Bench, CodeXGLUE)
C5  Fill-in-the-middle / infilling & static-analysis-guided decoding (FIM, InCoder, monitor-guided, type-constrained, SantaCoder)

### QA / reasoning / eval methodology (4)
Q1  Multi-hop QA datasets (HotpotQA, WikiHop/QAngaroo, 2Wiki, MuSiQue, HoVer, StrategyQA, Bamboogle, ComplexWebQ)
Q2  Multi-hop reasoning methods (graph readers DFGN/HDE/Cognitive-Graph; chain-of-thought/self-ask/decomposition; retrieve-and-read)
Q3  Commonsense/knowledge single-doc benchmarks + likelihood-based MC eval (HellaSwag, ARC, PIQA, WinoGrande, OpenBookQA, BoolQ, lm-eval-harness, byte-normalization)
Q4  Long-context evaluation & utilization (lost-in-the-middle, needle-in-haystack, RULER, LongBench, ZeroSCROLLS)

### Corpora (2)
D1  Wikipedia / hyperlink corpora (WikiLinkGraphs, WikiText, KILT, enwiki dumps, WikiText-103, DBpedia)
D2  Scientific citation corpora + citation-informed embeddings (unarXive, S2ORC, OGB/ogbn-arxiv, SPECTER/SciNCL/SciBERT, citation recommendation)

NOTE: refine/insert slices after code briefs land:
- VE banks (arch brief) -> may need a dedicated "value/residual embeddings" mini-slice under O5
- polar_express (Muon brief) -> orthogonalization-polynomial citations in O1
- bitmask kernel (kernel brief) -> block-sparse kernel citations in A6
- DocumentContext KV handling (generation brief) -> may sharpen R3/R4 framing

## ═══ REFINEMENTS from code briefs (round-2 fleet MUST cover these) ═══
Each slice agent gets the relevant code brief(s) + told to go DEEP on specifics below.

O1 Muon/orthogonalization: **Polar Express arXiv:2505.16932** (orthogonalization polynomial, non-stationary coeffs) + **NorMuon** (search — variance-reduced Muon; verify published vs coinage) + Muon + Muon-scalable(Kimi) + spectral/steepest-descent-under-schatten-norm + Bernstein modular-norm + Newton-Schulz orthogonalization lineage + Shampoo/SOAP.
O2 cautious optimizers: **C-Optim / "Cautious Optimizers" (Liang 2024)** (cautious weight decay/update masking) + Adafactor (rank-1 2nd moment, NorMuon's variance-reduction ancestor) + sign-based (Lion, signSGD).
O5b VALUE EMBEDDINGS + speedrun tricks (NEW dedicated slice): modded-nanoGPT value embeddings (find canonical source/writeup), QK-norm (Query-Key Normalization, Henry 2020), U-Net-style skip connections in transformers, embedding/residual highway (x0), learnable residual scaling, bigram hash embedding, zero-init output projection / ReZero / SkipInit, muP (Yang maximal-update). These are individually citeable.
O6 arch components: half-truncated/partial RoPE + RoPE base/theta tuning (also NTK-aware/YaRN scaling), parameter-free RMSNorm, Gemma-2 logit soft-capping [have gemma2], MTP = **Multi-Token Prediction (Gloeckle 2024, DeepSeek-V3)**.
O7 systems: Liger Kernels, FP8 training (Transformer Engine), FSDP/ZeRO, gradient checkpointing, torch.compile/TorchInductor [have pytorch2].
A6 kernels: Triton (Tillet 2019), online-softmax (Milakov & Gimelshein 2018), block-sparse GPU attention, FlashAttention-3, softmax numerical stability under masking.
A3 SSM/linear (context: sparse-attn alternatives): Mamba, S4, RWKV, Performer, Linear Transformers — brief, positioning only.
P3/curriculum: difficulty/density-based curriculum, load-balancing/straggler mitigation in DDP data-parallel (relevant to density bucketing), sample packing best-fit.
Graph-walk mini (under G6): DeepWalk, node2vec (biased p/q walk), PageRank + Personalized PageRank / RWR (contrast our teleport-to-uniform), GraphSAGE neighbor sampling, METIS/graph partitioning (worker Voronoi sharding).
Eval-method (Q3+): lm-evaluation-harness (Gao/EleutherAI), byte-length-normalized MC likelihood scoring, bootstrap CI, counterfactual/placebo eval, train-test contamination/decontamination (dedup, MinHash, Lee 2022 dedup).
Iterative/active retrieval (R-family): IRCoT, Self-Ask, FLARE (active retrieval), DSP, ITER-RETGEN — multi-hop retrieve-and-read (contrast our recursive link-following generation).
Static analysis for code (C5+): tree-sitter, import/dependency resolution, type-constrained decoding, grammar-constrained decoding.
Entity linking (G5+): wikification, mention detection, entity disambiguation (for link-target resolution + title index / constrained trie decoding).

## STATE (code briefs saved to notes/code_briefs/): masks, kernels, traversal, packing_density, link_detectors, architecture, muon_optimizer, generation_retrieval, eval_harness, data_pipelines, merged_multisource. PENDING: training_loop.
