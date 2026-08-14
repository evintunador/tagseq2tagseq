## guo2021graphcodebert — GraphCodeBERT: Pre-training Code Representations with Data Flow

Guo, Ren, Lu, Feng, Tang, Liu, Zhou, Duan, Svyatkovskiy, Fu, Tufano, Deng, Clement, Drain,
Sundaresan, Yin, Jiang, Zhou. ICLR 2021. arXiv:2009.08366.

Grounding: paper read via ar5iv (HTML render of 2009.08366); our side verified against the two
assigned code briefs (`paper/notes/code_briefs/masks.md`, `link_detectors.md`) and
`related_work_notes.md`. Claims below are marked (confirmed) from the paper text vs. (infer) where
I reason beyond it.

### What the paper actually does

GraphCodeBERT is the first code pre-trained model to inject *semantic* structure — data flow —
rather than treating code as a flat token sequence, and it does so through the attention mask, which
is exactly the mechanism our project uses. This is why it is the "same-mechanism" comparator.

**Data flow as a graph.** For a function, they parse the AST, extract the variable occurrences, and
build a *data-flow graph*: nodes are variables (one node per variable *occurrence*), and a directed
edge ⟨v_j → v_i⟩ means "the value of v_i comes from v_j" (a where-the-value-comes-from relation).
They deliberately reject the AST itself as the structure to encode, arguing it imposes an
"unnecessarily deep hierarchy" and that data flow is more compact and directly semantic. The input
sequence is the concatenation `[CLS], comment tokens W, [SEP], code tokens C, [SEP], variable nodes V`
— i.e. the data-flow nodes are appended as extra "tokens" after the code, with a special position
embedding marking them.

**Graph-guided masked attention (the core mechanism).** Structure enters as an additive mask matrix
M in the self-attention logits: `softmax(QKᵀ/√d + M)`, with `M_ij = 0` if attention is allowed and
`−∞` otherwise. The rules (confirmed):
- `[CLS]`/`[SEP]` attend to everything.
- comment+code tokens (W∪C) attend freely among themselves (standard dense encoder attention).
- **node→node**: variable node v_i may attend v_j only if the directed data-flow edge ⟨v_j,v_i⟩∈E
  (or same node). This is the graph edge realized as an attention permission.
- **node↔code**: a node v_i and a code token c_j attend each other only if the variable was
  *identified from* that token — a set E′ linking each variable occurrence to its source token span.

So the graph is not message-passed by a GNN; it is a hard binary permission over which
node/token pairs may exchange attention — structurally the same idea as our grant mask.

**Three pre-training objectives** (confirmed):
1. **MLM** — standard BERT masking on the comment+code tokens (15%; 80/10/10 mask/random/keep).
2. **Edge Prediction (EdgePred)** — sample 20% of data-flow nodes, mask their edges in M, and predict
   the masked edges. Edge probability = sigmoid of the dot product of the two node representations;
   positive and negative candidate edges are balanced. This teaches the model the data-flow structure
   itself.
3. **Node Alignment (NodeAlign)** — sample 20% of nodes, mask the *node↔code-token* edges (E′), and
   predict which code token each variable was identified from. This aligns the variable's
   representation to its lexical source. This is the objective with no analog in our design and is
   worth close attention (see below).

**Setup & scale** (confirmed): 12-layer / 768-hidden / 12-head Transformer (BERT-base shape),
**initialized from CodeBERT** (`feng2020codebert`, itself RoBERTa-based). Max sequence length 512,
max 128 data-flow nodes. Pre-trained on **CodeSearchNet** (`husain2019codesearchnet`): 2.3M functions
paired with NL docs across six languages (Ruby, JS, Go, Python, Java, PHP). ~200K steps, batch 1024,
Adam lr 2e-4, ~83 GPU-hours; structure objectives alternated with MLM.

**Results that matter** (confirmed):
- **NL code search** (mean MRR over 6 langs): **0.713** vs CodeBERT 0.693 vs RoBERTa ~0.617. Per-lang
  gains are uniform (Ruby 0.679→0.703, Python 0.672→0.692, etc.).
- **Clone detection** (BigCloneBench): F1 **0.950** vs CodeBERT 0.941.
- **Code translation** (CodeTrans): Java→C# BLEU 80.58 / EM 59.4; C#→Java BLEU 72.64 / EM 58.8.
- **Code refinement** (bug-fix): small EM 17.3 (BLEU 80.02), medium EM 9.1 (BLEU 91.31).
- **Ablation on code search MRR**: full 0.713 → −EdgePred 0.707 → −NodeAlign 0.703 → **−DataFlow 0.693**.
  The whole data-flow contribution is ~+2.0 MRR points; the two structure objectives together are
  ~+1.0 of that, i.e. the mask alone (data flow present, no structure loss) already buys roughly half
  the gain, and the structure-aware losses buy the rest.

They also report an attention analysis: in code search the model "prefers structure-level attentions
over token-level attentions," i.e. it actually routes probability mass through the granted edges.

### Methodology: theirs vs. ours

The shared thesis, stated on the nose: **structure is injected by permitting attention along
dependency edges, at pre-training time, not by a bolt-on GNN or an inference-time retrieval hop.** On
the brief's key axis — *train-on-structure vs. retrieve-at-inference; attention edge vs. GNN edge vs.
cached-KV vs. training-pair* — GraphCodeBERT and TS2TS are on the *same* side: both are **attention-edge,
train-on-structure** methods. That is rare among our comparators (most of §"graph-aware pretraining"
in `related_work_notes.md` use a GNN edge, a soft bias, or a contrastive pair). This is the tightest
mechanistic ally we have, which makes the *divergences* the load-bearing part of the comparison.

Divergences, each tied to our source:

1. **Encoder + MLM vs. decoder + causal AR.** GraphCodeBERT is a bidirectional encoder; its mask is
   *symmetric except along directed data-flow edges*, and its objective is denoising (MLM + edge/node
   recovery). TS2TS is a decoder-only LM: our `cross_doc_link` mask is `M=(q≥k)&(same_doc OR in_grant)`
   — **causal is never relaxed** (`masks.md` §"Formal semantics", cross_doc_mask.py:417-423), and the
   grant is *asymmetric* A→B only. GraphCodeBERT can afford bidirectional edges because MLM is not
   autoregressive; we cannot, and our whole grant machinery is built to add cross-document reach
   *without* breaking causality. This is the single biggest structural difference.

2. **Nodes-as-appended-tokens vs. edges-over-in-place-document-tokens.** GraphCodeBERT literally
   materializes each variable as an *extra token* appended after the code (128 node slots), then wires
   node↔token and node↔node edges among those extra positions. TS2TS has no separate node tokens: the
   "nodes" are whole documents packed in place, and the edge is a rectangle in the T×T mask
   ([link_end_pos, A.end) × [B.start, B.end); `masks.md` §"Formal semantics"). Their graph lives in a
   128-slot side-channel; ours lives in the token stream itself.

3. **Intra-function dataflow vs. inter-document link.** GraphCodeBERT's edges are *within one function*
   (≤512 tokens, ≤128 vars). Our edges are *between documents* in a 32k packed sequence — a hyperlink,
   an import, a `\cite` (`link_detectors.md`: 11 detector syntaxes). Their "where the value comes from"
   is a variable-to-variable relation; our "where the value comes from" is a document-to-target-document
   fetch. We are, almost exactly, GraphCodeBERT's graph-attention lifted two levels up: from
   intra-function variable dataflow to corpus-level import/citation/hyperlink dataflow.

4. **How the edge is obtained.** They run a full AST parse offline to build the dataflow graph. We do
   *online* tokenizer-decoupled detection at train and generation time (`link_detectors.md`
   §Protocol), and additionally support **Option B baked graph-edge grants** (`masks.md` §"Option B")
   where training/graph-eval precompute `link_to_target` once and rehydrate per batch — closer in
   spirit to GraphCodeBERT's offline-parsed graph, but for us it's an optimization, not the only mode
   (generation always re-detects from text).

5. **Structure-aware auxiliary losses vs. none.** GraphCodeBERT adds EdgePred + NodeAlign as explicit
   objectives that *supervise the graph*. TS2TS has **no auxiliary structural loss at all** — the only
   objective is next-token prediction; the graph enters purely through the mask. Their ablation shows
   the mask-alone already delivers ~half the gain (0.693→~0.703 region) and the losses add the rest.
   This is a live design question for us (below).

6. **Hard binary permission — shared.** Both use a hard 0/−∞ mask, not a learned soft edge bias
   (contrast `ying2021graphormer` / graph-transformer soft biases in `related_work_notes.md`). Our
   novelty over their mask is the *engineering*: bit-packed grants with pointwise membership and the
   Triton BIM block taxonomy (`masks.md` §"Novel/publishable" 1,3) — GraphCodeBERT never needed this
   because a 512×(512+128) dense mask is trivial; ours must scale a data-defined mask to 32k.

Not shared at all: cached-KV reuse (we forgo it for train/infer symmetry) and training-pair/contrastive
signal — GraphCodeBERT touches neither, so those axes don't apply.

### Predictions & open questions for our method

- **The mask alone should already move the needle, and the structure-loss is the tunable extra.**
  Their ablation isolates exactly our compute-control question. Data-flow-off→on is +2.0 MRR; of that,
  ~+1.0 is the mask with *no* structure loss and ~+1.0 is the two auxiliary objectives. Prediction:
  our `cross_doc_link` vs. `doc_causal` (mask-only, no aux loss) should show a *positive but modest*
  effect, and there may be headroom we are leaving on the table by having no analog of EdgePred/
  NodeAlign. This is the sharpest actionable transfer: **consider whether an auxiliary link-prediction
  loss (predict the target doc / the link edge) would recover the "structure-objective" half of their
  gain** — a possible ablation arm.

- **Where the effect should be strong.** GraphCodeBERT's largest relative wins are on tasks that
  genuinely require following a dependency (code translation EM, clone detection) and its attention
  analysis shows it *routes* through structure edges. Prediction: our effect should be strongest on
  eval where the answer literally lives in the linked target — multi-hop QA with the supporting doc as
  a link (`yang2018hotpotqa` in our notes), cross-file code completion (`ding2023crosscodeeval`,
  `liu2024repobench`) — and near-zero on single-document controls (HellaSwag/ARC/PIQA in our notes),
  exactly the control structure we already planned.

- **Uniformity across domains.** Their per-language gains are strikingly *uniform* (every one of six
  languages improves by roughly the same margin). If our linking bias is a real structural prior and
  not a corpus artifact, we should likewise see *consistent* cross-vs-concat gains across wiki / arXiv
  / thestack rather than one corpus carrying the whole effect. A per-corpus breakdown that is uniform
  would be strong evidence; a single-corpus spike would suggest an artifact.

- **Node-alignment as a diagnostic.** NodeAlign supervises "which token did this variable come from."
  Our analog is "does the model actually attend into the granted target span rather than ignoring it."
  Their finding that the model *prefers* structure attention suggests we should measure attention mass
  landing inside grant rectangles — if it's near zero, the grant is decorative. (infer)

- **Open question we might resolve for them, and vice versa.** GraphCodeBERT never tests whether the
  structure edge helps *generation* (it's an encoder; its "translation/refinement" tasks bolt a
  decoder on top). Our decoder-native, generation-time link fetch (`link_detectors.md`: generation uses
  live text detection, not the baked graph) is precisely the setting they could not probe — so we
  can answer "does a graph-attention edge help autoregressive generation, not just representation."
  Conversely, their EdgePred/NodeAlign result answers a question our design leaves open: whether
  explicitly *supervising* the edge beats leaving it implicit in the LM loss.

### Gotchas

- **The mask is only as good as the graph extractor.** GraphCodeBERT's edges come from a clean offline
  AST parse; ours come from 11 online detectors with known brittle spots — markdown hardcoded to GPT-2
  token id 16151, arXiv exact-byte-title matching, Java source-root ambiguity, Python candidate
  over-generation (`link_detectors.md` §"Reviewer-attackable"). A silent detector miss = an empty
  grant = we quietly test `doc_causal` while thinking we test `cross_doc_link`. GraphCodeBERT never
  faced this because its graph was deterministic. **Verify non-empty grant density before trusting any
  cross-vs-causal delta** — this is the analog of their "does the model attend to structure" check.

- **Node/edge budget truncation.** They cap at 128 nodes per function; we cap at **max_grants=256**
  (`masks.md` §"max_grants") with *positional* truncation (later links dropped first, not by
  importance). Beyond the cap the mask is silently biased. If a packed 32k window has >256 links, our
  measured effect *understates* the true one — matching their implicit warning that structure coverage
  must be complete to read the ablation honestly.

- **Train/eval mask parity.** Their ablation is only interpretable because the mask is identical across
  conditions. Our brief explicitly flags that Flex vs. Triton vs. dense-viz are *three*
  reimplementations that must agree and that `max_grants` must match train and eval or the effect is
  understated (`masks.md` §"Reviewer-attackable" 6). This is a direct "this broke conceptually for the
  ablation to be valid" warning — enforce it.

- **Position handling.** GraphCodeBERT gives nodes a *special position embedding* precisely because
  appended nodes have no natural sequence position — they engineered position handling for the injected
  structure. We do the opposite: RoPE is **not reset per document** and A reads B at a
  packing-order-dependent relative offset (`masks.md` §"Reviewer-attackable" 1). Their care here is a
  warning that arbitrary relative positions between a source and its structurally-linked target can
  matter; our large untrained RoPE offsets across a 32k gap are a plausible failure regime
  (cf. `chen2023positioninterpolation` in our notes). (infer)

- **Small absolute margins.** Their headline gain is ~2 MRR points (0.693→0.713) — real but small, and
  they needed 6-language uniformity + ablation + attention analysis to make it convincing. Expect our
  linking effect to be similarly modest in magnitude; plan the statistical machinery (bootstrap CI,
  paired Δnll, `koehn2004statsig`/`dror2018hitchhiker` in our notes) accordingly rather than expecting
  a large headline number.

### Missed citations worth adding

Checked against `paper/bib/refs.bib` (grep). GraphCodeBERT's own reference list contains several works
directly on our train-on-structure / attention-edge axis that are **not currently in refs.bib**:

- **feng2020codebert** — Feng et al., "CodeBERT: A Pre-Trained Model for Programming and Natural
  Languages," arXiv:2002.08155 (EMNLP Findings 2020). **Genuinely missing** (only appears as an
  author-name substring inside other entries). This is a real gap: GraphCodeBERT is *initialized from
  CodeBERT* and it's the canonical bimodal code-NL encoder baseline — our "concat/no-structure code
  encoder" pole should cite it. High priority.

- **allamanis2018learning** — Allamanis, Brockschmidt, Khademi, "Learning to Represent Programs with
  Graphs," ICLR 2018 (arXiv:1711.00740). Missing. The foundational program-graph paper: encodes code
  as a graph with data-flow/AST edges and runs a GNN over it — the *GNN-edge* counterpart to our and
  GraphCodeBERT's *attention-edge*, and the origin of "data flow as edges" for neural code. Strong add
  for the "GNN edge vs. attention edge" contrast.

- **hellendoorn2020great** — Hellendoorn, Sutton, Singh, Maniatis, Bieber, "Global Relational Models of
  Source Code," ICLR 2020. Missing. GREAT injects program-graph relations as *relational attention
  biases* in a Transformer — the closest prior to "code structure as an attention-level signal" and a
  direct antecedent of GraphCodeBERT's mask; sits squarely on our attention-edge axis
  (cf. `diao2023relationalattention` we already cite).

- **alon2019code2seq** — Alon, Brody, Levy, Yahav, "code2seq: Generating Sequences from Structured
  Representations of Code," ICLR 2019 (arXiv:1808.01400). Missing. Encodes code via sampled AST paths
  for generation — the AST-path structural-representation alternative that GraphCodeBERT explicitly
  argues *against* (they reject deep AST hierarchy for flat data flow); useful for the "which structure
  to encode" framing.

- **roziere2020transcoder** — Roziere, Lachaux, Chanussot, Lample, "Unsupervised Translation of
  Programming Languages" (TransCoder), NeurIPS 2020 (arXiv:2006.03511). Missing. The unsupervised
  code-translation model that is the standard comparator on GraphCodeBERT's CodeTrans task; relevant if
  we position against code-translation eval. Lower priority than the three above.

(I did not confirm these arXiv ids against the live arXiv — the ids are as cited by GraphCodeBERT /
standard usage; verify before adding, per brief.)

---
Done: wrote guo2021graphcodebert deep-dive; verified GraphCodeBERT's method/numbers via ar5iv and its
mechanism against our masks.md/link_detectors.md; flagged feng2020codebert + 4 others as missing.
