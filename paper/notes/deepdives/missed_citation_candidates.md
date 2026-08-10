# Missed-citation candidates from the deep-dive round

Works cited by our comparator papers that are NOT in refs.bib, proposed by the deep-dive
agents as potentially worth adding. **9 high-consensus / arXiv-verified entries have
already been added to refs.bib** (abbas2023semdedup, chan2022datadistributional, feng2020codebert, geva2021ffnkeyvalue, lee2022deduplicating, min2019necessitate, muennighoff2023scaling, press2022alibi, xiong2020wklm). The rest below are unverified
candidates — grep refs.bib and confirm the arXiv id before adding any. Priority tags (HIGH/MED/
LOW) and the source paper are as proposed by the agents. Grouped by source paper.

---

## from yasunaga2022linkbert (LinkBERT) — LOW priority (biomed/QA infra, only if we add those tracks)
gu2021pubmedbert  2007.15779  BLURB benchmark / domain-specific pretraining
jin2021medqa      2009.13081  MedQA-USMLE
fisch2019mrqa     ?           MRQA shared task (multi-doc QA eval track)
## from staniszewski2025structured (SPLiCe)
chan2022datadistributional  2205.05055  MED priority — burstiness/data-distributional basis for in-context learning; underpins why related-doc packing helps
redpajama                   ?           LOW — corpus (Together Computer), software cite
geng2023openllama           ?           LOW — software cite, no arXiv
## from khandelwal2020knnlm (kNN-LM)
grave2017continuouscache   ?  MED — continuous cache LM (kNN-LM stacks it; retrieval-memory lineage)
grave2017unboundedcache    ?  MED — unbounded cache / kNN-over-cached-states, direct kNN-LM ancestor
kaiser2017rareevents       ?  MED — learning to remember rare events, differentiable trained kNN memory
sprechmann2018mbpa         ?  LOW — memory-based parameter adaptation
## from wu2022memorizing (Memorizing Transformers)
sukhbaatar2021expirespan  2105.06548  MED — Expire-Span, learned forgetting (complements our DocumentContext eviction)
fan2020feedback           2002.09402  LOW — Feedback Transformer
polu2020theoremproving    2009.03393  LOW — GPT-f theorem proving (verbatim-reuse regime)
## from zhao2024analysing (sequence composition)
soboleva2023slimpajama    ?           LOW — SlimPajama corpus (likely no arXiv, Cerebras/HF)
chan2022datadistributional 2205.05055 *** HIGH now — corroborated by 2 papers (SPLiCe + this); burstiness→ICL, explains why related-doc packing works
## from shi2024incontext (In-Context Pretraining)
levine2022inductivebias   2110.04541  MED — pretraining context design shapes ICL (foundational for our packing thesis)
abbas2023semdedup         2303.09540  *** HIGH — dedup for packing; flagged uncited in our own notes, now proposed 2x
zhao2021calibrate         2102.09690  LOW — ICL eval calibration/fragility
## from decao2021genre (GENRE)
hokamp2017gridbeam        1704.07138  MED — grid beam search, lexical-constraint decoding origin (we lack this lineage)
post2018fastlexconstraint 1804.06609  MED — fast lexically-constrained decoding
ganea2017deepjoint        1704.04920  LOW — neural entity linking (joint)
raiman2018deeptype        1802.01021  LOW — DeepType entity linking
vanhulst2020rel           ?           LOW — REL entity linking toolkit
## from dejong2022tome (TOME/Mention Memory)
dhingra2020drkit          2002.10640  MED — DrKIT, differentiable KB traversal for multi-hop
sun2021opql               2102.07043  LOW — OPQL, KB-in-parameters query
fitzgerald2021moleman     2106.07352  LOW — MOLEMAN, mention-level entity representations
## from yasunaga2022dragon (DRAGON)
xiong2020wklm             1912.09637  *** HIGH — "Pretrained Encyclopedia" (WKLM): trains on Wikipedia HYPERLINK structure = direct antecedent we lack
feng2020mhgrn             2005.00646  LOW — MHGRN multi-hop graph reasoning
lin2019kagnet             1909.02151  LOW — KagNet
yao2019kgbert             1909.03193  LOW — KG-BERT link prediction
## from guo2021graphcodebert (GraphCodeBERT)
feng2020codebert          2002.08155  *** HIGH — CodeBERT, the model GraphCodeBERT inits from; foundational code-LM we lack
allamanis2018learning     1711.00740  MED — Learning to Represent Programs w/ Graphs (program-graph GNN-edge, structural code prior art)
hellendoorn2020great      ?           MED — GREAT: relational attention on code (global relational-attention analog)
alon2019code2seq          1808.01400  LOW — code2seq
roziere2020transcoder     2006.03511  LOW — TransCoder
## from zhang2023repocoder (RepoCoder)
svyatkovskiy2020intellicode 2005.08025 LOW — IntelliCode Compose (code completion product)
robertson2009bm25         -           MED — BM25 probabilistic relevance (lexical-retrieval anchor we lack; book/journal cite)
ren2020codebleu           2009.10297  LOW — CodeBLEU metric
## from borgeaud2022retro (RETRO)
gao2020pile               2101.00027  LOW — The Pile (eval corpus, only if Pile-style eval adopted)
rae2021gopher             2112.11446  LOW — Gopher/MassiveText (scale ref)
## from caciularu2021cdlm (CDLM)
zhou2020crossdocattention 2010.01263  MED — cross-document attention for multi-hop QA
asai2020reasoningpaths    1911.10470  MED — learning to retrieve reasoning paths (multi-hop)
zhao2020transformerxh     -           MED — Transformer-XH (eXtra-Hop attention); ICLR/OpenReview only, NO arXiv id (don't invent)
conneau2019xlm            1901.07291  LOW — XLM cross-lingual pretraining
ginzburg2021dcs           2106.01186  LOW — document cross-referencing similarity
## from liu2020kbert (K-BERT)
joshi2019spanbert         1907.10529  MED — SpanBERT span-boundary objective (relevant to boundary/FIM masking)
bosselut2019comet         1906.05317  LOW — COMET generative KG
## from guo2024deepseekcoder (DeepSeek-Coder)
lee2022deduplicating      2107.06499  *** HIGH — foundational near-dedup (== the dedup gap flagged 3x: SemDeDup-adjacent, our weak-dedup reviewer risk)
du2022glm                 2103.10360  LOW — GLM blank-infilling attention mask
## from wu2022emat (EMAT)
lewis2021paq              2102.07033  MED — PAQ (65M QA pairs), the frozen-KV memory corpus; same RAG author lineage
chen2022qamat             ?           LOW — QAMAT, concurrent QA-KV transformer
geva2021ffnkeyvalue       2012.14913  MED — Transformer FFN layers are key-value memories (foundational for KV-memory framing)
## from liu2024repobench (RepoBench)
feng2020codebert          2002.08155  *** HIGH (2nd proposal: GraphCodeBERT + RepoBench) — foundational code-LM
zhou2023docprompting      2207.05987  LOW — DocPrompting (retrieval from docs for code gen)
## from yang2018hotpotqa (HotpotQA)
min2019necessitate        1906.02900  MED — canonical "compositional questions solvable single-hop" shortcut paper (our multi-hop validity + placebo motivation)
jiang2019avoiding         1906.07132  LOW — avoiding reasoning shortcuts in multi-hop
chen2019understanding     1904.12106  LOW — understanding dataset design for multi-hop
clark2018simple           1710.10723  LOW — BiDAF++ baseline architecture
## from wang2024flashmask (FlashMask): no high-value adds (kernel slice already thorough)
## from yuan2025nsa (NSA)  [NOTE: refs.bib dup yuan2025nsa removed; canonical key = yuan2025nativesparse]
tang2024quest             2406.10774  MED — Quest (query-aware KV block selection); note NOT the data-selection "Quest"
xiao2023streamingllm      2309.17453  MED — StreamingLLM attention sinks (always-keep initial blocks; ties to our sentinel-LSE guard)
wu2024retrievalhead       2404.15574  LOW — retrieval heads
## from kocetkov2022stack (The Stack)
lee2022deduplicating      2107.06499  *** HIGH (4th proposal) — near-dedup improves LMs; = our known dedup gap
kandpal2022deduplicating  2202.06539  MED — dedup mitigates memorization (contamination story)
carlini2023quantifying    2202.07646  MED — quantifying memorization (leakage-protocol support)
broder1997resemblance     -           LOW — MinHash origin (book/journal)
## from liu2024lostmiddle (Lost in the Middle)
press2022alibi            2108.12409  MED — ALiBi positional encoding; relevant to positions-across-packed-docs + used by a model here
## from cheng2024draco: no confidently-missing refs
## from wu2024repoformer (Repoformer)
mallen2023whennottotrust  2212.10511  MED — when-not-to-retrieve (selective retrieval decision, mirrors our firing gate)
kadavath2022knowwhattheyknow 2207.05221 LOW — models know what they know (self-eval)
wang2023skr               2310.05002  LOW — self-knowledge-guided retrieval
zhou2023docprompting      2207.05987  (dup w/ RepoBench proposal) — DocPrompting
## from taylor2022galactica (Galactica)
muennighoff2023scaling    2305.16264  *** HIGH — Scaling Data-Constrained LMs; formalizes repeated-tokens/multi-epoch (we train multi-epoch); directly shapes our budget claims
aribandi2021ext5          2111.10952  LOW — ExT5 (prompt/task mixing)
## from pagliardini2023sparseflash (SCFA)
kitaev2020reformer        2001.04451  CHECK — appears inside SCFA note text; grep refs.bib (likely already present from A2 slice)
andoni2015lsh             -           LOW — LSH theory
## from xu2024retrievalmeetslong
jiang2022retrievalasattention 2212.02027  MED — "Retrieval as Attention" (retrieval as computation inside one transformer — conceptually close)
ratner2023parallelcontextwindows 2212.10947  LOW — Parallel Context Windows
press2022alibi            2108.12409  (2nd proposal, w/ LostInMiddle) — ALiBi
## from gutierrez2024hipporag
sarthi2024raptor          2401.18059  MED — RAPTOR recursive tree summarization retrieval
chen2024densex            2312.06648  LOW — Dense X / proposition retrieval
press2023selfask          2210.03350  CHECK — Self-Ask; note refs has press2023bamboogle (same paper?) — verify before add
