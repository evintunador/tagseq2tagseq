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
