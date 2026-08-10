# Round-2 review agents still to launch (20-concurrent cap hit)
# Launched so far (20): R1 R2 R3 R4 R5 R6 G1 G2 G3 G4 G6 G7 | P1 P2 P3 A1 A2 A3 A5 A6

## PENDING (launch as slots free):
O1  Muon/orthogonalization/Polar Express/NorMuon  [muon_optimizer.md] — PRIORITY
O2  preconditioned & cautious optimizers (C-Optim, Adafactor, Sophia, Lion) [muon_optimizer.md]
O3  WSD & LR schedules (MiniCPM, DeepSeek, cosine, cooldown) [training_loop.md]
O5  speedrun recipes + VALUE EMBEDDINGS + U-Net skips + muP [architecture.md] — PRIORITY
O6  arch components: QK-norm, partial-RoPE, MTP [architecture.md]
O7  systems: Liger, FP8, FSDP/ZeRO, NCCL, portable ckpt, torch.compile [training_loop.md]
C1  code LLM pretraining & corpora (Codex, CodeGen, InCoder, CodeLlama, StarCoder2, Stack v2, CodeT5+) [data_pipelines.md]
C2  repo-level via inference-retrieval (RepoCoder, CoCoMIC, RepoFusion, RepoHyper, R2C2, kNM-LM, ReAcc) [eval_harness.md]
C3  repo-level via pretraining structure / topological ordering (DeepSeek-Coder topo, LongCoder, RepoFormer, CodeGPT repo) [merged_multisource.md]
C4  cross-file & repo code benchmarks (RepoBench, CrossCodeEval, RepoEval, EvoCodeBench, CoLT, ASE-2025, R2C2-Bench) [eval_harness.md]
C5  FIM/infilling + static-analysis-guided decoding (FIM, InCoder, SantaCoder, monitor-guided, type-constrained, tree-sitter) [link_detectors.md]
Q1  multi-hop QA datasets deeper (HotpotQA, WikiHop, 2Wiki, MuSiQue, HoVer, StrategyQA, Bamboogle, ComplexWebQ, MultiHop-RAG) [eval_harness.md]
Q2  multi-hop reasoning methods (DFGN/HDE/CogGraph deeper, CoT, self-ask, decompose, retrieve-and-read, GNN readers) [eval_harness.md]
Q3  MC likelihood eval methodology (lm-eval-harness, byte/length norm, GPT-3 scoring, contamination/decontam, bootstrap) [eval_harness.md]
Q4  long-context eval & utilization (lost-in-the-middle, needle, RULER, LongBench, ZeroSCROLLS, InfiniteBench) [generation_retrieval.md]
D1  wiki/hyperlink corpora deeper (WikiLinkGraphs, WikiText, KILT, enwiki, DBpedia, Wikidata, ClueWeb anchors) [data_pipelines.md]
D2  scientific citation corpora + citation-embedding (unarXive, S2ORC, OGB, SPECTER/SciNCL/SciBERT deeper, citation recommendation, PeS2o) [data_pipelines.md]
G5  entity linking / wikification / constrained decoding (entity disambiguation, GENRE autoregressive entity retrieval, trie-constrained decoding, mention detection) [link_detectors.md]
