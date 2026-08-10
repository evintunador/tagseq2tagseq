# RAG & KV-cache retrieval (agent ad1d3e80) — the PRIORITY slice

## ★ MYSTERY PAPER VERDICT (author remembered Google/DeepMind cached-per-doc-KV, frozen corpus, no code)
- **Most likely = wu2022memorizing "Memorizing Transformers"** (Wu, Rabe, Hutchins, Szegedy; Google; ICLR 2022 spotlight; 2203.08913). ONLY paper caching literal (k,v) attention states into a NON-differentiable frozen external memory, retrieved by kNN into a designated attention layer, NOT trained on the store, memory populated from doc corpora (C4/arXiv/GitHub/PG-19). No official code. Mismatches: token-level granularity (not per-doc), and the model IS trained to use the kNN layer. → If "KV cache" is load-bearing, THIS is it.
- **Closest to "per-document knowledge corpus" framing = dejong2022tome "Mention Memory / TOME"** (Google, ICLR 2022, 2110.06176). Precomputes corpus-wide table (150M Wikipedia mention reps), frozen swappable store, injected via "memory attention" layers, no official code. Mismatch: learned embedding vectors at mention granularity, not raw per-doc KV; model pretrained to use it.
- Same Google line: dejong2023lumen (LUMEN, ICML 2023, 2301.10448), dejong2023glimmer (2306.10231, preprint).
- RECOMMENDATION: cite Memorizing Transformers as primary "cached-KV frozen knowledge store, inference-only" prior art; TOME/LUMEN as "precomputed per-corpus memory attention." Our novelty framing: cached KV at document/node granularity + graph-structured cross-doc attention mask applied in BOTH training AND inference (generated link deterministically fetches target node) — none combine these.

```bibtex
@inproceedings{lewis2020rag,
  title = {Retrieval-Augmented Generation for Knowledge-Intensive {NLP} Tasks},
  author = {Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, volume = {33}, pages = {9459--9474}, year = {2020},
  eprint = {2005.11401}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{guu2020realm,
  title = {{REALM}: Retrieval-Augmented Language Model Pre-Training},
  author = {Guu, Kelvin and Lee, Kenton and Tung, Zora and Pasupat, Panupong and Chang, Ming-Wei},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)}, volume = {119}, pages = {3929--3938}, year = {2020}, publisher = {PMLR},
  eprint = {2002.08909}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@article{izacard2023atlas,
  title = {Atlas: Few-shot Learning with Retrieval Augmented Language Models},
  author = {Izacard, Gautier and Lewis, Patrick and Lomeli, Maria and Hosseini, Lucas and Petroni, Fabio and Schick, Timo and Dwivedi-Yu, Jane and Joulin, Armand and Riedel, Sebastian and Grave, Edouard},
  journal = {Journal of Machine Learning Research}, volume = {24}, number = {251}, pages = {1--43}, year = {2023},
  eprint = {2208.03299}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{izacard2021fid,
  title = {Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering},
  author = {Izacard, Gautier and Grave, Edouard},
  booktitle = {Proceedings of the 16th Conference of the European Chapter of the ACL (EACL)}, pages = {874--880}, year = {2021}, publisher = {ACL},
  eprint = {2007.01282}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{borgeaud2022retro,
  title = {Improving Language Models by Retrieving from Trillions of Tokens},
  author = {Borgeaud, Sebastian and Mensch, Arthur and Hoffmann, Jordan and Cai, Trevor and Rutherford, Eliza and Millican, Katie and van den Driessche, George and Lespiau, Jean-Baptiste and Damoc, Bogdan and Clark, Aidan and de Las Casas, Diego and Guy, Aurelia and Menick, Jacob and Ring, Roman and Hennigan, Tom and Huang, Saffron and Maggiore, Loren and Jones, Chris and Cassirer, Albin and Brock, Andy and Paganini, Michela and Irving, Geoffrey and Vinyals, Oriol and Osindero, Simon and Simonyan, Karen and Rae, Jack W. and Elsen, Erich and Sifre, Laurent},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning (ICML)}, volume = {162}, pages = {2206--2240}, year = {2022}, publisher = {PMLR},
  eprint = {2112.04426}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{khandelwal2020knnlm,
  title = {Generalization through Memorization: Nearest Neighbor Language Models},
  author = {Khandelwal, Urvashi and Levy, Omer and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2020},
  eprint = {1911.00172}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{shi2024replug,
  title = {{REPLUG}: Retrieval-Augmented Black-Box Language Models},
  author = {Shi, Weijia and Min, Sewon and Yasunaga, Michihiro and Seo, Minjoon and James, Rich and Lewis, Mike and Zettlemoyer, Luke and Yih, Wen-tau},
  booktitle = {Proceedings of NAACL-HLT}, pages = {8371--8384}, year = {2024}, publisher = {ACL},
  eprint = {2301.12652}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{asai2024selfrag,
  title = {Self-{RAG}: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  author = {Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and Hajishirzi, Hannaneh},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2024},
  eprint = {2310.11511}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@article{ram2023incontextralm,
  title = {In-Context Retrieval-Augmented Language Models},
  author = {Ram, Ori and Levine, Yoav and Dalmedigos, Itay and Muhlgay, Dor and Shashua, Amnon and Leyton-Brown, Kevin and Shoham, Yoav},
  journal = {Transactions of the Association for Computational Linguistics}, volume = {11}, pages = {1316--1331}, year = {2023},
  eprint = {2302.00083}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{gim2024promptcache,
  title = {Prompt Cache: Modular Attention Reuse for Low-Latency Inference},
  author = {Gim, In and Chen, Guojun and Lee, Seung-seob and Sarda, Nikhil and Khandelwal, Anurag and Zhong, Lin},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys)}, year = {2024},
  eprint = {2311.04934}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{yao2025cacheblend,
  title = {CacheBlend: Fast Large Language Model Serving for {RAG} with Cached Knowledge Fusion},
  author = {Yao, Jiayi and Li, Hanchen and Liu, Yuhan and Ray, Siddhant and Cheng, Yihua and Zhang, Qizheng and Du, Kuntai and Lu, Shan and Jiang, Junchen},
  booktitle = {Proceedings of the Twentieth European Conference on Computer Systems (EuroSys)}, year = {2025},
  eprint = {2405.16444}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{ma2025blockattention,
  title = {Block-Attention for Efficient Prefilling},
  author = {Ma, Dongyang and Wang, Yan and Tian, Lan},
  booktitle = {The Thirteenth International Conference on Learning Representations (ICLR)}, year = {2025},
  eprint = {2409.15355}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{chan2025cag,
  title = {Don't Do {RAG}: When Cache-Augmented Generation is All You Need for Knowledge Tasks},
  author = {Chan, Brian J. and Chen, Chao-Ting and Cheng, Jui-Hung and Huang, Hen-Hsen},
  booktitle = {Companion Proceedings of the ACM Web Conference (WWW Companion)}, year = {2025},
  eprint = {2412.15605}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{wu2022memorizing,
  title = {Memorizing Transformers},
  author = {Wu, Yuhuai and Rabe, Markus N. and Hutchins, DeLesley and Szegedy, Christian},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2022}, note = {Spotlight},
  eprint = {2203.08913}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{dejong2022tome,
  title = {Mention Memory: Incorporating Textual Knowledge into Transformers through Entity Mention Attention},
  author = {de Jong, Michiel and Zemlyanskiy, Yury and FitzGerald, Nicholas and Sha, Fei and Cohen, William W.},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2022}, note = {TOME},
  eprint = {2110.06176}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{dejong2023lumen,
  title = {Pre-computed Memory or On-the-fly Encoding? A Hybrid Approach to Retrieval Augmentation Makes the Most of Your Compute},
  author = {de Jong, Michiel and Zemlyanskiy, Yury and FitzGerald, Nicholas and Ainslie, Joshua and Sanghai, Sumit and Sha, Fei and Cohen, William W.},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)}, year = {2023}, note = {LUMEN},
  eprint = {2301.10448}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@article{dejong2023glimmer,
  title = {{GLIMMER}: Generalized Late-Interaction Memory Reranker},
  author = {de Jong, Michiel and Zemlyanskiy, Yury and FitzGerald, Nicholas and Sanghai, Sumit and Cohen, William W. and Ainslie, Joshua},
  journal = {arXiv preprint arXiv:2306.10231}, year = {2023}
}
```

NOTES (train-on-structure vs retrieve-at-inference; cached-KV?):
- lewis2020rag: DPR+BART, passages concatenated as text, generator+query-enc trained, index frozen. No cached KV. Train+infer retrieval.
- guu2020realm: learned retriever baked into MLM PRE-training, backprop through retrieval. Text prepend, no cached KV. (unique: retrieval at pretraining)
- izacard2023atlas: Contriever+T5 FiD, jointly trained. No cached KV.
- izacard2021fid: FiD, passages encoded independently, decoder attends over concat. No cached KV. Backbone of LUMEN/GLIMMER.
- borgeaud2022retro: CLOSEST train-time comparator. kNN chunks from trillion-token DB via chunked cross-attention (CCA). Trained w/ retrieval from scratch; frozen retriever; neighbor encodings precomputed/cached & cross-attended. Chunk-level, not link/graph.
- khandelwal2020knnlm: THE KV-cache exemplar. Frozen LM + datastore of (hidden-state→next-token); interpolate with kNN. Inference-only, no retrieval training. Sharpest contrast.
- shi2024replug: frozen black-box LM, prepend+ensemble. Inference-time, no cached KV.
- asai2024selfrag: trains LM to decide WHEN to retrieve + reflection tokens. Text-level. FLAG ICLR2024 venue unconfirmed.
- ram2023incontextralm: frozen LM, prepend retrieved docs, optional trained reranker. Inference-only pole (w/ REPLUG).
- gim2024promptcache: precompute KV for reusable prompt segments, adjust positions. Inference-only serving opt, nothing trained. Caches actual k/v tensors, splices into attention. Inference-only precursor to reusing doc KV.
- yao2025cacheblend: precompute per-chunk KV, selectively recompute cross-chunk tokens, fuse. Inference-only. Closest inference analogue to "fetch target doc into attention" but APPROXIMATES rather than TRAINS cross-doc interaction. FLAG EuroSys2025 venue consensus not on arXiv.
- ma2025blockattention: block-local KV + final query block attends across; FINE-TUNES model to adapt to block mask, re-indexes positions. MOST relevant: custom block-structured mask at BOTH train+inference — parallels ours; difference = ours graph/link-structured (generated link fetches specific target node) vs flat retrieved blocks. Title = "Efficient Prefilling", ICLR 2025.
- chan2025cag: preload whole KB, precompute one monolithic corpus KV cache, append query. Inference-only. No per-doc caches selected by retrieval, no structured mask.
- wu2022memorizing / bertsch2023unlimiformer / mohtashami2023landmark / tworkowski2023focused: see long-context slice (04). Cached-KV external memory attention family.
- dejong2022tome/lumen/glimmer: Google precomputed-corpus-memory line (mystery candidates).

FLAGS: asai selfrag ICLR2024, yao cacheblend EuroSys2025 unconfirmed; glimmer no venue (preprint). RETRO+kNN-LM also belong to KV-cache discussion, cross-ref.
NOTE overlap w/ slice 04 (long-context): wu2022memorizing, bertsch2023unlimiformer, mohtashami2023landmark, tworkowski2023focused, borgeaud2022retro — dedup on merge.
