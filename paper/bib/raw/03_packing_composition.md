# Packing & sequence composition (agent ab2ba112) — all verified vs arXiv

KEY: shi2024incontext = MOST IMPORTANT prior art (In-Context Pretraining, Meta).

@inproceedings{shi2024incontext, 2310.10638, ICLR 2024 — pack related docs via ANN retrieval + TSP-style ordering; STANDARD causal attention, NO cross-doc attention. We differ: explicit graph edges + mask granting cross-doc attention.}
@inproceedings{staniszewski2025structured, 2312.17296, AAAI 2025 (SPLiCe) — retrieval-collated long examples; SPLiCe-Repo uses repo/dir structure (closest to edge traversal); no attention along edges.}
@inproceedings{zhao2024analysing, 2402.13991, ACL 2024 — intra-document causal masking; treats cross-doc attention as HARMFUL, masks entirely. We selectively RE-ENABLE along link edges. Key baseline motivating block-diagonal isolation.}
@inproceedings{ding2024fewer, 2404.10830, ICML 2024 — best-fit bin-packing for doc integrity; orthogonal to relatedness/attention.}
@inproceedings{groeneveld2024olmo, 2402.00838, ACL 2024 — open pretraining recipe (standard packing). FLAG masking in body/code.}
@article{brown2020language, 2005.14165, NeurIPS 2020 — GPT-3, canonical concat-with-delimiter packing baseline. FLAG specifics in body.}
@article{touvron2023llama, 2302.13971, 2023 — standard packing recipe, contrast point.}
@article{shoeybi2019megatron, 1909.08053, 2019 — reset-attention-mask/reset-position-ids = doc-boundary masking we generalize. FLAG feature in codebase.}

Full bibtex bodies stored below:

```bibtex
@inproceedings{shi2024incontext,
  title = {In-Context Pretraining: Language Modeling Beyond Document Boundaries},
  author = {Shi, Weijia and Min, Sewon and Lomeli, Maria and Zhou, Chunting and Li, Margaret and Szilvasy, Gergely and James, Rich and Lin, Xi Victoria and Smith, Noah A. and Zettlemoyer, Luke and Yih, Scott and Lewis, Mike},
  booktitle = {The Twelfth International Conference on Learning Representations (ICLR)}, year = {2024},
  eprint = {2310.10638}, archivePrefix = {arXiv}, primaryClass = {cs.CL}, url = {https://arxiv.org/abs/2310.10638}
}
@inproceedings{staniszewski2025structured,
  title = {Structured Packing in {LLM} Training Improves Long Context Utilization},
  author = {Staniszewski, Konrad and Tworkowski, Szymon and Jaszczur, Sebastian and Zhao, Yu and Michalewski, Henryk and Kuci{\'n}ski, {\L}ukasz and Mi{\l}o{\'s}, Piotr},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence}, year = {2025},
  eprint = {2312.17296}, archivePrefix = {arXiv}, primaryClass = {cs.CL}, url = {https://arxiv.org/abs/2312.17296}
}
@inproceedings{zhao2024analysing,
  title = {Analysing the Impact of Sequence Composition on Language Model Pre-training},
  author = {Zhao, Yu and Qu, Yuanbin and Staniszewski, Konrad and Tworkowski, Szymon and Liu, Wei and Mi{\l}o{\'s}, Piotr and Wu, Yuxiang and Minervini, Pasquale},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)}, year = {2024},
  eprint = {2402.13991}, archivePrefix = {arXiv}, primaryClass = {cs.CL}, url = {https://arxiv.org/abs/2402.13991}
}
@inproceedings{ding2024fewer,
  title = {Fewer Truncations Improve Language Modeling},
  author = {Ding, Hantian and Wang, Zijian and Paolini, Giovanni and Kumar, Varun and Deoras, Anoop and Roth, Dan and Soatto, Stefano},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)}, year = {2024},
  eprint = {2404.10830}, archivePrefix = {arXiv}, primaryClass = {cs.CL}, url = {https://arxiv.org/abs/2404.10830}
}
@inproceedings{groeneveld2024olmo,
  title = {{OLMo}: Accelerating the Science of Language Models},
  author = {Groeneveld, Dirk and Beltagy, Iz and Walsh, Pete and others},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)}, year = {2024},
  eprint = {2402.00838}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@article{brown2020language,
  title = {Language Models are Few-Shot Learners},
  author = {Brown, Tom B. and others},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)}, volume = {33}, year = {2020},
  eprint = {2005.14165}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@article{touvron2023llama,
  title = {{LLaMA}: Open and Efficient Foundation Language Models},
  author = {Touvron, Hugo and others}, year = {2023},
  eprint = {2302.13971}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@article{shoeybi2019megatron,
  title = {Megatron-{LM}: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author = {Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan}, year = {2019},
  eprint = {1909.08053}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
```
