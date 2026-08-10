# Code / repo-level LM + benchmarks (agent a83d2e68) — 20 works, verified vs arXiv/DBLP/ACL

CLOSEST CODE PRIOR ART = guo2024deepseekcoder: §2.2 topological sort of files by import graph for pretraining data, then FLAT concat + plain causal attention (NO edge-keyed cross-file attention) = exactly our gap.
Inference-retrieval CONTRAST class: repocoder, crosscodeeval, cocomic, repofusion, r2c2coder, repohyper, rlpg, monitor(MGD).
Pretraining-packing (flat, no edge attn): starcoder, starcoder2, deepseekcoder.
Single-file baselines: humaneval, humanevalpack(octopack), mbpp, codexglue.
NOTE: StackV2 == StarCoder2 SAME paper (2402.19173) — one key only.

```bibtex
@article{kocetkov2022stack,
  title = {The Stack: 3 TB of permissively licensed source code},
  author = {Kocetkov, Denis and Li, Raymond and Ben Allal, Loubna and Li, Jia and Mou, Chenghao and Mu{\~n}oz Ferrandis, Carlos and Jernite, Yacine and Mitchell, Margaret and Hughes, Sean and Wolf, Thomas and Bahdanau, Dzmitry and von Werra, Leandro and de Vries, Harm},
  journal = {arXiv preprint arXiv:2211.15533}, year = {2022}, eprint = {2211.15533}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@article{li2023starcoder,
  title = {StarCoder: may the source be with you!},
  author = {Li, Raymond and Ben Allal, Loubna and Zi, Yangtian and Muennighoff, Niklas and Kocetkov, Denis and others},
  journal = {arXiv preprint arXiv:2305.06161}, year = {2023}, eprint = {2305.06161}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@article{lozhkov2024starcoder2,
  title = {StarCoder 2 and The Stack v2: The Next Generation},
  author = {Lozhkov, Anton and Li, Raymond and Ben Allal, Loubna and Cassano, Federico and Lamy-Poirier, Joel and others},
  journal = {arXiv preprint arXiv:2402.19173}, year = {2024}, eprint = {2402.19173}, archivePrefix = {arXiv}, primaryClass = {cs.SE}
}
@inproceedings{liu2024repobench,
  title = {RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems},
  author = {Liu, Tianyang and Xu, Canwen and McAuley, Julian},
  booktitle = {The Twelfth International Conference on Learning Representations (ICLR)}, year = {2024},
  eprint = {2306.03091}, archivePrefix = {arXiv}, primaryClass = {cs.CL}, url = {https://arxiv.org/abs/2306.03091}
}
@inproceedings{ding2023crosscodeeval,
  title = {CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion},
  author = {Ding, Yangruibo and Wang, Zijian and Ahmad, Wasi Uddin and Ding, Hantian and Tan, Ming and Jain, Nihal and Ramanathan, Murali Krishna and Nallapati, Ramesh and Bhatia, Parminder and Roth, Dan and Xiang, Bing},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS) Datasets and Benchmarks Track}, year = {2023},
  eprint = {2310.11248}, archivePrefix = {arXiv}, primaryClass = {cs.LG}, url = {https://arxiv.org/abs/2310.11248}
}
@inproceedings{zhang2023repocoder,
  title = {RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation},
  author = {Zhang, Fengji and Chen, Bei and Zhang, Yue and Keung, Jacky and Liu, Jin and Zan, Daoguang and Mao, Yi and Lou, Jian-Guang and Chen, Weizhu},
  booktitle = {Proceedings of the 2023 Conference on EMNLP}, year = {2023},
  eprint = {2303.12570}, archivePrefix = {arXiv}, primaryClass = {cs.CL}, url = {https://arxiv.org/abs/2303.12570}
}
@inproceedings{ding2024cocomic,
  title = {CoCoMIC: Code Completion by Jointly Modeling In-file and Cross-file Context},
  author = {Ding, Yangruibo and Wang, Zijian and Ahmad, Wasi Uddin and Ramanathan, Murali Krishna and Nallapati, Ramesh and Bhatia, Parminder and Roth, Dan and Xiang, Bing},
  booktitle = {Proceedings of LREC-COLING 2024}, pages = {3433--3445}, year = {2024},
  eprint = {2212.10007}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@article{shrivastava2023repofusion,
  title = {RepoFusion: Training Code Models to Understand Your Repository},
  author = {Shrivastava, Disha and Kocetkov, Denis and de Vries, Harm and Bahdanau, Dzmitry and Scholak, Torsten},
  journal = {arXiv preprint arXiv:2306.10998}, year = {2023}, eprint = {2306.10998}, archivePrefix = {arXiv}, primaryClass = {cs.LG}
}
@article{deng2024r2c2coder,
  title = {R2C2-Coder: Enhancing and Benchmarking Real-world Repository-level Code Completion Abilities of Code Large Language Models},
  author = {Deng, Ken and Liu, Jiaheng and Zhu, He and Liu, Congnan and others},
  journal = {arXiv preprint arXiv:2406.01359}, year = {2024}, eprint = {2406.01359}, archivePrefix = {arXiv}, primaryClass = {cs.SE}
}
@article{phan2024repohyper,
  title = {RepoHyper: Search-Expand-Refine on Semantic Graphs for Repository-Level Code Completion},
  author = {Phan, Huy N. and Phan, Hoang N. and Nguyen, Tien N. and Bui, Nghi D. Q.},
  journal = {arXiv preprint arXiv:2403.06095}, year = {2024}, eprint = {2403.06095}, archivePrefix = {arXiv}, primaryClass = {cs.SE}
}
@article{bairi2024codeplan,
  title = {CodePlan: Repository-level Coding using LLMs and Planning},
  author = {Bairi, Ramakrishna and Sonwane, Atharv and Kanade, Aditya and D C, Vageesh and Iyer, Arun and Parthasarathy, Suresh and Rajamani, Sriram and Ashok, B. and Shet, Shashank},
  journal = {arXiv preprint arXiv:2309.12499}, year = {2023}, eprint = {2309.12499}, archivePrefix = {arXiv}, primaryClass = {cs.SE}
}
@article{guo2024deepseekcoder,
  title = {{DeepSeek-Coder}: When the Large Language Model Meets Programming -- The Rise of Code Intelligence},
  author = {Guo, Daya and Zhu, Qihao and Yang, Dejian and Xie, Zhenda and Dong, Kai and Zhang, Wentao and Chen, Guanting and Bi, Xiao and Wu, Y. and Li, Y. K. and Luo, Fuli and Xiong, Yingfei and Liang, Wenfeng},
  journal = {arXiv preprint arXiv:2401.14196}, year = {2024}, eprint = {2401.14196}, archivePrefix = {arXiv}, primaryClass = {cs.SE}, url = {https://arxiv.org/abs/2401.14196}
}
@inproceedings{shrivastava2023rlpg,
  title = {Repository-Level Prompt Generation for Large Language Models of Code},
  author = {Shrivastava, Disha and Larochelle, Hugo and Tarlow, Daniel},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)}, year = {2023},
  eprint = {2206.12839}, archivePrefix = {arXiv}, primaryClass = {cs.LG}
}
@inproceedings{agrawal2023monitor,
  title = {Guiding Language Models of Code with Global Context using Monitors},
  author = {Agrawal, Lakshya A and Kanade, Aditya and Goyal, Navin and Lahiri, Shuvendu K. and Rajamani, Sriram K.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2023},
  eprint = {2306.10763}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{wang2023codet5plus,
  title = {{CodeT5+}: Open Code Large Language Models for Code Understanding and Generation},
  author = {Wang, Yue and Le, Hung and Gotmare, Akhilesh Deepak and Bui, Nghi D. Q. and Li, Junnan and Hoi, Steven C. H.},
  booktitle = {Proceedings of the 2023 Conference on EMNLP}, pages = {1069--1088}, year = {2023}, publisher = {ACL}
}
@article{chen2021humaneval,
  title = {Evaluating Large Language Models Trained on Code},
  author = {Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and others},
  journal = {arXiv preprint arXiv:2107.03374}, year = {2021}, eprint = {2107.03374}, archivePrefix = {arXiv}, primaryClass = {cs.LG}
}
@article{muennighoff2024octopack,
  title = {OctoPack: Instruction Tuning Code Large Language Models},
  author = {Muennighoff, Niklas and Liu, Qian and Zebaze, Armel and Zheng, Qinkai and Hui, Binyuan and Zhuo, Terry Yue and Singh, Swayam and Tang, Xiangru and von Werra, Leandro and Longpre, Shayne},
  journal = {arXiv preprint arXiv:2308.07124}, year = {2024}, eprint = {2308.07124}, archivePrefix = {arXiv}, primaryClass = {cs.CL},
  note = {Introduces HumanEvalPack; ICLR 2024}
}
@article{austin2021mbpp,
  title = {Program Synthesis with Large Language Models},
  author = {Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and Sutton, Charles},
  journal = {arXiv preprint arXiv:2108.07732}, year = {2021}, eprint = {2108.07732}, archivePrefix = {arXiv}, primaryClass = {cs.PL}
}
@inproceedings{lu2021codexglue,
  title = {{CodeXGLUE}: A Machine Learning Benchmark Dataset for Code Understanding and Generation},
  author = {Lu, Shuai and Guo, Daya and Ren, Shuo and Huang, Junjie and Svyatkovskiy, Alexey and others},
  booktitle = {Proceedings of NeurIPS Datasets and Benchmarks}, year = {2021},
  eprint = {2102.04664}, archivePrefix = {arXiv}, primaryClass = {cs.SE}
}
```

FLAGS: octopack ICLR2024 & codexglue NeurIPS2021 venues not on arXiv page; codet5+ "single-file" inferred; starcoder2 author roster truncated (verify tail).
