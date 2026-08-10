# Architecture / optimizer / training infra (agent a1a769f3) — verified vs arXiv/github

PRIORITY (Muon + modded-nanoGPT), author-recommended @misc forms:

```bibtex
@misc{jordan2024muon,
  author = {Keller Jordan and Yuchen Jin and Vlado Boza and Jiacheng You and Franz Cesista and Laker Newhouse and Jeremy Bernstein},
  title  = {Muon: An optimizer for hidden layers in neural networks}, year = {2024},
  url    = {https://kellerjordan.github.io/posts/muon/}
}
@misc{liu2025muonscalable,
  title  = {Muon is Scalable for {LLM} Training},
  author = {Jingyuan Liu and Jianlin Su and Xingcheng Yao and Zhejun Jiang and Guokun Lai and Yulun Du and Yidao Qin and Weixin Xu and Enzhe Lu and Junjie Yan and Yanru Chen and Huabin Zheng and Yibo Liu and Shaowei Liu and Bohong Yin and Weiran He and Han Zhu and Yuzhi Wang and Jianzhou Wang and Mengnan Dong and Zheng Zhang and Yongsheng Kang and Hao Zhang and Xinran Xu and Yutao Zhang and Yuxin Wu and Xinyu Zhou and Zhilin Yang},
  year = {2025}, eprint = {2502.16982}, archivePrefix = {arXiv}, primaryClass = {cs.LG}, url = {https://arxiv.org/abs/2502.16982}
}
@misc{jordan2024moddednanogpt,
  author = {Keller Jordan and Jeremy Bernstein and Brendan Rappazzo and {@fernbear.bsky.social} and Boza Vlado and You Jiacheng and Franz Cesista and Braden Koszarsky and {@Grad62304977}},
  title  = {modded-nanogpt: Speedrunning the NanoGPT baseline}, year = {2024},
  url    = {https://github.com/KellerJordan/modded-nanogpt}
}
@misc{karpathy2022nanogpt,
  author = {Andrej Karpathy}, title = {nanoGPT}, year = {2022}, url = {https://github.com/karpathy/nanoGPT}
}
@inproceedings{vaswani2017attention,
  title = {Attention Is All You Need},
  author = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and {\L}ukasz Kaiser and Illia Polosukhin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2017}, url = {https://arxiv.org/abs/1706.03762}
}
@article{radford2019gpt2,
  title = {Language Models are Unsupervised Multitask Learners},
  author = {Alec Radford and Jeffrey Wu and Rewon Child and David Luan and Dario Amodei and Ilya Sutskever},
  journal = {OpenAI technical report}, year = {2019}
}
@inproceedings{brown2020gpt3,
  title = {Language Models are Few-Shot Learners},
  author = {Tom B. Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared Kaplan and Prafulla Dhariwal and Arvind Neelakantan and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2020}, url = {https://arxiv.org/abs/2005.14165}
}
@article{su2021roformer,
  title = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author = {Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu},
  journal = {arXiv preprint arXiv:2104.09864}, year = {2021}, url = {https://arxiv.org/abs/2104.09864}
}
@inproceedings{zhang2019rmsnorm,
  title = {Root Mean Square Layer Normalization}, author = {Biao Zhang and Rico Sennrich},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2019}, url = {https://arxiv.org/abs/1910.07467}
}
@article{so2021primer,
  title = {Primer: Searching for Efficient Transformers for Language Modeling},
  author = {David R. So and Wojciech Ma\'{n}ke and Hanxiao Liu and Zihang Dai and Noam Shazeer and Quoc V. Le},
  journal = {arXiv preprint arXiv:2109.08668}, year = {2021}, url = {https://arxiv.org/abs/2109.08668}
}
@article{shazeer2020glu,
  title = {GLU Variants Improve Transformer}, author = {Noam Shazeer},
  journal = {arXiv preprint arXiv:2002.05202}, year = {2020}, url = {https://arxiv.org/abs/2002.05202}
}
@article{gemmateam2024gemma2,
  title = {Gemma 2: Improving Open Language Models at a Practical Size}, author = {{Gemma Team}},
  journal = {arXiv preprint arXiv:2408.00118}, year = {2024}, url = {https://arxiv.org/abs/2408.00118}
}
@inproceedings{press2017tying,
  title = {Using the Output Embedding to Improve Language Models}, author = {Ofir Press and Lior Wolf},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the ACL (EACL)}, year = {2017}, url = {https://arxiv.org/abs/1608.05859}
}
@inproceedings{inan2017tying,
  title = {Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling},
  author = {Hakan Inan and Khashayar Khosravi and Richard Socher},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2017}, url = {https://arxiv.org/abs/1611.01462}
}
@inproceedings{kingma2015adam,
  title = {Adam: A Method for Stochastic Optimization}, author = {Diederik P. Kingma and Jimmy Ba},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2015}, url = {https://arxiv.org/abs/1412.6980}
}
@inproceedings{loshchilov2019adamw,
  title = {Decoupled Weight Decay Regularization}, author = {Ilya Loshchilov and Frank Hutter},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2019}, url = {https://arxiv.org/abs/1711.05101}
}
@inproceedings{micikevicius2018mixed,
  title = {Mixed Precision Training},
  author = {Paulius Micikevicius and Sharan Narang and Jonah Alben and Gregory Diamos and Erich Elsen and David Garcia and Boris Ginsburg and Michael Houston and Oleksii Kuchaiev and Ganesh Venkatesh and Hao Wu},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2018}, url = {https://arxiv.org/abs/1710.03740}
}
@article{kaplan2020scaling,
  title = {Scaling Laws for Neural Language Models},
  author = {Jared Kaplan and Sam McCandlish and Tom Henighan and Tom B. Brown and Benjamin Chess and Rewon Child and Scott Gray and Alec Radford and Jeffrey Wu and Dario Amodei},
  journal = {arXiv preprint arXiv:2001.08361}, year = {2020}, url = {https://arxiv.org/abs/2001.08361}
}
@article{hoffmann2022chinchilla,
  title = {Training Compute-Optimal Large Language Models},
  author = {Jordan Hoffmann and Sebastian Borgeaud and Arthur Mensch and others},
  journal = {arXiv preprint arXiv:2203.15556}, year = {2022}, url = {https://arxiv.org/abs/2203.15556}
}
@inproceedings{ainslie2023gqa,
  title = {GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author = {Joshua Ainslie and James Lee-Thorp and Michiel de Jong and Yury Zemlyanskiy and Federico Lebr\'{o}n and Sumit Sanghai},
  booktitle = {Proceedings of the 2023 Conference on EMNLP}, year = {2023}, url = {https://arxiv.org/abs/2305.13245}
}
@inproceedings{paszke2019pytorch,
  title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author = {Adam Paszke and Sam Gross and Francisco Massa and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2019}, url = {https://arxiv.org/abs/1912.01703}
}
@article{li2020pytorchddp,
  title = {PyTorch Distributed: Experiences on Accelerating Data Parallel Training},
  author = {Shen Li and Yanli Zhao and Rohan Varma and Omkar Salpekar and Pieter Noordhuis and Teng Li and Adam Paszke and Jeff Smith and Brian Vaughan and Pritam Damania and Soumith Chintala},
  journal = {Proceedings of the VLDB Endowment}, volume = {13}, number = {12}, pages = {3005--3018}, year = {2020}, url = {https://arxiv.org/abs/2006.15704}
}
@inproceedings{ansel2024pytorch2,
  title = {PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation},
  author = {Jason Ansel and Edward Yang and Horace He and Natalia Gimelshein and Animesh Jain and Michael Voznesensky and Bin Bao and Peter Bell and David Berard and Evgeni Burovski and others},
  booktitle = {Proceedings of the 29th ACM ASPLOS}, year = {2024}, doi = {10.1145/3620665.3640366}
}
@misc{submitit,
  author = {J{\'e}r{\'e}my Rapin and Louis Martin and {Facebook AI Research}},
  title  = {submitit: A lightweight tool for submitting Python functions onto a Slurm cluster}, year = {2020},
  url    = {https://github.com/facebookincubator/submitit}
}
```

NOTES: Muon(2D weights); liu2025 = Muon-scales-to-LLM; modded-nanogpt/nanogpt = recipe lineage; vaswani=transformer; radford2019/brown2020=decoder-only; su2021=RoPE; zhang2019=RMSNorm; so2021=squared-ReLU MLP; shazeer2020=SwiGLU(if compared); gemma2=logit soft-cap; press/inan=weight tying; adam/adamw; micikevicius=bf16; kaplan/chinchilla=scaling/token budget; gqa(if used); pytorch/ddp/pytorch2(torch.compile)/submitit=systems.
FLAGS: ansel2024pytorch2 DOI/venue not web-confirmed (ACM 403); gemma2 soft-capping in body not abstract; press2017 correct title = "...to Improve Language Models".
