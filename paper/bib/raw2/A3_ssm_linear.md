# A3 SSM/linear-time (agent abaa65bf) — positioning-only, 10 verified vs arXiv
Contrast: all compress history into fixed recurrent state / global kernel summary → forecloses content-addressed cross-doc attention edge (linking doc must read SPECIFIC target KV = needs retained per-token KV = softmax+sparsity). None train on graph structure.

```bibtex
@inproceedings{gu2022s4,
  title = {Efficiently Modeling Long Sequences with Structured State Spaces},
  author = {Gu, Albert and Goel, Karan and R{\'e}, Christopher}, booktitle = {ICLR}, year = {2022}, eprint = {2111.00396}, archivePrefix = {arXiv}
}
@inproceedings{gu2023mamba,
  title = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author = {Gu, Albert and Dao, Tri}, booktitle = {Conference on Language Modeling (COLM)}, year = {2024}, eprint = {2312.00752}, archivePrefix = {arXiv}
}
@inproceedings{dao2024mamba2,
  title = {Transformers are {SSMs}: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author = {Dao, Tri and Gu, Albert}, booktitle = {ICML}, year = {2024}, eprint = {2405.21060}, archivePrefix = {arXiv}
}
@inproceedings{peng2023rwkv,
  title = {{RWKV}: Reinventing {RNNs} for the Transformer Era},
  author = {Peng, Bo and Alcaide, Eric and Anthony, Quentin and others}, booktitle = {Findings of the ACL: EMNLP 2023}, year = {2023}, eprint = {2305.13048}, archivePrefix = {arXiv}
}
@inproceedings{katharopoulos2020linear,
  title = {Transformers are {RNNs}: Fast Autoregressive Transformers with Linear Attention},
  author = {Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\c{c}}ois}, booktitle = {ICML}, year = {2020}, eprint = {2006.16236}, archivePrefix = {arXiv}
}
@inproceedings{choromanski2021performer,
  title = {Rethinking Attention with Performers},
  author = {Choromanski, Krzysztof and Likhosherstov, Valerii and Dohan, David and Song, Xingyou and Gane, Andreea and Sarl{\'o}s, Tam{\'a}s and Hawkins, Peter and Davis, Jared and Mohiuddin, Afroz and Kaiser, Lukasz and Belanger, David and Colwell, Lucy and Weller, Adrian}, booktitle = {ICLR}, year = {2021}, eprint = {2009.14794}, archivePrefix = {arXiv}
}
@article{sun2023retnet,
  title = {Retentive Network: A Successor to Transformer for Large Language Models},
  author = {Sun, Yutao and Dong, Li and Huang, Shaohan and Ma, Shuming and Xia, Yuqing and Xue, Jilong and Wang, Jianyong and Wei, Furu}, journal = {arXiv preprint arXiv:2307.08621}, year = {2023}
}
@inproceedings{poli2023hyena,
  title = {Hyena Hierarchy: Towards Larger Convolutional Language Models},
  author = {Poli, Michael and Massaroli, Stefano and Nguyen, Eric and Fu, Daniel Y. and Dao, Tri and Baccus, Stephen and Bengio, Yoshua and Ermon, Stefano and R{\'e}, Christopher}, booktitle = {ICML}, year = {2023}, eprint = {2302.10866}, archivePrefix = {arXiv}
}
@inproceedings{arora2024based,
  title = {Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff},
  author = {Arora, Simran and Eyuboglu, Sabri and Zhang, Michael and Timalsina, Aman and Alberti, Silas and Zinsley, Dylan and Zou, James and Rudra, Atri and R{\'e}, Christopher}, booktitle = {ICML}, year = {2024}, eprint = {2402.18668}, archivePrefix = {arXiv}
}
@article{de2024griffin,
  title = {Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models},
  author = {De, Soham and Smith, Samuel L. and Fernando, Anushan and Botev, Aleksandar and Cristian-Muraru, George and Gu, Albert and others}, journal = {arXiv preprint arXiv:2402.19427}, year = {2024}
}
```
NOTES: S4=canonical SSM no QK addressing. Mamba=selective (content-gated forgetting NOT content-addressed retrieval). Mamba2=SSD duality (structured low-rank attn, can't express arbitrary sparse cross-doc read). RWKV=RNN linear-attn no persistent KV. Katharopoulos=linear-attn root (summed outer-product discards individual KV). Performer=random-feature softmax approx (degrades sharp lookups). RetNet=decay retention (arXiv-only FLAG). Hyena=long conv/gating. Based=recall-throughput (studies recall limit, still local softmax). Griffin/Hawk=gated linear recurrence + LOCAL attn only (arXiv-only FLAG). arora2024based recall relevant to our long-range cross-doc argument.
