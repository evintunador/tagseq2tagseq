# R5 dense/late-interaction retrieval backbones (agent a5a242b6) — 12 verified vs arXiv
CONTEXT: TAGSeq2TAGSeq uses NO learned retriever; "retrieval"=deterministic graph-edge/identifier resolution. These = learned-similarity paradigm it departs from.

```bibtex
@inproceedings{karpukhin2020dpr,
  title = {Dense Passage Retrieval for Open-Domain Question Answering},
  author = {Karpukhin, Vladimir and O{\u{g}}uz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau},
  booktitle = {Proceedings of the 2020 Conference on EMNLP}, pages = {6769--6781}, year = {2020}, eprint = {2004.04906}, archivePrefix = {arXiv}
}
@inproceedings{khattab2020colbert,
  title = {ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT},
  author = {Khattab, Omar and Zaharia, Matei},
  booktitle = {Proceedings of the 43rd International ACM SIGIR}, pages = {39--48}, year = {2020}, eprint = {2004.12832}, archivePrefix = {arXiv}
}
@inproceedings{santhanam2022colbertv2,
  title = {ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction},
  author = {Santhanam, Keshav and Khattab, Omar and Saad-Falcon, Jon and Potts, Christopher and Zaharia, Matei},
  booktitle = {Proceedings of NAACL-HLT}, pages = {3715--3734}, year = {2022}, eprint = {2112.01488}, archivePrefix = {arXiv}
}
@article{izacard2022contriever,
  title = {Unsupervised Dense Information Retrieval with Contrastive Learning},
  author = {Izacard, Gautier and Caron, Mathilde and Hosseini, Lucas and Riedel, Sebastian and Bojanowski, Piotr and Joulin, Armand and Grave, Edouard},
  journal = {Transactions on Machine Learning Research (TMLR)}, year = {2022}, eprint = {2112.09118}, archivePrefix = {arXiv}
}
@inproceedings{ni2022gtr,
  title = {Large Dual Encoders Are Generalizable Retrievers},
  author = {Ni, Jianmo and Qu, Chen and Lu, Jing and Dai, Zhuyun and Hern{\'a}ndez {\'A}brego, Gustavo and Ma, Ji and Zhao, Vincent Y. and Luan, Yi and Hall, Keith B. and Chang, Ming-Wei and Yang, Yinfei},
  booktitle = {Proceedings of the 2022 Conference on EMNLP}, pages = {9844--9855}, year = {2022}, eprint = {2112.07899}, archivePrefix = {arXiv}
}
@inproceedings{xiong2021ance,
  title = {Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval},
  author = {Xiong, Lee and Xiong, Chenyan and Li, Ye and Tang, Kwok-Fung and Liu, Jialin and Bennett, Paul and Ahmed, Junaid and Overwijk, Arnold},
  booktitle = {ICLR}, year = {2021}, eprint = {2007.00808}, archivePrefix = {arXiv}
}
@inproceedings{formal2021splade,
  title = {SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking},
  author = {Formal, Thibault and Piwowarski, Benjamin and Clinchant, St{\'e}phane},
  booktitle = {Proceedings of the 44th International ACM SIGIR}, pages = {2288--2292}, year = {2021}, eprint = {2107.05720}, archivePrefix = {arXiv}
}
@article{wang2022e5,
  title = {Text Embeddings by Weakly-Supervised Contrastive Pre-training},
  author = {Wang, Liang and Yang, Nan and Huang, Xiaolong and Jiao, Binxing and Yang, Linjun and Jiang, Daxin and Majumder, Rangan and Wei, Furu},
  journal = {arXiv preprint arXiv:2212.03533}, year = {2022}, eprint = {2212.03533}, archivePrefix = {arXiv}
}
@inproceedings{reimers2019sentencebert,
  title = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {Proceedings of the 2019 Conference on EMNLP-IJCNLP}, pages = {3982--3992}, year = {2019}, eprint = {1908.10084}, archivePrefix = {arXiv}
}
@article{johnson2019faiss,
  title = {Billion-Scale Similarity Search with GPUs},
  author = {Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal = {IEEE Transactions on Big Data}, volume = {7}, number = {3}, pages = {535--547}, year = {2021}, eprint = {1702.08734}, archivePrefix = {arXiv}
}
@inproceedings{guo2020scann,
  title = {Accelerating Large-Scale Inference with Anisotropic Vector Quantization},
  author = {Guo, Ruiqi and Sun, Philip and Lindgren, Erik and Geng, Quan and Simcha, David and Chern, Felix and Kumar, Sanjiv},
  booktitle = {Proceedings of the 37th ICML}, pages = {3887--3896}, year = {2020}, eprint = {1908.10396}, archivePrefix = {arXiv}
}
@article{malkov2020hnsw,
  title = {Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs},
  author = {Malkov, Yu. A. and Yashunin, D. A.},
  journal = {IEEE TPAMI}, volume = {42}, number = {4}, pages = {824--836}, year = {2020}, eprint = {1603.09320}, archivePrefix = {arXiv}
}
```
NOTES: DPR=canonical learned dual-encoder (contrast: we resolve deterministically, train ON structure not retrieve-at-inference). ColBERT/v2=late-interaction MaxSim (token-level MATCHING not attention; no gradient across boundary — vs our attention EDGE in one forward). Contriever=unsupervised contrastive (mid-point: no labels like us but learned proxy vs our true edge). GTR=scale retriever (we spend params on generator, no retriever to scale). ANCE=hard-negative mining from own ANN (we have no negative-sampling/recall ceiling). SPLADE=sparse learned lexical. E5=modern MTEB embedding default. SentenceBERT=foundational bi-encoder ancestor. FAISS/ScaNN/HNSW=ANN index infra RAG needs (we need none; O(1) hashmap, no approx error; HNSW is itself a graph-over-embeddings — juxtapose w/ our semantic doc graph).
