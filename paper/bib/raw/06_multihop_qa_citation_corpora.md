# Multi-hop QA benchmarks + citation/wiki graph corpora (agent adf14453) — verified vs ACL Anthology/DBLP

Benchmarks: HotpotQA = HEADLINE source (2-hop, sentence-level supporting facts → build cross-doc setting). Plus WikiHop/MedHop, 2WikiMultiHopQA, MuSiQue, HoVer, StrategyQA, Bamboogle.
Graph readers (prior graph-multihop, inference-time): Cognitive Graph, HDE, DFGN.
Single-doc CONTROLS: HellaSwag, ARC, PIQA, WinoGrande, OpenBookQA, BoolQ.
Corpora: unarXive 2020 + 2022 (arXiv \cite citation graph — our training corpus), S2ORC, OGB (ogbn-arxiv), WikiLinkGraphs (wiki hyperlink graph), WikiText.
Citation-embedding LMs: SciBERT, SPECTER, SciNCL.

```bibtex
@inproceedings{yang-etal-2018-hotpotqa,
  title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author = {Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)}, year = {2018}, address = {Brussels, Belgium}, publisher = {ACL},
  url = {https://aclanthology.org/D18-1259/}, doi = {10.18653/v1/D18-1259}, pages = {2369--2380}
}
@article{welbl-etal-2018-constructing,
  title = {Constructing Datasets for Multi-hop Reading Comprehension Across Documents},
  author = {Welbl, Johannes and Stenetorp, Pontus and Riedel, Sebastian},
  journal = {Transactions of the Association for Computational Linguistics}, volume = {6}, year = {2018}, publisher = {MIT Press},
  url = {https://aclanthology.org/Q18-1021/}, doi = {10.1162/tacl_a_00021}, pages = {287--302}
}
@inproceedings{ho-etal-2020-constructing,
  title = {Constructing A Multi-hop {QA} Dataset for Comprehensive Evaluation of Reasoning Steps},
  author = {Ho, Xanh and Duong Nguyen, Anh-Khoa and Sugawara, Saku and Aizawa, Akiko},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics (COLING)}, year = {2020}, publisher = {ICCL},
  url = {https://aclanthology.org/2020.coling-main.580/}, doi = {10.18653/v1/2020.coling-main.580}, pages = {6609--6625}
}
@article{trivedi-etal-2022-musique,
  title = {{M}u{S}i{Q}ue: Multihop Questions via Single-hop Question Composition},
  author = {Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  journal = {Transactions of the Association for Computational Linguistics}, volume = {10}, year = {2022}, publisher = {MIT Press},
  url = {https://aclanthology.org/2022.tacl-1.31/}, doi = {10.1162/tacl_a_00475}, pages = {539--554}
}
@inproceedings{jiang-etal-2020-hover,
  title = {{H}o{V}er: A Dataset for Many-Hop Fact Extraction And Claim Verification},
  author = {Jiang, Yichen and Bordia, Shikha and Zhong, Zheng and Dognin, Charles and Singh, Maneesh and Bansal, Mohit},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020}, year = {2020}, publisher = {ACL},
  url = {https://aclanthology.org/2020.findings-emnlp.309/}, doi = {10.18653/v1/2020.findings-emnlp.309}, pages = {3441--3460}
}
@article{geva-etal-2021-aristotle,
  title = {Did {A}ristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies},
  author = {Geva, Mor and Khashabi, Daniel and Segal, Elad and Khot, Tushar and Roth, Dan and Berant, Jonathan},
  journal = {Transactions of the Association for Computational Linguistics}, volume = {9}, year = {2021}, publisher = {MIT Press},
  url = {https://aclanthology.org/2021.tacl-1.21/}, doi = {10.1162/tacl_a_00370}, pages = {346--361}
}
@inproceedings{press-etal-2023-measuring,
  title = {Measuring and Narrowing the Compositionality Gap in Language Models},
  author = {Press, Ofir and Zhang, Muru and Min, Sewon and Schmidt, Ludwig and Smith, Noah A. and Lewis, Mike},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023}, year = {2023}, publisher = {ACL},
  url = {https://aclanthology.org/2023.findings-emnlp.378/}, doi = {10.18653/v1/2023.findings-emnlp.378}, pages = {5687--5711}
}
@inproceedings{ding-etal-2019-cognitive,
  title = {Cognitive Graph for Multi-Hop Reading Comprehension at Scale},
  author = {Ding, Ming and Zhou, Chang and Chen, Qibin and Yang, Hongxia and Tang, Jie},
  booktitle = {Proceedings of the 57th Annual Meeting of the ACL}, year = {2019}, publisher = {ACL},
  url = {https://aclanthology.org/P19-1259/}, doi = {10.18653/v1/P19-1259}, pages = {2694--2703}
}
@inproceedings{tu-etal-2019-multi,
  title = {Multi-hop Reading Comprehension across Multiple Documents by Reasoning over Heterogeneous Graphs},
  author = {Tu, Ming and Wang, Guangtao and Huang, Jing and Tang, Yun and He, Xiaodong and Zhou, Bowen},
  booktitle = {Proceedings of the 57th Annual Meeting of the ACL}, year = {2019}, publisher = {ACL},
  url = {https://aclanthology.org/P19-1260/}, doi = {10.18653/v1/P19-1260}, pages = {2704--2713}
}
@inproceedings{qiu-etal-2019-dynamically,
  title = {Dynamically Fused Graph Network for Multi-hop Reasoning},
  author = {Qiu, Lin and Xiao, Yunxuan and Qu, Yanru and Zhou, Hao and Li, Lei and Zhang, Weinan and Yu, Yong},
  booktitle = {Proceedings of the 57th Annual Meeting of the ACL}, year = {2019}, publisher = {ACL},
  url = {https://aclanthology.org/P19-1617/}, doi = {10.18653/v1/P19-1617}, pages = {6140--6150}
}
@inproceedings{zellers-etal-2019-hellaswag,
  title = {{H}ella{S}wag: Can a Machine Really Finish Your Sentence?},
  author = {Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
  booktitle = {Proceedings of the 57th Annual Meeting of the ACL}, year = {2019}, publisher = {ACL},
  url = {https://aclanthology.org/P19-1472/}, doi = {10.18653/v1/P19-1472}, pages = {4791--4800}
}
@article{clark2018arc,
  title = {Think you have Solved Question Answering? Try {ARC}, the {AI2} Reasoning Challenge},
  author = {Clark, Peter and Cowhey, Isaac and Etzioni, Oren and Khot, Tushar and Sabharwal, Ashish and Schoenick, Carissa and Tafjord, Oyvind},
  journal = {arXiv preprint arXiv:1803.05457}, year = {2018}
}
@inproceedings{bisk2020piqa,
  title = {{PIQA}: Reasoning about Physical Commonsense in Natural Language},
  author = {Bisk, Yonatan and Zellers, Rowan and Le Bras, Ronan and Gao, Jianfeng and Choi, Yejin},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence}, volume = {34}, number = {05}, pages = {7432--7439}, year = {2020}
}
@inproceedings{sakaguchi2020winogrande,
  title = {{W}ino{G}rande: An Adversarial {W}inograd Schema Challenge at Scale},
  author = {Sakaguchi, Keisuke and Le Bras, Ronan and Bhagavatula, Chandra and Choi, Yejin},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence}, volume = {34}, number = {05}, pages = {8732--8740}, year = {2020}
}
@inproceedings{mihaylov-etal-2018-suit,
  title = {Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering},
  author = {Mihaylov, Todor and Clark, Peter and Khot, Tushar and Sabharwal, Ashish},
  booktitle = {Proceedings of the 2018 Conference on EMNLP}, year = {2018}, publisher = {ACL},
  url = {https://aclanthology.org/D18-1260/}, doi = {10.18653/v1/D18-1260}, pages = {2381--2391}
}
@inproceedings{clark-etal-2019-boolq,
  title = {{B}ool{Q}: Exploring the Surprising Difficulty of Natural Yes/No Questions},
  author = {Clark, Christopher and Lee, Kenton and Chang, Ming-Wei and Kwiatkowski, Tom and Collins, Michael and Toutanova, Kristina},
  booktitle = {Proceedings of NAACL-HLT}, year = {2019}, publisher = {ACL},
  url = {https://aclanthology.org/N19-1300/}, doi = {10.18653/v1/N19-1300}, pages = {2924--2936}
}
@article{saier2020unarxive,
  title = {un{a}r{X}ive: a large scholarly data set with publications' full-text, annotated in-text citations, and links to metadata},
  author = {Saier, Tarek and F{\"a}rber, Michael},
  journal = {Scientometrics}, volume = {125}, number = {3}, pages = {3085--3108}, year = {2020}, publisher = {Springer}, doi = {10.1007/s11192-020-03382-z}
}
@inproceedings{saier2023unarxive,
  title = {un{a}r{X}ive 2022: All ar{X}iv Publications Pre-Processed for {NLP}, Including Structured Full-Text and Citation Network},
  author = {Saier, Tarek and Krause, Johan and F{\"a}rber, Michael},
  booktitle = {2023 ACM/IEEE Joint Conference on Digital Libraries (JCDL)}, pages = {66--70}, year = {2023}, doi = {10.1109/JCDL57899.2023.00020}, note = {arXiv:2303.14957}
}
@inproceedings{lo-etal-2020-s2orc,
  title = {{S}2{ORC}: The Semantic Scholar Open Research Corpus},
  author = {Lo, Kyle and Wang, Lucy Lu and Neumann, Mark and Kinney, Rodney and Weld, Daniel},
  booktitle = {Proceedings of the 58th Annual Meeting of the ACL}, year = {2020}, publisher = {ACL},
  url = {https://aclanthology.org/2020.acl-main.447/}, doi = {10.18653/v1/2020.acl-main.447}, pages = {4969--4983}
}
@inproceedings{hu2020ogb,
  title = {Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author = {Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, volume = {33}, pages = {22118--22133}, year = {2020}
}
@inproceedings{consonni2019wikilinkgraphs,
  title = {{W}iki{L}ink{G}raphs: A Complete, Longitudinal and Multi-Language Dataset of the {W}ikipedia Link Networks},
  author = {Consonni, Cristian and Laniado, David and Montresor, Alberto},
  booktitle = {Proceedings of the International AAAI Conference on Web and Social Media (ICWSM)}, volume = {13}, pages = {598--607}, year = {2019}
}
@inproceedings{merity2017pointer,
  title = {Pointer Sentinel Mixture Models},
  author = {Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
  booktitle = {5th International Conference on Learning Representations (ICLR)}, year = {2017}
}
@inproceedings{beltagy-etal-2019-scibert,
  title = {{S}ci{BERT}: A Pretrained Language Model for Scientific Text},
  author = {Beltagy, Iz and Lo, Kyle and Cohan, Arman},
  booktitle = {Proceedings of the 2019 Conference on EMNLP-IJCNLP}, year = {2019}, publisher = {ACL},
  url = {https://aclanthology.org/D19-1371/}, doi = {10.18653/v1/D19-1371}, pages = {3615--3620}
}
@inproceedings{cohan-etal-2020-specter,
  title = {{SPECTER}: Document-level Representation Learning using Citation-informed Transformers},
  author = {Cohan, Arman and Feldman, Sergey and Beltagy, Iz and Downey, Doug and Weld, Daniel},
  booktitle = {Proceedings of the 58th Annual Meeting of the ACL}, year = {2020}, publisher = {ACL},
  url = {https://aclanthology.org/2020.acl-main.207/}, doi = {10.18653/v1/2020.acl-main.207}, pages = {2270--2282}
}
@inproceedings{ostendorff-etal-2022-neighborhood,
  title = {Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings},
  author = {Ostendorff, Malte and Rethmeier, Nils and Augenstein, Isabelle and Gipp, Bela and Rehm, Georg},
  booktitle = {Proceedings of the 2022 Conference on EMNLP}, year = {2022}, publisher = {ACL},
  url = {https://aclanthology.org/2022.emnlp-main.802/}, doi = {10.18653/v1/2022.emnlp-main.802}, pages = {11670--11688}
}
```

NOTES:
- HotpotQA: HEADLINE. 2-hop over 2 wiki paragraphs + sentence-level supporting facts → construct cross-doc setting (answer para + linked supporting doc). Test cross-doc attention over graph edge vs flat concat.
- welbl (WikiHop/MedHop): canonical "reasoning across linked documents" origin.
- ho (2WikiMultiHopQA): harder, evidence-path annotated, mitigates HotpotQA shortcuts.
- trivedi (MuSiQue): 2-4 hop by composition; stress-test >2 hops.
- jiang (HoVer): many-hop claim verification (extends QA→verification).
- geva (StrategyQA): implicit reasoning control. press (Bamboogle): clean 2-hop compositionality probe.
- ding/tu/qiu (Cognitive Graph/HDE/DFGN): prior graph-multihop READERS at inference; we replace bolt-on GNN reader with native attention across TAG edges in decoder-only LM.
- CONTROLS (single-doc, should be unaffected): zellers HellaSwag, clark2018 ARC, bisk PIQA, sakaguchi WinoGrande, mihaylov OpenBookQA, clark2019 BoolQ.
- saier2020/2023 unarXive: arXiv \cite citation graph = our training corpus (nodes=papers, edges=citations). Cite whichever release ingested (likely 2022/2023).
- lo S2ORC: larger citation-graph alt. hu OGB (ogbn-arxiv): canonical TAG citation graph, GNN comparability.
- consonni WikiLinkGraphs: wiki hyperlink graph underlying our wiki corpus. merity WikiText: conventional wiki LM benchmark.
- beltagy SciBERT (sci text, ignores citation graph), cohan SPECTER + ostendorff SciNCL (use citation edges for DOC EMBEDDINGS; we inject link edges into generative LM via cross-doc attention).

FLAGS: SimpleWiki has NO canonical paper (cite as dataset/URL). PIQA/WinoGrande/OGB/WikiLinkGraphs/WikiText proceedings pages filled from standard refs, reverify. ARC=AI2 techreport (no peer venue). WinoGrande authoritative venue AAAI2020.
NOTE overlap: consonni2019wikilinkgraphs ALSO in slice 02 (graph-aware) — dedup on merge.
