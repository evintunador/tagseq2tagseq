# Long-context / sparse attention / retrieval-into-attention (agent a4e8bc5e)

```bibtex
@misc{flexattention_blog2024,
  author = {Horace He and Driss Guessous and Yanbo Liang and Joy Dong},
  title  = {{FlexAttention}: The Flexibility of {PyTorch} with the Performance of {FlashAttention}},
  howpublished = {PyTorch Blog}, year = {2024}, month = aug,
  note = {\url{https://pytorch.org/blog/flexattention/}}
}
@article{flexattention_paper2024,
  author = {Juechu Dong and Boyuan Feng and Driss Guessous and Yanbo Liang and Horace He},
  title  = {Flex Attention: A Programming Model for Generating Optimized Attention Kernels},
  journal = {arXiv preprint arXiv:2412.05496}, year = {2024},
  eprint = {2412.05496}, archivePrefix = {arXiv}, primaryClass = {cs.LG}
}
@inproceedings{dao2022flashattention,
  author = {Tri Dao and Daniel Y. Fu and Stefano Ermon and Atri Rudra and Christopher R{\'e}},
  title  = {{FlashAttention}: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2022},
  eprint = {2205.14135}, archivePrefix = {arXiv}, primaryClass = {cs.LG}
}
@inproceedings{dao2023flashattention2,
  author = {Tri Dao}, title = {{FlashAttention-2}: Faster Attention with Better Parallelism and Work Partitioning},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2024},
  eprint = {2307.08691}, archivePrefix = {arXiv}, primaryClass = {cs.LG}
}
@article{child2019sparsetransformers,
  author = {Rewon Child and Scott Gray and Alec Radford and Ilya Sutskever},
  title  = {Generating Long Sequences with Sparse Transformers},
  journal = {arXiv preprint arXiv:1904.10509}, year = {2019},
  eprint = {1904.10509}, archivePrefix = {arXiv}, primaryClass = {cs.LG}
}
@article{beltagy2020longformer,
  author = {Iz Beltagy and Matthew E. Peters and Arman Cohan},
  title  = {{Longformer}: The Long-Document Transformer},
  journal = {arXiv preprint arXiv:2004.05150}, year = {2020},
  eprint = {2004.05150}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{zaheer2020bigbird,
  author = {Manzil Zaheer and Guru Guruganesh and Avinava Dubey and Joshua Ainslie and Chris Alberti and Santiago Onta{\~n}{\'o}n and Philip Pham and Anirudh Ravula and Qifan Wang and Li Yang and Amr Ahmed},
  title  = {{Big Bird}: Transformers for Longer Sequences},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2020},
  eprint = {2007.14062}, archivePrefix = {arXiv}, primaryClass = {cs.LG}
}
@inproceedings{mohtashami2023landmark,
  author = {Amirkeivan Mohtashami and Martin Jaggi},
  title  = {Landmark Attention: Random-Access Infinite Context Length for Transformers},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2023},
  eprint = {2305.16300}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@article{ainslie2023colt5,
  author = {Joshua Ainslie and Tao Lei and Michiel de Jong and Santiago Onta{\~n}{\'o}n and Siddhartha Brahma and Yury Zemlyanskiy and David Uthus and Mandy Guo and James Lee-Thorp and Yi Tay and Yun-Hsuan Sung and Sumit Sanghai},
  title  = {{CoLT5}: Faster Long-Range Transformers with Conditional Computation},
  journal = {arXiv preprint arXiv:2303.09752}, year = {2023},
  eprint = {2303.09752}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@article{liu2023ringattention,
  author = {Hao Liu and Matei Zaharia and Pieter Abbeel},
  title  = {Ring Attention with Blockwise Transformers for Near-Infinite Context},
  journal = {arXiv preprint arXiv:2310.01889}, year = {2023},
  eprint = {2310.01889}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{wu2022memorizing,
  author = {Yuhuai Wu and Markus N. Rabe and DeLesley Hutchins and Christian Szegedy},
  title  = {Memorizing Transformers},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2022},
  eprint = {2203.08913}, archivePrefix = {arXiv}, primaryClass = {cs.LG}
}
@inproceedings{borgeaud2022retro,
  author = {Sebastian Borgeaud and Arthur Mensch and Jordan Hoffmann and Trevor Cai and Eliza Rutherford and Katie Millican and George van den Driessche and Jean-Baptiste Lespiau and Bogdan Damoc and Aidan Clark and Diego de Las Casas and Aurelia Guy and Jacob Menick and Roman Ring and Tom Hennigan and Saffron Huang and Loren Maggiore and Chris Jones and Albin Cassirer and Andy Brock and Michela Paganini and Geoffrey Irving and Oriol Vinyals and Simon Osindero and Karen Simonyan and Jack W. Rae and Erich Elsen and Laurent Sifre},
  title  = {Improving Language Models by Retrieving from Trillions of Tokens},
  booktitle = {International Conference on Machine Learning (ICML)}, year = {2022},
  eprint = {2112.04426}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{bertsch2023unlimiformer,
  author = {Amanda Bertsch and Uri Alon and Graham Neubig and Matthew R. Gormley},
  title  = {{Unlimiformer}: Long-Range Transformers with Unlimited Length Input},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2023},
  eprint = {2305.01625}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{tworkowski2023focused,
  author = {Szymon Tworkowski and Konrad Staniszewski and Miko{\l}aj Pacek and Yuhuai Wu and Henryk Michalewski and Piotr Mi{\l}o{\'s}},
  title  = {Focused Transformer: Contrastive Training for Context Scaling},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2023},
  eprint = {2307.03170}, archivePrefix = {arXiv}, primaryClass = {cs.CL}
}
@inproceedings{ying2021graphormer,
  author = {Chengxuan Ying and Tianle Cai and Shengjie Luo and Shuxin Zheng and Guolin Ke and Di He and Yanming Shen and Tie-Yan Liu},
  title  = {Do Transformers Really Perform Bad for Graph Representation?},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2021},
  eprint = {2106.05234}, archivePrefix = {arXiv}, primaryClass = {cs.LG}
}
```

NOTES:
- flexattention_blog2024 / _paper2024: THE API we build on. mask_mod predicate compiled to BlockMask skipping fully-masked KV blocks. Blog shows doc-masking (block-diagonal); we generalize to block-SPARSE-GRAPH mask lighting off-diagonal blocks along link edges. FLAG blog byline unverified.
- dao flashattention 1/2: kernel foundation; FlexAttention = FlashAttn + fused mask_mod. Makes 32k sparse training tractable.
- child2019 / beltagy2020 / zaheer2020: fixed-pattern sparse attention. Ours is DATA-defined by link graph, not geometric heuristic.
- mohtashami2023 landmark: learned block retrieval at inference, single sequence — not pretraining over external doc graph.
- ainslie2023colt5: conditional computation axis (we don't use).
- liu2023ringattention: distributed blockwise; complementary to our mask.
- wu2022memorizing: ARCHETYPAL retrieval-into-attention CONTRAST — non-differentiable frozen KV memory, kNN, inference augmentation. We train cross-doc attn as differentiable pretraining objective along known edges.
- borgeaud2022retro (RETRO, DeepMind): frozen datastore, embedding-sim neighbors, chunked cross-attn. We pack actual linked docs in-sequence, learned end-to-end.
- bertsch2023unlimiformer: inference kNN over encoder states; encoder-decoder.
- tworkowski2023focused (LongLLaMA): trains to improve memory attention but context still external kNN memory at inference.
- ying2021graphormer: attention over graph via shortest-path/edge BIAS terms (soft, dense, small graphs); we use HARD block-sparse mask over long token seqs.
