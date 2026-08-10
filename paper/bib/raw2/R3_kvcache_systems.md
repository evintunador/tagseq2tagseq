# R3 KV-cache reuse / serving systems (agent acb1280e) — 10 verified, dedup'd vs round1
CONTRAST: TAGSeq2TAGSeq does full O(T²) recompute every gen step (RoPE shift on insert makes paged KV reuse INCORRECT). These = the systems "what we deliberately do NOT do."

```bibtex
@inproceedings{kwon2023pagedattention,
  title = {Efficient Memory Management for Large Language Model Serving with {PagedAttention}},
  author = {Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E. and Zhang, Hao and Stoica, Ion}, booktitle = {Proceedings of the 29th SOSP}, year = {2023}, eprint = {2309.06180}, archivePrefix = {arXiv}
}
@inproceedings{zheng2024sglang,
  title = {{SGLang}: Efficient Execution of Structured Language Model Programs},
  author = {Zheng, Lianmin and Yin, Liangsheng and Xie, Zhiqiang and Sun, Chuyue and Huang, Jeff and Yu, Cody Hao and Cao, Shiyi and Kozyrakis, Christos and Stoica, Ion and Gonzalez, Joseph E. and Barrett, Clark and Sheng, Ying}, booktitle = {NeurIPS}, year = {2024}, eprint = {2312.07104}, archivePrefix = {arXiv}
}
@article{jin2024ragcache,
  title = {{RAGCache}: Efficient Knowledge Caching for Retrieval-Augmented Generation},
  author = {Jin, Chao and Zhang, Zili and Jiang, Xuanlin and Liu, Fangyue and Liu, Xin and Liu, Xuanzhe and Jin, Xin}, journal = {arXiv preprint arXiv:2404.12457}, year = {2024}
}
@inproceedings{ye2024chunkattention,
  title = {{ChunkAttention}: Efficient Self-Attention with Prefix-Aware {KV} Cache and Two-Phase Partition},
  author = {Ye, Lu and Tao, Ze and Huang, Yong and Li, Yang}, booktitle = {Proceedings of the 62nd ACL}, year = {2024}, eprint = {2402.15220}, archivePrefix = {arXiv}
}
@article{juravsky2024hydragen,
  title = {Hydragen: High-Throughput {LLM} Inference with Shared Prefixes},
  author = {Juravsky, Jordan and Brown, Bradley and Ehrlich, Ryan and Fu, Daniel Y. and R{\'e}, Christopher and Mirhoseini, Azalia}, journal = {arXiv preprint arXiv:2402.05099}, year = {2024}
}
@article{hu2024epic,
  title = {{EPIC}: Efficient Position-Independent Caching for Serving Large Language Models},
  author = {Hu, Junhao and Huang, Wenrui and Wang, Weidong and Wang, Haoyi and Hu, Tiancheng and Zhang, Qin and Feng, Hao and Chen, Xusheng and Shan, Yizhou and Xie, Tao}, journal = {arXiv preprint arXiv:2410.15332}, year = {2024}
}
@inproceedings{liu2024cachegen,
  title = {{CacheGen}: {KV} Cache Compression and Streaming for Fast Large Language Model Serving},
  author = {Liu, Yuhan and Li, Hanchen and Cheng, Yihua and Ray, Siddhant and Huang, Yuyang and Zhang, Qizheng and Du, Kuntai and Yao, Jiayi and Lu, Shan and Ananthanarayanan, Ganesh and Maire, Michael and Hoffmann, Henry and Holtzman, Ari and Jiang, Junchen}, booktitle = {Proceedings of ACM SIGCOMM 2024}, year = {2024}, eprint = {2310.07240}, archivePrefix = {arXiv}
}
@inproceedings{yang2025ape,
  title = {{APE}: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding},
  author = {Yang, Xinyu and Chen, Tianqi and Chen, Beidi}, booktitle = {ICLR}, year = {2025}, eprint = {2502.05431}, archivePrefix = {arXiv}
}
@inproceedings{yu2024pensieve,
  title = {Stateful Large Language Model Serving with {Pensieve}},
  author = {Yu, Lingfan and Lin, Jinkun and Li, Jinyang}, booktitle = {Proceedings of the 2024 ACM SoCC}, year = {2024}, eprint = {2312.05516}, archivePrefix = {arXiv}
}
@inproceedings{ye2025flashinfer,
  title = {{FlashInfer}: Efficient and Customizable Attention Engine for {LLM} Inference Serving},
  author = {Ye, Zihao and Chen, Lequn and Lai, Ruihang and Lin, Wuwei and Zhang, Yineng and Wang, Stephanie and Chen, Tianqi and Kasikci, Baris and Grover, Vinod and Krishnamurthy, Arvind and Ceze, Luis}, booktitle = {Proceedings of MLSys}, year = {2025}, eprint = {2501.01005}, archivePrefix = {arXiv}
}
```
NOTES: PagedAttention=canonical "what we do NOT do" (RoPE shift makes paged reuse incorrect). SGLang RadixAttention=exact-prefix KV reuse (strongest contrast). RAGCache=RAG retrieved-doc KV cache (we materialize+recompute for train/infer symmetry). ChunkAttention=trie prefix KV. Hydragen=shared-prefix batched. **EPIC=position-independent caching — CLOSEST analogue to our RoPE problem (cite: PIC's position-shift problem is exactly why naive KV cache incorrect in insertion-based retrieval)**. CacheGen=KV compression/transport (adjacent). APE=parallel-encode + align (approximates cross-chunk attn to enable reuse; we compute exact + full recompute; targets CAG). Pensieve=stateful multi-turn KV. FlashInfer=customizable attn engine (pair w/ A6, contrast to our bespoke BIM kernels). FLAGS: RAGCache/Hydragen/EPIC arXiv preprints.
