# TAGSeq2TAGSeq

Trains language models on graph-structured text data. Documents are nodes in a graph; edges are hyperlinks (Wikipedia) or import dependencies (Python code). The data pipeline traverses this graph to build packed token sequences, and a custom FlexAttention mask enforces document-level causal isolation and optional cross-document link awareness.

---

## Datasets

Two datasets are supported out of the box:

| Dataset | Source | Graph edges | Pretokenized location |
|---------|--------|-------------|----------------------|
| **SimpleWiki** | `simplewiki-20251027-cirrussearch-content.json.gz` | Markdown hyperlinks | `data/pretokenized_datasets/simplewiki/` |
| **The Stack (10M)** | `bigcode/the-stack-dedup` (Python, 10M files) | Python import statements | `data/pretokenized_datasets/stack_10m/` |

---

## Full Pipeline

### Wikipedia (SimpleWiki)

**1. Extract dump → markdown articles**

The raw dump is at `/fss/evin_t/wiki_dumps/simplewiki-20251027-cirrussearch-content.json.gz`.

```bash
python -m data.wiki_graph_extractor.dump_extractor \
    /fss/evin_t/wiki_dumps/simplewiki-20251027-cirrussearch-content.json.gz \
    -o data/wiki_articles \
    -p 60
```

Produces ~275,000 `.md` files in `data/wiki_articles/` organised into per-letter subdirectories.

**2. Build link graph**

```bash
python -m data.wiki_graph_extractor.build_graph \
    data/wiki_articles \
    -o data/wiki_articles/graph.jsonl \
    -p 60
```

Produces `data/wiki_articles/graph.jsonl` (~275k nodes, ~2.3M edges) plus `graph_stats.json` and `graph_degree_dist.png`.

**3. Pretokenize**

```bash
python -m data.pretokenize \
    data/wiki_articles \
    data/wiki_articles/graph.jsonl \
    -o data/pretokenized_datasets/simplewiki \
    -p 60
```

Produces `shard_000000.bin` (~108M tokens), `tokenized_graph.jsonl`, and `metadata.json` in `data/pretokenized_datasets/simplewiki/`.

---

### The Stack (10M Python files)

**1. Download 10M samples**

Requires a HuggingFace token with read access to `bigcode/the-stack-dedup`.

```bash
HF_TOKEN=<your_token> python data/github_graph_extractor/download_sample.py \
    --limit 10000000 \
    -o data/github_graph_extractor/sample_10M.jsonl
```

Streams ~56 GB of Python source files.

**2. Build import dependency graph**

Must be run from inside `data/github_graph_extractor/` (uses relative imports):

```bash
cd data/github_graph_extractor
python build_graph_streaming.py \
    sample_10M.jsonl \
    -o graph_10M.jsonl \
    -p 8 \
    --bucket-workers 8
cd -
```

Produces `graph_10M.jsonl` (~2.4M nodes) plus statistics and a degree-distribution plot.

**3. Pretokenize**

```bash
python -m data.pretokenize_stack \
    data/github_graph_extractor/sample_10M.jsonl \
    data/github_graph_extractor/graph_10M.jsonl \
    -o data/pretokenized_datasets/stack_10m \
    -p 60
```

Produces 6 binary shards, `tokenized_graph.jsonl`, and `metadata.json` in `data/pretokenized_datasets/stack_10m/`.

---

## Visualisation

### Inspect packed batches (text)

```bash
python demo_traversal.py <dataset_dir> --strategy dfs --token-budget 2048
```

Prints a packed-batch summary: doc spans, graph connectivity within the batch, and decoded text snippets.

### Attention mask images

`block_mask_creator.py` renders the FlexAttention mask for a real batch and saves a PNG to `artifacts/`. Output filenames include the dataset name to avoid collisions.

```bash
# doc_causal mask — Wikipedia
python block_mask_creator.py data/pretokenized_datasets/simplewiki \
    --mask-type doc_causal --strategy dfs --seed 42

# cross-document link mask — Wikipedia (uses markdown link detector)
python block_mask_creator.py data/pretokenized_datasets/simplewiki \
    --mask-type cross_doc_link --link-detector markdown --strategy dfs --seed 42

# doc_causal mask — The Stack
python block_mask_creator.py data/pretokenized_datasets/stack_10m \
    --mask-type doc_causal --strategy dfs --seed 42

# cross-document link mask — The Stack (uses Python import detector)
python block_mask_creator.py data/pretokenized_datasets/stack_10m \
    --mask-type cross_doc_link --link-detector python --strategy dfs --seed 42
```

Output images (in `artifacts/`):

```
mask_viz_simplewiki_doc_causal_seed42.png
mask_viz_simplewiki_cross_doc_link_seed42.png
mask_viz_stack_10m_doc_causal_seed42.png
mask_viz_stack_10m_cross_doc_link_seed42.png
```

Available mask types: `doc_causal`, `causal`, `full`, `doc_bidirectional`, `cross_doc_link`.
Available strategies: `dfs`, `bfs`, `random_walk`, `random`.

---

## Training

```bash
python main.py \
    --config configs/baseline.yaml \
    --dataset-dir <dataset_dir> \
    --strategy dfs \
    --model.compile false
```

> **Note:** `--model.compile false` is required — `torch.compile` currently conflicts with FlexAttention's `create_block_mask`. Run artifacts are saved to timestamped directories under `runs/`.

**Wikipedia:**
```bash
python main.py --config configs/baseline.yaml \
    --dataset-dir data/pretokenized_datasets/simplewiki \
    --strategy dfs --model.compile false
```

**The Stack:**
```bash
python main.py --config configs/baseline.yaml \
    --dataset-dir data/pretokenized_datasets/stack_10m \
    --strategy dfs --model.compile false
```

Key config options (`configs/baseline.yaml`):

| Key | Default | Description |
|-----|---------|-------------|
| `model.model_dim` | 768 | Hidden dimension |
| `model.num_layers` | 12 | Transformer layers |
| `model.max_seq_len` | 2048 | Token budget per batch |
| `model.mask_type` | `doc_causal` | Attention mask strategy |
| `data.strategy` | `random` | Graph traversal strategy |
| `optimizer.muon_lr` | 0.02 | LR for 2D backbone weights (Muon) |
| `optimizer.adamw_lr` | 0.0003 | LR for embeddings/norms (AdamW) |
| `train_loop.val_interval` | 50 | Steps between validation passes |
