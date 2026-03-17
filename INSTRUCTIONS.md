# Pipeline Instructions

Full instructions for data preparation, training, and generation. See [README.md](README.md) for a conceptual overview.

---

## Available Checkpoints

> This table is a temporarily maintained personal reference — it tracks runs on the author's cluster and will go stale as training progresses. It lives here rather than in the README because it's operational bookkeeping, not project documentation.

| Checkpoint | Architecture | Dataset | Context | Mask | Steps | Val loss |
|-----------|-------------|---------|---------|------|-------|----------|
| `runs/20260224_212158/checkpoints/best_model.pt` | 12L / 768D | SimpleWiki | 2k | `doc_causal` | 38,500 | 2.07 |
| `runs/20260308_012514/checkpoints/best_model.pt` | 36L / 1280D | SimpleWiki | 32k | `doc_causal` | 10,200 | 3.923 |
| `runs/20260308_012516/checkpoints/best_model.pt` | 36L / 1280D | SimpleWiki | 32k | `cross_doc_link` (md) | 12,200 | 3.905 |
| `runs/20260308_012518/checkpoints/best_model.pt` | 36L / 1280D | Stack 10M | 32k | `doc_causal` | 14,700 | 2.271 |
| `runs/20260308_012521/checkpoints/best_model.pt` | 36L / 1280D | Stack 10M | 32k | `cross_doc_link` (py) | 14,900 | 2.291 |
| `runs/run_20260311_184203_685319/checkpoints/best_model.pt` | 24L / 1024D | Stack 100M | 32k | `cross_doc_link` (py) | 3,000 | 1.430 |

> The Stack 100M checkpoint is partially trained (~16% of planned steps), currently being continued in a LR-cooldown run.

---

## Datasets

Four pretokenized datasets are ready to use on `/fss`:

| Dataset | Graph edges | Nodes | Tokens | Shards | Pretokenized location |
|---------|-------------|-------|--------|--------|-----------------------|
| **SimpleWiki** | Markdown hyperlinks | 275k | ~108M | 1 | `data/pretokenized_datasets/simplewiki/` |
| **EnWikiSource** | Markdown hyperlinks | 662k | ~612M | 1 | `data/pretokenized_datasets/enwikisource/` |
| **The Stack (10M)** | Python imports | 2.38M | ~7B | 6 | `data/pretokenized_datasets/stack_10m/` |
| **The Stack (100M)** | Python imports | 3.56M | ~8.7B | 9 | `data/pretokenized_datasets/stack_100m/` |

SimpleWiki and EnWikiSource use `--model.link_detector markdown`; both Stack datasets use `--model.link_detector python`.

Raw dumps and JSONL source files:

| Dataset | Raw source |
|---------|-----------|
| SimpleWiki | `/fss/evin_t/wiki_dumps/simplewiki-20251027-cirrussearch-content.json.gz` |
| EnWikiSource | `/fss/evin_t/wiki_dumps/enwikisource-20251027-cirrussearch-content.json.gz` |
| The Stack (10M) | `data/github_graph_extractor/sample_10M.jsonl` + `graph_10M.jsonl` |
| The Stack (100M) | `data/github_graph_extractor/sample_100M.jsonl` + `graph_100M.jsonl` |

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

The `--layout-policy` flag controls per-document token decoration:

| Value | Behaviour |
|-------|-----------|
| `null` (default) | No decoration — raw body tokens only |
| `bos-eos` | Wrap each document body with BOS/EOS tokens |
| `identifier-prefix` | Prepend `# {raw_identifier}\n\n` before each body |

```bash
# Default (no decoration)
python demo_traversal.py data/pretokenized_datasets/simplewiki --strategy dfs

# With identifier prefix (e.g. "# Water\n\n..." before each article)
python demo_traversal.py data/pretokenized_datasets/simplewiki \
    --strategy dfs --layout-policy identifier-prefix

# The Stack with identifier prefix (e.g. "# repo:src/file.py\n\n...")
python demo_traversal.py data/pretokenized_datasets/stack_10m \
    --strategy dfs --layout-policy identifier-prefix
```

### Attention mask images

`model/graph_traversal/block_mask_creator.py` renders the FlexAttention mask for a real batch and saves a PNG to `artifacts/`. Run as a module from the project root:

```bash
# doc_causal mask — Wikipedia
python -m model.graph_traversal.block_mask_creator data/pretokenized_datasets/simplewiki \
    --mask-type doc_causal --strategy bfs --seed 42

# cross-document link mask — Wikipedia (markdown link detector)
python -m model.graph_traversal.block_mask_creator data/pretokenized_datasets/simplewiki \
    --mask-type cross_doc_link --link-detector markdown --strategy bfs --seed 42

# doc_causal mask — The Stack
python -m model.graph_traversal.block_mask_creator data/pretokenized_datasets/stack_10m \
    --mask-type doc_causal --strategy bfs --seed 42

# cross-document link mask — The Stack (Python import detector)
python -m model.graph_traversal.block_mask_creator data/pretokenized_datasets/stack_10m \
    --mask-type cross_doc_link --link-detector python --strategy bfs --seed 42
```

Available mask types: `doc_causal`, `causal`, `full`, `doc_bidirectional`, `cross_doc_link`.
Available strategies: `dfs`, `bfs`, `random_walk`, `random`.
`--link-detector` is only used with `cross_doc_link`: `markdown` for Wikipedia, `python` for TheStack.

---

## Training

Run artifacts are saved to timestamped directories under `runs/`.

### Baseline runs (doc_causal, random traversal)

The baseline uses document-causal masking (each document attends only to itself) with random graph traversal. `data.strategy` defaults to `random` in `baseline.yaml` so no override is needed.

```bash
# SimpleWiki (~108M tokens, fast iteration)
python main.py --config configs/baseline.yaml \
    --dataset-dir data/pretokenized_datasets/simplewiki

# EnWikiSource (~612M tokens, longer literary texts)
python main.py --config configs/baseline.yaml \
    --dataset-dir data/pretokenized_datasets/enwikisource

# The Stack 10M (~7B tokens)
python main.py --config configs/baseline.yaml \
    --dataset-dir data/pretokenized_datasets/stack_10m

# The Stack 100M (~8.7B tokens, full Python corpus)
python main.py --config configs/baseline.yaml \
    --dataset-dir data/pretokenized_datasets/stack_100m
```

### Cross-document runs (cross_doc_link, BFS traversal)

BFS traversal places linked documents adjacently in the packed sequence, which is required for cross-doc attention to be meaningful. Set `model.link_detector` to match the dataset.

**Wikipedia / WikiSource** (`--model.link_detector markdown`):
```bash
python main.py --config configs/baseline.yaml \
    --dataset-dir data/pretokenized_datasets/enwikisource \
    --strategy bfs \
    --model.mask_type cross_doc_link \
    --model.link_detector markdown
```

**The Stack** (`--model.link_detector python`):
```bash
python main.py --config configs/baseline.yaml \
    --dataset-dir data/pretokenized_datasets/stack_100m \
    --strategy bfs \
    --model.mask_type cross_doc_link \
    --model.link_detector python
```

### Key config options (`configs/baseline.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `model.model_dim` | 768 | Hidden dimension |
| `model.num_layers` | 12 | Transformer layers |
| `model.max_seq_len` | 2048 | Token budget per batch |
| `model.mask_type` | `doc_causal` | Attention mask strategy |
| `model.link_detector` | *(unset)* | Required when `mask_type` is `cross_doc_link`: `markdown` or `python` |
| `data.strategy` | `random` | Graph traversal strategy |
| `optimizer.muon_lr` | 0.02 | LR for 2D backbone weights (Muon) |
| `optimizer.adamw_lr` | 0.0003 | LR for embeddings/norms (AdamW) |
| `train_loop.val_interval` | 50 | Steps between validation passes |

For larger models and longer contexts use `configs/large_32k.yaml` (36L/1280D, 32k context,
fitted for a single A100 80GB with `torch.compile`).

### Multi-node SLURM training via `launch_slurm.py`

Use `launch_slurm.py` instead of `main.py` for multi-node runs. It wraps submitit and handles
distributed process setup automatically. Config overrides use **dotted-key notation** — the argparse
shorthand flags (`--dataset-dir`, `--strategy`, etc.) are not defined in the launcher; pass
everything as `--section.key value` so the YAML config is never silently overridden.

```bash
# 2 nodes × 8 GPUs — Stack 100M (the canonical large run)
python launch_slurm.py \
    --nodes 2 --gpus-per-node 8 --time 48:00:00 \
    --config configs/stack_100m_32k.yaml \
    --data.dataset_dir data/pretokenized_datasets/stack_100m

# 1 node × 4 GPUs — quick iteration on EnWikiSource
python launch_slurm.py \
    --nodes 1 --gpus-per-node 4 --time 12:00:00 \
    --config configs/large_32k.yaml \
    --data.dataset_dir data/pretokenized_datasets/enwikisource \
    --model.mask_type cross_doc_link --model.link_detector markdown
```

`--no-tail` suppresses log following after submission. Logs land in the run's `logs/` subdirectory.

---

## Generation

After training, generate text from a checkpoint using `generate.py`. The script auto-reads
`hyperparameters.json` from the run directory to reconstruct the architecture, tokenizer,
link detector, and layout policy — no manual config needed.

### Stack 100M model (24L/1024D, cross-doc, 32k context)

This checkpoint was trained with `identifier_prefix_bos_eos` layout policy, so every document
seen during training began with `<BOS># path/to/file.py\n\n`. Pass `--root-identifier` with a
plausible filename or the model will see an empty `# \n\n` header and immediately generate EOS.

```bash
python generate.py \
    --checkpoint runs/run_20260311_184203_685319/checkpoints/best_model.pt \
    --dataset data/pretokenized_datasets/stack_100m \
    --root-identifier "trainer.py" \
    --prompt "import torch
from torch.optim import Adam
from model import ResNet

def train_epoch(model, loader, optimizer, criterion, device):" \
    --max-new-tokens 500 \
    --max-link-depth 2 \
    --repetition-penalty 1.1 \
    --temperature 0.9
```

**Repetition penalty tuning for this checkpoint:** at ~3,000 steps the model is partially trained
and prone to repetition loops. The default `--repetition-penalty 1.3` is too aggressive for code
(it penalises legitimate re-use of variable names like `d_model` or `optimizer`), causing premature
EOS. Use `1.05–1.15` for code generation. `1.0` disables the penalty entirely but risks infinite
repetition loops.

### Stack 10M models (36L/1280D, 32k context)

```bash
# doc_causal variant
python generate.py \
    --checkpoint runs/20260308_012518/checkpoints/best_model.pt \
    --root-identifier "sort.py" \
    --prompt "def merge_sort(arr):" \
    --max-link-depth 0 \
    --max-new-tokens 400

# cross_doc_link variant (imports resolved against corpus)
python generate.py \
    --checkpoint runs/20260308_012521/checkpoints/best_model.pt \
    --dataset data/pretokenized_datasets/stack_10m \
    --root-identifier "main.py" \
    --prompt "import numpy as np

def softmax(x):" \
    --max-link-depth 2 \
    --max-new-tokens 400
```

### SimpleWiki baseline (12L/768D, doc_causal, 2k context)

```bash
python generate.py \
    --checkpoint runs/20260224_212158/checkpoints/best_model.pt \
    --prompt "Python is a high-level programming language." \
    --dataset data/pretokenized_datasets/simplewiki \
    --max-link-depth 2 \
    --max-new-tokens 300
```

With `--dataset` provided, links detected in generated text are looked up in the corpus. Matched
documents are inserted into the attention context before the active document. Set
`--allow-generation-fallback` to also generate aux docs for links not found in the corpus.

### Key generation options

| Flag | Default | Description |
|------|---------|-------------|
| `--root-identifier` | `""` | Filename / identifier for the root document header (e.g. `attention.py`). **Required** for checkpoints trained with `identifier_prefix_bos_eos` layout policy (all Stack 100M runs) — without it the model sees an empty `# \n\n` header and generates EOS immediately. |
| `--max-link-depth` | `2` | `0` = single-doc baseline; `≥1` enables aux doc insertion |
| `--repetition-penalty` | `1.3` | Values `>1` reduce probability of already-seen tokens. Use `1.05–1.15` for code; `1.3` is appropriate for prose but too aggressive for code (penalises legitimate variable name reuse). |
| `--temperature` | `0.8` | Sampling temperature |
| `--top-k` | `50` | Top-k sampling |
| `--max-display-tokens` | `200` | Truncate displayed text per doc; full links list still shown |
| `--allow-generation-fallback` | off (with `--dataset`) | Generate aux docs for unresolved links |
| `--no-color` | off | Disable ANSI colour for piping/logs |

FlexAttention is compiled on first use (`dynamic=True` for variable-length inference contexts).
Compiled kernels are cached in `.torch_compile_cache/` and reused on subsequent runs.
Override the cache location with `TORCHINDUCTOR_CACHE_DIR`.
