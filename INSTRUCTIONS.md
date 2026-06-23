# Pipeline Instructions

Full instructions for data preparation, training, and generation. See [README.md](README.md) for a conceptual overview.

---

## Available Checkpoints

> **All checkpoint paths below are broken.** They were trained against the old normalization scheme (identifier hashes derived from the canonical form, not the raw string). New checkpoints must be trained against the re-processed datasets before generation examples here will work.

| Checkpoint | Architecture | Dataset | Context | Mask | Steps | Val loss |
|-----------|-------------|---------|---------|------|-------|----------|
| `runs/20260224_212158/checkpoints/best_model.pt` | 12L / 768D | SimpleWiki | 2k | `doc_causal` | 38,500 | 2.07 |
| `runs/20260308_012514/checkpoints/best_model.pt` | 36L / 1280D | SimpleWiki | 32k | `doc_causal` | 10,200 | 3.923 |
| `runs/20260308_012516/checkpoints/best_model.pt` | 36L / 1280D | SimpleWiki | 32k | `cross_doc_link` (md) | 12,200 | 3.905 |
| `runs/20260308_012518/checkpoints/best_model.pt` | 36L / 1280D | Stack 10M | 32k | `doc_causal` | 14,700 | 2.271 |
| `runs/20260308_012521/checkpoints/best_model.pt` | 36L / 1280D | Stack 10M | 32k | `cross_doc_link` (py) | 14,900 | 2.291 |
| `runs/run_20260311_184203_685319/checkpoints/best_model.pt` | 24L / 1024D | Stack 100M | 32k | `cross_doc_link` (py) | 3,000 | 1.430 |

---

## Datasets

Pretokenized datasets live outside all worktrees at `/fss-data/evin_t/tagseq2tagseq_artifacts/`:

| Dataset | Graph edges | Nodes | Shards | Pretokenized location |
|---------|-------------|-------|--------|-----------------------|
| **SimpleWiki** | Markdown hyperlinks | 282k | 1 | `/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/simplewiki/` |
| **TheStack** | Python imports | 3.56M | 9 | `/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/` |
| **arXiv** | `\cite{}` citations | 2.20M | — | `/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/arxiv/` |

SimpleWiki uses `--model.link_detector markdown`; TheStack uses `--model.link_detector python`; arXiv uses `--model.link_detector arxiv` (config: `configs/arxiv_cross_doc.yaml`).

All three datasets have been split into train/val/test subdirectories (see [Graph Splitting](#graph-splitting) below).

Raw dumps and intermediate graph files (as built on disk):

| Dataset | Raw source | Graph |
|---------|-----------|-------|
| SimpleWiki | `/fss-data/evin_t/tagseq2tagseq_artifacts/wiki_downloads/simplewiki-20260621-cirrussearch-content.json.gz` | `/fss-data/evin_t/tagseq2tagseq_artifacts/wiki_extracted/graph.jsonl` |
| TheStack | `/fss/evin_t/tagseq2tagseq/data/github_graph_extractor/sample_100M.jsonl` | `/fss-data/evin_t/tagseq2tagseq_artifacts/graphs/thestack/graph_100M.jsonl` |
| arXiv | unarXive 2024 (see [arXiv pipeline](#arxiv-unarxive-2024)) | `/fss-data/evin_t/tagseq2tagseq_artifacts/graphs/arxiv_run_legacy_20260616/graph.jsonl` |

Extracted article files (reusable across pipeline runs — no need to re-extract from dumps):

| Dataset | Articles directory |
|---------|--------------------|
| SimpleWiki | `/fss-data/evin_t/tagseq2tagseq_artifacts/wiki_extracted/` |

---

## Full Pipeline

### Wikipedia (SimpleWiki)

**1. Extract dump → markdown articles**

Only needed if the articles directory doesn't already exist. The extracted files at
`/fss-data/evin_t/tagseq2tagseq_artifacts/wiki_extracted/` are reusable.

```bash
python -m data.wiki_graph_extractor.dump_extractor \
    /fss-data/evin_t/tagseq2tagseq_artifacts/wiki_downloads/simplewiki-20260621-cirrussearch-content.json.gz \
    -o /fss-data/evin_t/tagseq2tagseq_artifacts/wiki_extracted \
    -p 60
```

Produces ~282,000 `.md` files organised into per-letter subdirectories.

**2. Build link graph**

```bash
python -m data.wiki_graph_extractor.build_graph \
    /fss-data/evin_t/tagseq2tagseq_artifacts/wiki_extracted \
    -o /fss-data/evin_t/tagseq2tagseq_artifacts/wiki_extracted/graph.jsonl \
    -p 40
```

Produces `graph.jsonl` (~282k nodes) plus `_stats.json` and `_degree_dist.png`.

**3. Pretokenize**

```bash
python -m data.pretokenize \
    /fss-data/evin_t/tagseq2tagseq_artifacts/wiki_extracted \
    /fss-data/evin_t/tagseq2tagseq_artifacts/wiki_extracted/graph.jsonl \
    -o /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/simplewiki \
    -p 40
```

Produces `shard_000000.bin`, `tokenized_graph.jsonl`, and `metadata.json`.

---

### TheStack (~100M Python files)

**1. Download samples**

Requires a HuggingFace token with read access to `bigcode/the-stack-dedup`.

```bash
HF_TOKEN=<your_token> python data/github_graph_extractor/download_sample.py \
    --limit 100000000 \
    -o /fss/evin_t/tagseq2tagseq/data/github_graph_extractor/sample_100M.jsonl
```

**2. Build import dependency graph**

Run from inside `data/github_graph_extractor/` (standalone script, uses project-root sys.path injection):

```bash
cd data/github_graph_extractor
python build_graph_streaming.py \
    /fss/evin_t/tagseq2tagseq/data/github_graph_extractor/sample_100M.jsonl \
    -o /fss-data/evin_t/tagseq2tagseq_artifacts/graphs/thestack/graph_100M.jsonl \
    -p 32 \
    --bucket-workers 8
cd -
```

Produces `graph_100M.jsonl` (~3.56M nodes) plus stats and a degree-distribution plot.

**3. Pretokenize**

```bash
python -m data.pretokenize_stack \
    /fss/evin_t/tagseq2tagseq/data/github_graph_extractor/sample_100M.jsonl \
    /fss-data/evin_t/tagseq2tagseq_artifacts/graphs/thestack/graph_100M.jsonl \
    -o /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack \
    -p 40
```

Produces 9 binary shards, `tokenized_graph.jsonl`, and `metadata.json`.

---

### arXiv (unarXive 2024)

Four CPU scripts under `data/arxiv_graph_extractor/`, each fully arg-driven (no hard-coded
paths). They are heavy/long, so run them on a compute node — e.g. wrap any command below in
`srun --cpus-per-task=32 --mem=64G --time=8:00:00 <cmd>`. Write all outputs under `<artifacts>`
on a bulk-I/O filesystem (here `/fss-data`), never `/fss`. `<corpus>` below is the extracted
unarXive directory `<artifacts>/raw/unarxive_2024/extracted/processed_unarxive_extended_data`.

**1. Download + extract unarXive 2024** (105 GB tarball → 371 GB JSONL shards)

```bash
# HF token in env (e.g. export HF_TOKEN=...); dataset is public (MIT).
hf download ines-besrour/unarxive_2024 --repo-type dataset \
    --include "unarXive_2024.tar.gz.part_*" --local-dir <artifacts>/raw/unarxive_2024/parts
cat <artifacts>/raw/unarxive_2024/parts/unarXive_2024.tar.gz.part_a? \
    | tar -xzf - -C <artifacts>/raw/unarxive_2024/extracted
```

**2. Build the arXiv↔OpenAlex map** (enriched citation resolution)

```bash
python -m data.arxiv_graph_extractor.build_openalex_map \
    --out <artifacts>/graphs/arxiv_openalex_map.jsonl --workers 32
```

Streams the OpenAlex works snapshot (~639 GB) over the public S3 **HTTPS** endpoint — no bulk
download, no credentials (the `aws` CLI is not installed on compute nodes). Writes ~3M
arxiv↔OpenAlex pairs.

**3. (optional) Measure graph density** before the full extract:

```bash
python -m data.arxiv_graph_extractor.measure_density \
    --corpus-dir <corpus> --out <artifacts>/graphs/arxiv_density_report.json --workers 32
```

**4. Extract graph + rehydrated content**

```bash
python -m data.arxiv_graph_extractor.extract \
    --corpus-dir <corpus> \
    --oa-map <artifacts>/graphs/arxiv_openalex_map.jsonl \
    -o <artifacts>/graphs/arxiv --workers 32
```

Two passes + merge: resolves citations (direct arXiv id, then OpenAlex map) to in-corpus
titles, rewrites `{{cite:...}}` → `\cite{Title}`, rehydrates bodies to faithful LaTeX, and
writes `<run-dir>/{graph.jsonl, content.jsonl}` (~2.20M nodes, mean out-degree 4.3) plus a
`reproducibility/` folder (git state + invocation). `-o` is the run directory; `--out-dir`
is accepted as an alias. The existing pre-`ReproducibilityManager` run lives at
`graphs/arxiv_run_legacy_20260616/`.

**5. Pretokenize**, then split as in [Graph Splitting](#graph-splitting):

```bash
python -m data.pretokenize_arxiv \
    <artifacts>/graphs/arxiv/content.jsonl \
    <artifacts>/graphs/arxiv/graph.jsonl \
    -o <artifacts>/pretokenized_datasets/arxiv -p 48
```

arXiv has four condition configs like the other datasets:
`configs/arxiv_{cross_doc,doc_causal,doc_concat_link,doc_concatenated}.yaml`.

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
python demo_traversal.py /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/simplewiki --strategy dfs

# With identifier prefix (e.g. "# Water\n\n..." before each article)
python demo_traversal.py /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/simplewiki \
    --strategy dfs --layout-policy identifier-prefix

# TheStack with identifier prefix (e.g. "# repo:src/file.py\n\n...")
python demo_traversal.py /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack \
    --strategy dfs --layout-policy identifier-prefix
```

### Attention mask images

`model/graph_traversal/block_mask_creator.py` renders the FlexAttention mask for a real batch and saves a PNG to `artifacts/`. Run as a module from the project root:

```bash
# doc_causal mask — Wikipedia
python -m model.graph_traversal.block_mask_creator \
    /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/simplewiki \
    --mask-type doc_causal --strategy bfs --seed 42

# cross-document link mask — Wikipedia (markdown link detector)
python -m model.graph_traversal.block_mask_creator \
    /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/simplewiki \
    --mask-type cross_doc_link --link-detector markdown --strategy bfs --seed 42

# doc_causal mask — TheStack
python -m model.graph_traversal.block_mask_creator \
    /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack \
    --mask-type doc_causal --strategy bfs --seed 42

# cross-document link mask — TheStack (Python import detector)
python -m model.graph_traversal.block_mask_creator \
    /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack \
    --mask-type cross_doc_link --link-detector python --strategy bfs --seed 42
```

Available mask types: `doc_causal`, `causal`, `full`, `doc_bidirectional`, `cross_doc_link`.
Available strategies: `dfs`, `bfs`, `random_walk`, `random`.
`--link-detector` is only used with `cross_doc_link`: `markdown` for Wikipedia, `python` for TheStack.

---

## Graph Splitting

Before training, split each dataset into five disjoint subdirectories. Each subdir is a
self-contained dataset (its own `tokenized_graph.jsonl` + `metadata.json` sharing the parent's
binary shards by absolute path). Cross-split edges are dropped so each split has no knowledge
of the others.

| Split | Fraction | Purpose |
|-------|----------|---------|
| `train` | ~90% | Training data |
| `val_community` | 2.5% | BFS-identified subgraphs; link structure intact. Periodic val loss during training; post-training `community_pack_perplexity`. |
| `val_random` | 2.5% | Uniformly sampled isolated nodes. Post-training `held_out_perplexity`. |
| `test_community` | 2.5% | Same structure as val_community; held back until paper submission. |
| `test_random` | 2.5% | Same structure as val_random; held back until paper submission. |

`val_community` is the periodic validation loss shown on the training loss curve.
`val_random` provides a complementary perplexity measure on structurally isolated documents.

```bash
# Split simplewiki (282k nodes, ~3 s)
python -m data.split_graph \
    --dataset-dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/simplewiki

# Split thestack (3.56M nodes, ~60 s)
python -m data.split_graph \
    --dataset-dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack

# Dry-run to preview split counts without writing
python -m data.split_graph \
    --dataset-dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/simplewiki \
    --dry-run
```

Output is written to `dataset_dir/splits/{train,val_community,val_random,test_community,test_random}/`.
Re-running is safe — it overwrites in place.

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--val-frac` | 0.025 | Fraction of nodes for each val split (community + random each). |
| `--test-frac` | 0.025 | Fraction of nodes for each test split (community + random each). |
| `--community-size-min` | 50 | Discard BFS communities smaller than this (isolates go to train). |
| `--community-size-max` | 5000 | Cap BFS expansion per community at this many nodes. |
| `--seed` | 42 | RNG seed for reproducibility. |

Once splits exist, point the training config at them explicitly. The split key names must
match the filesystem directory names (they're used as subdirectory names by
`community_pack_perplexity` and as loader labels in the training loss log):

```yaml
# In your training config (e.g. configs/large_32k.yaml):
data:
  dataset_dir: /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack
  train_dir:   /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/train
  val_dirs:
    val_community: /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/val_community
    val_random:    /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/val_random
  test_dirs:
    test_community: /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/test_community
    test_random:    /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/test_random
```

`train_dir` drives the training `GraphIndex`. If absent, falls back to `dataset_dir` (full
graph — no split exclusion). `val_dirs` builds one live `PackedSequenceDataset` loader per
entry, evaluated every `val_interval` steps. `test_dirs` are evaluated only at the end of
training. `val_dirs` and `test_dirs` absent → val uses train graph with offset seed (old
behaviour, fine for quick experiments).

### Precomputed val packs (optional)

For large datasets you can pre-compute val packs just like train packs, getting a deterministic,
reproducible pack sequence for fair cross-checkpoint comparison:

```bash
# Pre-compute val_community packs (--n-buckets 1: no density sorting needed for eval)
CUDA_VISIBLE_DEVICES=0 python precompute_epochs.py \
    --dataset-dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/val_community \
    --output-dir  /fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_val_community \
    --n-epochs 1 --strategy bfs --local-seq-len 32768 \
    --n-buckets 1 --n-workers 4 --seed 42 \
    --link-detector python --layout-policy stochastic_identifier_prefix
```

Then in the config:
```yaml
data:
  val_dirs:
    val_community: /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/val_community
  val_epoch_dirs:
    val_community: [/fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_val_community/epoch_0]
```

When `val_epoch_dirs[name]` is set it takes precedence over `val_dirs[name]` for the loader,
but `val_dirs[name]` is still used as the `GraphIndex` source for the precomputed dataset.

---

## Density-Aware Batch Scheduling

Pre-computing epochs offline eliminates online link detection overhead (~1.3 s/step at 32k) and ensures all DDP ranks receive packs of the same attention-mask density at each step, eliminating rank-stall waste on multi-node InfiniBand runs.

**Currently only supported for TheStack** (identifier format `owner/repo:path`). Wikipedia must use the standard live `PackedSequenceDataset` path.

> **TODO — Wikipedia support:** The TheStack restriction exists because workers are partitioned by repo prefix, keeping linked files co-resident in the same shard so BFS traversal doesn't immediately hit boundaries. Wikipedia needs a graph-community partitioner (multi-source BFS Voronoi: seed one doc per worker, expand round-robin with a size cap to prevent hub nodes dominating, re-seed workers that exhaust their queue). See the module-level TODO in `data/epoch_precompute.py` for the full design.

---

### Step 1 — Pre-compute epochs

Run once before training. Both `doc_causal` and `cross_doc_link` configs use
`stochastic_identifier_prefix` as their training layout policy, so they share a
single set of precomputed packs. The `doc_causal` model just ignores the
`link_to_target` field; sharing packs keeps the baseline directly comparable.

Each epoch takes roughly 20 min on 1 GPU for TheStack at 32k seq_len (16 workers).

```bash
# TheStack train split — 5 epochs, shared by both doc_causal and cross_doc_link
CUDA_VISIBLE_DEVICES=0 python precompute_epochs.py \
    --dataset-dir  /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/train \
    --output-dir   /fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_bfs \
    --n-epochs     5 \
    --strategy     bfs \
    --local-seq-len 32768 \
    --n-buckets    32 \
    --n-workers    16 \
    --seed         42 \
    --link-detector python \
    --layout-policy stochastic_identifier_prefix
```

Each `epoch_{i}/` directory receives:
- `packs.parquet` — packed, snappy-compressed, sorted by bucket then pack_id
- `metadata.json` — n_buckets, n_packs, token_budget, strategy, seed, kv_method, …

The script is **resume-safe**: it skips any epoch whose `packs.parquet` already exists.

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--n-buckets` | 32 | Density quantile buckets. Use 8 for quick experiments, 32 for production. |
| `--n-workers` | 8 | Subprocess workers (one repo shard each). Each worker opens its own GraphIndex. |
| `--local-seq-len` | 32768 | Token budget per pack — must match `model.max_seq_len`. |
| `--layout-policy` | `null` | Must match the training layout policy (`stochastic_identifier_prefix` for production). |
| `--gpu-kv-pass` | off | Use GPU BlockMask instead of CPU analytical method for kv_block_count (36 ms/pack vs 1 ms/pack; only useful for verifying C==B on a real dataset). |

---

### Step 2 — Train with pre-computed epochs

Pass `--data.epoch_dirs` pointing at the pre-computed epoch directories. The training script automatically activates `BucketedPackDataset` and injects `bucket_state_fn` so dataset position is saved in every checkpoint.

```bash
# Multi-node SLURM — doc_causal baseline
python launch_slurm.py \
    --nodes 2 --gpus-per-node 4 --time 24:00:00 \
    --config configs/thestack_doc_causal.yaml \
    --data.dataset_dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/train \
    --data.val_dirs.val_community /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/val_community \
    --data.val_dirs.val_random /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/val_random \
    --data.epoch_dirs /fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_bfs/epoch_0,/fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_bfs/epoch_1,/fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_bfs/epoch_2

# Multi-node SLURM — cross_doc_link experimental (same packs, different mask)
python launch_slurm.py \
    --nodes 2 --gpus-per-node 4 --time 24:00:00 \
    --config configs/thestack_cross_doc.yaml \
    --data.dataset_dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/train \
    --data.val_dirs.val_community /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/val_community \
    --data.val_dirs.val_random /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/val_random \
    --data.epoch_dirs /fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_bfs/epoch_0,/fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_bfs/epoch_1,/fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_bfs/epoch_2

# Resume from checkpoint (BucketState is embedded in the checkpoint metadata)
python launch_slurm.py \
    --nodes 2 --gpus-per-node 4 --time 24:00:00 \
    --config configs/thestack_cross_doc.yaml \
    --data.dataset_dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/train \
    --data.epoch_dirs /fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_bfs/epoch_0,/fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_bfs/epoch_1 \
    --resume-from runs/<run_dir>/checkpoints/best_model.pt
```

Notes:
- `world_size` and `grad_accum` can change on resume — the `bucket_consumed` cursors remain valid regardless.
- If all `epoch_dirs` are exhausted, training raises `RuntimeError` asking you to precompute more epochs.
- The `max_grants_warmup` warning is expected if `max_grants_start < max_grants` in the config: bucketing was done at the final `max_grants`, so density balance is approximate during warmup.

---

### Step 3 — Visualise results

`visualize_epoch.py` generates three figures from a pre-computed epoch directory.

```bash
# Density overview + masks (no timing data needed)
python visualize_epoch.py \
    --epoch-dir   /fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_bfs/epoch_0 \
    --dataset-dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/train \
    --output-dir  /fss-data/evin_t/tagseq2tagseq_artifacts/artifacts/thestack_report

# Full report with training timing comparison
python visualize_epoch.py \
    --epoch-dir         /fss-data/evin_t/tagseq2tagseq_artifacts/schedules/thestack_bfs/epoch_0 \
    --dataset-dir       /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/train \
    --live-run          runs/<live_run_dir> \
    --precomputed-run   runs/<precomputed_run_dir> \
    --output-dir        /fss-data/evin_t/tagseq2tagseq_artifacts/artifacts/thestack_report
```

**Outputs:**

| File | Contents |
|------|---------|
| `density_overview.png` | kv_block_count histogram (stacked by bucket) + per-bucket violin. Shows the density spread and confirms bucket boundaries are well-separated. |
| `masks.png` | Block-level attention mask (256×256 grid of 128-token blocks) for the median pack from the sparsest and densest bucket. Each cell = one block pair; blue = non-empty; red lines = document boundaries. |
| `step_timing.png` | Per-step wall-clock time. Live steps are uniform steel-blue; pre-computed steps are colour-coded by which density bucket was drawn — sparse buckets (dark) are fast, dense buckets (yellow) are slow. |
| `step_timing_by_bucket.png` | Breakdown panel. Live: histogram of all step times with percentile markers (shows the broad random distribution). Precomputed: per-bucket mean ± 1 std with within-bucket CoV annotated (should be ≪ overall CoV — confirms ranks are well-matched within each step). |

---

## Training

Run artifacts are saved to timestamped directories under `runs/`.

### Baseline runs (doc_causal, random traversal)

The baseline uses document-causal masking (each document attends only to itself) with random graph traversal. `data.strategy` defaults to `random` in `baseline.yaml` so no override is needed.

```bash
# SimpleWiki (~108M tokens, fast iteration)
python main.py --config configs/baseline.yaml \
    --dataset-dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/simplewiki

# TheStack (~8.7B tokens, full Python corpus)
python main.py --config configs/baseline.yaml \
    --dataset-dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack
```

### Cross-document runs (cross_doc_link, BFS traversal)

BFS traversal places linked documents adjacently in the packed sequence, which is required for cross-doc attention to be meaningful. Set `model.link_detector` to match the dataset.

**Wikipedia** (`--model.link_detector markdown`):
```bash
python main.py --config configs/baseline.yaml \
    --dataset-dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/simplewiki \
    --strategy bfs \
    --model.mask_type cross_doc_link \
    --model.link_detector markdown
```

**TheStack** (`--model.link_detector python`):
```bash
python main.py --config configs/baseline.yaml \
    --dataset-dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack \
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
# 2 nodes × 8 GPUs — TheStack (the canonical large run)
python launch_slurm.py \
    --nodes 2 --gpus-per-node 8 --time 48:00:00 \
    --config configs/thestack_cross_doc.yaml \
    --data.dataset_dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack

# 1 node × 4 GPUs — quick iteration on SimpleWiki
python launch_slurm.py \
    --nodes 1 --gpus-per-node 4 --time 12:00:00 \
    --config configs/large_32k.yaml \
    --data.dataset_dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/simplewiki \
    --model.mask_type cross_doc_link --model.link_detector markdown
```

`--no-tail` suppresses log following after submission. Logs land in the run's `logs/` subdirectory.

---

## Generation

> **Checkpoint paths in the examples below are broken** — they were trained on the old normalizer. Update these paths once new checkpoints are trained against the re-processed datasets.

After training, generate text from a checkpoint using `generate.py`. The script auto-reads
`hyperparameters.json` from the run directory to reconstruct the architecture, tokenizer,
link detector, and layout policy — no manual config needed.

### TheStack model (cross-doc, 32k context)

Checkpoints trained with `identifier_prefix_bos_eos` layout policy require `--root-identifier`
or the model sees an empty `# \n\n` header and immediately generates EOS.

```bash
python generate.py \
    --checkpoint runs/<run_dir>/checkpoints/best_model.pt \
    --dataset /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack \
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

**Repetition penalty for code:** use `1.05–1.15`. The default `1.3` is too aggressive — it
penalises legitimate variable name reuse (e.g. `d_model`, `optimizer`), causing premature EOS.
`1.0` disables the penalty but risks infinite repetition loops.

### SimpleWiki model (doc_causal or cross-doc)

```bash
python generate.py \
    --checkpoint runs/<run_dir>/checkpoints/best_model.pt \
    --prompt "Python is a high-level programming language." \
    --dataset /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/simplewiki \
    --max-link-depth 2 \
    --max-new-tokens 300
```

With `--dataset` provided, links detected in generated text are looked up in the corpus. Matched
documents are inserted into the attention context before the active document. Set
`--allow-generation-fallback` to also generate aux docs for links not found in the corpus.

### Key generation options

| Flag | Default | Description |
|------|---------|-------------|
| `--root-identifier` | `""` | Filename / identifier for the root document header (e.g. `attention.py`). **Required** for checkpoints trained with `identifier_prefix_bos_eos` layout policy — without it the model sees an empty `# \n\n` header and generates EOS immediately. |
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
