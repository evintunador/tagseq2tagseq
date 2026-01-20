# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This experiment trains language models on graph-structured Wikipedia data (DAGs). The system implements custom data loading pipelines that traverse a document graph to construct packed sequences with flexible attention patterns. The core innovation is treating documents as nodes in a graph and using various traversal strategies (BFS, DFS, random walk) to create training batches.

## Key Commands

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/data/test_dataset.py
pytest tests/data/test_traversal.py
pytest tests/data/test_pack_sampler.py
pytest tests/data/test_pretokenize.py
```

### Data Pipeline

```bash
# Extract Wikipedia dump to Markdown files (in data/wiki_graph_extractor/)
cd data/wiki_graph_extractor
python dump_extractor.py <dump_path> -o <output_dir> --limit <N>

# Build the link graph from extracted Markdown
python build_graph.py <output_dir>/ --output graph.jsonl

# Pre-tokenize the graph dataset into binary shards
cd ../..
python -m data.pretokenize <graph_jsonl_path> <markdown_dir> --output-dir <pretokenized_output>

# Inspect packed batches without training
python demo_traversal.py <pretokenized_dataset_path> --strategy random_walk --token-budget 2048
```

### Training
```bash
# Run training with default config
python main.py --dataset-dir data/pretokenized_datasets/simplewiki_full

# Run with custom settings
python main.py --dataset-dir <pretokenized_dataset_path> --strategy random_walk --max-seq-len 2048
```

### Model Testing & Benchmarking
```bash
# Run tunalab's built-in module tests (from parent tunalab framework)
# Tests are defined within the module files themselves (e.g., tunalab/modules/training_module.py)
```

## Architecture

### Data Pipeline Flow

1. **Wiki Extraction** (`data/wiki_graph_extractor/`): Converts Wikipedia Cirrus dumps to Markdown files, preserving internal links
2. **Graph Building**: Extracts link structure to create a directed graph (JSONL format)
3. **Pre-tokenization** (`data/pretokenize.py`): Tokenizes all documents and stores them in memory-mapped binary shards
4. **Runtime Data Loading**: Graph traversal + packing + collation happens dynamically during training

### Core Components

**Graph & Storage Layer** (`data/dataset.py`):
- `GraphIndex`: Loads graph structure from JSONL, provides title↔id mapping and neighbor queries
- `PretokShardedBackend`: Memory-maps binary token shards for zero-copy token access

**Traversal Strategies** (`data/traversal.py`):
- `TraversalStrategy` protocol: Interface for graph walking algorithms
- `RandomSelectionStrategy`: Uniform random sampling (baseline, ignores graph)
- `RandomWalkStrategy`: Markovian walk over edges with optional restarts
- `BFSStrategy`, `DFSStrategy`: Breadth/depth-first traversal with per-pack visited tracking
- `CompositeTraversalStrategy`: Mixes multiple strategies within a pack

**Packing & Sampling** (`data/pack_sampler.py`):
- `PackBatchSampler`: Uses a traversal strategy to select documents, enforces token/doc budgets, handles deduplication
- `DocPlacement`: Describes how a single document contributes tokens to a pack (start, length, truncation)
- Order modes: `"as_traversed"` vs `"prefer_targets_first"` (reorders so link targets appear before linkers)

**Layout Policies** (`data/layout.py`):
- `DocLayoutPolicy` protocol: Adds prefix/suffix decoration tokens to documents
- `NullLayoutPolicy`: No decoration
- `BOSEOSLayoutPolicy`: Wraps each document with BOS/EOS tokens

**Collation** (`data/collate.py`):
- `build_packed_batch()`: Materializes a list of `DocPlacement` into actual token tensors
- `DocSpan`: Metadata describing where each document appears in the final packed sequence
- Returns `{"tokens": Tensor, "doc_spans": List[DocSpan]}`

**Dataset** (`data/packed_dataset.py`):
- `PackedSequenceDataset`: Yields packed batches by calling the sampler and collate function

**Model Architecture** (`tunalab/modules/`):

The model follows a three-tier architecture pattern that separates concerns:

1. **`DS2DSBackbone`** (nn.Module): The "spine" of the transformer
   - Pure transformer layer stack with skip connections
   - Intentionally excludes embeddings and output head
   - Forward pass: pre-embedded inputs → hidden states
   - Facilitates multimodal extensions and weight sharing

2. **`DS2DSTrainingModule`** (nn.Module): Training wrapper implementing "batch in, loss out"
   - Combines backbone with embeddings, normalization, and fused CE loss
   - Two construction methods:
     - `from_config(...)`: Recommended factory method for standard training
     - `__init__(...)`: Direct construction with pre-built components
   - `to_inference_model()`: Converts to DS2DSModel after training
   - Forward pass: batch dict → loss dict

3. **`DS2DSModel`** (NOT nn.Module): Inference and evaluation interface
   - Holds references to trained components without nn.Module ceremony
   - Provides stub methods for graph-aware generation and evaluation
   - Methods to implement:
     - `forward_inference()`: tokens → logits
     - `generate()`: Graph-aware text generation
     - `@register_handler` methods for benchmark evaluation
   - Helper methods: `eval()`, `train()`, `to(device)`

**Supporting Modules**:
- `block_mask_creator.py`: Creates custom attention masks based on document boundaries and graph structure

### Key Design Patterns

**Two-Phase Data Strategy**: Pre-tokenization separates the expensive tokenization work from training-time graph traversal. This allows experimenting with different traversal strategies without re-tokenizing.

**Pluggable Traversal**: The `TraversalStrategy` protocol separates "how to walk the graph" from "how to enforce budgets/deduplication". The sampler handles budgets; the strategy just proposes next documents.

**Lazy Loading**: Binary shards are memory-mapped, so only accessed pages are loaded into RAM.

**Packed Sequences**: Multiple documents are concatenated into a single sequence to maximize GPU utilization. The `doc_spans` metadata tracks boundaries for custom attention patterns.

**FlexAttention Integration**: The training module accepts a `block_mask_creator` callable that receives batch metadata (including `doc_spans`) and returns a `BlockMask` for PyTorch's FlexAttention API. This enables document-aware and graph-aware attention patterns.

**Three-Tier Architecture**: The separation of Backbone (nn.Module) → TrainingModule (nn.Module) → Model (NOT nn.Module) follows the "backbone as spine" metaphor:
- Backbone contains only the transformer layers (the "spine")
- TrainingModule wraps backbone for training with "batch in, loss out" interface
- Model provides inference/evaluation without nn.Module overhead
- This enables clean weight sharing, multimodal extensions, and separate inference logic

## Configuration

- `tunalab.yaml`: Minimal config referencing tunalab packs (currently just `["nlp"]`)
- `main.py` CLI args override config values (e.g., `--dataset-dir`, `--strategy`, `--max-seq-len`)
- Most hyperparameters live in tunalab's configuration system (see `tunalab.configuration.compose_config`)

## Dependencies

This experiment relies on the parent `tunalab` framework for:
- Configuration management (`tunalab.configuration`)
- Distributed training utilities (`tunalab.distributed`)
- Logging and reproducibility (`tunalab.tracking`, `tunalab.reproducibility`)
- Neural network modules (transformer layers, norms, losses)
- Training loops (`tunalab.smart_train`)

External dependencies include PyTorch, NumPy, tiktoken (for tokenization), and tqdm (for progress bars).

## Important Notes

- **Batch Size**: FlexAttention currently requires batch size 1. Each "batch" is a single packed sequence containing multiple documents.
- **Token Budget**: The `token_budget` parameter in `PackBatchSampler` controls the max sequence length (including all prefix/suffix decorations).
- **Doc Budget**: Optional per-document token limit to prevent any single document from dominating a pack.
- **Overflow Policy**: When a pack would exceed the token budget, `"truncate"` mode trims individual documents to fit.
- **Memory-Mapped Shards**: Binary shard files are memory-mapped for efficient access. Always call `backend.close()` when done.
- **Hash Cleaning**: During pre-tokenization, 6-character hex hashes are stripped from link targets in Markdown (e.g., `[Link](Title_a1b2c3)` becomes `[Link](Title)`) to avoid polluting the model with implementation details.

## File Structure

```
├── main.py                          # Training entry point
├── demo_traversal.py                # Inspection tool for packed batches
├── block_mask_creator.py            # Custom attention mask logic
├── tunalab.yaml                     # Experiment config
├── data/
│   ├── dataset.py                   # GraphIndex, PretokShardedBackend
│   ├── traversal.py                 # Traversal strategy implementations
│   ├── pack_sampler.py              # PackBatchSampler, DocPlacement
│   ├── layout.py                    # Layout policies (BOS/EOS decoration)
│   ├── collate.py                   # Batch materialization
│   ├── packed_dataset.py            # PackedSequenceDataset
│   ├── pretokenize.py               # Pre-tokenization script
│   └── wiki_graph_extractor/        # Wikipedia dump extraction tools
│       ├── dump_extractor.py
│       ├── build_graph.py
│       └── extract.py               # Wikitext→Markdown cleaning logic
├── model/
│   ├── model.py                     # DS2DSModel (inference/eval)
│   └── modules/
│       ├── training_module.py       # DS2DSTrainingModule
│       ├── backbone.py              # DS2DSBackbone
│       └── layer.py                 # Transformer layer
└── tests/
    ├── data/
│   └── model/
```

## Common Workflows

### Adding a New Traversal Strategy

1. Implement the `TraversalStrategy` protocol in `data/traversal.py`
2. Add a factory lambda in `main.py` and `demo_traversal.py`
3. Write tests in `tests/data/test_traversal.py`

### Modifying the Attention Pattern

1. Edit `block_mask_creator.py` to change the mask creation logic
2. The `create_doc_causal_block_mask` function receives `tokens` and `doc_spans`
3. Use `doc_spans` to identify document boundaries and implement custom attention rules

### Implementing Inference and Evaluation

After training, convert the training module to an inference model:

```python
# After training
trained_module = result['model']  # from smart_train
inference_model = trained_module.to_inference_model()

# Implement the stub methods in tunalab/modules/model.py:
# 1. forward_inference(tokens, doc_spans) -> logits
# 2. generate(...) -> str (graph-aware generation)
# 3. @register_handler methods for evaluation
```

The DS2DSModel class provides skeleton implementations with detailed TODOs for:
- Converting standard eval formats to packed sequences
- Handling graph-aware generation logic
- Computing perplexity and other benchmark metrics

### Debugging Packed Batches

Use `demo_traversal.py` to inspect what a batch looks like before training:
```bash
python demo_traversal.py <dataset_dir> --strategy random_walk --use-bos-eos --order-mode prefer_targets_first
```

This will print:
- Token counts and shapes
- Document spans with start/end/truncation info
- Graph connectivity within the batch
- Decoded text snippets (if tokenizer metadata is available)
