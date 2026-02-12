# GitHub Graph Extractor

A tool to build intra-repository dependency graphs from GitHub repository data. This extracts Python repositories from the [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup) dataset and builds a graph showing file-to-file relationships within the same repository based on import dependencies.

## Features

- **Streams Repository Data**: Downloads and samples Python repositories from The Stack dataset
- **Intra-Repository Dependency Extraction**: Analyzes import statements to build file-to-file relationships within the same repository
- **Parallel Processing**: Utilizes multiple CPU cores for efficient processing
- **Filtered Graph Generation**: Only includes files with 2+ intra-repository dependencies
- **Graph Structure**: Creates JSONL files with file nodes showing internal repository relationships
- **Statistics & Visualization**: Generates degree distribution plots and graph statistics

## Installation

Install dependencies in a virtual environment:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Sample Data

Download a sample of Python repositories from The Stack dataset with production-ready streaming:

```bash
python download_sample.py --limit 100000
```

The script is optimized for very large datasets (10M+ records) with:
- **Memory-efficient streaming**: No memory accumulation, processes items one-by-one
- **Full content preservation**: No truncation of file contents
- **Resume capability**: Automatically resumes from partially downloaded files
- **Progress checkpoints**: Regular progress reporting and intermediate saves
- **Error handling**: Robust handling of network failures and malformed data
- **Disk space monitoring**: Checks available space before starting

**Examples:**
```bash
# Small sample for testing
python download_sample.py --limit 10000

# Medium sample (default)
python download_sample.py --limit 100000

# Large sample
python download_sample.py --limit 1000000

# Very large sample (production-ready)
python download_sample.py --limit 10000000

# Resume interrupted download
python download_sample.py --limit 10000000  # Will resume if sample_10M.jsonl exists

# Custom output file
python download_sample.py --limit 5000000 --output my_sample.jsonl
```

**Command-line options:**
- `-o, --output`: Output file path (default: auto-generated based on limit)
- `--limit`: Number of items to sample (default: 100,000)
- `--no-resume`: Don't resume from existing file, start fresh
- `--checkpoint-interval`: Progress report frequency (default: 100,000)
- `--max-retries`: Network failure retries (default: 3)
- `-v, --verbose`: Enable detailed logging
- `--limit`: Number of repositories to sample (default: 100,000, max: 10,000,000)

### 2. Build the Dependency Graph

Process the sampled data to create a dependency graph using the streaming graph builder:

```bash
python build_graph_streaming.py sample_100k.jsonl
# or whatever filename was generated
```

This will create:
- `graph.jsonl`: The dependency graph in JSONL format
- `graph_stats.json`: Statistical information about the graph
- `graph_degree_dist.png`: Degree distribution visualization

**Command-line options:**
- `input_file`: Path to the sampled data JSONL file
- `-o, --output`: Output path for the graph (default: `graph.jsonl` in input file directory)
- `--limit`: Limit number of repositories to process (for testing)
- `--batch-size`: Number of files to process per batch (default: 10,000)
- `--num-buckets`: Number of buckets for partitioning (default: 64)
- `--bucket-workers`: Number of parallel bucket workers (default: min(CPU cores, 8))
- `--no-stats`: Skip expensive statistics computation for very large datasets
- `-q, --quiet`: Suppress progress reporting

**Note:** The streaming version is optimized for large datasets (10M+ records) and uses parallel processing with bucket partitioning for memory efficiency.

## Graph Format

The output `graph.jsonl` contains one JSON object per line, where each object represents a file with 2+ intra-repository dependencies:

```json
{
  "title": "normalized_repo_name_hash:file_path",
  "char_count": 1234,
  "links_in_repo": 5,
  "outgoing": ["other_file_in_same_repo"],
  "incoming": ["files_that_import_this"]
}
```

- **title**: Normalized repository name with hash, followed by the file path within that repository
- **char_count**: Character count of the file content
- **links_in_repo**: Number of files this file is connected to within the same repository
- **outgoing**: Files within the same repository that this file imports from
- **incoming**: Files within the same repository that import from this file

## How Dependencies are Extracted

The extractor analyzes Python import statements to find intra-repository file dependencies:

- `import module_name` → Creates dependency link if module maps to another file in same repository
- `from module_name import ...` → Creates dependency link if module maps to another file in same repository
- Relative imports (`from .module import ...`) are also handled

Only files with 2+ intra-repository dependencies are included in the final graph. The system maps module names to file paths within each repository to establish file-to-file relationships.

## Differences from Wiki Graph Extractor

- **Data Source**: Uses The Stack dataset instead of Wikipedia dumps
- **Link Type**: Intra-repository file dependencies instead of inter-repository package dependencies
- **Node Types**: Individual files (not repositories) as nodes
- **Filtering**: Only includes files with 2+ connections within the same repository
- **Processing**: Maps imports to specific files within repositories instead of external package dependencies

## Authentication

To access The Stack dataset, you may need to:

1. Install huggingface_hub: `pip install huggingface_hub`
2. Login with: `huggingface-cli login`

This ensures access to gated datasets.
