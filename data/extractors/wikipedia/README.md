# DAGWikiExtractor

A streamlined Python tool for processing Wikipedia Cirrus dumps into a clean, link-preserving dataset for training LLMs. This script converts Wikipedia articles from their raw `source_text` in a Cirrus dump into individual Markdown files, making sure that all internal wiki links are converted to standard Markdown links.

This project was originally adapted from the much more complex [attardi/wikiextractor](https://github.com/attardi/wikiextractor) but has been heavily refactored to rely on pre-expanded Cirrus dumps, which dramatically simplifies the extraction and cleaning process.

## Features

-   **Processes Cirrus Dumps**: Directly reads the gzipped, line-delimited JSON format of Wikipedia Cirrus dumps.
-   **Parallel Processing**: Utilizes multiple CPU cores to process articles in parallel, dramatically speeding up extraction.
-   **Link Preservation**: Converts `[[wiki links]]` and `[[wiki links|with custom text]]` into standard `[Markdown links](wiki_links)`.
-   **Comprehensive Cleaning**: Removes a wide variety of wikitext noise, including:
    -   Template calls (`{{...}}`)
    -   Wikitables (`{|...|}`)
    -   External links (`[http...]`)
    -   Reference tags (`<ref>...`)
    -   Unwanted sections (e.g., "References", "See also", "External links").
-   **Simple & Fast**: With no need for a complex template expansion engine, the script is fast and has no external dependencies.
-   **One File Per Article**: Each Wikipedia article is saved as a cleanly formatted `.md` file.
-   **Sharded Output**: Organizes output files into subdirectories (e.g., `A/`, `B/`, `C/`) based on the first letter of the article title, improving filesystem performance and navigability for very large datasets.

## Installation

This project uses `tqdm` for progress bars. It is recommended to install dependencies inside a virtual environment.

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download a Wikipedia Dump

This tool is designed to work with Cirrus dumps, which contain the pre-expanded `source_text` we need.

1.  Go to the [Wikimedia Cirrus Dumps page](https://dumps.wikimedia.org/other/cirrussearch/).
2.  Select a recent date (e.g., `20251101/`).
3.  Download a `...-cirrussearch-content.json.gz` file for your language of choice (e.g., `enwiki-20251101-cirrussearch-content.json.gz` for English).

### 2. Run the Extractor

You can provide one or more dump files or directories as input. If you provide a directory, the script will automatically find all `.json.gz` files inside it and process them in chronological order.

```bash
python dump_extractor.py <path_to_dump_file_or_dir>...
```

**Examples:**

```bash
# Process a single dump file
python dump_extractor.py data/simplewiki-20251027-cirrussearch-content.json.gz

# Process multiple specific dump files
python dump_extractor.py dump1.json.gz dump2.json.gz

# Process all dumps within a directory
python dump_extractor.py data/

# Process multiple dumps and limit to the first 100 articles found in each
python dump_extractor.py data/ --limit 100
```

### Command-Line Options

-   `inputs`: (Required) One or more paths to input Wikipedia Cirrus dump files (`.json.gz`) or directories containing them.
-   `-o, --output`: The directory where extracted Markdown files will be saved. Defaults to `output`.
-   `--limit`: An optional integer to limit the number of articles to process *from each dump file*. Very useful for testing.
-   `-p, --processes`: The number of worker processes to use. Defaults to one less than the number of CPU cores.
-   `-q, --quiet`: Suppress progress reporting during extraction.

## 3. Build the Link Graph

After extracting the articles into Markdown files, you can generate a link graph in the memory-efficient JSON Lines format. This file will contain every article title along with its corresponding outgoing and incoming links.

```bash
# Run the graph builder on the output directory
python build_graph.py output/

# You can also specify a different output file and a limit for testing
python build_graph.py output/ --output my_graph.jsonl --limit 10000
```

### Graph Builder Options

-   `input_dir`: (Required) The directory containing the extracted `.md` files.
-   `-o, --output`: The path for the output graph file. Defaults to `graph.jsonl`.
-   `--limit`: An optional integer to limit the number of articles to process for the graph. Very useful for testing.
-   `-p, --processes`: The number of worker processes to use. Defaults to one less than the number of CPU cores.
-   `-q, --quiet`: Suppress progress reporting.