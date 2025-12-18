# dagseq2dagseq

This experiment explores training language models on graph-structured data (specifically DAGs from Wiki dumps). It implements custom data loading pipelines that traverse the text-attributed document graph to construct packed sequences.

## Structure

- `data/`: Contains the custom dataset, sampler, and traversal logic.
- `demo_traversal.py`: A script to demonstrate and inspect the graph traversal and packing logic without running a full training loop.
- `main.py`: The primary training script (currently a template).
- `tunalab.yaml`: Experiment configuration.

## Usage

### Inspecting Data

To see how the graph traversal works and what the packed batches look like:

```bash
python experiments/dagseq2dagseq/demo_traversal.py /path/to/pretokenized/dataset --strategy random_walk
```

### Training (Template)

To run the training script (once configured):

```bash
python experiments/dagseq2dagseq/main.py --dataset-dir /path/to/pretokenized/dataset
```
