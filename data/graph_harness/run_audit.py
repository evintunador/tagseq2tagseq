"""
CLI: audit a built dataset's graph quality (checkpoint-free).

Usage:
    python -m data.graph_harness.run_audit <dataset_dir> [--json]

Prints the structural graph audit (node/edge counts, degree distribution,
dangling/self-link/isolated/reciprocal rates, warnings). Works on ANY pretokenized
dataset dir — Python, Go, wiki, arxiv — since it reads only the graph metadata.

Optional resolvability sampling (needs a detector) is invoked via the library API
`audit_graph` + a resolvability sample; kept out of this bare CLI to stay
model-free and fast. See graph_harness.auditor.
"""
from __future__ import annotations

import argparse
import json
import sys

from data.graph_harness.auditor import audit_graph


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_dir", help="pretokenized dataset directory")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args(argv)

    audit = audit_graph(args.dataset_dir)
    if args.json:
        from dataclasses import asdict
        d = asdict(audit)
        d.pop("resolvability", None)
        print(json.dumps(d, indent=2))
    else:
        print(audit.summary())
    # Non-zero exit if the graph is structurally broken (edgeless or heavy dangling).
    broken = audit.n_edges == 0 or audit.dangling_edge_rate > 0.05
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
