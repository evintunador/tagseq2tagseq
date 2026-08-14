# Code briefs

Twelve line-referenced technical summaries of the TAGSeq2TAGSeq subsystems, written
to ground the paper (method section + appendices) in the actual implementation. Each
brief distills one subsystem: what it does, the exact mechanism, concrete parameters,
and reviewer-relevant subtleties.

## Staleness / drift protocol
These briefs are a snapshot of the code, so they go stale as the code evolves. Each
file starts with a `<!-- PROVENANCE -->` header recording (a) the commit it was
written against and (b) the source paths it covers, with a ready-to-run drift check:

    git diff --stat <pin-commit>..HEAD -- <covered paths>

Empty output ⇒ the brief is still faithful to the code. Non-empty ⇒ the covered
sources moved; re-verify the affected claims against source (or regenerate the brief)
before relying on it. All briefs were pinned to the same commit, `6134163` (main,
2026-08-07). To check every brief at once:

    for f in *.md; do
      paths=$(sed -n 's/^Covered sources: //p' "$f")
      [ -n "$paths" ] && { echo "== $f =="; git diff --stat 6134163..HEAD -- $paths; }
    done

## Contents
masks, kernels, traversal, packing_density, link_detectors, architecture,
muon_optimizer, generation_retrieval, eval_harness, training_loop, data_pipelines,
merged_multisource.
