# TAGSeq2TAGSeq — paper

LaTeX source for the TAGSeq2TAGSeq paper.

## Layout
- `main.tex` — top-level document (NeurIPS-style single-column preamble; swap for target venue).
- `sections/` — one `.tex` per section, `\input` from `main.tex`.
- `bib/refs.bib` — consolidated BibTeX (assembled from the literature-review pass).
- `figures/` — figures (symlink or copy from ../docs/images and ../artifacts as needed).
- `PROJECT_SUMMARY_for_litreview.md` — the brief handed to the lit-review sub-agents.

## Build
```bash
cd paper && latexmk -pdf main.tex   # or: pdflatex main && bibtex main && pdflatex main x2
```

## Source material (in repo root)
- README.md — method overview + mask figures
- RESULTS*.md — all quantitative results
- INSTRUCTIONS.md — full pipeline
- docs/handoff_*.md — design notes


## Literature review status (2026-08-10)
Two-round review complete, grounded in a 12-agent code-exploration pass
(`notes/code_briefs/*.md`, line-referenced per subsystem):
- `bib/refs.bib` — **531 verified entries** (118 round-1 broad + 413 round-2 deep),
  no duplicate keys, brace-balanced.
- `notes/LITREVIEW_PLAN.md`, `notes/REVIEWER_SHARED_BRIEF.md`, `notes/ROUND1_KEYS.txt` — review scaffolding.
- `sections/02_related_work.tex` — 9 paragraphs, ~168 citations, all keys resolve.
Every note draws the train-on-structure vs retrieve-at-inference contrast. Some entries
carry inline `% FLAG` venue/page caveats to reverify before camera-ready.


## Appendices (2026-08-10)
Implementation detail lives in `sections/appendices/` (indexed by `sections/A_appendix.tex`);
main text stays lean and points here. All grounded in `notes/code_briefs/`, factual, no
result-number dependencies, all \cite keys resolve:
- A `A_recipe_lineage` — inherited architecture & optimizer recipe (NanoGPT-speedrun lineage,
  explicitly *adopted, not claimed*; VE banks / bigram / MTP noted **disabled in all reported runs**).
- B `B_kernels` — custom block-sparse (BIM) Triton attention kernels.
- C `C_link_detection` — online link-detector protocol + per-language detectors.
- D `D_traversal` — BFS/DFS/random-walk/random packing algorithms + traversal↔DAG-gate coupling.
- E `E_density_scheduling` — kv_block_count proxy + quantile bucketing + per-rank density match.
- F `F_dataset_construction` — per-modality graph extraction, normalization, splitting.
- Z `Z_additional_results` — TODO placeholder for final tables.
refs.bib now 533 entries. VE banks + modded-nanoGPT provenance removed from main text
(main claims scoped to contributions); the lineage is documented only in Appendix A.

## Main-text draft (2026-08-23)
§1 intro, §3 method (formal masks, both figures), §4 datasets, §5 setup, §6 results
(+ new link-density subsection), §7 analysis, §8 conclusion, abstract all in prose.
Framing: the thesis is causality-preserving native corpus-fetching (unmeasured, stated
as the central open measurement); the density law is a *capability* finding; arXiv is
scoped out (32k window); the wiki community_pack negative is under audit and unclaimed.
Ledger: 109 grounded keys (`provenance/ledger.yaml`); Appendix Z tables are all `\val{}`.
Remaining `\fillin{}` blanks + every cluster-side follow-up: `TODOS.md` "Paper —
cluster-reliant items". Author: Evin Tunador, HOPPR. Bib: all 11 FLAGs verified.

## Notes layout
- `bib/refs.bib` — bibliography, source of truth (533 entries).
- `notes/related_work_notes.md` — annotated bibliography (per-work relationship notes, 6 themes).
- `notes/code_briefs/` — 12 subsystem briefs, each pinned to a commit with a drift-check (see its README).
