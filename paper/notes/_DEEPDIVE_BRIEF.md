# Deep-dive brief: one paper, cross-referenced against the TAGSeq2TAGSeq codebase

You produce a detailed, self-contained analysis of ONE assigned paper for the authors of
TAGSeq2TAGSeq. This is NOT a bibliography note — go deep. Two grounding sources:

1. THE PAPER: research it on the web (arxiv.org abstract+PDF, semantic scholar, the
   paper's own related-work). Read enough to describe its actual method, setup, and
   findings accurately — not just the abstract.
2. OUR CODEBASE + PAPER (repo root /fss/evin_t/tagseq2tagseq): read the relevant
   code briefs under paper/notes/code_briefs/ (they carry a PROVENANCE header naming
   the exact source files — read those source files too where it matters), the
   consolidated notes at paper/notes/related_work_notes.md, and grep the actual source
   to verify how OUR method really works before comparing.

Our project in one line: pretrain a decoder-only LM on a text-attributed graph
(documents = nodes; hyperlinks / imports / citations = edges) by (a) packing
graph-topologically-close documents into one 32k sequence via graph traversal and (b) a
custom sparse attention mask that grants a linking document read-access, from the link
position onward, into the document it links to — used identically in pretraining AND at
inference (a generated link fetches the target doc into the attention context). Compute-
control masks (concat variants) isolate the linking inductive bias from raw FLOPs.

Produce these sections (markdown), specific and technical:
## <cite_key> — <Paper title>
### What the paper actually does
  Method, training/eval setup, scale, and the concrete result numbers that matter.
### Methodology: theirs vs. ours
  Precise compare-and-contrast against OUR implementation (cite our code briefs / source).
  Nail the axis: train-on-structure vs retrieve-at-inference; attention edge vs GNN edge
  vs cached-KV vs training-pair signal; what they share with us and where they diverge.
### Predictions & open questions for our method
  From THEIR findings, what should we expect OUR method to do? (scaling behavior,
  where the effect should be strong/weak, ablation outcomes, failure regimes.) What
  open question of theirs might our design resolve, or vice versa?
### Gotchas
  Pitfalls their experience warns us about (eval artifacts, contamination, tuning traps,
  metric choices, things that broke for them and likely will for us).
### Missed citations worth adding
  Scan the paper's OWN references for works relevant to OUR project that are NOT yet in
  our lit review. For each: bib-key-style name, arXiv id if findable, one line on why it
  matters to us. (I will verify before adding — do NOT claim they're already in refs.bib.)
  Check the current set first: grep paper/bib/refs.bib. Only list genuinely-missing ones.

Rules: verify claims against real sources; distinguish what you confirmed from what you
infer. Do not invent arXiv ids or result numbers. Write to the output file given in your
task with the Write tool. Be thorough but do not pad. End with a one-line confirmation.
