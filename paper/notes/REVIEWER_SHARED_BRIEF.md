# Shared brief for round-2 literature-review agents

You are one of ~30+ agents doing a DEEP, NARROW literature review for the
TAGSeq2TAGSeq paper. Before searching, READ:
1. /fss/evin_t/tagseq2tagseq/paper/PROJECT_SUMMARY_for_litreview.md (what the project is)
2. /fss/evin_t/tagseq2tagseq/paper/notes/ROUND1_KEYS.txt (118 keys already collected — do NOT
   repeat these; go deeper/adjacent/newer and ADD)
3. the specific code brief(s) named in your task under
   /fss/evin_t/tagseq2tagseq/paper/notes/code_briefs/ (these contain the EXACT mechanisms +
   specific methods/arXiv ids the paper uses — anchor your search on them)

The project in one line: pretrain a decoder-only LM on a TEXT-ATTRIBUTED GRAPH (documents =
nodes; hyperlinks/imports/citations = edges) by (a) packing graph-topologically-close docs into
one 32k sequence via graph traversal, and (b) a custom sparse attention mask that grants a
linking doc read-access into the doc it links to — used in BOTH pretraining AND inference
(a generated link fetches the target doc into the attention context). Compute-control masks
(concat variants) isolate the linking inductive bias from raw FLOPs. Custom Triton/FlexAttention
kernels; Muon(NorMuon+Polar Express) optimizer; modded-nanoGPT lineage; merged 11-source model.

CONTRAST AXIS for every note: does the work TRAIN on the linked structure, or only use
links/retrieval at INFERENCE? and does it use cached KV / an attention edge / a GNN message-passing
edge / a training-pair signal? State the precise relationship to TAGSeq2TAGSeq.

RETURN: a single markdown doc — one ```bibtex fenced block with 6-15 VERIFIED entries (correct
authors/year/venue/arXiv id; verify each against arxiv.org / ACL Anthology / DBLP / the paper —
do NOT hallucinate; FLAG anything unverified), then "## Notes" keyed by cite-key with a 2-4
sentence relationship note each. Prefer canonical + recent. Deduplicate against ROUND1_KEYS.txt.
Your text IS the deliverable (returned to the orchestrator, not shown to a user).
