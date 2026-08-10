## saier2023unarxive — unarXive 2022: All arXiv Publications Pre-Processed for NLP, Including Structured Full-Text and Citation Network

Saier, Krause & Färber, JCDL 2023 (arXiv:2303.14957). This is OUR scientific
citation-graph corpus: it supplies the paper *nodes* (structured LaTeX full-text)
and the raw `\cite` markers we turn into graph *edges*. The comparison that matters
here is not architectural (it is a dataset, not a method) but a **data-provenance and
citation-resolution-coverage** comparison: what the corpus resolves, what it drops,
and how *our* extraction layer re-resolves citations on top of it.

> Provenance caveat worth flagging up front: our in-repo arXiv extractor
> (`data/arxiv_graph_extractor/extract.py`, `measure_density.py`) and its docstrings
> say **"unarXive 2024"** / `processed_unarxive_extended_data`, i.e. we actually run
> against a *later* (2024 "extended") release of the unarXive line, not the exact 2022
> snapshot this paper describes. The pipeline, format, and resolution mechanism are the
> ones this paper introduced; the concrete corpus we ingested is a newer drop of it.
> Every number I cite as "the paper's" is from the 2022 paper; every number I cite as
> "ours" is from our own logs against the release we ingested. I did not find an
> in-repo `extract_summary.json`/`arxiv_density_report.json` to confirm our edge count
> (the filesystem search timed out), so our edge/degree figures below are marked
> unverified where they are not in a log.

### What the paper actually does

unarXive 2022 is a full re-processing of essentially all of arXiv into a
machine-readable NLP corpus with three things its predecessors lacked together:
structured full-text, preserved mathematical/non-text content, and a resolved
in-text citation network. Confirmed figures from the paper:

- **Scale.** 1,881,346 papers ("1.9 M"), spanning 32 years, ~182.6 M paragraphs.
  Disciplinary skew is heavy: ~57% physics, ~20% mathematics, ~17% computer science,
  ~5% everything else. Only the permissively-licensed subset (~165k papers, ~9%) is
  openly redistributable; the full set is restricted-access.
- **Reference/citation volume.** 63,367,836 bibliography entries and 133,744,613
  in-text citation markers across the corpus.
- **Resolution (the key number for us).** Of those, **28,135,565 references (44.4%)**
  and **64,547,944 in-text citation markers (48.3%)** are successfully linked to a
  target work. A citation network edge exists only for a *linked* reference.
- **Text pipeline.** Three stages, following S2ORC-LaTeX and unarXive 2020:
  (1) `latexpand` flattens each submission's LaTeX into one `.tex`; (2) **Tralics**
  converts LaTeX→XML; (3) XML→JSON. The JSON keeps section title/number/type and
  nesting, content-block types (paragraph/listing/proof/…), and non-text content in a
  `ref_entries` table — **formulas retained as LaTeX** (`{"type":"formula","latex":...}`),
  figures/tables linked to captions. This is exactly the placeholder scheme our
  extractor rehydrates (`{{formula:uuid}}`, `{{figure|table:uuid}}`, `{{cite:hexkey}}`).
- **Reference resolution method.** Bibliographies are parsed with **GROBID** (replacing
  unarXive 2020's Neural-ParsCit), yielding title/authors/year/venue plus
  heuristically-extracted identifiers (DOI, arXiv id) across more citation styles. The
  parsed reference is matched against **OpenAlex** metadata (replacing Microsoft
  Academic Graph, discontinued). An OpenAlex hit carries DOI/PMID/arXiv-id, which is
  how a `\cite` becomes a resolvable target. The paper reports **no precision/recall/F1**
  for match quality — 44.4%/48.3% are *coverage*, not accuracy.
- **Vs. predecessors (paper's Table 1, arXiv 1991–2020 directly-comparable column).**
  unarXive 2022 44.4% citation-network completeness vs unarXive 2020 42.6% vs
  S2ORC-LaTeX 31.1%; sizes 1.9 M vs 1.2 M vs 1.5 M docs. S2ORC's headline 69.4% is
  its *PDF* pipeline over 12 M docs and is explicitly not comparable (no LaTeX/math
  structure). Takeaway: unarXive 2022 is the most-complete *LaTeX-structured* arXiv
  citation graph available, but ~55% of references still resolve to nothing.

### How WE turn it into training edges (vs. what the corpus hands us)

Our extractor (`data/arxiv_graph_extractor/extract.py`, verified) does **not** trust
the corpus's own citation network wholesale. It re-derives edges with a stricter,
*in-corpus-only* rule and a two-tier resolver:

1. **Direct arXiv id** — `bib_entries[*].contained_arXiv_ids` / `.ids.arxiv_id`,
   canonicalized (version stripped, `canonical_arxiv_id`). This is the ~**14.5%** of
   citations that carry a usable arXiv id directly. `measure_density.py` was our Phase-0
   gate that measured exactly this direct-id-only density.
2. **OpenAlex bridge** — `bib_entries[*].ids.open_alex_id` → our own
   `arxiv_openalex_map.jsonl` (built by `build_openalex_map.py`, streaming the ~639 GB
   OpenAlex `works` snapshot and keeping every work with an arXiv source location) →
   arXiv id. This lifts coverage from ~14.5% to ~**66%** *of citations we can map to an
   in-corpus arXiv target* (per the data-pipelines brief and the extract.py docstring).

Crucial differences from the corpus's native network:

- **Edge iff resolved AND in-corpus.** `_resolve_bib_to_arxiv` returns an id only if it
  is a key in `_ARXIV_TO_TITLE` (a titled, in-corpus paper). A citation to a real paper
  that is outside our node set produces **no edge**.
- **Out-of-corpus `\cite` is *deleted from the text*.** `sub_cite` returns `""` for an
  unresolved/out-of-corpus marker — the citation marker is removed entirely (no dangling
  `\cite{}`, no textual noise, no edge). In-corpus markers are rewritten to LaTeX-native
  `\cite{<cited paper's title>}`. This is deliberate: the link target the model learns
  to emit is the **title string** (`raw_identifier`), and `ArxivCiteDetector`
  (`model/graph_traversal/arxiv_cite_detector.py`) re-detects `\cite{Title}` at runtime
  and matches it verbatim against the target node's `raw_identifier`. Link positions are
  never stored — re-detected from tokens, per the detector-must-agree-with-extractor
  invariant.
- **Title-collision disambiguation.** When two papers share a title, the second gets
  `" (arxiv:YYYY.NNNNN)"` appended so the match stays exact and unique (~0.13% of nodes,
  logged as "title collisions"). Node identity itself is by `normalize_arxiv(paper_id)`
  = version-stripped canonical id + 6-char hash, so nodes never collide even when titles
  do; the title is only the *fetch key*.

Why the OpenAlex bridge exists at all is important for our results narrative: the
`build_openalex_map.py` docstring states the direct-id graph is **hub-dominated** — a
few universally-cited papers form a 58.7% giant component that *fragments under the
training no-repeats rule*. Mapping OpenAlex ids recovers non-hub **lateral** edges, which
is what makes BFS community-packing produce diverse, non-degenerate packs rather than
star graphs around a handful of megacited papers.

**Our realized graph (from our logs, verified):** the ingested arXiv graph has
**1,976,517 nodes** (`slurm_logs/precompute_arxiv*.log`) — slightly *more* than the 2022
paper's 1.88 M, consistent with our using a newer "extended" release. The val_community
split alone (from `docs/handoff_arxiv_commpack_sparsity.md`, verified) holds 55,026 nodes
with 302,003 in-split outgoing edges and 46,629 nodes with ≥1 in-split outgoing edge —
i.e. the graph is dense enough to pack, contrary to an early "arxiv is too sparse" scare
that turned out to be an eval-budget bug (see Gotchas). I did not find a corpus-wide
`extract_summary.json`, so corpus-wide edge count / mean out-degree for our run is
**unverified** here.

### Citation-resolution coverage/bias — what it implies for the arXiv arm

This is the crux the task asks for. Coverage is not missing-at-random; the losses stack
and they bias *which* edges survive into training:

1. **Two multiplicative funnels.** The paper resolves ~44–48% of citations to *any*
   target. On top of that, **we keep an edge only if the target is in-corpus** (an arXiv
   paper that is itself a node). Citations to non-arXiv venues (journals, books, NeurIPS/
   ACL proceedings without an arXiv preprint) resolve in OpenAlex but map to **no arXiv
   id** and are dropped. So our in-corpus edge yield (~66% of citations *per our
   pipeline's own framing*, which counts only arXiv-resolvable citations) sits on top of
   the corpus's own ~44% reference-resolution — the effective fraction of *all* real
   references that become trainable edges is lower than either number alone.
2. **Discipline bias inherited and amplified.** The corpus is 57% physics / 20% math /
   17% CS. arXiv-to-arXiv citation is far denser in ML/CS/hep than in math or in fields
   that cite books/journals. Our in-corpus rule therefore over-represents the
   arXiv-native, preprint-heavy subfields and under-represents math and cross-domain
   citation. The arXiv arm's cross-doc benefit will be **concentrated in the subfields
   where arXiv-to-arXiv citation is the norm**.
3. **Recency/venue bias.** arXiv ids in references are far more common for recent
   preprints; older or non-preprint work resolves less often, so surviving edges skew
   recent and skew toward work that circulates as preprints.
4. **Silent text deletion changes the node text, not just the graph.** Because
   out-of-corpus `\cite` markers are *removed from the body*, a physics paper that cites
   40 journal articles and 3 arXiv preprints is presented to the model as prose with 37
   citation markers excised. This is cleaner than dangling markers but it means the
   *node text itself* is a function of resolution coverage — a systematic, discipline-
   correlated perturbation of the training text, not only of the edge set.
5. **No resolution-quality metric anywhere.** The paper reports coverage but no
   precision. Our pipeline adds no verification either (direct-id is trusted; OpenAlex
   map is trusted). A GROBID mis-parse or an OpenAlex mismatch yields a **wrong edge**
   (paper A "cites" the wrong paper B) that is indistinguishable from a correct one. The
   data-pipelines brief flags "arXiv resolution rate UNMEASURED beyond coverage" as
   reviewer-attackable. Note the wiki arm's resolution rate is *entirely* unlogged, so
   arXiv is actually our **best-instrumented** source here — but only for coverage.

**Implication for the arXiv results arm.** Expect the cross-doc-link effect on arXiv to
be **real but the smallest of the three source families** — and our own re-run confirms
this: community-pack Δ (cross_doc_link vs doc_causal) is **+0.0039, 95% CI
[0.0023, 0.0059]** — strictly positive but an order of magnitude below wiki (+0.1595)
and well below thestack (+0.0761) (`docs/handoff_arxiv_commpack_sparsity.md`, verified).
Two structural reasons, both traceable to this dataset: (a) arXiv docs are huge (median
~14.7k tok/doc), so at 32k only ~44.4% of linked pairs even *fit together* in one pack,
capping how many edges can be exercised per sequence; (b) the resolved edge set is
sparser and more hub-biased than wiki's hyperlink graph. The corpus's own incompleteness
(~55% of references unlinked) plus our in-corpus filter means the arXiv graph the model
sees is a **lower-recall shadow** of the true citation network — which bounds the ceiling
of any cross-document effect we can measure on it.

### Methodology: theirs vs. ours

unarXive is data, so the mechanistic axes from the brief apply only to what the *edge*
is once we build it. The relevant contrasts:

- **Resolution = deterministic hashmap, not learned retrieval.** Both the corpus (GROBID
  parse → OpenAlex match) and our layer (id/OpenAlex → in-corpus title) resolve citations
  by *symbolic identifier lookup*, not embedding similarity. This is consistent with our
  whole thesis: the "retrieval" is exact identifier resolution, and at inference a
  generated `\cite{Title}` is matched verbatim (`ArxivCiteDetector.index_doc_span`
  returns `raw_identifier`) and the target node is fetched into attention — no ANN, no
  scoring. Contrast Galactica (**taylor2022galactica**), which memorizes reference
  strings in-weights and generates `[START_REF]` without ever attending into the cited
  document.
- **Train-on-structure, not retrieve-at-inference.** The citation edge enters as a
  block in our `cross_doc_link` attention mask (`configs/arxiv_cross_doc.yaml`,
  `mask_type: cross_doc_link`), used identically in pretraining and generation. The four
  arXiv configs form the compute-control ladder: `cross_doc_link` (edge granted) vs
  `doc_causal` (BFS-packed, no cross-doc attention — the paired baseline) vs
  `doc_concatenated` / `doc_concat_link` (concat variants isolating packing/FLOPs from
  the linking bias). The corpus's citation network is never consumed as a GNN or a
  message-passing structure — it is only the *source* of which title-string a link token
  resolves to.
- **What we share with the corpus vs where we diverge.** Shared: node = paper full-text,
  edge = `\cite`, resolution via OpenAlex. Diverged: the corpus *keeps* every resolved
  reference including out-of-arXiv targets (its network points into all of OpenAlex);
  **we keep only arXiv→arXiv, in-corpus edges and delete the rest from the text**. Our
  graph is therefore an induced subgraph of the corpus's citation network, restricted to
  the arXiv-node-set, further filtered by our two-tier resolver's own recall.

### Predictions & open questions for our method

- **Effect strength tracks arXiv-to-arXiv citation density.** Predict the cross-doc-link
  gain is largest on CS/ML and hep papers (dense preprint-to-preprint citation) and
  weakest on math and applied/experimental papers that cite books/journals. A useful
  ablation: stratify the arXiv community-pack Δnll by the paper's `categories` field
  (we store it per node) — the effect should be visibly discipline-dependent. This is a
  cheap, high-value robustness slice we already have the metadata for.
- **Doc size caps the edge budget.** Because only ~44% of linked pairs fit in 32k, the
  arXiv arm is the one most likely to *underestimate* the true structural effect. If we
  ever raise context length, arXiv should benefit disproportionately relative to wiki
  (whose docs are small and nearly all co-fit). Predict the arXiv Δ scales with context
  length faster than wiki's.
- **Hub domination is a live failure regime.** The 58.7% giant-component / hub warning
  from `build_openalex_map.py` predicts that BFS/traversal packing on arXiv, if seeded
  naively, will keep re-fetching a few megacited survey papers. Watch for degree-skew in
  the packs; the OpenAlex lateral edges are the mitigation and should be verified to
  actually diversify packs (measure pack-internal degree distribution).
- **Resolution recall is a knob on the effect, not just noise.** Their open question
  (can citation-network completeness be pushed past ~44%?) is one our OpenAlex bridge
  *partially* answers for the arXiv-restricted case. Conversely, our design raises a
  question their static dataset can't: does a model trained on a *higher-recall* edge set
  (more of the true citations exercised) show a proportionally larger cross-doc gain?
  That would be direct evidence that the effect is edge-density-limited rather than
  saturated.

### Gotchas

- **The "arXiv is too sparse to pack" scare was an eval bug, not a data fact**
  (`docs/handoff_arxiv_commpack_sparsity.md`, verified). `run_community_pack_perplexity`
  silently fell back to a hardcoded **2048** token budget instead of 32768; arXiv's huge
  docs collapsed to n=5 scoreable packs and looked catastrophically sparse. Fixed
  2026-08-01 → n=392, Δ strictly positive. **Lesson for the paper: never read an
  arXiv-arm null as a data-sparsity result without checking the pack budget actually
  used.** Any per-source eval must resolve `backbone.max_seq_len`, not `model.max_seq_len`
  (which doesn't exist).
- **Coverage ≠ correctness.** 44.4%/48.3% (theirs) and ~66% (ours) are *linking rates*,
  not accuracy. A reviewer can ask "how many of your edges are wrong?" and neither the
  corpus nor our pipeline has an answer. Consider a small hand-audited precision estimate
  on a random edge sample before claiming the arXiv graph is high-quality.
- **We ingested a newer release than the cited paper.** Cite key is unarXive 2022 but
  the code targets `processed_unarxive_extended_data` ("2024"). If the paper's Methods
  section cites saier2023unarxive as *the* corpus, either (a) confirm the exact release
  we tokenized and cite it, or (b) note explicitly that we used a later extended drop of
  the unarXive line. The 1,976,517-node count (vs the paper's 1.88 M) is the tell.
- **Silent text deletion is a hidden confound.** Removing out-of-corpus `\cite` markers
  edits the node text in a discipline-correlated way. If we ever compare arXiv against an
  edgeless flat baseline (e.g. peS2o), the baseline text will contain those markers and
  ours won't — a text-distribution difference that is *not* the linking bias. Match the
  text preprocessing across conditions or the comparison is contaminated.
- **Non-arXiv-native fields are near-invisible.** Biomedical/chemistry/economics work is
  both scarce in arXiv and cites non-arXiv venues; the arXiv arm says essentially nothing
  about those domains. Don't over-generalize an arXiv result to "scientific text."
- **Title as fetch-key is brittle to normalization.** `ArxivCiteDetector` matches the
  decoded `\cite{Title}` byte-for-byte against `raw_identifier`. Titles are whitespace-
  collapsed at extraction (`re.sub(r"\s+"," ",title)`); any tokenizer round-trip that
  perturbs whitespace/unicode in a title silently breaks the match (no edge fires at
  inference). This is the arXiv analog of the wiki raw-vs-hashed asymmetry.

### Missed citations worth adding

I checked `paper/bib/refs.bib`. Already present: saier2020unarxive, saier2023unarxive,
lo2020s2orc, soldaini2023pes2o, priem2022openalex, sinha2015mag, zhang2019oag,
ammar2018literaturegraph, clement2019arxiv, hu2020ogb, sen2008collective. Genuinely
missing tooling/method works that this paper *depends on* and that matter to our
resolution pipeline:

- **GROBID** — Lopez, "GROBID: Combining Automatic Bibliographic Data Recognition and
  Extraction..." (2009; software widely cited as Lopez 2009 / the GROBID repo). The
  actual reference-parser behind unarXive 2022's citation resolution and thus behind the
  edges we train on. arXiv id: none (software; JCDL/TPDL 2009 paper). Matters: it is the
  upstream determinant of our arXiv edge recall/precision — worth one citation when we
  describe resolution provenance.
- **Tralics** — Grimm, "Tralics, a LaTeX to XML translator" (2003, INRIA report). The
  LaTeX→XML converter producing the structured full-text (formulas-as-LaTeX, sections)
  that constitutes our arXiv *node text*. No arXiv id. Matters: explains why our node
  text is faithful LaTeX with retained math, which is unusual vs PDF-parsed corpora.
- **latexpand** — the LaTeX-flattening step (part of the TeXLive `latexpand` utility; no
  paper). Minor; mention only if we detail the pipeline. Likely *not* worth a bib entry.
- **Färber & Lauscher / Neural-ParsCit line** — unarXive 2020 used Neural-ParsCit
  (Prasad et al., "Neural ParsCit", 2018) for reference parsing; the 2022 paper's move to
  GROBID is a methodological delta. Prasad et al. 2018, arXiv:1809.??? (Neural ParsCit,
  Int. J. Digital Libraries). Only worth adding if we discuss the parser-quality
  evolution; otherwise skip.

(I did not find GROBID or Tralics in refs.bib; verify before adding. The S2ORC / MAG /
OAG / OpenAlex / Semantic-Scholar-graph cluster the brief would expect is already
present, so nothing to add there.)

---
Confirmed against sources: paper numbers via arXiv abstract + ar5iv full text
(2303.14957); our pipeline via `data/arxiv_graph_extractor/{extract,build_openalex_map,measure_density}.py`,
`model/graph_traversal/arxiv_cite_detector.py`, `data/normalization.py`, arXiv configs,
and `slurm_logs/precompute_arxiv*.log` + `docs/handoff_arxiv_commpack_sparsity.md`.
Unverified (flagged inline): corpus-wide edge count/mean-degree for our ingested release;
the exact unarXive release version behind `processed_unarxive_extended_data`.
