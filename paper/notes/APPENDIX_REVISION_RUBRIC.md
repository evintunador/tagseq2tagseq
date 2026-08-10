# Rubric for revising an appendix (each agent: ONE appendix file only)

You are improving a standalone appendix of the TAGSeq2TAGSeq paper. Read, in order:
1. The appendix .tex file you are assigned (this is what you edit, in place, with Write).
2. Its code brief in /fss/evin_t/tagseq2tagseq/paper/notes/code_briefs/.
3. The ACTUAL SOURCE FILES listed in your task — grep/read them to verify claims and mine missing detail.
4. /fss/evin_t/tagseq2tagseq/paper/notes/REVIEWER_SHARED_BRIEF.md for the one-line project framing.

Apply this rubric:

CUT (unnecessary / not paper-register):
- Internal identifiers presented as if meaningful to a reader: bare file names, class names,
  variable names, kernel version tags (v10/v12/v18), config-key names, run-dir names, commit
  hashes. Describe the MECHANISM, not the symbol. (A single parenthetical "(implemented in
  <file>)" is fine once; a wall of them is not.)
- "the code does X", "the brief says", "TODO", "we flag", chatty hedging, first-person process
  narration. Use present-tense declarative scientific voice.
- Provenance/history of dead ends and version evolution UNLESS it teaches something (a
  documented NaN bug that motivates a design choice is worth one sentence; a changelog is not).
- Duplication with the main text or other appendices (traversal↔density↔masks overlap: keep
  each fact in exactly one appendix; cross-reference with \Cref instead of repeating).

FIX (phrasing/rigor):
- Turn "we do the same thing but faster" into precise statements (what op, what complexity,
  what was measured vs argued). Distinguish MEASURED numbers (say on what hardware, and that
  they are as-measured, not re-run here) from analytic/claimed properties.
- Define notation before use; make any algorithm block self-contained and reader-followable.
- Hedge honestly on unverified/limitation points, but in reviewer-facing register.

ADD (missing & worth it — find in source):
- Concrete parameters a reader needs to reproduce or judge: block sizes, budgets, caps,
  default values, complexity, memory footprints, the exact predicate/formula. Pull real numbers
  from the source rather than leaving them vague.
- Any correctness-critical invariant or subtlety in the code that the appendix omits and a
  reviewer would want (edge cases, what guarantees a claim, where it can break).
- A crisp statement of what is novel here vs standard, and the contrast to prior art already
  in the paragraph.

CITATIONS: only use \cite keys that already exist in /fss/evin_t/tagseq2tagseq/paper/bib/refs.bib
(grep it to confirm). If you want a cite you can't confirm, write \todo{cite ...} — NEVER invent a key.
Keep the existing \section/\label. Do NOT edit any file other than your assigned appendix.
Keep it tight: aim to make it SHORTER and denser, not longer. Return a 3-5 line changelog of what you cut/fixed/added.
