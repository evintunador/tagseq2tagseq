"""HotpotQA bridge cross-doc port.

Maps HotpotQA (fullwiki) bridge questions into the canonical CrossDocExample
schema, reproducing field-for-field the item construction of
eval/nlp_benchmarks.py::run_hotpotqa_cross_doc:
  * context  = article A's supporting sentences, HTML links → markdown so the
    MarkdownLinkDetector fires on the natural [text](B title) link, followed by
    the "\\nQuestion: ...\\nAnswer: " scaffold.
  * target   = the answer.
  * aux      = article B's supporting sentences as plain text; its
    raw_identifier is B's title (== the markdown detector's extracted
    target_str, matched exactly by MarkdownLinkDetector.index_doc_span).

Routing this port through the Tier-2 harness gives HotpotQA the same
placebo/derangement control the code ports already have: the harness swaps in a
DIFFERENT bridge example's article-B text under this example's own B title, so
the grant still fires but attention reads the wrong supporting article.

Only the NATIVE scope is meaningful — this is QA, not code, so there are no
aux-symbol use-sites; full_file is left None and use-scopes are unsupported.

The corpus/link pre-filters (both supporting articles present; a rendered
](B title) marker in article A) are applied here so emitted examples can fire,
keeping the harness fire-rate meaningful — exactly as run_hotpotqa_cross_doc
pre-filters before counting.
"""
from __future__ import annotations

from typing import List, Optional

from ..schema import AuxDoc, CrossDocExample, PortAdapter


def _load_hotpotqa_bridge(max_examples: Optional[int]) -> List[CrossDocExample]:
    from eval.nlp_benchmarks import (
        _load_hotpotqa_corpus,
        _hotpotqa_bridge_examples,
        _hotpotqa_titles,
        _hotpotqa_supporting_sent_ids,
        _html_links_to_markdown,
        _strip_html_links,
    )

    corpus = _load_hotpotqa_corpus()
    # Pull extra raw examples: many are dropped by the corpus/link pre-filters,
    # so cap emitted examples separately from the raw scan (None = all bridge).
    raw = _hotpotqa_bridge_examples(None, cache_dir=None)

    out: List[CrossDocExample] = []
    for ex in raw:
        a_title, b_title = _hotpotqa_titles(ex)
        if a_title is None or b_title is None:
            continue
        a_sents_raw = corpus.get(a_title.lower())
        b_sents_raw = corpus.get(b_title.lower())
        if a_sents_raw is None or b_sents_raw is None:
            continue

        a_ids = _hotpotqa_supporting_sent_ids(ex, a_title)
        b_ids = _hotpotqa_supporting_sent_ids(ex, b_title)

        def _pick_raw(sents, ids):
            picked = [sents[i] for i in ids if i < len(sents)]
            return picked if picked else [sents[0]]

        a_sents_md = [_html_links_to_markdown(s) for s in _pick_raw(a_sents_raw, a_ids)]
        b_sents_plain = [_strip_html_links(s) for s in _pick_raw(b_sents_raw, b_ids)]

        # Pre-filter: article A must contain a rendered ](B title) link, else the
        # grant can never fire and the example is identical to the flat baseline.
        marker = f"]({b_title})"
        if not any(marker in s for s in a_sents_md):
            continue

        a_text_md = " ".join(a_sents_md)
        b_text_plain = " ".join(b_sents_plain)
        context = a_text_md + "\nQuestion: " + ex["question"] + "\nAnswer: "

        out.append(CrossDocExample(
            repo="hotpotqa",
            file_path=a_title,
            context=context,
            target=ex["answer"],
            aux=(AuxDoc(path=b_title, content=b_text_plain),),
            meta={"id": ex.get("id"), "level": ex.get("level"),
                  "a_title": a_title, "b_title": b_title},
            full_file=None,
        ))
        if max_examples is not None and len(out) >= max_examples:
            break
    return out


def _markdown_detector(decode_fn):
    from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
    return MarkdownLinkDetector(decode_fn)


HOTPOTQA_BRIDGE = PortAdapter(
    name="hotpotqa_bridge",
    language="markdown",
    examples_fn=_load_hotpotqa_bridge,
    # The markdown detector matches raw_identifier == the [text](title) target,
    # i.e. article B's title, which we stored as the AuxDoc path.
    identifier_fn=lambda repo, path, content: path,
    detector_factory=_markdown_detector,
)
