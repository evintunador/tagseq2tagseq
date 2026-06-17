"""
Canonical identifier normalization for TAGSeq2TAGSeq datasets.

All normed_identifier values written to graph.jsonl and used in the pretokenized
corpus are produced by the functions here. Use these at both data-extraction time
and model inference time so that lookup keys always match.

Three flavors cover the three corpus types:
  normalize_wiki_title    — Wikipedia article titles
  normalize_repo_name     — GitHub repository names (user/repo)
  normalize_package_name  — Python package/module names (dot-separated)

All three append a 6-char MD5 hash of the *original raw string* so that titles
that differ only in punctuation (e.g. "A+B" vs "A-B", both body → "a_b") remain
distinguishable.
"""
import hashlib
import html
import re


def identifier_hash(raw: str) -> str:
    """Return a 6-character lowercase hex MD5 hash of the raw (unnormalized) string."""
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:6]


def _norm_body(text: str) -> str:
    """Shared normalization body applied after any source-specific pre-processing.

    Steps: lowercase + strip + spaces→underscores, replace non-[a-z0-9-_] with
    underscores, collapse repeated underscores, strip leading/trailing underscores.
    Hyphens are preserved so that hyphenated titles round-trip correctly.
    """
    text = text.lower().strip().replace(" ", "_")
    text = re.sub(r"[^a-z0-9\-_]", "_", text)
    text = re.sub(r"__+", "_", text)
    text = text.strip("_")
    return text


def normalize_wiki_title(raw: str) -> str:
    """
    Normalize a Wikipedia article title to a normed_identifier string.

    Pipeline: HTML-unescape → _norm_body → cap body at 193 chars →
    append '_' + identifier_hash(original raw).
    """
    h = identifier_hash(raw)
    decoded = html.unescape(raw)
    body = _norm_body(decoded)
    if len(body) > 193:
        body = body[:193]
    return f"{body}_{h}"


def normalize_repo_name(raw: str) -> str:
    """
    Normalize a GitHub repository name (e.g. 'user/repo-name') to a normed_identifier.

    Slashes and dashes are replaced with underscores before _norm_body runs.
    Hash is of the original raw string (before any transformation).
    """
    h = identifier_hash(raw)
    pre = raw.replace("/", "_").replace("-", "_")
    body = _norm_body(pre)
    return f"{body}_{h}"


def normalize_package_name(raw: str) -> str:
    """
    Normalize a Python package/module name (e.g. 'os.path') to a normed_identifier.

    Dots are replaced with underscores before _norm_body runs.
    Hash is of the original raw string (before any transformation).
    """
    h = identifier_hash(raw)
    pre = raw.replace(".", "_")
    body = _norm_body(pre)
    return f"{body}_{h}"


_ARXIV_VERSION_RE = re.compile(r"v\d+$")


def canonical_arxiv_id(raw: str) -> str:
    """Strip a trailing version suffix (e.g. 'v2') and surrounding whitespace.

    A paper's node and every citation edge that points to it must derive from the
    same id, so all arXiv ids are canonicalized to their version-less form before
    use. New-style ids are 'YYYY.NNNNN'; old-style 'archive/NNNNNNN' canonicalize
    likewise (the version suffix, if any, is removed).
    """
    return _ARXIV_VERSION_RE.sub("", raw.strip())


def normalize_arxiv(raw: str) -> str:
    """
    Normalize an arXiv id (e.g. '2401.12345' or '2401.12345v2') to a normed_identifier.

    The id is canonicalized (version suffix stripped) first, then dots and slashes
    are replaced with underscores before _norm_body runs. The hash is of the
    *canonical* id (not the raw, version-bearing string) so that a paper node and
    any citation edge that references it always produce the same normed_identifier.

    Because arXiv ids are globally unique, this guarantees a unique normed_identifier
    per paper even when two papers share an identical title.
    """
    canon = canonical_arxiv_id(raw)
    h = identifier_hash(canon)
    pre = canon.replace(".", "_").replace("/", "_")
    body = _norm_body(pre)
    return f"{body}_{h}"


def strip_hash(normed: str) -> str:
    """Remove the trailing '_[0-9a-f]{6}' hash suffix from a normed_identifier."""
    return re.sub(r"_[0-9a-f]{6}$", "", normed)
