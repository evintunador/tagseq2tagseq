import hashlib
import json
import re
import pytest
from pathlib import Path

from data.document_sources import (
    MarkdownDirectorySource,
    StackJSONLSource,
    _normalize_repo_name,
)


# ---------------------------------------------------------------------------
# _normalize_repo_name
# ---------------------------------------------------------------------------

class TestNormalizeRepoName:
    def test_slash_becomes_underscore(self):
        assert _normalize_repo_name("user/repo").startswith("user_repo_")

    def test_dash_becomes_underscore(self):
        assert _normalize_repo_name("my-user/my-repo").startswith("my_user_my_repo_")

    def test_lowercase(self):
        result = _normalize_repo_name("MyUser/MyRepo")
        assert result == result.lower()

    def test_appends_six_char_hash(self):
        result = _normalize_repo_name("user/repo")
        parts = result.rsplit("_", 1)
        assert len(parts) == 2
        assert len(parts[1]) == 6
        assert re.fullmatch(r"[0-9a-f]{6}", parts[1])

    def test_hash_is_deterministic(self):
        assert _normalize_repo_name("user/repo") == _normalize_repo_name("user/repo")

    def test_matches_extract_py_logic(self):
        # Reproduce the exact logic from github_graph_extractor/extract.py
        # to ensure the mirror stays in sync.
        repo = "Phil65/prettyqt"
        clean = repo.replace("/", "_").replace("-", "_").lower()
        clean = re.sub(r"[^a-z0-9\-_]", "_", clean)
        clean = re.sub(r"__+", "_", clean)
        clean = clean.strip("_")
        h = hashlib.md5(clean.encode("utf-8")).hexdigest()[:6]
        expected = f"{clean}_{h}"
        assert _normalize_repo_name(repo) == expected


# ---------------------------------------------------------------------------
# MarkdownDirectorySource
# ---------------------------------------------------------------------------

@pytest.fixture
def markdown_dir(tmp_path):
    files = {
        "Alpha": "# Alpha\nContent of alpha with [link](Beta).\nAlpha",
        "Beta": "# Beta\nContent of beta.\nBeta",
        "Gamma": "# Gamma\nSome text here.\nGamma",
    }
    for name, content in files.items():
        (tmp_path / f"{name}.md").write_text(content, encoding="utf-8")
    return tmp_path, files


class TestMarkdownDirectorySource:
    def test_len(self, markdown_dir):
        directory, _ = markdown_dir
        source = MarkdownDirectorySource(directory)
        assert len(source) == 3

    def test_yields_correct_normed_ids(self, markdown_dir):
        directory, _ = markdown_dir
        source = MarkdownDirectorySource(directory)
        normed_ids = {normed_id for normed_id, _ in source}
        assert normed_ids == {"Alpha", "Beta", "Gamma"}

    def test_yields_correct_content(self, markdown_dir):
        directory, raw = markdown_dir
        source = MarkdownDirectorySource(directory)
        results = dict(source)
        assert "Content of beta." in results["Beta"]

    def test_strips_raw_title_marker_from_last_line(self, markdown_dir):
        directory, _ = markdown_dir
        source = MarkdownDirectorySource(directory)
        results = dict(source)
        # Last line (raw title marker) should not appear in yielded content
        assert not results["Beta"].endswith("Beta")
        assert not results["Gamma"].endswith("Gamma")

    def test_raw_link_targets_passed_through_unchanged(self, markdown_dir):
        directory, _ = markdown_dir
        source = MarkdownDirectorySource(directory)
        results = dict(source)
        # Raw link target ](Beta) should pass through verbatim
        assert "](Beta)" in results["Alpha"]

    def test_empty_directory(self, tmp_path):
        source = MarkdownDirectorySource(tmp_path)
        assert len(source) == 0
        assert list(source) == []

    def test_iterable_multiple_times(self, markdown_dir):
        directory, _ = markdown_dir
        source = MarkdownDirectorySource(directory)
        first = list(source)
        second = list(source)
        assert len(first) == len(second) == 3


# ---------------------------------------------------------------------------
# StackJSONLSource
# ---------------------------------------------------------------------------

def _make_jsonl(tmp_path, records):
    path = tmp_path / "sample.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def _graph_title(repo, path):
    return f"{_normalize_repo_name(repo)}:{path}"


@pytest.fixture
def stack_data(tmp_path):
    records = [
        {
            "max_stars_repo_name": "user/repo-alpha",
            "max_stars_repo_path": "src/main.py",
            "content": "def main(): pass",
        },
        {
            "max_stars_repo_name": "user/repo-alpha",
            "max_stars_repo_path": "src/utils.py",
            "content": "def helper(): pass",
        },
        {
            "max_stars_repo_name": "other/project",
            "max_stars_repo_path": "app.py",
            "content": "# app",
        },
        # Record NOT in the graph — should be skipped
        {
            "max_stars_repo_name": "unrelated/repo",
            "max_stars_repo_path": "script.py",
            "content": "print('hi')",
        },
    ]
    jsonl_path = _make_jsonl(tmp_path, records)
    graph_normed_ids = {
        _graph_title("user/repo-alpha", "src/main.py"),
        _graph_title("user/repo-alpha", "src/utils.py"),
        _graph_title("other/project", "app.py"),
    }
    return jsonl_path, graph_normed_ids, records


class TestStackJSONLSource:
    def test_len_equals_graph_normed_ids(self, stack_data):
        jsonl_path, graph_normed_ids, _ = stack_data
        source = StackJSONLSource(jsonl_path, graph_normed_ids)
        assert len(source) == len(graph_normed_ids) == 3

    def test_yields_only_graph_members(self, stack_data):
        jsonl_path, graph_normed_ids, _ = stack_data
        source = StackJSONLSource(jsonl_path, graph_normed_ids)
        titles = {normed_id for normed_id, _ in source}
        assert titles == graph_normed_ids

    def test_does_not_yield_non_graph_records(self, stack_data):
        jsonl_path, graph_normed_ids, _ = stack_data
        source = StackJSONLSource(jsonl_path, graph_normed_ids)
        normed_ids = {normed_id for normed_id, _ in source}
        excluded = _graph_title("unrelated/repo", "script.py")
        assert excluded not in normed_ids

    def test_yields_correct_content(self, stack_data):
        jsonl_path, graph_normed_ids, _ = stack_data
        source = StackJSONLSource(jsonl_path, graph_normed_ids)
        results = dict(source)
        main_title = _graph_title("user/repo-alpha", "src/main.py")
        assert results[main_title] == "def main(): pass"

    def test_iterable_multiple_times(self, stack_data):
        jsonl_path, graph_normed_ids, _ = stack_data
        source = StackJSONLSource(jsonl_path, graph_normed_ids)
        first = list(source)
        second = list(source)
        assert sorted(nid for nid, _ in first) == sorted(nid for nid, _ in second)

    def test_empty_graph_normed_ids_yields_nothing(self, stack_data):
        jsonl_path, _, _ = stack_data
        source = StackJSONLSource(jsonl_path, set())
        assert list(source) == []

    def test_skips_records_missing_fields(self, tmp_path):
        records = [
            {"max_stars_repo_name": "", "max_stars_repo_path": "x.py", "content": "x"},
            {"max_stars_repo_name": "user/repo", "max_stars_repo_path": "", "content": "x"},
            {"max_stars_repo_name": "user/repo", "max_stars_repo_path": "x.py", "content": ""},
        ]
        jsonl_path = _make_jsonl(tmp_path, records)
        # Even if all titles were in the graph, empty fields should be skipped
        all_titles = {_graph_title("user/repo", "x.py")}
        source = StackJSONLSource(jsonl_path, all_titles)
        assert list(source) == []
