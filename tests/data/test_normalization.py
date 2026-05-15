"""
Tests for data.normalization — the canonical identifier normalization pipeline.
"""
import hashlib
import pytest

from data.normalization import (
    identifier_hash,
    normalize_wiki_title,
    normalize_repo_name,
    normalize_package_name,
    strip_hash,
)


def _raw_hash(raw: str) -> str:
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:6]


class TestIdentifierHash:
    def test_length(self):
        assert len(identifier_hash("anything")) == 6

    def test_hex(self):
        h = identifier_hash("Test")
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert identifier_hash("abc") == identifier_hash("abc")

    def test_hashes_raw(self):
        # Must hash the original raw string, not a normalized form
        assert identifier_hash("A+B") == _raw_hash("A+B")
        assert identifier_hash("A+B") != identifier_hash("A-B")


class TestNormalizeWikiTitle:
    def test_basic(self):
        result = normalize_wiki_title("Python programming")
        h = _raw_hash("Python programming")
        assert result == f"python_programming_{h}"

    def test_html_entities_decoded(self):
        result = normalize_wiki_title("C&amp;C Studio")
        h = _raw_hash("C&amp;C Studio")
        assert result == f"c_c_studio_{h}"

    def test_hyphens_preserved(self):
        result = normalize_wiki_title("Python-3")
        h = _raw_hash("Python-3")
        assert result == f"python-3_{h}"

    def test_special_chars_become_underscore(self):
        result = normalize_wiki_title("C++ Tutorial")
        h = _raw_hash("C++ Tutorial")
        assert result == f"c_tutorial_{h}"

    def test_underscores_collapsed(self):
        result = normalize_wiki_title("Multiple   Spaces")
        h = _raw_hash("Multiple   Spaces")
        assert result == f"multiple_spaces_{h}"

    def test_leading_trailing_stripped(self):
        result = normalize_wiki_title("  Title  ")
        h = _raw_hash("  Title  ")
        assert result == f"title_{h}"

    def test_length_cap(self):
        long_raw = "A" * 300
        result = normalize_wiki_title(long_raw)
        body = strip_hash(result)
        assert len(body) <= 193

    def test_hash_of_raw_not_normalized(self):
        # Two titles that normalize to the same body must differ by hash
        r1 = normalize_wiki_title("A+B")
        r2 = normalize_wiki_title("A-B")
        h1 = _raw_hash("A+B")
        h2 = _raw_hash("A-B")
        assert r1.endswith(f"_{h1}")
        assert r2.endswith(f"_{h2}")
        assert h1 != h2


class TestNormalizeRepoName:
    def test_slash_to_underscore(self):
        result = normalize_repo_name("user/repo")
        h = _raw_hash("user/repo")
        assert result == f"user_repo_{h}"

    def test_dash_to_underscore(self):
        result = normalize_repo_name("my-user/my-repo")
        h = _raw_hash("my-user/my-repo")
        assert result == f"my_user_my_repo_{h}"

    def test_lowercase(self):
        result = normalize_repo_name("MyUser/MyRepo")
        assert result == result.lower()

    def test_six_char_hash(self):
        result = normalize_repo_name("user/repo")
        parts = result.rsplit("_", 1)
        assert len(parts[1]) == 6

    def test_hash_of_raw(self):
        repo = "Phil65/prettyqt"
        result = normalize_repo_name(repo)
        assert result.endswith(f"_{_raw_hash(repo)}")

    def test_deterministic(self):
        assert normalize_repo_name("user/repo") == normalize_repo_name("user/repo")


class TestNormalizePackageName:
    def test_dot_to_underscore(self):
        result = normalize_package_name("os.path")
        h = _raw_hash("os.path")
        assert result == f"os_path_{h}"

    def test_lowercase(self):
        result = normalize_package_name("MyPackage")
        assert result.startswith("mypackage_")

    def test_hash_of_raw(self):
        pkg = "os.path"
        result = normalize_package_name(pkg)
        assert result.endswith(f"_{_raw_hash(pkg)}")

    def test_deterministic(self):
        assert normalize_package_name("os.path") == normalize_package_name("os.path")


class TestStripHash:
    def test_strips_valid_suffix(self):
        assert strip_hash("python_a7f8c3") == "python"
        assert strip_hash("test_title_123abc") == "test_title"

    def test_no_suffix(self):
        assert strip_hash("no_hash_here") == "no_hash_here"

    def test_too_short_not_stripped(self):
        assert strip_hash("test_12345") == "test_12345"

    def test_non_hex_not_stripped(self):
        assert strip_hash("test_gggggg") == "test_gggggg"
