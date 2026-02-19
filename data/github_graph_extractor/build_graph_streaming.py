#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GitHub Graph Builder (Streaming Version): Build intra-repository dependency graphs
from GitHub repository data with memory-efficient streaming processing.

This module processes large GitHub repository datasets to create dependency graphs
showing file-to-file relationships within the same repository. The streaming
architecture handles datasets of any size efficiently using:

- Batched processing with multiprocessing for Phase 1 import extraction
- Hash-based bucket partitioning for memory-efficient Phase 2 graph construction
- Parallel bucket processing using ProcessPoolExecutor
- Optional fast JSON parsing via orjson (falls back to stdlib json)

Statistics computation provides exact metrics for final graph nodes only,
computed via a second streaming pass over the input file for accuracy.
"""

import argparse
import json as stdlib_json  # Always use stdlib json for file writing
import logging
import os
import tempfile
import gc
import re
import hashlib
import heapq
from collections import Counter
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from timeit import default_timer

from extract import extract_file_imports, normalize_repository_name


# =============================================================================
# Optional faster JSON
# =============================================================================
try:
    import orjson as _json

    def json_loads(s: str):
        return _json.loads(s)

    def json_dumps(obj) -> str:
        return _json.dumps(obj, option=_json.OPT_SORT_KEYS).decode("utf-8")

    JSON_BACKEND = "orjson"
except Exception:
    import json as _json

    def json_loads(s: str):
        return _json.loads(s)

    def json_dumps(obj) -> str:
        return _json.dumps(obj, sort_keys=True)

    JSON_BACKEND = "json"


# =============================================================================
# Helpers
# =============================================================================
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\S+")


def _det_hash32(s: str) -> int:
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def _median_fallback(a):
    """Median like numpy (avg of two middle elements for even length)."""
    if not a:
        return 0.0
    s = sorted(a)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _word_count(content: str) -> int:
    return len(_WORD_RE.findall(content)) if content else 0


def _line_stats(content: str):
    # returns: (max_line_len, avg_line_len, num_lines)
    if not content:
        return 0, 0.0, 0
    lines = content.splitlines()
    if not lines:
        L = len(content)
        return L, float(L), 1
    total = 0
    mx = 0
    for ln in lines:
        L = len(ln)
        total += L
        if L > mx:
            mx = L
    return mx, (total / len(lines)) if lines else 0.0, len(lines)


def _non_alnum_ratio(content: str) -> float:
    if not content:
        return 0.0
    alnumish = 0
    for ch in content:
        if ch.isalnum() or ch == "_":
            alnumish += 1
    return 1.0 - (alnumish / len(content))


def _content_signature(content: str, prefix_chars: int = 4096) -> str:
    """Near-dup signal: normalize whitespace on a prefix and hash."""
    if not content:
        return ""
    prefix = content[:prefix_chars]
    norm = _WS_RE.sub(" ", prefix).strip()
    return hashlib.blake2b(norm.encode("utf-8", "ignore"), digest_size=8).hexdigest()


def _is_test_path(p: str) -> bool:
    s = p.lower()
    return (
        "/test/" in s or "/tests/" in s or "\\test\\" in s or "\\tests\\" in s
        or s.endswith("_test.py") or s.endswith("_spec.py")
        or "/spec/" in s or "\\spec\\" in s
    )


def _ext(p: str) -> str:
    _, e = os.path.splitext(os.path.basename(p))
    return e.lower() or "<none>"


def _pow2_bucket(n: int) -> str:
    # buckets: 0, 1-1, 2-3, 4-7, 8-15, ...
    if n <= 0:
        return "0"
    b = 1
    while b * 2 <= n:
        b *= 2
    lo = b
    hi = b * 2 - 1
    return f"{lo}-{hi}"


def _hist_quantile(hist: Counter, q: float) -> str:
    total = sum(hist.values())
    if total == 0:
        return "n/a"
    target = total * q
    cum = 0

    def bucket_key(lbl: str):
        if lbl == "0":
            return 0
        return int(lbl.split("-", 1)[0])

    for lbl in sorted(hist.keys(), key=bucket_key):
        cum += hist[lbl]
        if cum >= target:
            return lbl
    return sorted(hist.keys(), key=bucket_key)[-1]


def _topk_pairs(mapping: dict, k: int = 10):
    return heapq.nlargest(k, mapping.items(), key=lambda x: x[1])


def _path_to_module_name(file_path: str) -> str:
    module_path = file_path[:-3] if file_path.endswith(".py") else file_path
    return module_path.replace("/", ".").replace("\\", ".").strip(".")


def _resolve_import_to_file(imported_module: str, module_to_file: dict, importing_file: str) -> str:
    if imported_module.startswith("."):
        dot_count = len(imported_module) - len(imported_module.lstrip("."))
        base_module = imported_module[dot_count:]

        import_dir = os.path.dirname(importing_file)
        for _ in range(dot_count - 1):
            import_dir = os.path.dirname(import_dir)

        if base_module:
            full_module_path = os.path.join(import_dir, base_module.replace(".", "/"))
        else:
            full_module_path = import_dir

        full_module_path = full_module_path.rstrip("/")

        for file_path in module_to_file.values():
            if file_path == full_module_path + ".py" or file_path.startswith(full_module_path + "/"):
                return file_path
        return None

    if imported_module in module_to_file:
        return module_to_file[imported_module]

    for module_name, file_path in module_to_file.items():
        if module_name.startswith(imported_module + "."):
            return file_path
        if imported_module.startswith(module_name + "."):
            return file_path

    return None


# =============================================================================
# Worker: Phase 1 (must read content to extract imports; also captures cheap exact sizes)
# =============================================================================
def process_file_worker_line(line: str):
    """
    Worker: parse one JSONL line, extract imports, return compact record.
    Content is required here for extract_file_imports; we also store exact char_count.
    """
    try:
        file_data = json_loads(line)
        repo_name = file_data.get("max_stars_repo_name", "")
        file_path = file_data.get("max_stars_repo_path", "")
        content = file_data.get("content", "") or ""

        if not repo_name or not file_path:
            return None

        imported_modules = extract_file_imports(content, file_path, repo_name)
        return {
            "repo_name": repo_name,
            "file_path": file_path,
            "imported_modules": list(imported_modules),
            "import_count": int(len(imported_modules)),
            "char_count": int(len(content)),
        }
    except Exception as e:
        logging.warning(f"Could not process file: {e}")
        return None


# =============================================================================
# Phase 2 bucket processor (must be top-level for Windows multiprocessing)
# =============================================================================
def process_bucket_file(bucket_file: str):
    if not os.path.exists(bucket_file) or os.path.getsize(bucket_file) == 0:
        return {}

    repo_files = {}  # repo_name -> list tuples

    with open(bucket_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                file_data = json_loads(line)
            except Exception:
                continue

            repo_name = file_data.get("repo_name")
            file_path = file_data.get("file_path")
            imported_modules = file_data.get("imported_modules", [])
            char_count = int(file_data.get("char_count", 0) or 0)
            import_count = int(file_data.get("import_count", 0) or 0)

            if not repo_name or not file_path:
                continue

            repo_files.setdefault(repo_name, []).append(
                (file_path, set(imported_modules), char_count, import_count)
            )

    final_graph = {}

    for repo_name, files in repo_files.items():
        if len(files) < 2:
            continue

        module_to_file = {}
        file_to_imports = {}
        norm_repo = normalize_repository_name(repo_name)

        for file_path, imported_modules, char_count, import_count in files:
            file_to_imports[file_path] = imported_modules
            module_name = _path_to_module_name(file_path)
            module_to_file[module_name] = file_path
            if file_path.endswith("/__init__.py"):
                parent_module = _path_to_module_name(file_path[:-12])
                if parent_module:
                    module_to_file[parent_module] = file_path

        repo_graph = {}
        path_to_key = {}

        for file_path, imported_modules, char_count, import_count in files:
            k = f"{norm_repo}:{file_path}"
            path_to_key[file_path] = k
            repo_graph[k] = {
                "title": k,
                "char_count": char_count,
                "import_count": import_count,
                "outgoing": [],
                "incoming": [],
                "links_in_repo": 0,
            }

        # outgoing
        for file_path, imported_modules in file_to_imports.items():
            outgoing_links = set()
            for imported_module in imported_modules:
                target_file = _resolve_import_to_file(imported_module, module_to_file, file_path)
                if target_file and target_file != file_path:
                    outgoing_links.add(target_file)
            if outgoing_links:
                repo_graph[path_to_key[file_path]]["outgoing"] = sorted(outgoing_links)

        # incoming (O(E))
        for source_key, data in repo_graph.items():
            source_path = source_key.split(":", 1)[1]
            for target_file_path in data["outgoing"]:
                target_key = path_to_key.get(target_file_path)
                if target_key:
                    repo_graph[target_key]["incoming"].append(source_path)

        for data in repo_graph.values():
            data["incoming"] = sorted(set(data["incoming"]))
            data["links_in_repo"] = len(data["outgoing"]) + len(data["incoming"])

        final_graph.update(repo_graph)

    return final_graph


# =============================================================================
# Streaming Graph Builder
# =============================================================================
class StreamingGraphBuilder:
    def __init__(self, batch_size=50000, max_memory_gb=8, num_buckets=256, bucket_workers=None):
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        self.num_buckets = num_buckets
        self.bucket_workers = bucket_workers or min(cpu_count(), 8)

        self.temp_dir = tempfile.mkdtemp(prefix="github_graph_")
        self.phase1_dir = os.path.join(self.temp_dir, "phase1_batches")
        self.bucket_dir = os.path.join(self.temp_dir, "repo_buckets")
        os.makedirs(self.phase1_dir, exist_ok=True)
        os.makedirs(self.bucket_dir, exist_ok=True)

        self.nodes_before_filter = None

    def build_graph_streaming(self, input_file, output_file, processes=8, bucket_workers=None):
        if bucket_workers is not None:
            self.bucket_workers = bucket_workers

        logging.info(
            f"Starting streaming graph build with batch_size={self.batch_size}, "
            f"bucket_workers={self.bucket_workers}, json_backend={JSON_BACKEND}"
        )

        batch_files = self._process_files_in_batches(input_file, processes)
        filtered_graph = self._merge_batch_results(batch_files, output_file)
        self._cleanup_temp_files(batch_files)
        return filtered_graph

    def _process_files_in_batches(self, input_file, processes):
        batch_files = []
        batch_num = 0
        current_batch_lines = []

        worker_procs = min(processes, 4)
        logging.info(f"Phase 1: using {worker_procs} worker processes")

        with Pool(processes=worker_procs) as pool:
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    current_batch_lines.append(line)

                    if len(current_batch_lines) >= self.batch_size:
                        batch_files.append(self._process_batch_lines(pool, current_batch_lines, batch_num))
                        current_batch_lines = []
                        batch_num += 1
                        logging.info(f"Processed batch {batch_num}, total files so far: {batch_num * self.batch_size}")

                if current_batch_lines:
                    batch_files.append(self._process_batch_lines(pool, current_batch_lines, batch_num))

        logging.info(f"Completed processing {len(batch_files)} batches")
        return batch_files

    def _process_batch_lines(self, pool: Pool, batch_lines, batch_num: int):
        batch_file = os.path.join(self.phase1_dir, f"batch_{batch_num}.jsonl")
        results_iter = pool.imap_unordered(process_file_worker_line, batch_lines, chunksize=500)

        with open(batch_file, "w", encoding="utf-8") as out:
            for file_data in results_iter:
                if file_data is None:
                    continue
                out.write(json_dumps(file_data) + "\n")

        del batch_lines
        gc.collect()
        return batch_file

    def _merge_batch_results(self, batch_files, output_file):
        logging.info(f"Phase 2: partitioning {len(batch_files)} batch files into {self.num_buckets} repo buckets...")

        bucket_paths = [os.path.join(self.bucket_dir, f"bucket_{i:04d}.jsonl") for i in range(self.num_buckets)]
        bucket_fhs = [open(p, "w", encoding="utf-8") for p in bucket_paths]
        repo_bucket_cache = {}

        total_partitioned = 0
        try:
            for batch_file in batch_files:
                with open(batch_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json_loads(line)
                        except Exception:
                            continue

                        repo_name = obj.get("repo_name")
                        if not repo_name:
                            continue

                        idx = repo_bucket_cache.get(repo_name)
                        if idx is None:
                            idx = _det_hash32(repo_name) % self.num_buckets
                            repo_bucket_cache[repo_name] = idx

                        bucket_fhs[idx].write(line + "\n")
                        total_partitioned += 1
        finally:
            for fh in bucket_fhs:
                fh.close()

        logging.info(f"Partitioned {total_partitioned} file records into buckets")

        bucket_files_to_process = [p for p in bucket_paths if os.path.exists(p) and os.path.getsize(p) > 0]
        max_workers = self.bucket_workers
        logging.info(f"Phase 2: processing {len(bucket_files_to_process)} buckets in parallel with {max_workers} workers")

        final_graph = {}
        failed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(process_bucket_file, bf): bf for bf in bucket_files_to_process}
            for fut in as_completed(futures):
                bf = futures[fut]
                try:
                    bucket_graph = fut.result()
                    if bucket_graph:
                        final_graph.update(bucket_graph)
                except Exception as e:
                    failed += 1
                    logging.warning(f"Bucket failed ({bf}): {e}")

        if failed:
            logging.warning(f"Phase 2: {failed}/{len(bucket_files_to_process)} buckets failed. Graph may be incomplete.")

        self.nodes_before_filter = len(final_graph)
        logging.info(f"Built graph with {self.nodes_before_filter} total files before filtering")

        filtered_graph = {k: v for k, v in final_graph.items() if v.get("links_in_repo", 0) >= 2}
        logging.info(f"After filtering: {len(filtered_graph)} files with 2+ links")

        self._write_final_graph(filtered_graph, output_file)
        return filtered_graph

    def _write_final_graph(self, graph, output_file):
        sorted_keys = sorted(graph.keys())
        with open(output_file, "w", encoding="utf-8") as f:
            for file_key in sorted_keys:
                d = graph[file_key]
                node_data = {
                    "title": file_key,
                    "char_count": int(d.get("char_count", 0) or 0),
                    "import_count": int(d.get("import_count", 0) or 0),
                    "links_in_repo": int(d.get("links_in_repo", 0) or 0),
                    "outgoing": d.get("outgoing", []) or [],
                    "incoming": d.get("incoming", []) or [],
                }
                f.write(json_dumps(node_data) + "\n")

    def _cleanup_temp_files(self, batch_files):
        for batch_file in batch_files:
            try:
                os.remove(batch_file)
            except Exception:
                pass

        try:
            for name in os.listdir(self.bucket_dir):
                try:
                    os.remove(os.path.join(self.bucket_dir, name))
                except Exception:
                    pass
            os.rmdir(self.bucket_dir)
        except Exception:
            pass

        try:
            os.rmdir(self.phase1_dir)
        except Exception:
            pass

        try:
            os.rmdir(self.temp_dir)
        except Exception:
            pass


# =============================================================================
# Exact content stats on filtered nodes via second pass over input
# =============================================================================
def _compute_exact_content_stats_from_input(input_file: str, filtered_keys: set, sig_sample_mod: int = 100):
    """
    Stream the original input JSONL and compute exact content metrics ONLY for filtered nodes.
    Uses normalize_repository_name(repo_name) + ":" + file_path to match filtered_keys.
    """
    repo_norm_cache = {}

    matched = 0
    missing_content = 0

    total_chars = 0
    total_words = 0
    total_lines = 0

    max_chars = 0
    max_words = 0

    empty_content = 0
    tiny_files = 0  # <50 chars OR <10 words

    # distributions (pow2 buckets)
    chars_hist = Counter()
    words_hist = Counter()
    max_line_hist = Counter()
    avg_line_hist = Counter()

    # hygiene
    long_line_over_1000 = 0
    high_non_alnum_over_0_35 = 0
    high_char_per_word_over_12 = 0

    # duplication (sample-based, exact on sampled subset)
    sig_sample = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json_loads(line)
            except Exception:
                continue

            repo_name = obj.get("max_stars_repo_name", "")
            file_path = obj.get("max_stars_repo_path", "")
            if not repo_name or not file_path:
                continue

            norm_repo = repo_norm_cache.get(repo_name)
            if norm_repo is None:
                norm_repo = normalize_repository_name(repo_name)
                repo_norm_cache[repo_name] = norm_repo

            key = f"{norm_repo}:{file_path}"
            if key not in filtered_keys:
                continue

            matched += 1
            content = obj.get("content", "")
            if content is None:
                content = ""
            content = content or ""

            cc = len(content)
            wc = _word_count(content)
            mll, anl, nl = _line_stats(content)
            nar = _non_alnum_ratio(content)

            total_chars += cc
            total_words += wc
            total_lines += nl

            if cc > max_chars:
                max_chars = cc
            if wc > max_words:
                max_words = wc

            if cc == 0:
                empty_content += 1
                missing_content += 1
            if cc < 50 or wc < 10:
                tiny_files += 1

            chars_hist[_pow2_bucket(cc)] += 1
            words_hist[_pow2_bucket(wc)] += 1
            max_line_hist[_pow2_bucket(mll)] += 1
            avg_line_hist[_pow2_bucket(int(anl))] += 1

            if mll > 1000:
                long_line_over_1000 += 1
            if nar > 0.35:
                high_non_alnum_over_0_35 += 1
            if wc > 0 and (cc / wc) > 12:
                high_char_per_word_over_12 += 1

            # deterministic signature sampling (bounded memory)
            if content and (_det_hash32(key) % sig_sample_mod) == 0:
                sig = _content_signature(content)
                if sig:
                    sig_sample.append(sig)

    dup_rate = 0.0
    if sig_sample:
        c = Counter(sig_sample)
        dupes = sum(v - 1 for v in c.values() if v > 1)
        dup_rate = dupes / len(sig_sample)

    return {
        "matched_final_records": matched,
        "missing_content_records": missing_content,

        "content_total_chars_final": total_chars,
        "content_total_words_final": total_words,
        "content_total_lines_final": total_lines,
        "content_max_chars_final": max_chars,
        "content_max_words_final": max_words,

        "content_empty_pct_final": None,  # filled by caller with n
        "content_tiny_pct_final": None,

        "content_chars_hist_pow2_final": dict(chars_hist),
        "content_words_hist_pow2_final": dict(words_hist),
        "max_line_length_hist_pow2_final": dict(max_line_hist),
        "avg_line_length_hist_pow2_final": dict(avg_line_hist),

        "content_chars_p50_bucket": _hist_quantile(chars_hist, 0.50),
        "content_chars_p90_bucket": _hist_quantile(chars_hist, 0.90),
        "content_chars_p99_bucket": _hist_quantile(chars_hist, 0.99),
        "content_words_p50_bucket": _hist_quantile(words_hist, 0.50),
        "content_words_p90_bucket": _hist_quantile(words_hist, 0.90),
        "content_words_p99_bucket": _hist_quantile(words_hist, 0.99),

        "pct_max_line_over_1000_final": None,
        "pct_high_non_alnum_over_0_35_final": None,
        "pct_high_char_per_word_over_12_final": None,

        "dup_signature_sample_size": len(sig_sample),
        "dup_signature_sample_dupe_rate": dup_rate,
    }


# =============================================================================
# Statistics (final nodes only; exact content metrics via second pass)
# =============================================================================
def compute_and_save_stats(graph_data, jsonl_output_path, input_file, skip_plots=False, nodes_before_filter=None):
    logging.info("Computing graph statistics... (final nodes only, exact content stats)")

    n = len(graph_data)
    filtered_keys = set(graph_data.keys())

    # ----------------------------
    # Graph composition (no content needed)
    # ----------------------------
    ext_counts = Counter()
    test_count = 0
    repo_set = set()
    path_to_key = {}

    for key in filtered_keys:
        repo, path = key.split(":", 1)
        repo_set.add(repo)
        path_to_key[path] = key
        ext_counts[_ext(path)] += 1
        if _is_test_path(path):
            test_count += 1

    # Internal edges + degrees (filtered-only)
    in_deg = {k: 0 for k in filtered_keys}
    out_deg = {}
    edges_internal = 0
    import_counts = []
    zero_imports = 0

    for key, data in graph_data.items():
        ic = int(data.get("import_count", 0) or 0)
        import_counts.append(ic)
        if ic == 0:
            zero_imports += 1

        out_deg_val = 0
        for target_path in (data.get("outgoing", []) or []):
            target_key = path_to_key.get(target_path)
            if target_key in in_deg:
                in_deg[target_key] += 1
                edges_internal += 1
                out_deg_val += 1
        out_deg[key] = out_deg_val

    in_degrees_list = list(in_deg.values())
    out_degrees_list = list(out_deg.values())

    in_deg_hist = Counter(_pow2_bucket(d) for d in in_degrees_list)
    out_deg_hist = Counter(_pow2_bucket(d) for d in out_degrees_list)

    kept_pct = None
    if nodes_before_filter is not None and nodes_before_filter > 0:
        kept_pct = n / nodes_before_filter

    # ----------------------------
    # Exact content stats from input (second pass)
    # ----------------------------
    content_stats = _compute_exact_content_stats_from_input(
        input_file=input_file,
        filtered_keys=filtered_keys,
        sig_sample_mod=100,  # ~1% sample for dup signature
    )

    # Fill percentages that require n
    content_stats["content_empty_pct_final"] = (content_stats["missing_content_records"] / n) if n else 0.0


    # Build final stats
    stats = {
        # graph counts (final)
        "graph_nodes_final": n,
        "graph_edges_internal_final": edges_internal,
        "graph_repos_final": len(repo_set),
        "graph_nodes_before_filter": nodes_before_filter,
        "graph_kept_pct": kept_pct,

        # composition
        "file_extensions_counts_final": dict(ext_counts),
        "tests_pct_final": (test_count / n) if n else 0.0,

        # imports
        "imports_avg_per_record_final": (sum(import_counts) / len(import_counts)) if import_counts else 0.0,
        "imports_median_per_record_final": float(_median_fallback(import_counts)) if import_counts else 0.0,
        "imports_zero_pct_final": (zero_imports / n) if n else 0.0,

        # degree stats (final internal edges only)
        "in_degree_hist_pow2_final": dict(in_deg_hist),
        "out_degree_hist_pow2_final": dict(out_deg_hist),
        "in_degree_avg_final": (sum(in_degrees_list) / n) if n else 0.0,
        "out_degree_avg_final": (sum(out_degrees_list) / n) if n else 0.0,
        "in_degree_median_final": float(_median_fallback(in_degrees_list)) if in_degrees_list else 0.0,
        "out_degree_median_final": float(_median_fallback(out_degrees_list)) if out_degrees_list else 0.0,
        "pct_in_degree_zero_final": (sum(1 for d in in_degrees_list if d == 0) / n) if n else 0.0,
        "pct_out_degree_zero_final": (sum(1 for d in out_degrees_list if d == 0) / n) if n else 0.0,

        "top_in_degree_files_final": _topk_pairs(in_deg, 10),
        "top_out_degree_files_final": _topk_pairs(out_deg, 10),
    }

    # merge content stats and compute averages
    stats.update(content_stats)
    stats["content_avg_chars_per_record_final"] = (stats["content_total_chars_final"] / n) if n else 0.0
    stats["content_avg_words_per_record_final"] = (stats["content_total_words_final"] / n) if n else 0.0
    stats["content_avg_lines_per_record_final"] = (stats["content_total_lines_final"] / n) if n else 0.0

    # Sanity: how many final keys did we actually see in input?
    stats["content_coverage_pct_final"] = (content_stats["matched_final_records"] / n) if n else 0.0

    base_path = os.path.splitext(jsonl_output_path)[0]
    stats_path = f"{base_path}_stats.json"
    try:
        with open(stats_path, "w", encoding="utf-8") as f:
            stdlib_json.dump(stats, f, indent=2)
        logging.info(f"Saved stats to {stats_path}")
    except Exception as e:
        logging.error(f"Failed to save stats json: {e}")

    # Optional plotting (can be heavy for huge graphs; disable with --no-plots)
    if not skip_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.hist(out_degrees_list, bins=50, log=True)
            ax1.set_title("Out-degree Distribution (Log Scale)")
            ax1.set_xlabel("Number of File Dependencies")
            ax1.set_ylabel("Frequency")

            ax2.hist(in_degrees_list, bins=50, log=True)
            ax2.set_title("In-degree Distribution (Log Scale)")
            ax2.set_xlabel("Number of Files Depending on This")
            ax2.set_ylabel("Frequency")

            plt.tight_layout()
            plot_path = f"{base_path}_degree_dist.png"
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Saved degree distribution plot to {plot_path}")
        except ImportError:
            logging.info("Matplotlib not available, skipping plots")
        except Exception as e:
            logging.error(f"Failed to save plots: {e}")
    else:
        logging.info("Skipping plots as requested")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        prog="GitHubGraphBuilderStreaming",
        description="Memory-efficient streaming graph builder for large GitHub datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", help="JSONL file containing sampled repository data.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file for the graph (default: '<input>_streaming_graph.jsonl')",
    )
    parser.add_argument("--batch-size", type=int, default=50000, help="Number of files to process per batch (default: 50,000)")
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=min(max(cpu_count() - 1, 1), 8),
        help="Number of worker processes to use in Phase 1 (default: 8, capped safely)",
    )
    parser.add_argument("--buckets", type=int, default=256, help="Number of repo hash buckets for Phase 2 (default: 256)")
    parser.add_argument("--bucket-workers", type=int, default=min(cpu_count(), 8), help="Workers for Phase 2 bucket processing (default: min(cpu_count(), 8))")
    parser.add_argument("--no-stats", action="store_true", help="Skip statistics computation (useful for very large datasets)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation (faster / avoids matplotlib dependency)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress reporting")

    args = parser.parse_args()

    if args.output is None:
        input_dir = os.path.dirname(args.input_file)
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output = os.path.join(input_dir, f"{base_name}_streaming_graph.jsonl")

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    if not os.path.exists(args.input_file):
        logging.error(f"Input file not found: {args.input_file}")
        return

    start_time = default_timer()

    builder = StreamingGraphBuilder(
        batch_size=args.batch_size,
        num_buckets=args.buckets,
        bucket_workers=args.bucket_workers,
    )
    final_graph = builder.build_graph_streaming(args.input_file, args.output, processes=args.processes)

    duration = default_timer() - start_time
    logging.info(f"Streaming graph build completed in {duration:.2f}s")
    logging.info(f"Final graph: {len(final_graph)} nodes saved to {args.output}")

    if not args.no_stats:
        compute_and_save_stats(
            final_graph,
            args.output,
            input_file=args.input_file,
            skip_plots=args.no_plots,
            nodes_before_filter=builder.nodes_before_filter,
        )
    else:
        logging.info("Skipping statistics computation as requested (--no-stats)")


if __name__ == "__main__":
    main()
