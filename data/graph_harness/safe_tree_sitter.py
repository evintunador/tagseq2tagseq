"""Timeout-guarded tree-sitter parsing for the graph builders.

Some real Stack files hang tree-sitter's C parser indefinitely on certain
constructs (e.g. tree_sitter_kotlin 1.1.0 goes super-linear on Kotlin
array-literal annotation syntax `@[Ann]`, even the well-formed form). The hang
is inside the C `parse()` call, which does NOT respond to Python signals, and the
0.26 `progress_callback` cancellation path segfaults — so the only robust guard is
to run parsing in a separate process the parent can hard-kill on timeout.

`SafeParser` keeps ONE long-lived worker process (spawned once, not per file) that
imports the grammar and parses whatever bytes it's handed, returning the extracted
data over a pipe. If a file exceeds `timeout_s`, the parent kills the worker, logs
the skip, and restarts it for the next file. Amortized cost is a normal in-process
parse plus IPC; only the rare pathological file pays the kill/restart.

The worker runs a caller-supplied `extract(parser, source_bytes) -> picklable`
function (module-level, so it's importable in the spawned worker), keeping this
helper language-agnostic — each builder passes its own extractor.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _worker_loop(grammar_module: str, extract_qualname: str, in_q, out_q):
    """Persistent worker: build parser once, then parse items until sentinel."""
    import importlib
    from tree_sitter import Language, Parser

    gm = importlib.import_module(grammar_module)
    parser = Parser(Language(gm.language()))

    mod_name, fn_name = extract_qualname.rsplit(".", 1)
    extract = getattr(importlib.import_module(mod_name), fn_name)

    while True:
        item = in_q.get()
        if item is None:  # sentinel
            return
        src_bytes = item
        try:
            result = extract(parser, src_bytes)
            out_q.put((True, result))
        except Exception as e:  # noqa: BLE001
            out_q.put((False, f"{type(e).__name__}: {e}"))


class SafeParser:
    """Timeout-guarded parser backed by a killable worker process.

    Args:
        grammar_module: importable module exposing `.language()` (e.g.
            "tree_sitter_kotlin").
        extract_qualname: fully-qualified name of a module-level
            `extract(parser, src_bytes) -> picklable` function.
        timeout_s: per-file wall-clock budget; a file exceeding it is skipped.
    """

    def __init__(self, grammar_module: str, extract_qualname: str,
                 timeout_s: float = 20.0):
        self._grammar_module = grammar_module
        self._extract_qualname = extract_qualname
        self._timeout_s = timeout_s
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._in_q = None
        self._out_q = None
        self.n_skipped = 0
        self.n_ok = 0
        self._start()

    def _start(self):
        self._in_q = self._ctx.Queue()
        self._out_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker_loop,
            args=(self._grammar_module, self._extract_qualname,
                  self._in_q, self._out_q),
            daemon=True,
        )
        self._proc.start()

    def _restart(self):
        try:
            if self._proc is not None and self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        self._start()

    def parse(self, source: str, label: str = "") -> Optional[Any]:
        """Return the extractor's result, or None if the file timed out/errored."""
        src_bytes = source.encode("utf-8", errors="replace")
        self._in_q.put(src_bytes)
        try:
            ok, payload = self._out_q.get(timeout=self._timeout_s)
        except Exception:  # queue.Empty on timeout
            self.n_skipped += 1
            logger.warning("SafeParser: parse TIMEOUT (>%.0fs), skipping %s",
                           self._timeout_s, label or "<file>")
            self._restart()
            return None
        if not ok:
            self.n_skipped += 1
            logger.warning("SafeParser: parse error on %s: %s", label or "<file>", payload)
            return None
        self.n_ok += 1
        return payload

    def close(self):
        try:
            self._in_q.put(None)
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
        except Exception:  # noqa: BLE001
            pass
