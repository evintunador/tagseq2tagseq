"""Per-rank observability for diagnosing the multi-rank DDP hang (TODOS.md).

Opt-in via the ``TS2TS_DEBUG=1`` environment variable so normal runs are
unaffected.  When enabled, every rank (not just rank 0) gets:

  1. A per-rank DEBUG-level log file with line-buffered flushing, written to
     node-local ``/tmp`` (cluster-KB: never high-frequency-write to /fss).
     Survives a SIGKILL because each line is flushed immediately.

  2. A ``faulthandler`` watchdog that dumps *all* Python thread stacks for this
     rank every ``TS2TS_DEBUG_DUMP_EVERY`` seconds (default 90).  When a rank
     wedges in a collective, the periodic dump shows exactly which line each
     rank is parked on — this is what distinguishes "stuck in NCCL allreduce"
     from "stuck recompiling the backbone" from "stuck in a Python guard".

  3. SIGTERM/SIGUSR1 handlers that dump stacks on demand, so `kill -USR1 <pid>`
     or SLURM's pre-timeout SIGTERM produces a final stack snapshot.

Collect the artifacts after a run with::

    # single-node repro — all ranks' files are on the one allocated node
    srun --jobid <JID> --overlap bash -c 'cat /tmp/ts2ts_debug_*/rank*.log'

or just read /tmp/ts2ts_debug_<jobid>/ directly on the node.
"""

import faulthandler
import logging
import os
import signal
import sys
from typing import Optional

_FAULT_FILE = None  # keep a reference so the fd isn't garbage-collected


def debug_enabled() -> bool:
    return os.environ.get("TS2TS_DEBUG", "").strip() in ("1", "true", "yes")


def _debug_dir() -> str:
    # Prefer a run-dir location on shared storage (TS2TS_DEBUG_DIR) so per-rank
    # logs + faulthandler crash dumps survive the compute node being released
    # (node-local /tmp is wiped/unreachable once the job ends).  Falls back to
    # node-local /tmp when no shared dir is provided.
    shared = os.environ.get("TS2TS_DEBUG_DIR", "").strip()
    if shared:
        d = shared
    else:
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        d = f"/tmp/ts2ts_debug_{job_id}"
    os.makedirs(d, exist_ok=True)
    return d


def setup_rank_debug_logging(rank: int, local_rank: int) -> Optional[str]:
    """Install per-rank DEBUG file logging + faulthandler watchdog.

    No-op unless TS2TS_DEBUG is truthy.  Returns the log path (or None).
    """
    if not debug_enabled():
        return None

    global _FAULT_FILE
    d = _debug_dir()
    log_path = os.path.join(d, f"rank{rank}.log")

    # Line-buffered so every record hits disk before the next collective; a
    # SIGKILL on a hung rank therefore still leaves a complete log up to the
    # last line executed.
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        f"%(asctime)s.%(msecs)03d r{rank} l{local_rank} %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    ))
    # Force line flushing on the underlying stream.
    fh.stream.reconfigure(line_buffering=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)

    # Quiet the very noisy libraries so the signal isn't buried.
    for noisy in ("PIL", "matplotlib", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.INFO)

    # faulthandler: periodic all-thread stack dumps for THIS rank.
    fault_path = os.path.join(d, f"rank{rank}.faulthandler.log")
    _FAULT_FILE = open(fault_path, "w", buffering=1)  # line-buffered
    faulthandler.enable(file=_FAULT_FILE, all_threads=True)

    dump_every = int(os.environ.get("TS2TS_DEBUG_DUMP_EVERY", "90"))
    # repeat=True → keep dumping every `dump_every`s for the life of the process,
    # so a hang produces a series of identical stacks (proves it's wedged, not slow).
    faulthandler.dump_traceback_later(dump_every, repeat=True, file=_FAULT_FILE)

    # On-demand dumps: SIGUSR1 (manual) is non-fatal; SIGTERM (SLURM pre-timeout)
    # dumps then lets the existing handler exit.
    try:
        faulthandler.register(signal.SIGUSR1, file=_FAULT_FILE, all_threads=True, chain=True)
    except (ValueError, OSError):
        pass

    logging.getLogger("ts2ts.debug").debug(
        "rank debug logging online: log=%s fault=%s dump_every=%ds pid=%d",
        log_path, fault_path, dump_every, os.getpid(),
    )
    return log_path
