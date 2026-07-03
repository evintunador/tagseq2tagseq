#!/bin/bash
# Per-rank wrapper invoked by torchrun.
#
# Compile-cache strategy (CLAUDE.md "Multi-rank runs need a pre-warmed compile
# cache"): all ranks of a run share ONE cache dir, NOT per-rank dirs.  With
# TORCHINDUCTOR_COMPILE_THREADS=1 (set by launch_downnode.sh) inductor/triton
# take a FileLock per cache key, so exactly one rank compiles each kernel and
# the others block on the lock then read the artifact.  Per-rank dirs instead
# make all N ranks cold-compile the SAME kernels concurrently — which deadlocks
# the heavy custom cross-doc-link (BIM/FlexAttention) kernel (observed: rank0
# idle while ranks1-3 spin forever in compile).  doc_causal/doc_concatenated
# (lighter varlen kernels) tolerated per-rank dirs; cross_doc did not.
#
# Cache lives on /fss-data and is keyed by RUN_TAG so it persists across runs
# (a second run of the same tag is a warm cache hit = fast startup) but distinct
# configs don't collide.
set -euo pipefail
REPO=/fss/evin_t/tagseq2tagseq
RUN_TAG="$1"; shift

# NODE-LOCAL shared cache (/tmp = local ext4), shared by this node's 4 ranks.
# Must NOT be on /fss-data: that is NFS, and inductor/triton FileLocks taken by
# 3 nodes x 4 ranks = 12 procs concurrently on one NFS server DEADLOCK (observed:
# the node that grabs the NFS lock first compiles; the others' rank0 freezes
# while ranks1-3 spin forever).  Node-local /tmp gives reliable within-node
# FileLock serialization (1 rank compiles each kernel, the node's other 3 read)
# with ZERO cross-node lock contention.  Cost: each node compiles independently
# (no cache reuse across nodes) — fine, 3 parallel local compiles beats a deadlock.
SHARED_CACHE=/tmp/ts2ts_compile_cache_${RUN_TAG}
export TORCHINDUCTOR_CACHE_DIR=${SHARED_CACHE}/inductor
export TRITON_CACHE_DIR=${SHARED_CACHE}/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# main.py installs no console log handler, so logger.info (graph load, compile,
# cache-vs-mock decision) is invisible by default.  Force INFO logging to stderr
# so down-node runs are observable, then exec main.py via runpy.
export PYTHONUNBUFFERED=1
exec "$REPO/.venv/bin/python" -u -c "
import logging, sys, runpy
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                    stream=sys.stderr)
sys.argv = ['main.py'] + sys.argv[1:]
runpy.run_path('$REPO/main.py', run_name='__main__')
" "$@"
