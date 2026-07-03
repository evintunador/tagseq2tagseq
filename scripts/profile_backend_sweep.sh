#!/usr/bin/env bash
# Profiling sweep: {doc_causal, cross_doc_link} x {triton, flex} at 32k, 1 GPU.
# Sequential (never concurrent) so cold compiles don't corrupt each other.
# Each config gets its own compile-cache dir. Keeps compile=true throughout
# (flex+compile is the real FlexAttention path; flex+compile=false would be the
# dense O(T^2) fallback that OOMs at 32k — NOT what we benchmark).
set -u
cd /fss/evin_t/tagseq2tagseq

CACHE_ROOT=/fss-data/evin_t/tagseq2tagseq_artifacts/compile_cache/backend_sweep
STEPS=12          # max_optimizer_steps
WARMUP=3
ACTIVE=7
COMMON="--train_loop.profile.enabled true \
  --train_loop.profile.warmup_steps ${WARMUP} \
  --train_loop.profile.active_steps ${ACTIVE} \
  --train_loop.profile.model_internals true \
  --train_loop.max_optimizer_steps ${STEPS} \
  --eval.run_on_completion false"

run_one () {
  local name="$1" config="$2" backend="$3"
  local cache="${CACHE_ROOT}/${name}"
  mkdir -p "${cache}/inductor" "${cache}/triton"
  echo ""
  echo "########## SWEEP CONFIG: ${name}  (config=${config} backend=${backend}) ##########"
  TS2TS_SHARED_COMPILE_CACHE="${cache}" \
  TORCHINDUCTOR_CACHE_DIR="${cache}/inductor" \
  TRITON_CACHE_DIR="${cache}/triton" \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0 \
  python main.py --config "${config}" \
    --model.attention_backend "${backend}" \
    ${COMMON} 2>&1
  echo "########## END ${name} ##########"
}

run_one dc_triton      configs/thestack_doc_causal.yaml triton
run_one dc_flex        configs/thestack_doc_causal.yaml flex
run_one cdl_triton     configs/thestack_cross_doc.yaml  triton
run_one cdl_flex       configs/thestack_cross_doc.yaml  flex

echo ""
echo "########## SWEEP COMPLETE ##########"
