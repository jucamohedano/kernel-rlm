#!/usr/bin/env bash
# Launch trusted qualification only; the caller selects the visible physical GPU.
set -euo pipefail
if [[ $# != 4 ]]; then
  printf 'Usage: bash %s PYTHON UPSTREAM NEW_RUN_DIR {gpu-reference|gpu-controls|gpu-full}\n' "$0" >&2
  exit 2
fi
mode=$4
case "$mode" in
  gpu-reference|gpu-controls|gpu-full) ;;
  *) printf 'Unsupported GPU mode: %s\n' "$mode" >&2; exit 2 ;;
esac
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run=$3
case "$run" in
  /var/tmp/*) ;;
  *) printf 'Use a fresh directory under /var/tmp\n' >&2; exit 2 ;;
esac
umask 077
mkdir -- "$run"
mkdir -- "$run/tmp" "$run/inductor" "$run/triton" "$run/cuda-cache"
export TMPDIR="$run/tmp" TORCHINDUCTOR_CACHE_DIR="$run/inductor" TRITON_CACHE_DIR="$run/triton"
export CUDA_CACHE_PATH="$run/cuda-cache" TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=0
export PYTHONDONTWRITEBYTECODE=1
exec "$1" "$here/qualify.py" --mode "$mode" --upstream "$2" --work-dir "$run/tmp" \
  --receipt "$run/receipt.json" --nvcc "${CUDA_HOME:+$CUDA_HOME/bin/}nvcc"
