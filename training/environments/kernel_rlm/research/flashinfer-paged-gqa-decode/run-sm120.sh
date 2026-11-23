#!/usr/bin/env bash
# Use the existing interpreter/GPU and the preflight-verified headers. No installs.
set -euo pipefail
: "${PYTHON_BIN:?set to the existing CUDA-enabled Python interpreter}"
: "${CPATH:?export the preflight-verified Python.h include paths}"
: "${QUAL_SCRATCH:?set to an approved per-wave directory under /var/tmp}"
case "$QUAL_SCRATCH" in /var/tmp/?*) ;; *) echo 'QUAL_SCRATCH must be under /var/tmp' >&2; exit 2;; esac
mode="${1:?pass sm120-reference or sm120-native first}"
shift
case "$mode" in sm120-reference|sm120-native) ;; *) echo "invalid mode: $mode" >&2; exit 2;; esac
export CPATH
export NVIDIA_TF32_OVERRIDE=0
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$QUAL_SCRATCH/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$QUAL_SCRATCH/triton}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$QUAL_SCRATCH}"
export TMPDIR="${TMPDIR:-$QUAL_SCRATCH/tmp}"
mkdir -p "$QUAL_SCRATCH" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version --format=csv,noheader >&2
nvcc --version >&2
exec "$PYTHON_BIN" -B "$(dirname "$0")/qualify.py" --author-only --mode "$mode" "$@"
