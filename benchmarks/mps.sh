#!/bin/bash
# User-level CUDA MPS daemon (no root needed), for running several trainer processes on one GPU.
# Usage: source benchmarks/mps.sh start|stop
# Every process that should share the GPU through MPS needs these variables, which `source` exports.
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-${TMPDIR:-/tmp}/fast-mario-env-mps/pipe}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-${TMPDIR:-/tmp}/fast-mario-env-mps/log}
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
case "$1" in
  start) nvidia-cuda-mps-control -d ;;
  stop) echo quit | nvidia-cuda-mps-control ;;
  *) echo "usage: source $0 start|stop" ;;
esac
