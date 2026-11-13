#!/usr/bin/env bash
#SBATCH --job-name=transdiff-vit
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=transdiff-vit-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the TransDiff repository root}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
# Per-job compile caches: jobs sharing a node's /tmp/torchinductor_$USER crash (SIGSEGV)
# when they compile at the same time.
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR/triton"
trap 'rm -rf "$TORCHINDUCTOR_CACHE_DIR"' EXIT
# Activate your tested environment before sbatch, or export TRANSDIFF_PYTHON.
# For multiple GPUs, also set --gres=gpu:N at submission and NPROC=N.
"${TRANSDIFF_PYTHON:-python}" -m torch.distributed.run --standalone \
  --nproc_per_node="${NPROC:-1}" -m imagenet.run \
  --config imagenet/configs/vit_large_last3.json "$@"
