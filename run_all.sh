#!/bin/bash
# One command for the whole pilot: render episodes, run the three models (one GPU each, in parallel), score.
# Run from this folder on a Slurm login node after setup.sh. Extra sbatch options (e.g. the partition) go in
# SBATCH_ARGS; a Python that already has vLLM can be given as VLM_PY.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
sbatch ${SBATCH_ARGS:-} --wait render.sbatch
pids=()
for m in qwen3vl8b mistral24b glm41v9b; do
  sbatch ${SBATCH_ARGS:-} --wait --export=ALL,MODEL=$m agent.sbatch & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
env-hab/bin/python analyze.py --data data --results results --compare results_v1
