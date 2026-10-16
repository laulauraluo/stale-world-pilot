#!/bin/bash
#SBATCH --gres=gpu:1 -c 8 --mem=64G -t 6:00:00 -J rsp-agent -o logs/%x-%j.log
# Usage: sbatch $SBATCH_ARGS --export=ALL,MODEL=qwen3vl8b[,DATA=data,OUT=results,LIMIT=10] agent.sbatch
# Smoke tests: pass OUT=results_smoke, otherwise they overwrite results/<model>.jsonl.
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1   # models were downloaded by setup.sh
"${VLM_PY:-$PWD/env-vlm/bin/python}" agent.py --model "$MODEL" --data "${DATA:-data}" --out "${OUT:-results}" ${LIMIT:+--limit $LIMIT}
