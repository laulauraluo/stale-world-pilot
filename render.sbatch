#!/bin/bash
#SBATCH --gres=gpu:1 -c 4 --mem=16G -t 60 -J rsp-render -o logs/%x-%j.log
# Usage: sbatch $SBATCH_ARGS [--export=ALL,N=200,OUT=data] render.sbatch
cd "$SLURM_SUBMIT_DIR"
export __EGL_VENDOR_LIBRARY_DIRS=${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}  # the node's NVIDIA EGL driver
env-hab/bin/python sim.py --n "${N:-200}" --out "${OUT:-data}"
