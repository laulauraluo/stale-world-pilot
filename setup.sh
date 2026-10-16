#!/bin/bash
# One-time setup (needs conda and internet). Safe to re-run.
#   env-hab : habitat-sim 0.3.3 (headless, EGL) + trimesh, pillow, imageio-ffmpeg, matplotlib
#   env-vlm : vllm==0.11.0; skipped if VLM_PY already points to a Python with vLLM
#   models  : the three VLMs, downloaded into the Hugging Face cache (HF_HOME)
set -euo pipefail
cd "$(dirname "$0")"
[ -d env-hab ] || conda create -y -p env-hab -c aihabitat -c conda-forge --override-channels \
  python=3.9 "habitat-sim=0.3.3=py3.9_headless_linux*" trimesh pillow numpy=1.26 libopengl libegl libglvnd \
  imageio-ffmpeg matplotlib
VLM_PY=${VLM_PY:-$PWD/env-vlm/bin/python}
if [ ! -x "$VLM_PY" ]; then
  python3 -m venv env-vlm && env-vlm/bin/pip install -q vllm==0.11.0
  VLM_PY=$PWD/env-vlm/bin/python
fi
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1   # all three models are public; never send a token
for m in Qwen/Qwen3-VL-8B-Instruct RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8 zai-org/GLM-4.1V-9B-Thinking; do
  "$(dirname "$VLM_PY")/hf" download "$m"
done
