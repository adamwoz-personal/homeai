#!/usr/bin/env bash
# Install ComfyUI and the bake-off models into ~/.local/share/homeai/comfyui (AVATAR_PLAN 3.4, 7.1).
#
# Why: the art modes need an image runner. It gets its own Python 3.12 venv with
# PyTorch's ROCm wheels, because the system has no ROCm and the project runs 3.14.
# Pinned here so a rebuild gets the same thing. Safe to re-run: downloads resume.
#
# Usage: tools/art/install_comfy.sh [--no-models]
set -euo pipefail
COMFY_TAG=v0.44.0
ROCM=rocm7.2
ROOT="$HOME/.local/share/homeai/comfyui"
APP="$ROOT/app"
HF=https://huggingface.co

mkdir -p "$ROOT"
if [ ! -d "$APP/.git" ]; then
  git clone --depth 1 --branch "$COMFY_TAG" https://github.com/comfyanonymous/ComfyUI "$APP"
fi
if [ ! -x "$ROOT/venv/bin/python" ]; then
  uv venv --python 3.12 "$ROOT/venv"
fi
uv pip install --python "$ROOT/venv/bin/python" torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/$ROCM"
grep -v -E '^(torch|torchvision|torchaudio)([<>=~ ]|$)' "$APP/requirements.txt" > "$ROOT/requirements.notorch.txt"
uv pip install --python "$ROOT/venv/bin/python" -r "$ROOT/requirements.notorch.txt"
"$ROOT/venv/bin/python" -c 'import torch; print("torch", torch.__version__, "hip", torch.version.hip)'

[ "${1:-}" = "--no-models" ] && exit 0
get() {  # repo, path in repo, folder under models/
  local name; name=$(basename "$2")
  mkdir -p "$APP/models/$3"
  curl -fL --retry 5 -C - -o "$APP/models/$3/$name" "$HF/$1/resolve/main/$2"
}
get Comfy-Org/Flux-2-Klein-4B split_files/diffusion_models/flux-2-klein-4b.safetensors diffusion_models
get Comfy-Org/Flux-2-Klein-4B split_files/text_encoders/qwen_3_4b.safetensors text_encoders
get Comfy-Org/Flux-2-Klein-4B split_files/vae/flux2-vae.safetensors vae
get Comfy-Org/Z-Image-Turbo split_files/diffusion_models/z_image_turbo_bf16.safetensors diffusion_models
get Comfy-Org/Z-Image-Turbo split_files/vae/ae.safetensors vae
get stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors checkpoints
