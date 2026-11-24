#!/usr/bin/env bash
# Install openWakeWord's training pipeline and its data, to train the "Hey Jeeves" wake word (plans/WAKE_WORD_PLAN.md).
#
# Why: no Jeeves model exists; openWakeWord trains one from synthetic speech (Piper's LibriTTS-R generator: hundreds
# of voices) against ~2000 hours of precomputed negative features. Its own Python 3.12 venv with PyTorch's ROCm
# wheels (as ComfyUI, tools/art/install_comfy.sh), because the project runs 3.14. Repos pinned; safe to re-run
# (downloads resume). Code in ~/.local/share/homeai/wake-train, data (~19 GB) in ~/.cache/homeai/wake-train.
#
# Usage: tools/wake/install_wake_train.sh [--no-data]
set -euo pipefail
ROCM=rocm7.2
OWW_COMMIT=368c037            # dscripka/openWakeWord main, 2026-11-23
# piper-phonemize 1.1.0 has no Linux cp312 wheel; piper-phonemize-cross 1.2.1 provides the same piper_phonemize module.
PIPER_TAG=v2.0.0              # rhasspy/piper-sample-generator: last layout train.py imports (generate_samples.py)
ROOT="$HOME/.local/share/homeai/wake-train"
DATA="$HOME/.cache/homeai/wake-train"
HF=https://huggingface.co
GH=https://github.com
mkdir -p "$ROOT/src" "$DATA"

clone() {  # url, folder, ref
  [ -d "$ROOT/src/$2/.git" ] || git clone -q "$1" "$ROOT/src/$2"
  git -C "$ROOT/src/$2" checkout -q "$3"
}
clone $GH/dscripka/openWakeWord.git openWakeWord "$OWW_COMMIT"
clone $GH/rhasspy/piper-sample-generator.git piper-sample-generator "$PIPER_TAG"
# The LibriTTS-R generator (rhasspy's own v2.0.0 release file) is a whole pickled model, which PyTorch >= 2.6
# refuses by default. Trusted source, fixed URL: allow it for that one load.
sed -i 's/model = torch.load(model_path)$/model = torch.load(model_path, weights_only=False)/' \
  "$ROOT/src/piper-sample-generator/generate_samples.py"

[ -x "$ROOT/venv/bin/python" ] || uv venv -q --python 3.12 "$ROOT/venv"
PY="$ROOT/venv/bin/python"
uv pip install -q --python "$PY" torch torchaudio --index-url "https://download.pytorch.org/whl/$ROCM"
uv pip install -q --python "$PY" "piper-phonemize-cross==1.2.1" webrtcvad mutagen torchinfo torchmetrics "speechbrain<1" \
  audiomentations torch-audiomentations acoustics pronouncing "datasets<3" "deep-phonemizer==0.0.19" pyyaml "scipy<1.15" \
  onnx onnxscript onnxruntime soundfile tqdm pyarrow
uv pip install -q --python "$PY" --no-deps -e "$ROOT/src/openWakeWord"
# Upstream bug: --convert_to_tflite has default="False" (a true string), so every training tried TFLite (needs
# TensorFlow, not installed) and exited 1 after saving the ONNX model. Make the default a real False.
sed -i 's/        default="False",/        default=False,/' "$ROOT/src/openWakeWord/openwakeword/train.py"
# torchaudio >= 2.9 has no load()/info() backends of its own: route them through soundfile (torchaudio_shim.py).
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cp "$(dirname "$0")/torchaudio_shim.py" "$SITE/homeai_torchaudio_shim.py"
echo "import homeai_torchaudio_shim" > "$SITE/homeai_torchaudio_shim.pth"

get() {  # url, destination
  [ -s "$2" ] && return 0
  mkdir -p "$(dirname "$2")"
  curl -fL --retry 5 -C - -o "$2.part" "$1" && mv "$2.part" "$2"
}
get "$GH/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt" \
  "$ROOT/src/piper-sample-generator/models/en_US-libritts_r-medium.pt"
for m in embedding_model.onnx melspectrogram.onnx; do
  get "$GH/dscripka/openWakeWord/releases/download/v0.5.1/$m" "$ROOT/src/openWakeWord/openwakeword/resources/models/$m"
done
"$PY" -c 'import torch; print("torch", torch.__version__, "gpu", torch.cuda.is_available())'

[ "${1:-}" = "--no-data" ] && exit 0
FEAT=$HF/datasets/davidscripka/openwakeword_features/resolve/main
get "$FEAT/validation_set_features.npy" "$DATA/validation_set_features.npy"                   # 185 MB, ~11 h
get "$FEAT/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" "$DATA/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"  # 17.3 GB
"$PY" -I "$(dirname "$0")/fetch_wake_data.py" "$DATA"
echo "wake-train ready: $ROOT, data $DATA"
