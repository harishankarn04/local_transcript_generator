#!/bin/bash
# One-time setup. Safe to re-run: anything already installed is skipped.
# Everything is installed inside ./asr-env (delete that folder to uninstall).
#
# Supported platforms:
#   macOS on Apple Silicon (M1 or newer)  -> MLX (Apple GPU)
#   Linux (x86_64 or arm64)               -> faster-whisper (NVIDIA GPU if present, else CPU)
set -e
cd "$(dirname "$0")"

ENV_DIR="$PWD/asr-env"
PYVER="3.12"
mkdir -p "$ENV_DIR"

export HF_HOME="$ENV_DIR/hf-cache"
export UV_CACHE_DIR="$ENV_DIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ENV_DIR/python"

# ---- detect platform --------------------------------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
  PLATFORM="mac"
  FFMPEG_HINT="brew install ffmpeg"
elif [ "$OS" = "Darwin" ]; then
  echo "Intel Macs are not supported (the required PyTorch version has no Intel Mac build)." >&2
  echo "Use an Apple Silicon Mac or Linux." >&2
  exit 1
elif [ "$OS" = "Linux" ]; then
  PLATFORM="linux"
  FFMPEG_HINT="sudo apt install ffmpeg   (or your distribution's equivalent)"
else
  echo "Unsupported platform: $OS $ARCH (supported: macOS Apple Silicon, Linux)." >&2
  exit 1
fi
HAS_NVIDIA=0
if [ "$PLATFORM" = "linux" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAS_NVIDIA=1
fi
echo "Platform: $PLATFORM ($ARCH)$([ "$HAS_NVIDIA" = 1 ] && echo ", NVIDIA GPU detected")"

echo "== 1/5 checking ffmpeg"
if ! command -v ffmpeg >/dev/null; then
  echo "ffmpeg not found. Install it with:  $FFMPEG_HINT" >&2
  exit 1
fi

echo "== 2/5 Python $PYVER environment"
if [ ! -x "$ENV_DIR/.uvtool/bin/uv" ]; then
  python3 -m venv "$ENV_DIR/.uvtool" || {
    echo "Could not create a virtual environment. On Debian/Ubuntu: sudo apt install python3-venv" >&2
    exit 1
  }
  "$ENV_DIR/.uvtool/bin/pip" install -q uv
fi
UV="$ENV_DIR/.uvtool/bin/uv"
[ -x "$ENV_DIR/venv/bin/python" ] || "$UV" venv --python "$PYVER" "$ENV_DIR/venv"
PY="$ENV_DIR/venv/bin/python"

echo "== 3/5 installing packages (a few GB, first time only)"
if [ "$PLATFORM" = "mac" ]; then
  "$UV" pip install --python "$PY" -r requirements.txt -r requirements-mac.txt
elif [ "$HAS_NVIDIA" = 1 ]; then
  "$UV" pip install --python "$PY" -r requirements.txt
else
  # No NVIDIA GPU: use the much smaller CPU-only build of PyTorch.
  "$UV" pip install --python "$PY" -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match
fi

echo "== 4/5 downloading speech + alignment models"
SETUP_PLATFORM="$PLATFORM" "$PY" -W ignore - <<'EOF'
import os
import whisperx
import config
if os.environ["SETUP_PLATFORM"] == "mac":
    from huggingface_hub import snapshot_download
    snapshot_download(config.WHISPER_MODEL_MLX)
else:
    from faster_whisper.utils import download_model
    download_model(config.WHISPER_MODEL_FASTER)
whisperx.load_align_model(language_code=config.LANGUAGE, device="cpu")
print("speech + alignment models ready")
EOF

echo "== 5/5 Hugging Face access for the speaker model"
if [ ! -s "$HF_HOME/token" ]; then
  echo "Not logged in. Create a READ token at https://huggingface.co/settings/tokens"
  echo "and paste it when asked (it stays inside $HF_HOME)."
  "$ENV_DIR/venv/bin/hf" auth login
fi
if ! "$PY" -W ignore - <<'EOF'
import config
from pyannote.audio import Pipeline
Pipeline.from_pretrained(config.DIARIZATION_MODEL)
print("speaker model ready")
EOF
then
  echo >&2
  echo "Could not download the speaker model. Open this page while logged in to" >&2
  echo "Hugging Face and click 'Agree / Access repository', then re-run ./setup.sh:" >&2
  echo "  https://huggingface.co/pyannote/speaker-diarization-community-1" >&2
  exit 1
fi

# the package download cache is no longer needed
rm -rf "$UV_CACHE_DIR"

echo
echo "Setup complete. Put videos in ./videos and run:  ./run.sh"
