#!/bin/bash
# One-time setup. Safe to re-run: anything already installed is skipped.
# Everything is installed inside ./asr-env (delete that folder to uninstall).
set -e
cd "$(dirname "$0")"

ENV_DIR="$PWD/asr-env"
PYVER="3.12"
mkdir -p "$ENV_DIR"

export HF_HOME="$ENV_DIR/hf-cache"
export UV_CACHE_DIR="$ENV_DIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ENV_DIR/python"

echo "== 1/5 checking ffmpeg"
if ! command -v ffmpeg >/dev/null; then
  echo "ffmpeg not found. Install it with:  brew install ffmpeg" >&2
  exit 1
fi

echo "== 2/5 Python $PYVER environment"
if [ ! -x "$ENV_DIR/.uvtool/bin/uv" ]; then
  python3 -m venv "$ENV_DIR/.uvtool"
  "$ENV_DIR/.uvtool/bin/pip" install -q uv
fi
UV="$ENV_DIR/.uvtool/bin/uv"
[ -x "$ENV_DIR/venv/bin/python" ] || "$UV" venv --python "$PYVER" "$ENV_DIR/venv"

echo "== 3/5 installing packages (a few GB, first time only)"
"$UV" pip install --python "$ENV_DIR/venv/bin/python" -r requirements.txt

PY="$ENV_DIR/venv/bin/python"

echo "== 4/5 downloading speech + alignment models"
"$PY" -W ignore - <<'EOF'
from huggingface_hub import snapshot_download
import whisperx
snapshot_download("mlx-community/whisper-large-v3-mlx")
whisperx.load_align_model(language_code="en", device="cpu")
print("speech + alignment models ready")
EOF

echo "== 5/5 Hugging Face access for the speaker model"
if [ ! -s "$HF_HOME/token" ]; then
  echo "Not logged in. Create a READ token at https://huggingface.co/settings/tokens"
  echo "and paste it when asked (it stays inside $HF_HOME)."
  "$ENV_DIR/venv/bin/hf" auth login
fi
if ! "$PY" -W ignore - <<'EOF'
from pyannote.audio import Pipeline
Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
print("speaker model ready")
EOF
then
  echo >&2
  echo "Could not download the speaker model. Open this page while logged in to" >&2
  echo "Hugging Face and click 'Agree / Access repository', then re-run ./setup.sh:" >&2
  echo "  https://huggingface.co/pyannote/speaker-diarization-community-1" >&2
  exit 1
fi

# the package download cache is no longer needed (saves ~1.3 GB)
rm -rf "$UV_CACHE_DIR"

echo
echo "Setup complete. Put videos in ./videos and run:  ./run.sh"
