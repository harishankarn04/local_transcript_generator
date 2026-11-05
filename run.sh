#!/bin/bash
# Run the transcription pipeline.
#   ./run.sh                    all videos in ./videos
#   ./run.sh interview.mp4       one video
#   ./run.sh --force            redo everything (ignore cache)
# Settings (speaker names, prompt, models) live in config.py.
set -e
cd "$(dirname "$0")"

ENV_DIR="asr-env"
if [ ! -x "$ENV_DIR/venv/bin/python" ]; then
  echo "Environment not found. Run ./setup.sh first." >&2
  exit 1
fi

# Keep every model download inside this folder.
export HF_HOME="$PWD/$ENV_DIR/hf-cache"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

exec "$ENV_DIR/venv/bin/python" -W ignore transcribe.py "$@"
