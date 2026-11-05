"""
Settings for the transcript generator. This is the file you edit.

After changing anything here just run ./run.sh again. Slow steps are cached,
so changing SPEAKER_NAMES re-runs in seconds and changing INITIAL_PROMPT only
redoes the transcription.
"""

# --- folders -----------------------------------------------------------------
INPUT_DIR = "videos"       # put your videos here
OUTPUT_DIR = "output"      # transcripts are written here: output/<video name>/
CACHE_DIR = ".cache"       # extracted audio + intermediate results (safe to delete)
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".wav", ".mp3", ".m4a"}

# --- speech to text ----------------------------------------------------------
# BACKEND: "auto" picks the best engine for the machine:
#   "mlx"            Apple Silicon Mac (runs on the Apple GPU)
#   "faster-whisper" Linux (NVIDIA GPU if present, otherwise CPU)
BACKEND = "auto"
# Whisper model for each backend. Faster, slightly less accurate options:
#   mlx: "mlx-community/whisper-large-v3-turbo"    faster-whisper: "large-v3-turbo"
WHISPER_MODEL_MLX = "mlx-community/whisper-large-v3-mlx"
WHISPER_MODEL_FASTER = "large-v3"
LANGUAGE = "en"

# Words Whisper tends to get wrong: names, places, jargon, slang.
# Keep it short and natural; a very long list can make Whisper hallucinate.
INITIAL_PROMPT = (
    "Conversation in English. Terms: ExampleName, ExampleCompany, ExampleCity."
)

# --- speaker detection -------------------------------------------------------
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
# Three people overall, but a given video may only have two of them talking,
# so the model picks a number between MIN and MAX. Set NUM_SPEAKERS to force an
# exact count (it overrides MIN/MAX).
MIN_SPEAKERS = 1
MAX_SPEAKERS = 3
NUM_SPEAKERS = None

# --- speaker names -----------------------------------------------------------
# The model only produces SPEAKER_00, SPEAKER_01 ... and the numbering is NOT
# the same between videos. Run once, read the speaker summary printed at the
# end (it shows sample lines for each speaker), then fill this in and re-run.
# Keys are video file names without the extension. "default" is the fallback.
SPEAKER_NAMES = {
    "default": {},
    # "video_1": {"SPEAKER_00": "Name A", "SPEAKER_02": "Name B", "SPEAKER_01": "Name C"},
    # "video_2": {"SPEAKER_00": "Name B", "SPEAKER_01": "Name C"},
}

# Start a new line when the same speaker pauses longer than this many seconds.
PAUSE_SPLIT_SECONDS = 4.0

# --- personal overrides ------------------------------------------------------
# Create a file called config_local.py next to this one to keep your own names,
# vocabulary and speaker names out of git. Anything defined there replaces the
# values above, e.g.:
#     INITIAL_PROMPT = "Interview. Terms: ..."
#     SPEAKER_NAMES = {"my_video": {"SPEAKER_00": "Asha"}}
try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass
