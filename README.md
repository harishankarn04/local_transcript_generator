# Local Transcript Generator

Turn video or audio recordings into **speaker-labelled transcripts**, entirely on your own computer (macOS on Apple Silicon, or Linux). No cloud transcription service: after the one-time model download, nothing leaves your machine.

```
[00:00:02] Interviewer: Welcome to the show. Could you tell us about your background?

[00:00:50] Guest: Thanks. I started out studying electronics ...
```

It chains three open-source tools:

| Step | Tool | Job |
|---|---|---|
| Speech to text | Whisper large-v3 via [mlx-whisper](https://github.com/ml-explore/mlx-examples) (Apple Silicon) or [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Linux) | Chosen automatically for your machine |
| Word timing | [WhisperX](https://github.com/m-bain/whisperX) | Aligns each word to the audio |
| Who is speaking | [pyannote.audio](https://github.com/pyannote/pyannote-audio) (`speaker-diarization-community-1`) | Splits the audio into speakers |

The result is merged into turns, and you can replace the generic `SPEAKER_00`, `SPEAKER_01` ... labels with real names.

## Requirements

- **One of these platforms** (`./setup.sh` detects which and installs the right packages):

  | Platform | Speech engine | Notes |
  |---|---|---|
  | macOS, Apple Silicon (M1 or newer), macOS 13+ | MLX on the Apple GPU | Developed and tested on an M1 with 16 GB RAM |
  | Linux (x86_64 or arm64) | faster-whisper | Uses an NVIDIA GPU if `nvidia-smi` works, otherwise the CPU (much slower). **Not yet tested** |

  Intel Macs and Windows are not supported.
- **~10 GB of free disk space** for Python packages and models (all kept inside the project folder).
- **ffmpeg**: `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` (or your distribution's equivalent) on Linux.
- Python 3 on your PATH (only used to bootstrap; the project installs its own Python 3.12). On Debian/Ubuntu you may also need `sudo apt install python3-venv`.
- A free **[Hugging Face](https://huggingface.co) account**. The speaker-detection model is free but gated, so you must accept its terms once (see below).

## Setup (one time)

**1. Get the code**

```bash
git clone git@github.com:harishankarn04/local_transcript_generator.git
cd local_transcript_generator
```

**2. Accept the speaker model's terms**

While logged in to Hugging Face, open
<https://huggingface.co/pyannote/speaker-diarization-community-1>
and click **Agree / Access repository**.

**3. Create a read token**

Go to <https://huggingface.co/settings/tokens>, create a token of type **Read**, and keep it handy.

**4. Run the installer**

```bash
./setup.sh
```

It detects your platform, creates a self-contained environment in `asr-env/`, installs the packages (`requirements.txt` for everything, plus `requirements-mac.txt` on Apple Silicon), downloads the models (several GB, first time only), and asks for your Hugging Face token. Type or paste it when prompted; it is stored only inside `asr-env/` and is never committed. It is safe to re-run if something is interrupted.

To uninstall, delete the project folder.

## Usage

Put your recordings in the `videos/` folder (`.mov`, `.mp4`, `.m4v`, `.mkv`, `.avi`, `.wav`, `.mp3`, `.m4a`), then:

```bash
./run.sh                    # every file in videos/
./run.sh interview.mp4      # a single file
./run.sh --force            # ignore the cache and redo everything
```

As a rough guide, an M1 with 16 GB handles about 8 minutes of audio in ~5 minutes (about 0.6x real time).

### Where things go

```
.
├── config.py            settings you can edit
├── config_local.py      (optional, git-ignored) your personal overrides
├── transcribe.py        the pipeline
├── run.sh / setup.sh
├── videos/              INPUT: put recordings here
├── output/              OUTPUT
│   └── <video name>/        one folder per video, kept separate
│       ├── transcript.txt     readable transcript
│       └── transcript.json    same, with start/end times per turn
├── .cache/              extracted audio + intermediate results (safe to delete)
└── asr-env/             Python, packages, models (created by setup.sh)
```

`videos/`, `output/`, `.cache/`, `asr-env/` and `config_local.py` are git-ignored, so your recordings, transcripts and personal settings are never committed.

## Naming the speakers

The model labels voices `SPEAKER_00`, `SPEAKER_01`, ... and the numbering **differs from video to video**. The workflow:

1. Run `./run.sh` once. At the end it prints, per video, each speaker's total time and a few sample lines.
2. Use those samples (and the timestamps in `transcript.txt`) to work out who is who.
3. Add the names to `SPEAKER_NAMES` in `config_local.py` (or `config.py`), keyed by the video's file name without its extension:

   ```python
   SPEAKER_NAMES = {
       "interview_part1": {"SPEAKER_00": "Asha", "SPEAKER_01": "Ravi", "SPEAKER_02": "Meera"},
       "interview_part2": {"SPEAKER_00": "Ravi", "SPEAKER_01": "Asha"},
   }
   ```
4. Run `./run.sh` again. Everything slow is cached, so this takes seconds.

## Settings (`config.py`)

| Setting | Purpose |
|---|---|
| `INPUT_DIR`, `OUTPUT_DIR`, `CACHE_DIR` | Where videos, results and cache live |
| `WHISPER_MODEL` | `whisper-large-v3-mlx` (most accurate) or `whisper-large-v3-turbo` (faster) |
| `LANGUAGE` | Spoken language code, e.g. `en` |
| `INITIAL_PROMPT` | Names, places and jargon Whisper should spell correctly. Keep it short |
| `MIN_SPEAKERS` / `MAX_SPEAKERS` | Range for automatic speaker count (a video may have fewer people than the group) |
| `NUM_SPEAKERS` | Force an exact number of speakers (overrides the range) |
| `SPEAKER_NAMES` | Per-video mapping from `SPEAKER_xx` to real names |
| `PAUSE_SPLIT_SECONDS` | Start a new line when one speaker pauses this long |

Put your own values in a `config_local.py` file next to `config.py`; it overrides `config.py` and stays out of git.

## Tips and limitations

- **Always proofread.** Whisper can mis-hear names and slang, and occasionally invents text during silence. The `INITIAL_PROMPT` helps with the first problem.
- **Overlapping speech** is the hardest case for any speaker-detection model; expect some mislabelled lines when people talk over each other.
- **Single-microphone recordings** (e.g. a phone in the middle of a room) are harder than separate mics.
- A very short interjection may be attributed to a neighbouring speaker.
- Warnings about `torchcodec` / ffmpeg libraries at startup are harmless; audio is fed to the models directly.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Environment not found` | Run `./setup.sh` first |
| `ffmpeg not found` | `brew install ffmpeg` |
| Speaker model download fails (401/403) | Accept the terms on the model page (setup step 2) and check your token is a Read token, then re-run `./setup.sh` |
| Out of memory on a long file | Close other apps. Diarization falls back to the CPU automatically |
| Want a clean slate | Delete `.cache/` (keeps models) or `asr-env/` (re-run `./setup.sh`) |

## Credits and licences

Built on [Whisper](https://github.com/openai/whisper), [MLX](https://github.com/ml-explore/mlx), [WhisperX](https://github.com/m-bain/whisperX) and [pyannote.audio](https://github.com/pyannote/pyannote-audio). The pyannote models are subject to their own licence and terms on Hugging Face; check them before commercial use.
