#!/usr/bin/env python3
"""
Local video -> transcript with speaker labels (Apple Silicon).

Per video:
  1. ffmpeg      -> 16 kHz mono audio
  2. mlx-whisper -> speech to text
  3. whisperx    -> word-level timestamps
  4. pyannote    -> who speaks when
  5. merge       -> "[HH:MM:SS] Name: text" turns

Settings live in config.py. Run through ./run.sh:
    ./run.sh                    every video in the videos/ folder
    ./run.sh interview.mp4       one video
    ./run.sh --force            ignore all caches

Output:
    output/<video>/transcript.txt     readable transcript
    output/<video>/transcript.json    same, with timestamps per turn
    output/all_transcripts.txt        every transcript in one file
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import config

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def short_hash(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8]


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def find_inputs(names):
    if not names:
        d = Path(config.INPUT_DIR)
        return [p for p in sorted(d.iterdir())
                if p.is_file() and p.suffix.lower() in config.VIDEO_EXTS] if d.is_dir() else []
    paths = []
    for n in names:
        p = Path(n)
        if not p.exists():
            p = Path(config.INPUT_DIR) / n
        if not p.exists():
            sys.exit(f"File not found: {n}")
        paths.append(p)
    return paths


def extract_audio(video, wav):
    if wav.exists():
        return
    log(f"extracting audio from {video.name}")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1",
         "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True)


def transcribe(audio, cache, force):
    if cache.exists() and not force:
        log("transcription: using cache")
        return json.loads(cache.read_text())
    import mlx_whisper
    log(f"transcribing with {config.WHISPER_MODEL} ...")
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=config.WHISPER_MODEL,
        language=config.LANGUAGE,
        initial_prompt=config.INITIAL_PROMPT or None,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    cache.write_text(json.dumps(result))
    log("transcription done")
    return result


def align_words(asr, audio):
    import whisperx
    log("aligning words ...")
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in asr["segments"]]
    model, meta = whisperx.load_align_model(language_code=config.LANGUAGE, device="cpu")
    aligned = whisperx.align(segs, model, meta, audio, "cpu", return_char_alignments=False)
    log("alignment done")
    return aligned


def diarize(audio, cache, force):
    if cache.exists() and not force:
        log("diarization: using cache")
        return json.loads(cache.read_text())
    import torch
    from pyannote.audio import Pipeline
    log(f"detecting speakers with {config.DIARIZATION_MODEL} ...")
    pipe = Pipeline.from_pretrained(config.DIARIZATION_MODEL)
    kwargs = ({"num_speakers": config.NUM_SPEAKERS} if config.NUM_SPEAKERS
              else {"min_speakers": config.MIN_SPEAKERS, "max_speakers": config.MAX_SPEAKERS})
    inp = {"waveform": torch.from_numpy(audio)[None], "sample_rate": 16000}
    try:
        pipe.to(torch.device("mps"))
        out = pipe(inp, **kwargs)
    except Exception as e:  # the GPU can run out of memory on long files
        log(f"GPU diarization failed ({type(e).__name__}), retrying on CPU")
        pipe.to(torch.device("cpu"))
        out = pipe(inp, **kwargs)
    ann = getattr(out, "speaker_diarization", out)
    rows = [{"start": t.start, "end": t.end, "speaker": s}
            for t, _, s in ann.itertracks(yield_label=True)]
    cache.write_text(json.dumps(rows))
    log(f"diarization done: {len(rows)} turns")
    return rows


def build_turns(aligned, diar_rows):
    """Merge words + speaker segments into consecutive per-speaker turns."""
    import pandas as pd
    import whisperx
    res = whisperx.assign_word_speakers(pd.DataFrame(diar_rows), aligned)
    turns, cur = [], None
    for seg in res["segments"]:
        seg_speaker = seg.get("speaker", "UNKNOWN")
        words = seg.get("words") or [{"word": seg["text"], "start": seg["start"],
                                      "end": seg["end"], "speaker": seg_speaker}]
        for w in words:
            sp = w.get("speaker") or (cur["speaker"] if cur else seg_speaker)
            start = w.get("start", cur["end"] if cur else seg["start"])
            end = w.get("end", start)
            text = w["word"].strip()
            if not text:
                continue
            if (cur is None or sp != cur["speaker"]
                    or start - cur["end"] > config.PAUSE_SPLIT_SECONDS):
                cur = {"speaker": sp, "start": start, "end": end, "words": [text]}
                turns.append(cur)
            else:
                cur["words"].append(text)
                cur["end"] = end
    for t in turns:
        t["text"] = " ".join(t.pop("words"))
    return turns


def name_for(stem, label):
    names = {**config.SPEAKER_NAMES.get("default", {}), **config.SPEAKER_NAMES.get(stem, {})}
    return names.get(label, label)


def write_outputs(stem, turns, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"[{hms(t['start'])}] {name_for(stem, t['speaker'])}: {t['text']}" for t in turns]
    txt = out_dir / "transcript.txt"
    txt.write_text("\n\n".join(lines) + "\n")
    data = [{"start": round(t["start"], 2), "end": round(t["end"], 2),
             "speaker": t["speaker"], "name": name_for(stem, t["speaker"]), "text": t["text"]}
            for t in turns]
    (out_dir / "transcript.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return txt


def write_combined(output_root):
    parts = [f"===== {d.name} =====\n\n{(d / 'transcript.txt').read_text()}"
             for d in sorted(output_root.iterdir())
             if d.is_dir() and (d / "transcript.txt").exists()]
    if len(parts) > 1:
        combined = output_root / "all_transcripts.txt"
        combined.write_text("\n\n".join(parts))
        log(f"combined transcript: {combined}")


def print_speaker_summary(stem, turns):
    stats = {}
    for t in turns:
        s = stats.setdefault(t["speaker"], {"sec": 0.0, "n": 0, "samples": []})
        s["sec"] += t["end"] - t["start"]
        s["n"] += 1
        if len(s["samples"]) < 3 and len(t["text"]) > 40:
            s["samples"].append(f"[{hms(t['start'])}] {t['text'][:90]}")
    print(f"\n--- speakers in {stem} ---")
    for label, s in sorted(stats.items(), key=lambda kv: -kv[1]["sec"]):
        print(f"{label} -> {name_for(stem, label)}: {s['sec'] / 60:.1f} min, {s['n']} turns")
        for line in s["samples"]:
            print(f"     {line}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="videos to process (default: all in the input folder)")
    ap.add_argument("--force", action="store_true", help="ignore all caches")
    ap.add_argument("--force-asr", action="store_true", help="redo transcription only")
    ap.add_argument("--force-diar", action="store_true", help="redo speaker detection only")
    args = ap.parse_args()

    videos = find_inputs(args.files)
    if not videos:
        sys.exit(f"No videos found in '{config.INPUT_DIR}/'. Put videos there or pass file names.")

    import soundfile as sf
    asr_key = short_hash(config.WHISPER_MODEL, config.LANGUAGE, config.INITIAL_PROMPT)
    diar_key = short_hash(config.DIARIZATION_MODEL, config.MIN_SPEAKERS,
                          config.MAX_SPEAKERS, config.NUM_SPEAKERS)
    output_root = Path(config.OUTPUT_DIR)

    for video in videos:
        stem = video.stem
        log(f"===== {video.name} =====")
        cache = Path(config.CACHE_DIR) / stem
        cache.mkdir(parents=True, exist_ok=True)

        wav = cache / "audio.wav"
        extract_audio(video, wav)
        audio, sr = sf.read(str(wav), dtype="float32")
        log(f"audio length {len(audio) / sr / 60:.1f} min")

        asr = transcribe(audio, cache / f"asr.{asr_key}.json", args.force or args.force_asr)
        aligned = align_words(asr, audio)
        diar = diarize(audio, cache / f"diar.{diar_key}.json", args.force or args.force_diar)
        turns = build_turns(aligned, diar)

        txt = write_outputs(stem, turns, output_root / stem)
        log(f"wrote {txt}")
        print_speaker_summary(stem, turns)

    write_combined(output_root)
    log("all done")


if __name__ == "__main__":
    main()
