"""
Arabic → Turkish Video Voiceover Pipeline  (with YouTube Downloader)
=====================================================================
Install dependencies:
    pip install openai-whisper transformers sentencepiece soundfile numpy torch TTS yt-dlp

FFmpeg must also be installed on your system:
    Windows → https://ffmpeg.org/download.html  (add ffmpeg to PATH)
    Mac     → brew install ffmpeg
    Linux   → sudo apt install ffmpeg

─────────────────────────────────────────────────────────────────────
USAGE — YouTube URL (recommended):
    python arabic_to_turkish_voiceover.py --url "https://www.youtube.com/watch?v=XXXX"

USAGE — Local file:
    python arabic_to_turkish_voiceover.py --input video.mp4

Optional flags:
    --whisper_model   tiny | base | small | medium | large-v3  (default: medium)
    --tts_model       coqui | gtts                             (default: coqui)
    --keep_bg_audio   Mix original audio quietly under voiceover
    --bg_volume       Background audio volume 0.0–1.0          (default: 0.15)
    --downloads_dir   Where to save everything                 (default: ~/Downloads)

─────────────────────────────────────────────────────────────────────
OUTPUT FILES  (all saved to ~/Downloads/<video_title>/)
    <title>.mp4                   original downloaded video
    <title>_no_audio.mp4          video with no sound
    <title>_turkish_voiceover.mp4 video with Turkish voiceover
    <title>_original_audio.mp3    extracted original Arabic audio
    <title>_turkish_voiceover.mp3 Turkish voiceover audio only
    <title>_transcript.json       bilingual Arabic/Turkish transcript
"""

import argparse
import os
import sys
import json
import subprocess
import tempfile
import shutil
import math
import re

import torch
import numpy as np
import soundfile as sf


# ══════════════════════════════════════════════════════════════════
# 1. ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a YouTube Arabic video and produce a Turkish voiceover"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",   help="YouTube video URL")
    group.add_argument("--input", help="Path to a local video file")

    parser.add_argument("--whisper_model", default="medium",
                        choices=["tiny","base","small","medium","large","large-v2","large-v3"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--tts_model", default="coqui", choices=["coqui","gtts"],
                        help="TTS engine: coqui (offline, higher quality) or gtts (online, faster)")
    parser.add_argument("--keep_bg_audio", action="store_true",
                        help="Keep original audio quietly in background under the voiceover")
    parser.add_argument("--bg_volume", type=float, default=0.15,
                        help="Volume of background audio 0.0-1.0 (default 0.15)")
    parser.add_argument("--downloads_dir", default=None,
                        help="Output folder (default: ~/Downloads)")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════
# 2. YOUTUBE DOWNLOAD
# ══════════════════════════════════════════════════════════════════

def sanitize_filename(name: str) -> str:
    """Remove characters unsafe in filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def download_youtube(url: str, downloads_dir: str) -> tuple[str, str]:
    """
    Download a YouTube video as best-quality MP4 using yt-dlp.
    Returns (video_path, video_title).
    """
    try:
        import yt_dlp
    except ImportError:
        print("Error: yt-dlp not installed. Run:  pip install yt-dlp")
        sys.exit(1)

    print(f"\n[DOWN] Fetching video info from YouTube...")

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        title = sanitize_filename(info.get("title", "video"))

    out_path = os.path.join(downloads_dir, f"{title}.mp4")

    print(f"       Title  : {title}")
    print(f"       Saving : {out_path}")
    print(f"       Downloading...")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_path,
        "merge_output_format": "mp4",
        "quiet": False,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # yt-dlp may add extra characters to the filename — find the file
    if not os.path.exists(out_path):
        matches = [f for f in os.listdir(downloads_dir)
                   if title[:30] in f and f.endswith(".mp4")]
        if matches:
            out_path = os.path.join(downloads_dir, matches[0])
        else:
            print("Error: Downloaded file not found in Downloads folder.")
            sys.exit(1)

    print(f"       Done -> {out_path}")
    return out_path, title


# ══════════════════════════════════════════════════════════════════
# 3. CREATE SILENT VIDEO COPY
# ══════════════════════════════════════════════════════════════════

def create_silent_video(video_path: str, out_path: str):
    """Copy the video with all audio streams removed."""
    print(f"\n[FILE] Creating silent video copy...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-an",           # strip all audio
        "-c:v", "copy",  # copy video stream (no re-encode, fast)
        out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"       Warning: {result.stderr[:200]}")
    else:
        print(f"       -> {out_path}")


# ══════════════════════════════════════════════════════════════════
# 4. EXTRACT ORIGINAL AUDIO AS MP3
# ══════════════════════════════════════════════════════════════════

def extract_original_audio_mp3(video_path: str, out_mp3: str):
    """Extract the original Arabic audio track to a 192k MP3."""
    print(f"\n[FILE] Extracting original audio as MP3...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", "192k",
        "-ar", "44100",
        out_mp3
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"       Warning: {result.stderr[:200]}")
    else:
        print(f"       -> {out_mp3}")


# ══════════════════════════════════════════════════════════════════
# 5. EXTRACT AUDIO FOR WHISPER (16kHz mono WAV)
# ══════════════════════════════════════════════════════════════════

def extract_audio_for_whisper(video_path: str, out_wav: str):
    """Extract 16kHz mono WAV — the format Whisper expects."""
    print(f"\n[1/5] Extracting audio for transcription...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        out_wav
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg error:", result.stderr)
        sys.exit(1)
    print(f"       -> {out_wav}")


# ══════════════════════════════════════════════════════════════════
# 6. TRANSCRIBE ARABIC WITH WHISPER
# ══════════════════════════════════════════════════════════════════

def transcribe_arabic(wav_path: str, model_name: str) -> list[dict]:
    """
    Transcribe Arabic audio using OpenAI Whisper.
    Runs 100% offline — no API key needed.
    Returns: [{"start": 0.0, "end": 2.4, "text": "..."}, ...]
    """
    print(f"\n[2/5] Transcribing Arabic with Whisper ({model_name})...")
    print("      Runs on CPU — allow 20-60 min for a 28-min video")

    import whisper
    model = whisper.load_model(model_name)
    result = model.transcribe(
        wav_path,
        language="ar",
        verbose=False,
        word_timestamps=False,
        task="transcribe"
    )

    segments = result["segments"]
    print(f"       Transcribed {len(segments)} segments")
    return segments


# ══════════════════════════════════════════════════════════════════
# 7. TRANSLATE ARABIC → TURKISH
# ══════════════════════════════════════════════════════════════════

def translate_segments(segments: list[dict]) -> list[dict]:
    """
    Translate each segment from Arabic to Turkish.
    Uses Helsinki-NLP/opus-mt-ar-tr — free, fully offline, no API key.
    Downloads ~300MB on first run, then cached locally.
    """
    print(f"\n[3/5] Translating Arabic -> Turkish...")

    from transformers import MarianMTModel, MarianTokenizer

    hf_model = "Helsinki-NLP/opus-mt-ar-tr"
    print(f"      Loading model (downloads once ~300MB)...")
    tokenizer = MarianTokenizer.from_pretrained(hf_model)
    tr_model  = MarianMTModel.from_pretrained(hf_model)
    tr_model.eval()

    texts      = [seg["text"].strip() for seg in segments]
    batch_size = 16
    all_tr     = []

    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        with torch.no_grad():
            outputs = tr_model.generate(**inputs, num_beams=4)
        all_tr.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        print(f"      {min(i+batch_size, len(texts))}/{len(texts)} segments translated", end="\r")

    print()

    translated = []
    for seg, tr_text in zip(segments, all_tr):
        translated.append({
            "start": seg["start"],
            "end":   seg["end"],
            "ar":    seg["text"].strip(),
            "tr":    tr_text.strip(),
        })

    print(f"      Translation complete ({len(translated)} segments)")
    return translated


# ══════════════════════════════════════════════════════════════════
# 8. GENERATE TURKISH TTS PER SEGMENT
# ══════════════════════════════════════════════════════════════════

def _write_silence(path: str, duration_seconds: float, sample_rate: int = 22050):
    n = max(1, int(duration_seconds * sample_rate))
    sf.write(path, np.zeros(n, dtype=np.float32), sample_rate)


def _gtts_segment(text: str, out_path: str):
    """gTTS fallback — uses Google's servers, requires internet."""
    from gtts import gTTS
    import time
    tts     = gTTS(text=text, lang="tr", slow=False)
    mp3_tmp = out_path.replace(".wav", "_tmp.mp3")
    tts.save(mp3_tmp)
    subprocess.run(["ffmpeg", "-y", "-i", mp3_tmp,
                    "-ar", "22050", "-ac", "1", out_path],
                   capture_output=True)
    os.remove(mp3_tmp)
    time.sleep(0.3)   # gentle rate-limit


def generate_tts_segments(translated: list[dict], work_dir: str,
                           engine_name: str) -> list[dict]:
    """Synthesize a Turkish WAV clip for every segment."""
    print(f"\n[4/5] Generating Turkish TTS ({engine_name})...")

    coqui_engine = None
    if engine_name == "coqui":
        try:
            from TTS.api import TTS as CoquiTTS
            coqui_engine = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
            print("      Loaded Coqui XTTS-v2 (offline, Turkish)")
        except Exception as e:
            print(f"      Coqui unavailable ({e}) -- falling back to gTTS")
            engine_name = "gtts"

    for i, seg in enumerate(translated):
        out_wav = os.path.join(work_dir, f"seg_{i:04d}.wav")
        text    = seg["tr"]

        if not text.strip():
            _write_silence(out_wav, seg["end"] - seg["start"])
        else:
            try:
                if engine_name == "coqui" and coqui_engine:
                    coqui_engine.tts_to_file(
                        text=text, language="tr",
                        speaker="Ana Florence", file_path=out_wav
                    )
                else:
                    _gtts_segment(text, out_wav)
            except Exception as e:
                print(f"\n      Warning seg {i}: {e} -- using silence")
                _write_silence(out_wav, seg["end"] - seg["start"])

        seg["tts_wav"] = out_wav
        print(f"      {i+1}/{len(translated)}", end="\r")

    print(f"\n      TTS complete")
    return translated


# ══════════════════════════════════════════════════════════════════
# 9. TIME-STRETCH SEGMENTS TO FIT THEIR ORIGINAL SLOTS
# ══════════════════════════════════════════════════════════════════

def _audio_duration(path: str) -> float:
    r = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _atempo_chain(ratio: float) -> list[str]:
    """Chain atempo filters — each limited to 0.5-2.0."""
    filters   = []
    remaining = ratio
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return filters


def time_stretch(in_wav: str, out_wav: str, target_sec: float):
    """Speed up or slow down a clip to exactly fit target_sec."""
    actual = _audio_duration(in_wav)
    if actual <= 0:
        shutil.copy(in_wav, out_wav)
        return
    ratio = max(0.4, min(3.0, actual / target_sec))
    subprocess.run([
        "ffmpeg", "-y", "-i", in_wav,
        "-filter:a", ",".join(_atempo_chain(ratio)),
        "-ar", "22050", out_wav
    ], capture_output=True)


# ══════════════════════════════════════════════════════════════════
# 10. ASSEMBLE COMPLETE TURKISH AUDIO TRACK
# ══════════════════════════════════════════════════════════════════

def _resample(data: np.ndarray, orig_sr: int, tgt_sr: int) -> np.ndarray:
    if orig_sr == tgt_sr:
        return data
    tgt_len = int(len(data) / orig_sr * tgt_sr)
    return np.interp(
        np.linspace(0, len(data)-1, tgt_len),
        np.arange(len(data)), data
    ).astype(np.float32)


def assemble_audio_track(segments: list[dict], work_dir: str,
                         total_dur: float) -> str:
    """
    Stretch each TTS segment to fit its original timeslot,
    then stitch everything together with silence in the gaps.
    """
    print(f"\n[5/5] Assembling Turkish audio track...")

    SR       = 22050
    audio    = np.zeros(int(math.ceil(total_dur * SR)), dtype=np.float32)

    for i, seg in enumerate(segments):
        stretched = os.path.join(work_dir, f"seg_{i:04d}_s.wav")
        time_stretch(seg["tts_wav"], stretched, seg["end"] - seg["start"])

        data, sr = sf.read(stretched, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SR:
            data = _resample(data, sr, SR)

        s = int(seg["start"] * SR)
        e = s + len(data)
        if e > len(audio):
            data = data[:len(audio) - s]
            e    = len(audio)
        audio[s:e] += data
        print(f"      {i+1}/{len(segments)}", end="\r")

    print()
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio /= peak

    out = os.path.join(work_dir, "turkish_voiceover.wav")
    sf.write(out, audio, SR)
    print(f"      Audio assembled")
    return out


# ══════════════════════════════════════════════════════════════════
# 11. EXPORT VOICEOVER AS STANDALONE MP3
# ══════════════════════════════════════════════════════════════════

def export_voiceover_mp3(wav_path: str, out_mp3: str):
    print(f"\n[FILE] Exporting Turkish voiceover MP3...")
    cmd = [
        "ffmpeg", "-y", "-i", wav_path,
        "-acodec", "libmp3lame", "-ab", "192k", "-ar", "44100",
        out_mp3
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"       Warning: {result.stderr[:200]}")
    else:
        print(f"       -> {out_mp3}")


# ══════════════════════════════════════════════════════════════════
# 12. MERGE VOICEOVER INTO VIDEO
# ══════════════════════════════════════════════════════════════════

def merge_into_video(video_path: str, voiceover_wav: str, out_path: str,
                     keep_bg: bool, bg_vol: float):
    print(f"\n[FILE] Merging voiceover into video...")

    if keep_bg:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", voiceover_wav,
            "-filter_complex",
            f"[0:a]volume={bg_vol}[bg];[1:a]volume=1.0[vo];[bg][vo]amix=inputs=2:duration=first[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            out_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", voiceover_wav,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            out_path
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg merge error:", result.stderr)
        sys.exit(1)
    print(f"       -> {out_path}")


# ══════════════════════════════════════════════════════════════════
# 13. HELPERS
# ══════════════════════════════════════════════════════════════════

def get_video_duration(path: str) -> float:
    r = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def print_summary(paths: dict):
    print("\n" + "=" * 62)
    print("  COMPLETE — files saved to:")
    print("  " + os.path.dirname(paths["original_video"]))
    print("=" * 62)
    labels = {
        "original_video":     "Original video            ",
        "silent_video":       "Video — no audio          ",
        "dubbed_video":       "Video — Turkish voiceover ",
        "original_audio_mp3": "Arabic audio (MP3)        ",
        "voiceover_mp3":      "Turkish voiceover (MP3)   ",
        "transcript":         "Bilingual transcript      ",
    }
    for key, label in labels.items():
        p = paths.get(key, "")
        if p and os.path.exists(p):
            mb = os.path.getsize(p) / 1_048_576
            print(f"  {label} {os.path.basename(p)}  ({mb:.1f} MB)")
        elif p:
            print(f"  {label} {os.path.basename(p)}  (not created)")
    print("=" * 62 + "\n")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Resolve downloads directory
    dl_dir = args.downloads_dir or os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(dl_dir, exist_ok=True)

    # ── Get the video
    if args.url:
        video_path, title = download_youtube(args.url, dl_dir)
    else:
        video_path = args.input
        if not os.path.exists(video_path):
            print(f"Error: File not found: {video_path}")
            sys.exit(1)
        title = sanitize_filename(os.path.splitext(os.path.basename(video_path))[0])
        dest  = os.path.join(dl_dir, f"{title}.mp4")
        if os.path.abspath(video_path) != os.path.abspath(dest):
            shutil.copy2(video_path, dest)
        video_path = dest

    # ── Create output subfolder: ~/Downloads/<title>/
    out_dir = os.path.join(dl_dir, title)
    os.makedirs(out_dir, exist_ok=True)

    # Move original into subfolder
    orig = os.path.join(out_dir, f"{title}.mp4")
    if os.path.abspath(video_path) != os.path.abspath(orig):
        shutil.move(video_path, orig)
    video_path = orig

    paths = {
        "original_video":     video_path,
        "silent_video":       os.path.join(out_dir, f"{title}_no_audio.mp4"),
        "dubbed_video":       os.path.join(out_dir, f"{title}_turkish_voiceover.mp4"),
        "original_audio_mp3": os.path.join(out_dir, f"{title}_original_audio.mp3"),
        "voiceover_mp3":      os.path.join(out_dir, f"{title}_turkish_voiceover.mp3"),
        "transcript":         os.path.join(out_dir, f"{title}_transcript.json"),
    }

    print(f"\n  Output folder: {out_dir}\n")

    # ── Quick exports (no ML needed)
    create_silent_video(video_path, paths["silent_video"])
    extract_original_audio_mp3(video_path, paths["original_audio_mp3"])

    # ── ML pipeline in a temp working directory
    work_dir = tempfile.mkdtemp(prefix="ar_tr_")
    try:
        total_dur = get_video_duration(video_path)
        print(f"\n  Video duration: {total_dur:.1f}s  ({total_dur/60:.1f} min)")

        raw_wav = os.path.join(work_dir, "audio_16k.wav")
        extract_audio_for_whisper(video_path, raw_wav)

        segments   = transcribe_arabic(raw_wav, args.whisper_model)
        translated = translate_segments(segments)

        with open(paths["transcript"], "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)
        print(f"      Transcript saved -> {paths['transcript']}")

        tts_segs      = generate_tts_segments(translated, work_dir, args.tts_model)
        voiceover_wav = assemble_audio_track(tts_segs, work_dir, total_dur)

        export_voiceover_mp3(voiceover_wav, paths["voiceover_mp3"])
        merge_into_video(video_path, voiceover_wav,
                         paths["dubbed_video"],
                         args.keep_bg_audio, args.bg_volume)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    print_summary(paths)


if __name__ == "__main__":
    main()
