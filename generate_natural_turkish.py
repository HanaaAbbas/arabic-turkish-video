"""
Natural Speed Turkish Audio Generator
======================================
Reads the existing _transcript.json and regenerates the Turkish voiceover
at completely natural speed — no stretching, no timing constraints.

Usage:
    python generate_natural_turkish.py --transcript "path/to/_transcript.json"

Output:
    A _turkish_natural.mp3 file saved next to the transcript file.
"""

import argparse
import os
import sys
import json
import subprocess
import tempfile
import shutil

import numpy as np
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", required=True,
                        help="Path to the _transcript.json file")
    return parser.parse_args()


def gtts_segment(text: str, out_path: str):
    """Synthesize one Turkish segment using gTTS at natural speed."""
    from gtts import gTTS
    import time
    tts     = gTTS(text=text, lang="tr", slow=False)
    mp3_tmp = out_path.replace(".wav", "_tmp.mp3")
    tts.save(mp3_tmp)
    subprocess.run([
        "ffmpeg", "-y", "-i", mp3_tmp,
        "-ar", "22050", "-ac", "1", out_path
    ], capture_output=True)
    os.remove(mp3_tmp)
    time.sleep(0.3)


def main():
    args = parse_args()

    if not os.path.exists(args.transcript):
        print(f"Error: Transcript not found: {args.transcript}")
        sys.exit(1)

    with open(args.transcript, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"Loaded {len(segments)} segments from transcript")

    out_dir  = os.path.dirname(args.transcript)
    base     = os.path.splitext(os.path.basename(args.transcript))[0]
    base     = base.replace("_transcript", "")
    out_mp3  = os.path.join(out_dir, f"{base}_turkish_natural.mp3")

    work_dir = tempfile.mkdtemp(prefix="natural_tts_")

    try:
        wav_files = []

        for i, seg in enumerate(segments):
            text = seg.get("tr", "").strip()
            out_wav = os.path.join(work_dir, f"seg_{i:04d}.wav")

            if not text:
                # Write a short silence for empty segments
                sf.write(out_wav, np.zeros(2205, dtype=np.float32), 22050)
            else:
                try:
                    gtts_segment(text, out_wav)
                except Exception as e:
                    print(f"  Warning seg {i}: {e} — using silence")
                    sf.write(out_wav, np.zeros(2205, dtype=np.float32), 22050)

            wav_files.append(out_wav)
            print(f"  TTS: {i+1}/{len(segments)}", end="\r")

        print(f"\n  All segments synthesized, concatenating...")

        # Write a file list for FFmpeg concat
        list_file = os.path.join(work_dir, "files.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for wav in wav_files:
                # FFmpeg concat list requires forward slashes and escaped paths
                safe = wav.replace("\\", "/")
                f.write(f"file '{safe}'\n")

        # Concatenate all WAVs into one, then convert to MP3
        concat_wav = os.path.join(work_dir, "concat.wav")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-ar", "22050", "-ac", "1",
            concat_wav
        ], capture_output=True)

        subprocess.run([
            "ffmpeg", "-y", "-i", concat_wav,
            "-acodec", "libmp3lame", "-ab", "192k", "-ar", "44100",
            out_mp3
        ], capture_output=True)

        if os.path.exists(out_mp3):
            mb = os.path.getsize(out_mp3) / 1_048_576
            print(f"\n  Done! Natural speed Turkish audio saved:")
            print(f"  {out_mp3}  ({mb:.1f} MB)")
        else:
            print("\n  Error: output file was not created.")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
