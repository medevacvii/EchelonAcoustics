import subprocess
import numpy as np
import tempfile
import os

TARGET_SR = 22050

def ffmpeg_preview_audio_bytes(audio_bytes, start_sec=0.0, duration_sec=5.0):
    """
    Streams a specific audio slice [start_sec, start_sec+duration_sec]
    using ffmpeg. Safe for large files.
    Returns (float32 audio array, sample_rate)
    """

    # Write audio bytes to a temporary file for ffmpeg
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", tmp_path,
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-vn",
        "-"
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = process.stdout.read(int(duration_sec * TARGET_SR) * 4)

    process.stdout.close()
    process.kill()
    os.remove(tmp_path)

    y = np.frombuffer(raw, dtype=np.float32)
    return y, TARGET_SR