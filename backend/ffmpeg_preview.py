import subprocess
import numpy as np
import tempfile
import os
import librosa

TARGET_SR = 22050

def ffmpeg_preview_audio_bytes(audio_bytes, duration_sec=5.0):
    """
    Streams only the first N seconds of audio using ffmpeg.
    Safe for huge files. Returns mono float32 array + sr.
    """

    # Write bytes to temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", "0",                     # always start at 0 sec
        "-t", str(duration_sec),        # read only N seconds
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