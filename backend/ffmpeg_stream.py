import subprocess
import numpy as np
import tempfile
import os

TARGET_SR = 22050

def ffmpeg_stream_audio_bytes(audio_bytes, chunk_duration_sec=5.0):
    """
    A true streaming decoder that handles huge audio files without memory spikes.
    Used instead of SoundFile/librosa for large files.
    """

    # Write audio bytes to a temporary file for FFmpeg to read
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # ffmpeg command:
    # -i tmp file
    # -f f32le → float32 raw PCM output
    # -ac 1 → mono forced
    # -ar TARGET_SR → resample on the fly
    # -vn → ignore video tracks
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", tmp_path,
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-vn",
        "-"
    ]

    # Start FFmpeg streaming
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    chunk_size = int(chunk_duration_sec * TARGET_SR)

    try:
        while True:
            raw = process.stdout.read(chunk_size * 4)  # float32 = 4 bytes
            if not raw:
                break

            # Convert raw bytes → float32 numpy array
            audio = np.frombuffer(raw, dtype=np.float32)

            if len(audio) == 0:
                break

            yield audio, TARGET_SR

    finally:
        process.stdout.close()
        process.kill()
        os.remove(tmp_path)