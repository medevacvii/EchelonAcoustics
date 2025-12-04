import io
import soundfile as sf
import numpy as np
import soxr
import librosa

TARGET_SR = 22050

def stream_audio_chunks(audio_bytes, chunk_duration=5.0):
    """
    Stream large audio safely in chunks without loading everything into memory.
    Uses soxr instead of librosa.resample to avoid missing 'resampy' dependency.
    """

    bio = io.BytesIO(audio_bytes)

    with sf.SoundFile(bio) as f:
        sr = f.samplerate
        frames_per_chunk = int(chunk_duration * sr)

        while True:
            data = f.read(frames_per_chunk, dtype="float32")

            if len(data) == 0:
                break

            # Safe resample using soxr (fast, installed on Streamlit)
            if sr != TARGET_SR:
                data = soxr.resample(data, sr, TARGET_SR)

            yield data, TARGET_SR