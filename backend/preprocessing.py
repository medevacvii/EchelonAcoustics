import io
import soundfile as sf
import numpy as np
import soxr    # FAST resampler, avoids librosa's resampy dependency

TARGET_SR = 22050

def stream_audio_chunks(audio_bytes, chunk_duration=5.0):
    """Yield audio chunks without loading full file into RAM."""

    bio = io.BytesIO(audio_bytes)

    with sf.SoundFile(bio) as f:
        sr = f.samplerate
        frames_per_chunk = int(chunk_duration * sr)

        while True:
            data = f.read(frames_per_chunk, dtype="float32")
            if len(data) == 0:
                break

            # SAFE resampling using soxr (no resampy dependency)
            if sr != TARGET_SR:
                data = soxr.resample(data, sr, TARGET_SR)

            yield data, TARGET_SR