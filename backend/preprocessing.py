import io
import librosa
import soundfile as sf
import numpy as np

TARGET_SR = 22050

def stream_audio_chunks(audio_bytes, chunk_duration=5.0):
    """Yield audio chunks without loading full file into RAM."""

    bio = io.BytesIO(audio_bytes)

    with sf.SoundFile(bio) as f:
        sr = f.samplerate
        frames_per_chunk = int(chunk_duration * sr)

        while True:
            # Read frames directly from disk-like buffer
            data = f.read(frames_per_chunk, dtype="float32")
            if len(data) == 0:
                break

            # Resample chunk-to-chunk
            if sr != TARGET_SR:
                data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)

            yield data, TARGET_SR