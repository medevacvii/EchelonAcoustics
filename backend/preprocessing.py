import io
import librosa
import soundfile as sf
import numpy as np

TARGET_SR = 22050
CHUNK_SEC = 5
CHUNK_SAMPLES = TARGET_SR * CHUNK_SEC

def load_audio(audio_bytes):
    """Load audio from raw bytes (Streamlit upload)."""
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_buffer, sr=TARGET_SR)
    return y, sr

def chunk_audio(y, sr):
    """Split audio into fixed-length chunks."""
    chunks = []
    total_samples = len(y)
    chunk_len = CHUNK_SAMPLES
    num_chunks = total_samples // chunk_len

    for i in range(num_chunks):
        start = i * chunk_len
        end = start + chunk_len
        chunk = y[start:end]
        chunks.append((chunk, start / sr, end / sr))

    return chunks

def stream_audio_chunks(audio_bytes, chunk_duration=5.0):
    """Yield audio chunks without loading full file into RAM."""

    # Wrap bytes into a file-like object
    import io
    bio = io.BytesIO(audio_bytes)

    with sf.SoundFile(bio) as f:
        sr = f.samplerate
        frames_per_chunk = int(chunk_duration * sr)

        while True:
            data = f.read(frames_per_chunk, dtype="float32")
            if len(data) == 0:
                break

            # Resample to model sample rate
            if sr != TARGET_SR:
                data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)

            yield data, TARGET_SR