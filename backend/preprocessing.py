import io
import librosa
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