import librosa
import numpy as np

TARGET_SR = 22050
WINDOW_SIZE = 5.0  # seconds

def load_audio(audio_bytes):
    y, sr = librosa.load(audio_bytes, sr=TARGET_SR)
    return y, sr

def chunk_audio(y, sr, sec=WINDOW_SIZE):
    samples = int(sec * sr)
    num_chunks = len(y) // samples
    chunks = []

    for i in range(num_chunks):
        start = i * samples
        end = start + samples
        chunks.append((y[start:end], i * sec, (i + 1) * sec))

    return chunks