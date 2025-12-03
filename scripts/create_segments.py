import os
import librosa
import numpy as np
import soundfile as sf

INPUT_DIR = "../data/raw_recordings"
OUTPUT_DIR = "../data/segments"
CHUNK_SEC = 5
SR = 22050

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if fname.endswith(".wav"):
        y, sr = librosa.load(os.path.join(INPUT_DIR, fname), sr=SR)
        chunk_len = CHUNK_SEC * SR
        num_chunks = len(y) // chunk_len

        for i in range(num_chunks):
            start = i * chunk_len
            end = start + chunk_len
            chunk = y[start:end]

            out_name = f"{fname.replace('.wav', '')}_chunk_{i}.wav"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            sf.write(out_path, chunk, SR)
            print("Created:", out_path)