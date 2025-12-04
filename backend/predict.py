import torch
import librosa
import numpy as np

from backend.preprocessing import stream_audio_chunks
from backend.model_loader import load_vgg_model

MODEL_PATH = "model/vgg_frog_model.pth"
LABEL_MAP_PATH = "model/label_mapping.json"

# Load model once at import time
model, label_map, device = load_vgg_model(MODEL_PATH, LABEL_MAP_PATH)

TARGET_SR = 22050
N_MELS = 128
FIXED_FRAMES = 128
CHUNK_SEC = 5.0
MIN_SAMPLES = 2048  # skip garbage 1-sample chunks etc.


def preprocess_mel(chunk, sr):
    if sr != TARGET_SR:
        chunk = librosa.resample(chunk, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    S = librosa.feature.melspectrogram(
        y=chunk,
        sr=TARGET_SR,
        n_mels=N_MELS
    )

    # Force width to FIXED_FRAMES
    current = S.shape[1]
    if current < FIXED_FRAMES:
        S = np.pad(S, ((0, 0), (0, FIXED_FRAMES - current)), mode="constant")
    else:
        S = S[:, :FIXED_FRAMES]

    S_db = librosa.power_to_db(S, ref=np.max)

    # **CRITICAL: normalize to match training**
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    S_db = S_db * 2 - 1  # [-1, 1]

    return torch.tensor(S_db, dtype=torch.float32).unsqueeze(0)

def analyze_audio(audio_bytes):
    """
    Stream over the uploaded audio, chunk it into 5-second windows,
    run the VGG model on each window, and return a list of detections.
    """
    detections = []
    start = 0.0

    for chunk, sr in stream_audio_chunks(audio_bytes, CHUNK_SEC):
        # Some very large/wonky files can yield tiny leftover chunks.
        if chunk is None or len(chunk) < MIN_SAMPLES:
            start += CHUNK_SEC
            continue

        x = preprocess_mel(chunk, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred = probs.max(dim=0)

        species = label_map[str(int(pred.item()))]

        detections.append({
            "start": start,
            "end": start + CHUNK_SEC,
            "species": species,
            "confidence": float(conf.item())
        })

        start += CHUNK_SEC

    return detections