import torch
import numpy as np
import librosa
from backend.preprocessing import stream_audio_chunks
from backend.model_loader import load_vgg_model

MODEL_PATH = "model/vgg_frog_model.pth"
LABEL_MAP_PATH = "model/label_mapping.json"

# Load ONCE globally – never reload per chunk
model, label_map, device = load_vgg_model(MODEL_PATH, LABEL_MAP_PATH)

N_MELS = 128
TARGET_SR = 22050

# Minimum samples needed for n_fft=2048
MIN_SAMPLES = 2048

def analyze_audio(audio_bytes):
    """
    Stream audio in fixed-size chunks and run inference.
    This version is SAFE for large files and avoids librosa crashes
    caused by incomplete final chunks.
    """
    detections = []

    CHUNK_SEC = 5.0       # inference resolution
    start_time = 0.0

    # Stream chunks safely
    for chunk, sr in stream_audio_chunks(audio_bytes, CHUNK_SEC):

        # Skip tiny trailing chunks that cause librosa crashes
        if len(chunk) < MIN_SAMPLES:
            start_time += CHUNK_SEC
            continue

        # Compute mel spectrogram safely
        S = librosa.feature.melspectrogram(
            y=chunk,
            sr=TARGET_SR,
            n_mels=N_MELS,
            n_fft=2048,
            hop_length=512
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Prepare tensor for model
        x = torch.tensor(S_dB, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred = probs.max(dim=0)

        species = label_map[str(int(pred.item()))]

        detections.append({
            "start": start_time,
            "end": start_time + CHUNK_SEC,
            "species": species,
            "confidence": float(conf.item())
        })

        start_time += CHUNK_SEC

    return detections