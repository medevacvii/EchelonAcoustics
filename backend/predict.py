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

def to_melspec(chunk):
    S = librosa.feature.melspectrogram(y=chunk, sr=TARGET_SR, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=np.max)


def analyze_audio(audio_bytes):
    """Stream audio in chunks & run inference without loading entire file."""
    detections = []

    start_time = 0.0
    CHUNK_SEC = 5.0

    # Stream chunks safely
    for chunk, sr in stream_audio_chunks(audio_bytes, CHUNK_SEC):

        # Convert chunk → mel-spectrogram
        S_dB = to_melspec(chunk)

        # Prepare model input
        x = torch.tensor(S_dB, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

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