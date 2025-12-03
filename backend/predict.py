import torch
import numpy as np
import librosa
from backend.preprocessing import load_audio, chunk_audio
from backend.model_loader import load_vgg_model

MODEL_PATH = "model/vgg_frog_model.pth"
LABEL_MAP_PATH = "model/label_mapping.json"

model, label_map, device = load_vgg_model(MODEL_PATH, LABEL_MAP_PATH)

N_MELS = 128
TARGET_SR = 22050

def to_melspec(chunk):
    S = librosa.feature.melspectrogram(y=chunk, sr=TARGET_SR, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


def analyze_audio(audio_bytes):
    y, sr = load_audio(audio_bytes)
    chunks = chunk_audio(y, sr)
    detections = []

    for chunk, start, end in chunks:

        # Convert WAV chunk → Mel spectrogram
        S_dB = to_melspec(chunk)

        # Shape → (1, 1, 128, time_frames)
        x = torch.tensor(S_dB, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred = probs.max(dim=0)

        species = label_map[str(int(pred.item()))]

        detections.append({
            "start": float(start),
            "end": float(end),
            "species": species,
            "confidence": float(conf.item())
        })

    return detections