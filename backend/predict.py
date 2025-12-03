import torch
import numpy as np
from backend.preprocessing import load_audio, chunk_audio
from backend.model_loader import load_vgg_model

MODEL_PATH = "model/vgg_frog_model.pth"
LABELS_PATH = "model/label_mapping.json"


model, label_map, device = load_vgg_model(MODEL_PATH, LABELS_PATH)

def analyze_audio(audio_bytes):
    y, sr = load_audio(audio_bytes)
    chunks = chunk_audio(y, sr)

    detections = []

    for chunk, start, end in chunks:
        x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)

        species = label_map[str(int(pred.item()))]

        detections.append({
            "start": float(start),
            "end": float(end),
            "species": species,
            "confidence": float(conf.item())
        })

    return detections