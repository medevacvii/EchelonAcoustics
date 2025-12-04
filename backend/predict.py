import torch
import librosa
from backend.preprocessing import stream_audio_chunks
from backend.model_loader import load_vgg_model

MODEL_PATH = "model/vgg_frog_model.pth"
LABEL_MAP_PATH = "model/label_mapping.json"

model, label_map, device = load_vgg_model(MODEL_PATH, LABEL_MAP_PATH)

TARGET_SR = 22050
N_MELS = 128
MIN_SAMPLES = 2048
CHUNK_SEC = 5.0

def analyze_audio(audio_bytes):
    detections = []
    start = 0.0

    for chunk, sr in stream_audio_chunks(audio_bytes, CHUNK_SEC):

        if chunk is None or len(chunk) < MIN_SAMPLES:
            start += CHUNK_SEC
            continue

        S = librosa.feature.melspectrogram(
            y=chunk,
            sr=TARGET_SR,
            n_mels=N_MELS,
            n_fft=2048,
            hop_length=512
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        x = torch.tensor(S_db).float().unsqueeze(0).unsqueeze(0).to(device)

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