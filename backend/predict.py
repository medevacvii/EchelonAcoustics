import os
import torch
import numpy as np
import librosa

from .model_loader import load_vgg_model
from .ffmpeg_stream import ffmpeg_stream_audio_bytes  # NEW: true streaming decoder

# ---------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "vgg_frog_model.pth")
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, "model", "label_mapping.json")

# Load model once
model, label_map, device = load_vgg_model(MODEL_PATH, LABEL_MAP_PATH)

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------
TARGET_SR = 22050
N_MELS = 128
FIXED_FRAMES = 128
MIN_SAMPLES = 2048  # skip garbage chunks
DEFAULT_CHUNK = 5.0  # seconds


# =============================================================================
# MEL PREPROCESSING WITH SILENCE GATE
# =============================================================================
def preprocess_mel(chunk, sr):
    """
    Convert raw audio chunk (float32) into model-ready (1, 128, 128) mel tensor.
    Includes energy gate → return None for silence.
    """

    # 1. Silence / low energy gate
    energy = np.mean(chunk ** 2)
    if energy < 1e-4:
        return None

    # 2. (Already mono + resampled by ffmpeg)
    #    So sr == TARGET_SR always.

    # 3. Mel spectrogram
    S = librosa.feature.melspectrogram(
        y=chunk,
        sr=TARGET_SR,
        n_mels=N_MELS
    )

    # 4. Force width to FIXED_FRAMES=128
    current = S.shape[1]
    if current < FIXED_FRAMES:
        pad = FIXED_FRAMES - current
        S = np.pad(S, ((0, 0), (0, pad)), mode="constant")
    else:
        S = S[:, :FIXED_FRAMES]

    # 5. Convert to dB
    S_db = librosa.power_to_db(S, ref=np.max)

    # 6. Normalize to [-1, 1]
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    S_db = S_db * 2 - 1

    # 7. Convert to tensor (1, 128, 128)
    return torch.tensor(S_db, dtype=torch.float32).unsqueeze(0)


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================
def analyze_audio(audio_bytes):
    """
    Streams large audio safely using ffmpeg.
    Splits into chunks, runs mel preprocessing, then model inference.
    Returns list of:
    { start, end, species, confidence }
    """

    detections = []
    start = 0.0

    # ------------------------------------------------------
    # Dynamic chunk size: smaller for very large files
    # ------------------------------------------------------
    file_size_mb = len(audio_bytes) / (1024 * 1024)

    if file_size_mb > 200:
        chunk_duration = 1.5
    elif file_size_mb > 100:
        chunk_duration = 3.0
    else:
        chunk_duration = DEFAULT_CHUNK

    # ------------------------------------------------------
    # FFmpeg true streaming chunk iterator
    # ------------------------------------------------------
    for chunk, sr in ffmpeg_stream_audio_bytes(audio_bytes, chunk_duration_sec=chunk_duration):

        # Skip tiny or invalid chunks
        if chunk is None or len(chunk) < MIN_SAMPLES:
            start += chunk_duration
            continue

        mel = preprocess_mel(chunk, sr)

        # Silence or noise
        if mel is None:
            detections.append({
                "start": start,
                "end": start + chunk_duration,
                "species": "no_frog",
                "confidence": 1.0
            })
            start += chunk_duration
            continue

        # Normal inference
        x = mel.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred = probs.max(dim=0)

        species = label_map[str(int(pred.item()))]

        detections.append({
            "start": start,
            "end": start + chunk_duration,
            "species": species,
            "confidence": float(conf.item())
        })

        start += chunk_duration

    return detections