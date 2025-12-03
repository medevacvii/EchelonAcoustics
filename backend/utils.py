import numpy as np
import librosa

TARGET_SR = 22050

def load_audio_file(path_or_bytes, sr=TARGET_SR):
    """
    Loads audio from a file path or raw bytes and resamples to TARGET_SR.
    """
    if isinstance(path_or_bytes, str):
        y, sr = librosa.load(path_or_bytes, sr=TARGET_SR)
    else:
        y, sr = librosa.load(path_or_bytes, sr=TARGET_SR)
    return y, sr


def pad_or_trim(y, length):
    """
    Ensure the audio length is exactly `length` samples by padding or trimming.
    """
    if len(y) < length:
        return np.pad(y, (0, length - len(y)))
    return y[:length]