import io
import librosa
import soundfile as sf
import numpy as np

TARGET_SR = 22050
MIN_SAMPLES = 2048  # required for n_fft safety

def stream_audio_chunks(audio_bytes, chunk_duration=5.0):
    """
    Yield audio chunks safely without loading entire file into RAM.
    Ensures each chunk has enough samples to compute a mel-spectrogram
    without causing librosa to crash.
    """

    bio = io.BytesIO(audio_bytes)

    with sf.SoundFile(bio) as f:
        sr = f.samplerate
        frames_per_chunk = int(chunk_duration * sr)

        while True:
            # Read from file buffer
            data = f.read(frames_per_chunk, dtype="float32")
            if len(data) == 0:
                break

            # Skip tiny / corrupted chunks (common at end of file)
            if len(data) < MIN_SAMPLES:
                continue

            # Resample safely if needed
            if sr != TARGET_SR:
                data = librosa.resample(
                    data,
                    orig_sr=sr,
                    target_sr=TARGET_SR,
                    res_type="kaiser_fast"
                )

            yield data, TARGET_SR