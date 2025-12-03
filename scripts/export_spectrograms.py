import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

INPUT_DIR = "../data/segments"
OUTPUT_DIR = "../data/spectrogram_images"
SR = 22050
N_MELS = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if fname.endswith(".wav"):
        y, sr = librosa.load(os.path.join(INPUT_DIR, fname), sr=SR)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_dB, sr=sr)
        plt.axis("off")

        out = fname.replace(".wav", ".png")
        plt.savefig(os.path.join(OUTPUT_DIR, out), bbox_inches="tight", pad_inches=0)
        plt.close()
        print("Exported:", out)