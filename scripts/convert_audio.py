import os
from pydub import AudioSegment

INPUT_DIR = "../data/raw_recordings"
OUTPUT_DIR = "../data/converted"
TARGET_SR = 22050

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if fname.lower().endswith((".wav", ".mp3", ".m4a", ".aac")):
        audio = AudioSegment.from_file(os.path.join(INPUT_DIR, fname))
        audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
        out_path = os.path.join(OUTPUT_DIR, fname.replace(".mp3", ".wav"))
        audio.export(out_path, format="wav")
        print("Converted:", out_path)