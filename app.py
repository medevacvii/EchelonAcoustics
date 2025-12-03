import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import io

from backend.predict import analyze_audio

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Frog Call Classifier",
    layout="wide"
)

# =============================================================================
# SAFE VISUALIZATION FUNCTION (FIRST 10s ONLY)
# =============================================================================
def get_visualization_data(audio_bytes, max_vis_sec=10):
    """Load ONLY the first few seconds for visualization (safe for huge files)."""
    try:
        bio = io.BytesIO(audio_bytes)

        with sf.SoundFile(bio) as f:
            sr = f.samplerate
            frames = min(len(f), sr * max_vis_sec)
            f.seek(0)
            y = f.read(frames, dtype="float32")

        # Resample for consistent display
        y_vis = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr_vis = 22050

        # Mel Spectrogram
        S = librosa.feature.melspectrogram(
            y=y_vis,
            sr=sr_vis,
            n_mels=128
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        return y_vis, sr_vis, S_dB

    except Exception as e:
        st.error(f"Visualization failed: {e}")
        return None, None, None


# =============================================================================
# HEADER
# =============================================================================
st.title("🐸 Frog Call Classifier")
st.write(
    "Upload frog audio recordings to analyze species, timestamps, "
    "and confidence scores. Supports very large audio files using safe streaming."
)

# =============================================================================
# FILE UPLOAD
# =============================================================================
uploaded_files = st.file_uploader(
    "Upload frog audio file(s) (WAV/MP3)",
    type=["wav", "mp3"],
    accept_multiple_files=True
)

MAX_MB = 200

if not uploaded_files:
    st.info("Upload one or more audio files to begin analysis.")
    st.stop()

file_names = [f.name for f in uploaded_files]

selected_file_name = st.selectbox("Select a file to inspect", file_names)

# =============================================================================
# PROCESS SELECTED FILE
# =============================================================================
for f in uploaded_files:

    if f.name != selected_file_name:
        continue

    if f.size > MAX_MB * 1024 * 1024:
        st.error(
            f"❌ File '{f.name}' is too large "
            f"({f.size/1024/1024:.1f} MB). Max allowed is {MAX_MB} MB."
        )
        st.stop()

    if f.size > 100 * 1024 * 1024:
        st.warning(
            f"⚠ '{f.name}' is large ({f.size/1024/1024:.1f} MB). "
            "Processing may take a while."
        )

    audio_bytes = f.read()

    # =============================================================================
    # AUDIO PLAYER
    # =============================================================================
    st.subheader("🎧 Full Audio Playback")
    st.audio(audio_bytes)

    # =============================================================================
    # SAFE WAVEFORM + SPECTROGRAM
    # =============================================================================
    st.subheader("📈 Waveform & Spectrogram Preview (first 10 seconds)")

    y_vis, sr_vis, S_vis = get_visualization_data(audio_bytes)

    if y_vis is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Waveform**")
            fig, ax = plt.subplots(figsize=(7, 3))
            librosa.display.waveshow(y_vis, sr=sr_vis, ax=ax)
            ax.set_title("Waveform (First 10 Seconds)")
            st.pyplot(fig)

        with col2:
            st.write("**Spectrogram**")
            fig, ax = plt.subplots(figsize=(7, 3))

            # Correct spectrogram display — capture image handle
            img = librosa.display.specshow(
                S_vis,
                sr=sr_vis,
                x_axis="time",
                y_axis="mel",
                ax=ax
            )

            ax.set_title("Mel Spectrogram (dB, First 10 Seconds)")
            fig.colorbar(img, ax=ax, format="%+2.f dB")
            st.pyplot(fig)

    # =============================================================================
    # STREAMING INFERENCE
    # =============================================================================
    st.subheader(f"📊 Model Detection Results for: {f.name}")

    with st.spinner("Analyzing full audio (streaming mode)..."):
        detections = analyze_audio(audio_bytes)

    df_raw = pd.DataFrame(detections)
    st.dataframe(df_raw, use_container_width=True)

    # =============================================================================
    # TIMELINE PLOT
    # =============================================================================
    st.subheader("📌 Species Timeline")

    if not df_raw.empty:
        fig, ax = plt.subplots(figsize=(12, 2))
        for _, row in df_raw.iterrows():
            ax.plot(
                [row["start"], row["end"]],
                [row["species"], row["species"]],
                linewidth=6
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Species")
        ax.set_title("Detected Calls Over Time")
        st.pyplot(fig)

    # =============================================================================
    # CONFIDENCE SUMMARY
    # =============================================================================
    st.subheader("📊 Average Confidence per Species")

    if not df_raw.empty:
        conf_summary = df_raw.groupby("species")["confidence"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(conf_summary["species"], conf_summary["confidence"], color="skyblue")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    st.divider()