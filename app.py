import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import io

from backend.predict import analyze_audio

st.set_page_config(
    page_title="Frog Call Classifier",
    layout="wide"
)

MIN_SAMPLES = 2048
VIS_SR = 22050

# =============================================================================
# SAFE VISUALIZATION FUNCTION (FIRST 10s ONLY)
# =============================================================================
def get_visualization_data(audio_bytes, max_vis_sec=10):
    """
    Load only the first few seconds for visualization (safe for huge files).
    Prevents librosa crashes from corrupted tail frames.
    """
    try:
        bio = io.BytesIO(audio_bytes)

        with sf.SoundFile(bio) as f:
            sr = f.samplerate
            frames = min(len(f), sr * max_vis_sec)
            f.seek(0)
            y = f.read(frames, dtype="float32")

        # Guard: skip if too small
        if len(y) < MIN_SAMPLES:
            return None, None, None

        # Resample for display
        y_vis = librosa.resample(
            y,
            orig_sr=sr,
            target_sr=VIS_SR,
            res_type="kaiser_fast"
        )

        # Mel Spectrogram
        S = librosa.feature.melspectrogram(
            y=y_vis,
            sr=VIS_SR,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        return y_vis, VIS_SR, S_dB

    except Exception as e:
        st.error(f"Visualization failed: {e}")
        return None, None, None


# =============================================================================
# SEGMENT AUDIO HELPER FOR PLAYBACK
# =============================================================================
def extract_segment_audio(audio_bytes, start_sec, end_sec):
    try:
        bio = io.BytesIO(audio_bytes)
        with sf.SoundFile(bio) as f:
            sr = f.samplerate
            start_frame = int(start_sec * sr)
            end_frame = int(end_sec * sr)
            num_frames = max(0, end_frame - start_frame)

            f.seek(start_frame)
            segment = f.read(num_frames, dtype="float32")

        if len(segment) == 0:
            return None

        out_bio = io.BytesIO()
        sf.write(out_bio, segment, sr, format="WAV")
        out_bio.seek(0)
        return out_bio.read()

    except Exception as e:
        st.error(f"Failed to extract segment audio: {e}")
        return None


# =============================================================================
# HEADER
# =============================================================================
st.title("🐸 Frog Call Classifier")
st.write("Upload frog audio recordings to analyze species, timestamps, and confidence scores.")


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

    # ---- File size checks ----
    if f.size > MAX_MB * 1024 * 1024:
        st.error(f"❌ File '{f.name}' is too large. Max allowed is {MAX_MB} MB.")
        st.stop()

    if f.size > 100 * 1024 * 1024:
        st.warning(f"⚠ File '{f.name}' is large and may take longer to process.")

    audio_bytes = f.read()

    # =============================================================================
    # AUDIO PLAYER
    # =============================================================================
    st.subheader("🎧 Full Audio Playback")
    st.audio(audio_bytes)

    # =============================================================================
    # SAFE WAVEFORM + SPECTROGRAM (first 10 seconds)
    # =============================================================================
    st.subheader("📈 Waveform & Spectrogram Preview (first 10 seconds)")

    y_vis, sr_vis, S_vis = get_visualization_data(audio_bytes)

    if y_vis is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Waveform**")
            fig, ax = plt.subplots(figsize=(7, 3))
            librosa.display.waveshow(y_vis, sr=sr_vis, ax=ax)
            ax.set_title("Waveform (Preview)")
            st.pyplot(fig)

        with col2:
            st.write("**Spectrogram**")
            fig, ax = plt.subplots(figsize=(7, 3))
            img = librosa.display.specshow(
                S_vis,
                sr=sr_vis,
                x_axis="time",
                y_axis="mel",
                ax=ax
            )
            ax.set_title("Spectrogram (Preview)")
            fig.colorbar(img, ax=ax, format="%+2.f dB")
            st.pyplot(fig)

    # =============================================================================
    # STREAMING INFERENCE
    # =============================================================================
    st.subheader(f"📊 Model Detection Results for: {f.name}")

    with st.spinner("Analyzing full audio..."):
        detections = analyze_audio(audio_bytes)

    df_raw = pd.DataFrame(detections)
    st.dataframe(df_raw, use_container_width=True)

    # =============================================================================
    # SPECIES TIMELINE
    # =============================================================================
    st.subheader("📌 Species Timeline")

    if not df_raw.empty:
        fig, ax = plt.subplots(figsize=(12, 2))
        for _, row in df_raw.iterrows():
            ax.plot([row["start"], row["end"]], [row["species"], row["species"]], linewidth=6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Species")
        ax.set_title("Detected Calls Over Time")
        st.pyplot(fig)

    # =============================================================================
    # CONFIDENCE SUMMARY
    # =============================================================================
    st.subheader("📊 Average Confidence per Species")

    if not df_raw.empty:
        conf_summary = (
            df_raw.groupby("species")["confidence"]
            .mean()
            .reset_index()
            .sort_values("confidence", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(conf_summary["species"], conf_summary["confidence"], color="skyblue")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Avg confidence")
        st.pyplot(fig)

    # =============================================================================
    # SEGMENT PLAYBACK
    # =============================================================================
    st.subheader("🎯 Play Detected Frog Calls")

    if df_raw.empty:
        st.info("No detections found in this file.")
    else:
        species_options = ["All species"] + sorted(df_raw["species"].unique())
        selected_species = st.selectbox("Filter detections by species", species_options)

        if selected_species == "All species":
            df_play = df_raw.copy()
        else:
            df_play = df_raw[df_raw["species"] == selected_species].copy()

        df_play = df_play.sort_values("start").reset_index(drop=True)

        if df_play.empty:
            st.warning("No detections for the selected species.")
        else:
            options = [
                f"{i+1}: {row['species']} {row['start']:.1f}–{row['end']:.1f}s (conf {row['confidence']:.2f})"
                for i, row in df_play.iterrows()
            ]

            selected_idx = st.selectbox("Select a detection to play", options, index=0)
            selected_row = df_play.iloc[options.index(selected_idx)]

            st.write(
                f"**Selected:** {selected_row['species']} from "
                f"{selected_row['start']:.2f}s to {selected_row['end']:.2f}s "
                f"(confidence {selected_row['confidence']:.3f})"
            )

            if st.button("▶️ Play this detected call"):
                seg_bytes = extract_segment_audio(
                    audio_bytes,
                    float(selected_row["start"]),
                    float(selected_row["end"])
                )
                if seg_bytes:
                    st.audio(seg_bytes, format="audio/wav")
                else:
                    st.error("Could not extract that segment.")

    st.divider()