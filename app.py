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

MIN_SAMPLES = 2048  # must be >= n_fft for safe spectrograms

# =============================================================================
# SAFE VISUALIZATION FUNCTION (FIRST 10s ONLY)
# =============================================================================
def get_visualization_data(audio_bytes, max_vis_sec=10):
    """Load ONLY the first few seconds for visualization (safe for huge files)."""
    try:
        bio = io.BytesIO(audio_bytes)

        with sf.SoundFile(bio) as f:
            sr = f.samplerate
            frames = min(len(f), int(sr * max_vis_sec))
            f.seek(0)
            y = f.read(frames, dtype="float32")

        # If we somehow got nothing, bail out
        if y is None or len(y) == 0:
            return None, None, None

        # Resample for consistent display
        y_vis = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr_vis = 22050

        # If the preview is too short for a spectrogram, return waveform only
        if len(y_vis) < MIN_SAMPLES:
            return y_vis, sr_vis, None

        # Compute mel spectrogram (no n_fft/hop_length override — use defaults)
        S = librosa.feature.melspectrogram(
            y=y_vis,
            sr=sr,
            n_mels=128
        )
        
        # Ensure width is reasonable for preview (cap extremely short frames)
        if S.shape[1] == 0:
            S = np.zeros((128, 1))

        S_db = librosa.power_to_db(S, ref=np.max)
        
        return y_vis, sr_vis, S_db

    except Exception as e:
        st.error(f"Visualization failed: {e}")
        return None, None, None


# =============================================================================
# SEGMENT AUDIO HELPER FOR PLAYBACK
# =============================================================================
def extract_segment_audio(audio_bytes, start_sec, end_sec):
    """
    Extract a precise audio segment [start_sec, end_sec] from the original file,
    without loading the full file into RAM.
    Returns raw WAV bytes suitable for st.audio().
    """
    try:
        bio = io.BytesIO(audio_bytes)
        with sf.SoundFile(bio) as f:
            sr = f.samplerate
            start_frame = int(start_sec * sr)
            end_frame = int(end_sec * sr)
            num_frames = max(0, end_frame - start_frame)

            f.seek(start_frame)
            segment = f.read(num_frames, dtype="float32")

        if segment is None or len(segment) == 0:
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

    # ---- File size checks ----
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

    # Read bytes once – reused for visualization and segment playback
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

        # Waveform
        with col1:
            st.write("**Waveform**")
            fig, ax = plt.subplots(figsize=(7, 3))
            librosa.display.waveshow(y_vis, sr=sr_vis, ax=ax)
            ax.set_title("Waveform (First 10 Seconds)")
            st.pyplot(fig)

        # Spectrogram (only if we have a valid S_vis)
        with col2:
            st.write("**Spectrogram**")
            if S_vis is not None:
                fig, ax = plt.subplots(figsize=(7, 3))
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
            else:
                st.info(
                    "Preview segment is too short for a spectrogram. "
                    "Waveform is shown instead."
                )

    # =============================================================================
    # STREAMING INFERENCE (BACKEND MODEL)
    # =============================================================================
    st.subheader(f"📊 Model Detection Results for: {f.name}")

    with st.spinner("Analyzing full audio (streaming mode)..."):
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
    # SEGMENT-LEVEL PLAYBACK
    # =============================================================================
    st.subheader("🎯 Play Detected Frog Calls")

    if df_raw.empty:
        st.info("No detections found in this file.")
    else:
        # Species filter
        species_options = ["All species"] + sorted(df_raw["species"].unique())
        selected_species = st.selectbox(
            "Filter detections by species",
            species_options,
            index=0
        )

        if selected_species == "All species":
            df_play = df_raw.copy()
        else:
            df_play = df_raw[df_raw["species"] == selected_species].copy()

        df_play = df_play.sort_values("start").reset_index(drop=True)

        if df_play.empty:
            st.warning("No detections for the selected species.")
        else:
            # Build human-readable options for each detection
            options = [
                f"{i+1}: {row['species']} "
                f"{row['start']:.1f}–{row['end']:.1f}s "
                f"(conf {row['confidence']:.2f})"
                for i, row in df_play.iterrows()
            ]

            selected_idx = st.selectbox(
                "Select a detection to play",
                options,
                index=0
            )

            selected_row = df_play.iloc[options.index(selected_idx)]

            st.write(
                f"**Selected detection:** {selected_row['species']} "
                f"from {selected_row['start']:.2f}s to {selected_row['end']:.2f}s "
                f"(confidence {selected_row['confidence']:.3f})"
            )

            if st.button("▶️ Play this detected call"):
                seg_bytes = extract_segment_audio(
                    audio_bytes,
                    float(selected_row["start"]),
                    float(selected_row["end"])
                )
                if seg_bytes is None:
                    st.error("Could not extract that segment from the audio.")
                else:
                    st.audio(seg_bytes, format="audio/wav")

    st.divider()