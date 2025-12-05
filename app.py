import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import librosa
import librosa.display
import soundfile as sf
import io
import os
import sys

# ---------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from backend.predict import analyze_audio  # streaming + VGG model

# ---------------------------------------------------------------------
# CONSTANTS / LIMITS
# ---------------------------------------------------------------------
MIN_SAMPLES = 2048
MAX_MB = 200                     
MAX_ROWS_TO_DISPLAY = 500        
MAX_SEGMENTS_FOR_TIMELINE = 500  
MAX_PLAY_OPTIONS = 300           
LARGE_FILE_BYTES = 150 * 1024 * 1024  


# =============================================================================
# SAFE VISUALIZATION FUNCTION (FIRST 10s ONLY)
# =============================================================================
def get_visualization_data(audio_bytes, max_vis_sec=10):
    try:
        bio = io.BytesIO(audio_bytes)
        with sf.SoundFile(bio) as f:
            sr = f.samplerate
            frames = min(len(f), int(sr * max_vis_sec))
            f.seek(0)
            y = f.read(frames, dtype="float32")

        if y is None or len(y) == 0:
            return None, None, None

        y_vis = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr_vis = 22050

        if len(y_vis) < MIN_SAMPLES:
            return y_vis, sr_vis, None

        S = librosa.feature.melspectrogram(y=y_vis, sr=sr_vis, n_mels=128)

        if S.shape[1] == 0:
            S = np.zeros((128, 1))

        S_db = librosa.power_to_db(S, ref=np.max)
        return y_vis, sr_vis, S_db

    except Exception as e:
        st.error(f"Visualization failed: {e}")
        return None, None, None


# =============================================================================
# SEGMENT AUDIO FOR PLAYBACK
# =============================================================================
def extract_segment_audio(audio_bytes, start_sec, end_sec):
    try:
        bio = io.BytesIO(audio_bytes)
        with sf.SoundFile(bio) as f:
            sr = f.samplerate
            start_frame = int(start_sec * sr)
            end_frame = int(end_sec * sr)
            f.seek(start_frame)
            segment = f.read(end_frame - start_frame, dtype="float32")

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
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Frog Call Classifier",
    layout="wide"
)

st.title("🐸 Frog Call Classifier")

st.write(
    "Upload frog audio recordings to analyze species calls, timestamps, "
    "confidence scores, and play individual segments. Optimized for large files."
)

# =============================================================================
# FILE UPLOAD
# =============================================================================
uploaded_files = st.file_uploader(
    "Upload frog audio file(s) (WAV/MP3)",
    type=["wav", "mp3"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload audio files to begin analysis.")
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
            f"⚠ Large file detected ({f.size/1024/1024:.1f} MB). "
            "Processing may take a while."
        )

    audio_bytes = f.read()

    # =============================================================================
    # AUDIO PLAYER
    # =============================================================================
    st.subheader("🎧 Full Audio Playback")
    st.audio(audio_bytes)

    # =============================================================================
    # WAVEFORM & SPECTROGRAM (SAFE)
    # =============================================================================
    st.subheader("📈 Waveform & Spectrogram (First 10 Seconds)")

    if len(audio_bytes) > LARGE_FILE_BYTES:
        st.warning("Skipping waveform & spectrogram preview due to file size.")
        y_vis = sr_vis = S_vis = None
    else:
        y_vis, sr_vis, S_vis = get_visualization_data(audio_bytes)

    if y_vis is not None:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7, 3))
            librosa.display.waveshow(y_vis, sr=sr_vis, ax=ax)
            ax.set_title("Waveform (Preview)")
            st.pyplot(fig)

        with col2:
            if S_vis is not None:
                fig, ax = plt.subplots(figsize=(7, 3))
                img = librosa.display.specshow(
                    S_vis, sr=sr_vis, x_axis="time", y_axis="mel", ax=ax
                )
                ax.set_title("Mel Spectrogram (Preview, dB)")
                fig.colorbar(img, ax=ax, format="%+2.f dB")
                st.pyplot(fig)

    # =============================================================================
    # RUN MODEL (STREAMING INFERENCE)
    # =============================================================================
    st.subheader(f"📊 Model Detection Results for: {f.name}")

    with st.spinner("Analyzing full audio..."):
        detections = analyze_audio(audio_bytes)

    df_raw = pd.DataFrame(detections)

    if df_raw.empty:
        st.warning("No detections found in this file.")
        st.stop()

    total_duration = float(df_raw["end"].max())

    # =============================================================================
    # PER-SPECIES CONFIDENCE FILTERS
    # =============================================================================
    st.subheader("🎚 Per-Species Confidence Filtering")

    species_list = sorted(df_raw["species"].unique())
    thresholds = {}

    cols = st.columns(len(species_list))
    for i, sp in enumerate(species_list):
        with cols[i]:
            thresholds[sp] = st.slider(
                f"{sp} min conf",
                min_value=0.0, max_value=1.0,
                value=0.0, step=0.05
            )

    # Apply filtering
    df_filtered = df_raw.copy()
    for sp, th in thresholds.items():
        df_filtered = df_filtered[
            ~((df_filtered["species"] == sp) & (df_filtered["confidence"] < th))
        ]

    if df_filtered.empty:
        st.error("All detections were filtered out by thresholds.")
        st.stop()

    # =============================================================================
    # CSV EXPORT
    # =============================================================================
    st.subheader("📄 Export Detections to CSV")

    csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download CSV (Filtered Detections)",
        data=csv_bytes,
        file_name=f"{selected_file_name}_detections.csv",
        mime="text/csv"
    )

    # =============================================================================
    # DETECTION TABLE (LIMITED)
    # =============================================================================
    st.subheader("📋 Detection Table (Filtered)")

    if len(df_filtered) > MAX_ROWS_TO_DISPLAY:
        st.warning(f"Showing first {MAX_ROWS_TO_DISPLAY} rows of {len(df_filtered)}.")
        st.dataframe(df_filtered.head(MAX_ROWS_TO_DISPLAY))
    else:
        st.dataframe(df_filtered)

    # =============================================================================
    # TIMELINE (PLOTLY, WINDOWED)
    # =============================================================================
    st.subheader("📌 Species Timeline (Interactive)")

    if len(df_filtered) > MAX_SEGMENTS_FOR_TIMELINE:
        st.warning(
            f"File contains {len(df_filtered)} filtered detections. "
            "Timeline is shown in windows for performance."
        )

        default_window = min(120.0, total_duration)
        window_size = st.slider(
            "Timeline window size (seconds)",
            min_value=30.0, max_value=float(total_duration),
            value=float(default_window), step=10.0
        )

        max_start = max(0.0, total_duration - window_size)
        window_start = st.slider(
            "Timeline start time (seconds)",
            min_value=0.0,
            max_value=float(max_start),
            value=0.0,
            step=5.0,
        )

        window_end = window_start + window_size

        df_timeline = df_filtered[
            (df_filtered["start"] >= window_start) &
            (df_filtered["end"] <= window_end)
        ].copy()

        st.write(
            f"Showing {len(df_timeline)} segments "
            f"from {window_start:.1f}s to {window_end:.1f}s"
        )

    else:
        df_timeline = df_filtered.copy()

    # -------- PLOTLY TIMELINE --------
    if not df_timeline.empty:
        fig = go.Figure()

        species_colors = {
            sp: f"rgba({50+hash(sp)%200}, {50+(hash(sp)*7)%200}, {50+(hash(sp)*13)%200}, 0.85)"
            for sp in species_list
        }

        for _, row in df_timeline.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["start"], row["end"]],
                y=[row["species"], row["species"]],
                mode="lines",
                line=dict(width=6, color=species_colors[row["species"]]),
                showlegend=False
            ))

        fig.update_layout(
            height=300,
            xaxis_title="Time (s)",
            yaxis_title="Species",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No detections in this timeline window.")

    # =============================================================================
    # PLAYBACK (FILTERED + WINDOWED)
    # =============================================================================
    st.subheader("🎯 Play Detected Calls")

    df_play = df_timeline.copy()

    species_options = ["All species"] + sorted(df_play["species"].unique())
    selected_species = st.selectbox(
        "Filter detections by species",
        species_options
    )

    if selected_species != "All species":
        df_play = df_play[df_play["species"] == selected_species]

    df_play = df_play.sort_values("start")

    if df_play.empty:
        st.warning("No detections for the selected species in this window.")
        st.stop()

    if len(df_play) > MAX_PLAY_OPTIONS:
        st.warning(f"Showing first {MAX_PLAY_OPTIONS} detections.")
        df_play = df_play.head(MAX_PLAY_OPTIONS)

    options = [
        f"{i+1}: {row['species']} {row['start']:.1f}–{row['end']:.1f}s (conf {row['confidence']:.2f})"
        for i, row in df_play.iterrows()
    ]

    selected_idx = st.selectbox("Select a detection to play", options)
    selected_row = df_play.iloc[options.index(selected_idx)]

    if st.button("▶️ Play detection"):
        seg_bytes = extract_segment_audio(
            audio_bytes,
            selected_row["start"],
            selected_row["end"]
        )
        if seg_bytes:
            st.audio(seg_bytes, format="audio/wav")
        else:
            st.error("Could not extract this segment.")

    st.divider()
