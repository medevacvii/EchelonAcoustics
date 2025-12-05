import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import librosa
import soundfile as sf
import io
import os
import sys

# ------------------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from backend.predict import analyze_audio
from backend.ffmpeg_preview import ffmpeg_preview_audio_bytes  # NEW for preview


# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------
MIN_SAMPLES = 2048
MAX_ROWS_TO_DISPLAY = 500
MAX_SEGMENTS_FOR_TIMELINE = 500
MAX_PLAY_OPTIONS = 300
PREVIEW_SEC = 5.0   # Preview duration (seconds)
WAVEFORM_HEIGHT = 250
SPEC_HEIGHT = 250


# ==============================================================================
# WAVEFORM + SPECTROGRAM PREVIEW (PLOTLY + FFmpeg Safe Extraction)
# ==============================================================================
def get_preview_plotly(audio_bytes, preview_sec=PREVIEW_SEC):
    """
    Safely decode ONLY the first few seconds of audio using FFmpeg,
    then generate waveform + mel spectrogram using Plotly.
    """
    try:
        y, sr = ffmpeg_preview_audio_bytes(audio_bytes, duration_sec=preview_sec)

        if y is None or len(y) < MIN_SAMPLES:
            return None, None

        # ------------------- Waveform (Plotly) -------------------
        t = np.linspace(0, len(y) / sr, len(y))
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=t, y=y, mode="lines"))
        fig_wave.update_layout(
            title=f"Waveform Preview (First {preview_sec} sec)",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=WAVEFORM_HEIGHT,
            template="plotly_white",
        )

        # ------------------- Spectrogram (Plotly) -------------------
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        fig_spec = go.Figure(
            data=go.Heatmap(z=S_db, colorscale="Viridis")
        )
        fig_spec.update_layout(
            title=f"Mel Spectrogram Preview (First {preview_sec} sec)",
            xaxis_title="Frame Index",
            yaxis_title="Mel Bin",
            height=SPEC_HEIGHT,
            template="plotly_white",
        )

        return fig_wave, fig_spec

    except Exception as e:
        st.error(f"Preview failed: {e}")
        return None, None


# ==============================================================================
# AUDIO SEGMENT EXTRACTION FOR PLAYBACK
# ==============================================================================
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

        output = io.BytesIO()
        sf.write(output, segment, sr, format="WAV")
        output.seek(0)
        return output.read()

    except Exception:
        return None


# ==============================================================================
# STREAMLIT PAGE CONFIG
# ==============================================================================
st.set_page_config(page_title="Frog Call Classifier (Plotly)", layout="wide")

st.title("🐸 Frog Call Classifier")
st.write("Fully optimized with FFmpeg streaming + Plotly-only UI for stability.")


# ==============================================================================
# FILE UPLOAD
# ==============================================================================
uploaded_files = st.file_uploader(
    "Upload frog audio file(s) (WAV/MP3)",
    type=["wav", "mp3"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload audio files to begin.")
    st.stop()

file_names = [f.name for f in uploaded_files]
selected_file_name = st.selectbox("Select a file to analyze", file_names)


# ==============================================================================
# PROCESS SELECTED FILE
# ==============================================================================
for f in uploaded_files:
    if f.name != selected_file_name:
        continue

    audio_bytes = f.read()

    # ---- Full Audio Playback ----
    st.subheader("🎧 Full Audio Playback")
    st.audio(audio_bytes)

    # ==============================================================================
    # PLOTLY WAVEFORM + SPECTROGRAM PREVIEW (ALWAYS ENABLED)
    # ==============================================================================
    st.subheader("📈 Audio Preview (First Few Seconds)")
    fig_wave, fig_spec = get_preview_plotly(audio_bytes)

    if fig_wave:
        st.plotly_chart(fig_wave, use_container_width=True)
        st.plotly_chart(fig_spec, use_container_width=True)
    else:
        st.info("Could not generate preview.")

    # ==============================================================================
    # MODEL INFERENCE (FFmpeg Streaming)
    # ==============================================================================
    st.subheader("🤖 Model Detection Results")

    with st.spinner("Running streaming inference..."):
        detections = analyze_audio(audio_bytes)

    df_raw = pd.DataFrame(detections)

    if df_raw.empty:
        st.error("No detections found.")
        st.stop()

    total_duration = float(df_raw["end"].max())

    # ==============================================================================
    # PER-SPECIES CONFIDENCE FILTERS
    # ==============================================================================
    st.subheader("🎚 Confidence Thresholds (Per Species)")

    species_list = sorted(df_raw["species"].unique())
    cols = st.columns(len(species_list))
    thresholds = {}

    for i, sp in enumerate(species_list):
        with cols[i]:
            thresholds[sp] = st.slider(
                sp,
                0.0, 1.0,
                0.0, 0.05
            )

    df_filtered = df_raw.copy()
    for sp, th in thresholds.items():
        df_filtered = df_filtered[
            ~((df_filtered["species"] == sp) & (df_filtered["confidence"] < th))
        ]

    if df_filtered.empty:
        st.error("All detections filtered out by thresholds.")
        st.stop()

    # ==============================================================================
    # CSV EXPORT
    # ==============================================================================
    st.subheader("⬇ Export")
    csv_data = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Detections (CSV)",
        csv_data,
        file_name=f"{selected_file_name}_detections.csv",
        mime="text/csv"
    )

    # ==============================================================================
    # TABLE VIEW
    # ==============================================================================
    st.subheader("📋 Detection Table")

    if len(df_filtered) > MAX_ROWS_TO_DISPLAY:
        st.warning(f"Showing first {MAX_ROWS_TO_DISPLAY} rows.")
        st.dataframe(df_filtered.head(MAX_ROWS_TO_DISPLAY))
    else:
        st.dataframe(df_filtered)

    # ==============================================================================
    # INTERACTIVE TIMELINE (PLOTLY)
    # ==============================================================================
    st.subheader("📌 Species Timeline")

    # Windowing for large audio
    if len(df_filtered) > MAX_SEGMENTS_FOR_TIMELINE:
        st.warning("Large number of detections — timeline windowing enabled.")

        window_size = st.slider(
            "Timeline window (seconds)",
            30.0, total_duration,
            120.0, step=10.0
        )
        max_start = max(0.0, total_duration - window_size)

        window_start = st.slider(
            "Start time (seconds)",
            0.0, max_start,
            0.0, step=5.0
        )
        window_end = window_start + window_size

        df_timeline = df_filtered[
            (df_filtered["start"] >= window_start) &
            (df_filtered["end"] <= window_end)
        ]
        st.write(f"Showing {len(df_timeline)} detections in this window.")
    else:
        df_timeline = df_filtered.copy()

    if df_timeline.empty:
        st.info("No detections in this window.")
    else:
        fig = go.Figure()

        species_colors = {
            sp: f"rgba({50+(hash(sp)%200)}, {50+((hash(sp)*7)%200)}, {50+((hash(sp)*13)%200)}, 0.85)"
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
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==============================================================================
    # PLAYBACK
    # ==============================================================================
    st.subheader("🎯 Play Detected Calls")

    df_play = df_timeline.copy()
    species_opt = ["All species"] + sorted(df_play["species"].unique())
    chosen_sp = st.selectbox("Filter species for playback", species_opt)

    if chosen_sp != "All species":
        df_play = df_play[df_play["species"] == chosen_sp]

    if df_play.empty:
        st.warning("No detections for playback.")
        st.stop()

    df_play = df_play.sort_values("start")

    if len(df_play) > MAX_PLAY_OPTIONS:
        st.warning(f"Showing first {MAX_PLAY_OPTIONS} detections.")
        df_play = df_play.head(MAX_PLAY_OPTIONS)

    opts = [
        f"{i+1}: {row['species']} {row['start']:.1f}–{row['end']:.1f}s "
        f"(conf {row['confidence']:.2f})"
        for i, row in df_play.iterrows()
    ]

    chosen_idx = st.selectbox("Select a detection to play", opts)
    idx = opts.index(chosen_idx)
    chosen_row = df_play.iloc[idx]

    if st.button("▶ Play selected call"):
        seg_bytes = extract_segment_audio(
            audio_bytes,
            chosen_row["start"],
            chosen_row["end"]
        )
        if seg_bytes:
            st.audio(seg_bytes, format="audio/wav")
        else:
            st.error("Unable to extract segment.")

    st.divider()