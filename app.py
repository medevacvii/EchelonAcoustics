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

from backend.predict import analyze_audio  # FFmpeg + model inference


# ------------------------------------------------------------------------------
# CONSTANTS / LIMITS
# ------------------------------------------------------------------------------
MIN_SAMPLES = 2048
MAX_ROWS_TO_DISPLAY = 500
MAX_SEGMENTS_FOR_TIMELINE = 500
MAX_PLAY_OPTIONS = 300
LARGE_FILE_BYTES = 150 * 1024 * 1024   # 150 MB threshold to skip preview
WAVEFORM_MAX_SEC = 10                  # preview limit


# ==============================================================================
# SAFE VISUALIZATION USING PLOTLY
# ==============================================================================
def get_waveform_and_spectrogram_plotly(audio_bytes, max_sec=WAVEFORM_MAX_SEC):
    """
    Reads only the first few seconds of audio and returns:
    - Plotly waveform figure
    - Plotly spectrogram figure
    More stable than matplotlib; safe for Streamlit Cloud.
    """

    try:
        bio = io.BytesIO(audio_bytes)

        with sf.SoundFile(bio) as f:
            sr = f.samplerate
            frames = min(len(f), int(sr * max_sec))
            f.seek(0)
            y = f.read(frames, dtype="float32")

        if y is None or len(y) == 0:
            return None, None

        # Force mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Waveform Plotly figure
        t = np.linspace(0, len(y)/sr, len(y))
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=t, y=y, mode="lines"))
        fig_wave.update_layout(
            title="Waveform (Preview)",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=250,
        )

        # Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        fig_spec = go.Figure(
            data=go.Heatmap(
                z=S_db,
                colorscale="Viridis"
            )
        )
        fig_spec.update_layout(
            title="Mel Spectrogram (Preview, dB)",
            xaxis_title="Frame Index",
            yaxis_title="Mel Bin",
            height=250,
        )

        return fig_wave, fig_spec

    except Exception as e:
        st.error(f"Plotly preview failed: {e}")
        return None, None


# ==============================================================================
# BACKEND AUDIO SEGMENT EXTRACTION FOR PLAYBACK
# ==============================================================================
def extract_segment_audio(audio_bytes, start_sec, end_sec):
    """
    Extracts a segment safely, independent of preview size.
    """
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

        out = io.BytesIO()
        sf.write(out, segment, sr, format="WAV")
        out.seek(0)
        return out.read()

    except Exception:
        return None


# ==============================================================================
# STREAMLIT PAGE CONFIG
# ==============================================================================
st.set_page_config(page_title="Frog Call Classifier", layout="wide")

st.title("🐸 Frog Call Classifier (Plotly-Only Edition)")
st.write("This version uses Plotly everywhere for maximum Streamlit Cloud stability.")


# ==============================================================================
# FILE UPLOADER
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
selected_file_name = st.selectbox("Select a file to inspect", file_names)


# ==============================================================================
# PROCESS USER-SELECTED FILE
# ==============================================================================
for f in uploaded_files:

    if f.name != selected_file_name:
        continue

    audio_bytes = f.read()

    st.subheader(f"🎧 Full Audio Playback — {f.name}")
    st.audio(audio_bytes)

    # ======================================================================
    # WAVEFORM + SPECTROGRAM PREVIEW (Plotly)
    # ======================================================================
    st.subheader("📈 Preview (Waveform + Spectrogram)")

    if len(audio_bytes) > LARGE_FILE_BYTES:
        st.warning("Preview skipped: file exceeds safe preview size.")
        fig_wave = fig_spec = None
    else:
        fig_wave, fig_spec = get_waveform_and_spectrogram_plotly(audio_bytes)

    if fig_wave:
        st.plotly_chart(fig_wave, use_container_width=True)
        st.plotly_chart(fig_spec, use_container_width=True)

    # ======================================================================
    # RUN MODEL VIA FFmpeg STREAMING
    # ======================================================================
    st.subheader("🤖 Model Detection Results")

    with st.spinner("Analyzing full audio (streaming inference)..."):
        detections = analyze_audio(audio_bytes)

    df_raw = pd.DataFrame(detections)

    if df_raw.empty:
        st.error("No detections found.")
        st.stop()

    total_duration = float(df_raw["end"].max())

    # ======================================================================
    # CONFIDENCE FILTERS (PER SPECIES)
    # ======================================================================
    st.subheader("🎚 Per-Species Confidence Threshold Filters")

    species_list = sorted(df_raw["species"].unique())
    cols = st.columns(len(species_list))
    thresholds = {}

    for i, sp in enumerate(species_list):
        with cols[i]:
            thresholds[sp] = st.slider(
                f"{sp}",
                0.0, 1.0,
                0.0, 0.05
            )

    df_filtered = df_raw.copy()
    for sp, th in thresholds.items():
        df_filtered = df_filtered[
            ~((df_filtered["species"] == sp) & (df_filtered["confidence"] < th))
        ]

    if df_filtered.empty:
        st.error("All detections filtered out.")
        st.stop()

    # ======================================================================
    # CSV EXPORT
    # ======================================================================
    csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Filtered Detections (CSV)",
        csv_bytes,
        file_name=f"{selected_file_name}_detections.csv",
        mime="text/csv"
    )

    # ======================================================================
    # TABLE VIEW (LIMITED)
    # ======================================================================
    st.subheader("📋 Detection Table")

    if len(df_filtered) > MAX_ROWS_TO_DISPLAY:
        st.warning(f"Showing first {MAX_ROWS_TO_DISPLAY} of {len(df_filtered)} rows.")
        st.dataframe(df_filtered.head(MAX_ROWS_TO_DISPLAY))
    else:
        st.dataframe(df_filtered)

    # ======================================================================
    # INTERACTIVE TIMELINE (PLOTLY)
    # ======================================================================
    st.subheader("📌 Species Timeline (Interactive Plotly)")

    # WINDOWED MODE FOR LARGE FILES
    if len(df_filtered) > MAX_SEGMENTS_FOR_TIMELINE:
        st.warning(f"Large file: {len(df_filtered)} detections. Timeline windowing enabled.")

        window_size = st.slider(
            "Window size (seconds)",
            30.0, total_duration, 120.0, step=10.0
        )
        window_start = st.slider(
            "Window start (seconds)",
            0.0, total_duration - window_size, 0.0, step=5.0
        )
        window_end = window_start + window_size

        df_timeline = df_filtered[
            (df_filtered["start"] >= window_start) &
            (df_filtered["end"] <= window_end)
        ]
        st.write(f"Showing {len(df_timeline)} detections in this window.")
    else:
        df_timeline = df_filtered

    if df_timeline.empty:
        st.info("No detections in this window.")
    else:

        fig = go.Figure()

        # Unique deterministic colors
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

    # ======================================================================
    # PLAYBACK (FILTERED + WINDOWED)
    # ======================================================================
    st.subheader("🎯 Play Detected Calls")

    df_play = df_timeline.copy()

    species_opt = ["All species"] + sorted(df_play["species"].unique())
    selected_species = st.selectbox("Filter by species", species_opt)

    if selected_species != "All species":
        df_play = df_play[df_play["species"] == selected_species]

    df_play = df_play.sort_values("start").reset_index(drop=True)

    if df_play.empty:
        st.warning("No detections available for playback.")
        st.stop()

    if len(df_play) > MAX_PLAY_OPTIONS:
        st.warning(f"Showing first {MAX_PLAY_OPTIONS} detections for playback.")
        df_play = df_play.head(MAX_PLAY_OPTIONS)

    opts = [
        f"{i+1}: {row['species']} {row['start']:.1f}–{row['end']:.1f}s (conf {row['confidence']:.2f})"
        for i, row in df_play.iterrows()
    ]

    selected_idx = st.selectbox("Select detection to play", opts)
    idx = opts.index(selected_idx)
    selected_row = df_play.iloc[idx]

    if st.button("▶ Play selected call"):
        seg = extract_segment_audio(audio_bytes, selected_row["start"], selected_row["end"])
        if seg:
            st.audio(seg, format="audio/wav")
        else:
            st.error("Unable to extract segment.")

    st.divider()