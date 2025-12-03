# app.py

import io
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

from backend.predict import analyze_audio  # backend module you just created


# =============================================================================
# STREAMLIT CONFIG
# =============================================================================

st.set_page_config(
    page_title="Frog Call Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🐸 Frog Call Classifier")
st.caption(
    "Upload frog audio recordings, run the model, and inspect species, "
    "timestamps, and confidence scores for single or multiple files."
)


# =============================================================================
# CACHED HELPERS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_audio(audio_bytes: bytes, sr: int | None = None):
    """
    Load raw audio bytes into waveform + sampling rate using librosa.
    """
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    return y, sr


@st.cache_data(show_spinner=False)
def run_model(audio_bytes: bytes) -> pd.DataFrame:
    """
    Call backend.analyze_audio and normalize to a DataFrame.
    """
    detections = analyze_audio(audio_bytes)
    df = pd.DataFrame(detections)

    expected_cols = ["start", "end", "species", "confidence"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Model output missing expected column: {col}")

    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)
    df["confidence"] = df["confidence"].astype(float)
    df["species"] = df["species"].astype(str)
    return df


def make_segment_audio(y: np.ndarray, sr: int, start_s: float, end_s: float) -> bytes:
    """
    Cut [start_s, end_s] from waveform and return as WAV bytes.
    """
    start_idx = int(max(0, start_s) * sr)
    end_idx = int(min(len(y), end_s * sr))
    segment = y[start_idx:end_idx]

    buf = io.BytesIO()
    sf.write(buf, segment, sr, format="WAV")
    buf.seek(0)
    return buf.read()


# =============================================================================
# SIDEBAR – GLOBAL CONTROLS
# =============================================================================

st.sidebar.header("Controls")

confidence_threshold = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Only detections with confidence ≥ this value will be shown.",
)

view_mode = st.sidebar.radio(
    "View mode",
    ["Single-file focus", "Batch summary"],
    help=(
        "Single-file focus lets you deep-dive into one recording. "
        "Batch summary aggregates detections across all uploaded files."
    ),
)

st.sidebar.markdown("---")
st.sidebar.write("**Instructions**")
st.sidebar.markdown(
    "- Upload one or more audio recordings (WAV/MP3).\n"
    "- Adjust the confidence threshold and species filters.\n"
    "- In single-file mode, explore waveform, spectrogram, timeline, and segments.\n"
    "- In batch summary, compare species and confidence across files."
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

if uploaded_files:

    file_names = [f.name for f in uploaded_files]

    # ---- File selector for multi-file mode ----
    selected_file_name = st.selectbox("Select a file to inspect", file_names)

    for f in uploaded_files:

        if f.name != selected_file_name:
            continue  # Only process one file at a time for display

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
        df_raw = run_model(audio_bytes)

        st.write(f"### Results for: {f.name}")
        st.dataframe(df_raw)

else:
    st.info("Upload one or more audio files to begin analysis.")


# =============================================================================
# RUN ANALYSIS FOR ALL FILES
# =============================================================================

def get_all_results():
    results_per_file: dict[str, dict] = {}
    for file in uploaded_files:
        # Must re-read bytes fresh each time; .read() exhausts the buffer
        audio_bytes = file.read()
        y, sr = load_audio(audio_bytes)
        df_raw = run_model(audio_bytes)
        results_per_file[file.name] = {
            "audio_bytes": audio_bytes,
            "waveform": y,
            "sr": sr,
            "df_raw": df_raw,
        }
    return results_per_file


with st.spinner("Running analysis on uploaded file(s)..."):
    all_results = get_all_results()


# =============================================================================
# FILTER HELPERS
# =============================================================================

def filter_detections(df: pd.DataFrame, min_conf: float, species_filter: list[str] | None):
    df_filtered = df[df["confidence"] >= min_conf].copy()
    if species_filter:
        df_filtered = df_filtered[df_filtered["species"].isin(species_filter)]
    return df_filtered


# =============================================================================
# SINGLE-FILE FOCUS MODE
# =============================================================================

if view_mode == "Single-file focus":
    st.subheader("Single-file Analysis")

    selected_file_name = st.selectbox("Select a file to inspect", file_names)
    data = all_results[selected_file_name]
    y, sr, df_raw = data["waveform"], data["sr"], data["df_raw"]

    available_species = sorted(df_raw["species"].unique())
    species_filter = st.multiselect(
        "Filter by species (optional)",
        options=available_species,
        default=available_species,
    )

    df = filter_detections(df_raw, confidence_threshold, species_filter)

    # AUDIO PLAYER (full file)
    st.markdown("### 🔊 Full Audio Playback")
    st.audio(data["audio_bytes"], format="audio/wav")

    col_wave, col_spec = st.columns(2, gap="large")

    # Waveform
    with col_wave:
        st.markdown("#### Waveform")
        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform")
        st.pyplot(fig, clear_figure=True)

    # Spectrogram
    with col_spec:
        st.markdown("#### Spectrogram")
        S = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        fig, ax = plt.subplots(figsize=(8, 3))
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="hz", ax=ax
        )
        ax.set_title("Spectrogram (dB)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig, clear_figure=True)

    # Timeline
    st.markdown("### 📈 Species Timeline")
    if df.empty:
        st.warning("No detections match the current filters/threshold.")
    else:
        unique_species = sorted(df["species"].unique())
        species_to_y = {sp: i for i, sp in enumerate(unique_species)}

        fig, ax = plt.subplots(figsize=(12, 3))
        for _, row in df.iterrows():
            y_pos = species_to_y[row["species"]]
            ax.plot(
                [row["start"], row["end"]],
                [y_pos, y_pos],
                linewidth=6,
            )
            mid_t = 0.5 * (row["start"] + row["end"])
            ax.text(
                mid_t,
                y_pos + 0.05,
                row["species"],
                fontsize=8,
                ha="center",
                va="bottom",
            )

        ax.set_yticks(list(species_to_y.values()))
        ax.set_yticklabels(list(species_to_y.keys()))
        ax.set_xlabel("Time (s)")
        ax.set_title("Timeline of Detected Calls")
        st.pyplot(fig, clear_figure=True)

    # Detections table
    st.markdown("### 📄 Detection Results")
    st.write(
        "Each row is a detected window with start/end time, model label (species), and confidence."
    )
    st.dataframe(df.sort_values(["start", "end"]), use_container_width=True)

    # Confidence summary
    st.markdown("### 📊 Average Confidence per Species")
    if not df.empty:
        conf_summary = df.groupby("species")["confidence"].mean().sort_values(ascending=False)
        st.bar_chart(conf_summary)
    else:
        st.info("No detections to summarize for current filters.")

    # Segment playback
    st.markdown("### 🎧 Play Detected Segments")
    if df.empty:
        st.info("No segments to play for current filters/threshold.")
    else:
        df_for_select = df.reset_index(drop=True)
        segment_index = st.number_input(
            "Segment index",
            min_value=0,
            max_value=len(df_for_select) - 1,
            value=0,
            step=1,
            help="Select which detected segment to preview.",
        )

        row = df_for_select.iloc[int(segment_index)]
        st.write(
            f"Selected segment: **{row['species']}**, "
            f"{row['start']:.2f}s → {row['end']:.2f}s, "
            f"confidence = {row['confidence']:.2f}"
        )

        segment_bytes = make_segment_audio(
            y, sr, start_s=row["start"], end_s=row["end"]
        )
        st.audio(segment_bytes, format="audio/wav")

        show_zoom = st.checkbox("Show zoomed waveform around segment", value=True)
        if show_zoom:
            margin = 0.25  # seconds
            zoom_start = max(0, row["start"] - margin)
            zoom_end = min(len(y) / sr, row["end"] + margin)
            fig, ax = plt.subplots(figsize=(10, 3))
            t = np.linspace(0, len(y) / sr, num=len(y))
            mask = (t >= zoom_start) & (t <= zoom_end)
            ax.plot(t[mask], y[mask])
            ax.set_title(
                f"Zoomed Waveform: {row['species']} "
                f"({zoom_start:.2f}s–{zoom_end:.2f}s)"
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig, clear_figure=True)


# =============================================================================
# BATCH SUMMARY MODE
# =============================================================================

elif view_mode == "Batch summary":
    st.subheader("Batch Summary Across Files")

    combined_rows: List[pd.DataFrame] = []
    for fname, data in all_results.items():
        df_raw = data["df_raw"]
        tmp = df_raw.copy()
        tmp["file"] = fname
        combined_rows.append(tmp)

    combined_df = pd.concat(combined_rows, ignore_index=True)

    all_species = sorted(combined_df["species"].unique())
    species_filter = st.multiselect(
        "Filter by species (optional)",
        options=all_species,
        default=all_species,
    )

    df_filtered = filter_detections(combined_df, confidence_threshold, species_filter)

    if df_filtered.empty:
        st.warning("No detections across any files match the current filters/threshold.")
        st.stop()

    st.markdown("### 📄 Detections Table (All Files)")
    st.dataframe(
        df_filtered.sort_values(["file", "start", "end"]),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Detections per Species")
        count_species = df_filtered["species"].value_counts()
        st.bar_chart(count_species)

    with col2:
        st.markdown("#### Average Confidence per Species")
        avg_conf = (
            df_filtered.groupby("species")["confidence"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(avg_conf)

    st.markdown("### 📊 Detections per File")
    detections_per_file = df_filtered.groupby("file")["species"].count()
    st.bar_chart(detections_per_file)

    st.markdown("### 📥 Export filtered detections")
    csv_data = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="frog_detections_filtered.csv",
        mime="text/csv",
    )