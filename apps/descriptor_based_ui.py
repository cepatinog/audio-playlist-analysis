import os
import random
import streamlit as st
import pandas as pd
import numpy as np

# Configuration

# Path to the final descriptors TSV file (adjust this path as needed)
DESCRIPTORS_TSV = os.path.join("final_descriptors.tsv")
# Path where the generated playlist will be saved
PLAYLIST_FILE = os.path.join("generated_playlist.m3u8")


# Load Final Descriptors

@st.cache_data
def load_descriptors(tsv_path):
    return pd.read_csv(tsv_path, sep="\t")

df = load_descriptors(DESCRIPTORS_TSV)
st.title("Descriptor-Based Playlist Generation")
st.write(f"Loaded final descriptors for **{len(df)}** tracks.")


# Sidebar Filters

st.sidebar.header("Filter Tracks")

# Tempo filter (BPM)
min_tempo = float(df['tempo_global_bpm'].min())
max_tempo = float(df['tempo_global_bpm'].max())
tempo_range = st.sidebar.slider("Tempo (BPM)", min_value=min_tempo, max_value=max_tempo,
                                value=(min_tempo, max_tempo), step=1.0)

# Voice/Instrumental filter
voice_option = st.sidebar.selectbox("Voice/Instrumental", options=["All", "voice", "instrumental"])

# Danceability filters
dance_signal_range = st.sidebar.slider("Danceability (Signal-based, [0,3])",
                                         min_value=0.0, max_value=3.0,
                                         value=(0.0, 3.0), step=0.1)
dance_classifier_range = st.sidebar.slider("Danceability (Classifier, [0,1])",
                                             min_value=0.0, max_value=1.0,
                                             value=(0.0, 1.0), step=0.05)

# Emotion: Arousal and Valence
arousal_range = st.sidebar.slider("Arousal (Expected range: [1,9])",
                                  min_value=1.0, max_value=9.0,
                                  value=(1.0, 9.0), step=0.1)
valence_range = st.sidebar.slider("Valence (Expected range: [1,9])",
                                  min_value=1.0, max_value=9.0,
                                  value=(1.0, 9.0), step=0.1)

# Key filter: get unique keys from the DataFrame
keys = list(df['key'].dropna().unique())
keys.sort()
key_filter = st.sidebar.selectbox("Key", options=["Any"] + keys)

# Scale filter
scales = list(df['scale'].dropna().unique())
scales.sort()
scale_filter = st.sidebar.selectbox("Scale", options=["Any"] + scales)

# Parent Genre filter: multi-select list of parent genres
parent_genres = list(df['parent_genre'].dropna().unique())
parent_genres.sort()
selected_parent_genres = st.sidebar.multiselect("Parent Genre", options=parent_genres, default=parent_genres)

# Additional options: maximum tracks and shuffle
max_tracks = st.sidebar.number_input("Max number of tracks (0 for all)", min_value=0, value=10, step=1)
shuffle = st.sidebar.checkbox("Shuffle tracks", value=False)


# Apply Filters

filtered_df = df.copy()

# Filter by tempo
filtered_df = filtered_df[(filtered_df['tempo_global_bpm'] >= tempo_range[0]) &
                          (filtered_df['tempo_global_bpm'] <= tempo_range[1])]

# Filter by voice/instrumental if not "All"
if voice_option != "All":
    filtered_df = filtered_df[filtered_df['voice_instrumental'] == voice_option]

# Filter by danceability (signal-based)
filtered_df = filtered_df[(filtered_df['danceability_signal'] >= dance_signal_range[0]) &
                          (filtered_df['danceability_signal'] <= dance_signal_range[1])]

# Filter by danceability (classifier)
filtered_df = filtered_df[(filtered_df['danceability_classifier'] >= dance_classifier_range[0]) &
                          (filtered_df['danceability_classifier'] <= dance_classifier_range[1])]

# Filter by emotion: arousal and valence
filtered_df = filtered_df[(filtered_df['arousal'] >= arousal_range[0]) &
                          (filtered_df['arousal'] <= arousal_range[1])]
filtered_df = filtered_df[(filtered_df['valence'] >= valence_range[0]) &
                          (filtered_df['valence'] <= valence_range[1])]

# Filter by key if specified
if key_filter != "Any":
    filtered_df = filtered_df[filtered_df['key'] == key_filter]

# Filter by scale if specified
if scale_filter != "Any":
    filtered_df = filtered_df[filtered_df['scale'] == scale_filter]

# Filter by parent genre if any selected
if selected_parent_genres:
    filtered_df = filtered_df[filtered_df['parent_genre'].isin(selected_parent_genres)]


# Generate Playlist and Display Results

if st.button("Generate Playlist"):
    result_df = filtered_df.copy()

    # Optionally shuffle
    if shuffle:
        result_df = result_df.sample(frac=1).reset_index(drop=True)

    # Limit number of tracks if max_tracks > 0
    if max_tracks > 0:
        result_df = result_df.head(max_tracks)

    st.subheader(f"Found {len(result_df)} tracks matching your filters.")
    st.dataframe(result_df)

    # Generate M3U8 playlist (store absolute or relative paths as desired)
    with open(PLAYLIST_FILE, "w") as f:
        for file_path in result_df["file"]:
            f.write(file_path + "\n")
    st.write("Playlist generated and saved to:", PLAYLIST_FILE)

    st.subheader("Audio Previews (first 5 tracks):")
    for idx, row in result_df.head(5).iterrows():
        st.write(f"**Track {idx+1}:** {row['predicted_genre']} ({row['tempo_global_bpm']} BPM)")
        st.audio(row["file"], format="audio/mp3", start_time=0)
