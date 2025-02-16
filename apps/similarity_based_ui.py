import os
import json
import numpy as np
import pandas as pd
import streamlit as st

# --- Configuration: paths ---
# We assume that the final embeddings were saved in a file inside the apps/playlist_data folder.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "apps", "playlist_data")
EMBEDDINGS_CSV = os.path.join(EMBEDDINGS_DIR, "final_embeddings.csv")

# --- Helper function to parse embeddings stored as JSON strings ---
def parse_embedding(emb_str):
    try:
        return json.loads(emb_str)
    except Exception as e:
        return None

# --- Load Embeddings ---
@st.cache(show_spinner=True)
def load_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    # Parse the embeddings from strings back into lists
    df["emb_discogs"] = df["emb_discogs"].apply(parse_embedding)
    df["emb_msd"] = df["emb_msd"].apply(parse_embedding)
    # Remove rows with missing embeddings
    df = df[df["emb_discogs"].notnull() & df["emb_msd"].notnull()].reset_index(drop=True)
    return df

df_emb = load_embeddings(EMBEDDINGS_CSV)

st.title("Track Similarity Playlist Generator")
st.write("Select a query track from the collection:")

# --- Let the user choose a query track ---
# For clarity we use the 'file' field (assumed to be the relative path to the audio file)
query_track = st.selectbox("Query Track", df_emb["file"].tolist())

# --- Compute similarity ---
if query_track:
    # Convert query embeddings to numpy arrays
    query_discogs = np.array(df_emb.loc[df_emb["file"] == query_track, "emb_discogs"].iloc[0])
    query_msd = np.array(df_emb.loc[df_emb["file"] == query_track, "emb_msd"].iloc[0])
    
    # Compute cosine similarities for all tracks
    discogs_sims = []
    msd_sims = []
    
    for _, row in df_emb.iterrows():
        # Convert embedding lists to numpy arrays
        emb_disc = np.array(row["emb_discogs"])
        emb_msd = np.array(row["emb_msd"])
        # Cosine similarity: dot / (norms)
        sim_disc = np.dot(query_discogs, emb_disc) / (np.linalg.norm(query_discogs) * np.linalg.norm(emb_disc) + 1e-10)
        sim_msd = np.dot(query_msd, emb_msd) / (np.linalg.norm(query_msd) * np.linalg.norm(emb_msd) + 1e-10)
        discogs_sims.append(sim_disc)
        msd_sims.append(sim_msd)
    
    # Append similarities to the DataFrame (make a copy so that cached df remains intact)
    df_sim = df_emb.copy()
    df_sim["discogs_similarity"] = discogs_sims
    df_sim["msd_similarity"] = msd_sims

    # Exclude the query track from the similar-list (set its similarity to -inf)
    df_sim.loc[df_sim["file"] == query_track, "discogs_similarity"] = -np.inf
    df_sim.loc[df_sim["file"] == query_track, "msd_similarity"] = -np.inf

    # Get top 10 similar tracks for each embedding type
    top10_discogs = df_sim.nlargest(10, "discogs_similarity")
    top10_msd = df_sim.nlargest(10, "msd_similarity")
    
    st.markdown("## Query Track")
    st.audio(query_track, format="audio/mp3")
    
    st.markdown("## Top 10 Similar Tracks (Discogs Embeddings)")
    for idx, row in top10_discogs.iterrows():
        st.write(f"**{row['file']}** (Similarity: {row['discogs_similarity']:.3f})")
        st.audio(row["file"], format="audio/mp3")
    
    st.markdown("## Top 10 Similar Tracks (MSD Embeddings)")
    for idx, row in top10_msd.iterrows():
        st.write(f"**{row['file']}** (Similarity: {row['msd_similarity']:.3f})")
        st.audio(row["file"], format="audio/mp3")
