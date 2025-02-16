# Final Report on Audio Analysis & Playlist Generation:

This report summarizes the design decisions, methodologies, observations, and personal reflections I developed over the course of the project. All source code, extracted features, statistical analyses, and UI implementations are included in the project repository for further review:

## 🚀 GitHub Repository
🔗 **[View the complete source code and resources on GitHub](https://github.com/cepatinog/audio-playlist-analysis.git)** 🔗

## 1. Overview
In this project, I developed a modular system to automatically extract meaningful musical features from a large audio collection, perform statistical analyses, and build interactive user interfaces for playlist generation. The system consists of three major components:

- **Feature Extraction:** I implemented a pipeline that uses Essentia and several pre-trained TensorFlow models to extract descriptors (tempo, key/scale, loudness, emotion, voice/instrumental classification, danceability, and music style activations) along with embeddings from two distinct models (Discogs‑Effnet and MSD‑MusicCNN).
- **Statistical Overview:** Extracted features were aggregated into final descriptor files and analyzed using Jupyter notebooks. Visualizations (histograms, scatter plots, and count plots) were generated to evaluate the distribution of tempo, loudness, key/scale, emotion, and genre.
- **Playlist Generation Interfaces:** I built two Streamlit apps:
  - A descriptor‑based UI that filters tracks by tempo, key, danceability, emotion, and genre.
  - A similarity‑based UI that computes cosine similarity between track embeddings (both Discogs‑Effnet and MSD‑MusicCNN) to create playlists of similar tracks.

---

## 2. Feature Extraction and Descriptor Calculation

### 2.1 Extraction Pipeline
My extraction pipeline is defined in `audio_analysis.py` and several dedicated modules (e.g., `extract_tempo.py`, `extract_key.py`, etc.). Key decisions include:

- **Tempo:** I opted for the TempoCNN method (using the `deeptemp-k16-3.pb` model) to provide a global BPM estimate and frame‑wise local estimates with confidence scores. This choice offers robust tempo information even in complex musical passages.

- **Key and Scale:** I computed key estimates using three profiles—Temperley, Krumhansl, and EDMA—to capture different tonal perceptions. For further processing, I chose to use the Temperley profile because its estimates (key, scale, and strength) were consistent and aligned well with my subjective listening tests (with ~53% agreement across profiles).

- **Loudness:** Using Essentia’s `LoudnessEBUR128` algorithm, I extracted momentary, short‑term, and integrated loudness (in LUFS) as well as loudness range. These measures allowed me to compare overall dynamics and track loudness consistency.

- **Genre and Embeddings:**
  - **Embeddings:** I extracted two separate embedding vectors: a 1280‑dimensional vector using the Discogs‑Effnet model (trained on 400 music styles) and a 200‑dimensional vector from MSD‑MusicCNN (trained on 50 music tags).
  - **Genre Predictions:** From the 400‑dimensional activations provided by Discogs‑Effnet, I predicted the style by taking the class with the highest activation. The full style label (e.g., “Electronic---Techno”) was then split to extract the parent genre for a higher‑level view.

- **Voice/Instrumental & Danceability:**
  - The voice/instrumental classifier outputs a softmax probability, and I assign a binary label (“voice” or “instrumental”) based on the maximum probability.
  - Danceability is estimated in two ways: a signal‑based measure (using Essentia’s algorithm, yielding a value roughly in [0, 3]) and a classifier‑based measure (using the Discogs‑Effnet embedding as input to a danceability classifier, outputting a probability in [0, 1]).

- **Emotion (Arousal/Valence):** I computed emotion descriptors using a regression model (`emomusic‑msd‑musicnn‑2`) that operates on an averaged MSD‑MusicCNN embedding. The predictions for arousal and valence are expected to fall within [1, 9].

### 2.2 Descriptor Files
Final descriptors (excluding raw embeddings) were saved into CSV and TSV files inside the `apps/playlist_data` folder. These files include fields such as file path, tempo, voice/instrumental label, danceability values, emotion (arousal and valence), key, scale, predicted genre, and parent genre.

Separately, I stored the two sets of embeddings (Discogs‑Effnet and MSD‑MusicCNN) in CSV/TSV files so that similarity‑based processing can operate independently.

---

## 3. Statistical Analysis & UI for Playlist Generation

### 3.1 Statistical Overview
Using Jupyter notebooks (e.g., `analysis_overview.ipynb`), I loaded the final descriptors and generated several plots:

- **Tempo & Loudness:** Histograms of global BPM and integrated loudness provided insights into the rhythmic and dynamic characteristics of the collection.
- **Key and Scale:** Count plots for key and scale across the three estimation profiles highlighted both consensus and variability in tonal analysis.
- **Emotion:** A scatter plot in the arousal/valence space revealed clusters of tracks with similar emotional characteristics.
- **Genre Distribution:** Due to the large number of specific styles (400), I aggregated genres by their parent category (e.g., “Electronic”, “Hip Hop”) to obtain a clearer picture of the collection’s style diversity.

### 3.2 User Interfaces
I implemented two Streamlit apps:

1. **Descriptor-Based UI (`descriptor_based_ui.py`):**
   - This interface lets users filter tracks by descriptors such as tempo range, danceability (both signal‑based and classifier‑based), emotion (arousal/valence), key, scale, and parent genre.
   - The app then displays a filtered list of tracks, generates an M3U8 playlist, and embeds audio previews.

2. **Similarity-Based UI (`similarity_based_ui.py`):**
   - In this app, users select a query track from the collection. 
   - The app computes cosine similarity between the query’s embeddings and those of all other tracks, using both the Discogs‑Effnet and MSD‑MusicCNN embeddings.
   - Two lists of the top 10 most similar tracks are produced—each accompanied by embedded audio previews—allowing users to listen and compare the results.

---

## 4. Observations & Reflections

### 4.1 Feature Quality and System Performance
- **Robust Features:**  
  The extraction modules performed reliably on most tracks. For example, clear rhythmic tracks consistently returned global BPM estimates (e.g., 128 BPM) that matched my auditory impressions. The loudness extraction produced realistic LUFS values, and emotion predictions for dynamic tracks were within the expected range.

- **Challenges:**  
  - **Ambiguous Content:** Some tracks with ambiguous tonality or hybrid instrumentation occasionally resulted in less stable key estimates or conflicting voice/instrumental predictions.  
  - **Danceability Differences:** I observed that signal‑based danceability values sometimes diverged from classifier outputs on borderline cases, suggesting that combining both methods might yield a more robust measure.

- **Embedding Similarity:**  
  In my similarity tests, the Discogs‑Effnet embeddings—which are trained on a comprehensive set of 400 styles—often captured stylistic similarities more effectively. For instance, tracks with subtle electronic textures and specific rhythmic patterns were more reliably clustered together using Discogs‑Effnet embeddings. By contrast, MSD‑MusicCNN embeddings, while valuable in capturing tag-related nuances (e.g., instrumental versus vocal cues), sometimes missed finer stylistic details.

### 4.2 Playlist Generation via Similarity
The similarity‑based UI provided two distinct playlists for each query track:

- **Discogs-Based Playlist:**  
  This playlist generally produced more coherent stylistic groupings. My listening tests indicated that tracks retrieved using Discogs‑Effnet embeddings shared similar instrumentation, production aesthetics, and overall vibe.

- **MSD-Based Playlist:**  
  While the MSD‑MusicCNN-based playlist was useful for exploring tag‐related attributes, it occasionally returned tracks that, despite similar tag activations, sounded less similar in terms of overall style.

Based on my tests, the Discogs‑Effnet embeddings appear to deliver superior performance in capturing perceptual similarity for playlist generation. I hypothesize that this is due to the model’s training on a diverse and detailed taxonomy of 400 music styles, which forces it to learn nuanced distinctions between genres.

---

## 5. Conclusion
In summary, this project demonstrates a robust system for extracting musical features, statistically analyzing a large audio collection, and interactively generating playlists based on both descriptor queries and track similarity. The modular design allowed me to combine multiple state‑of‑the‑art models to provide a rich set of descriptors and embeddings.

### Personal Reflections
- The integration of diverse features (tempo, key, loudness, emotion, and style) has proven invaluable for understanding the collection’s musical landscape.
- Although certain extraction modules (e.g., key or voice/instrumental classification) sometimes produced ambiguous results on borderline tracks, the overall performance is strong.
- My listening tests indicate that the Discogs‑Effnet embeddings most effectively capture musical similarity, likely due to their fine‑grained training on a large and varied set of styles.


