#!/usr/bin/env python3
"""
audio_analysis.py

This module processes an entire raw audio collection and extracts multiple features
using modular extractor functions. For each audio file, it loads the file once
using the helper function in load_audio.py, then:
  - Uses the stereo audio for loudness extraction,
  - Uses the mono audio (at 44100 Hz) for key extraction,
  - Uses the resampled mono audio (at 11025 Hz) for tempo extraction, and
  - Optionally, uses a mono version resampled to 16000 Hz for embedding extraction 
    (Discogs-Effnet and MSD-MusicCNN) and genre predictions.
    
The combined features are saved into a CSV file.
"""

import os
import csv
import traceback
from tqdm import tqdm
import essentia.standard as es
import numpy as np

# Import our feature extractor functions using relative imports.
from .extract_tempo import extract_tempo_features
from .extract_key import extract_key_features
from .extract_loudness import extract_loudness_features
from .extract_embeddings import extract_discogs_effnet_embeddings, extract_msd_musicnn_embeddings
from .extract_genre import extract_genre_features
from .extract_voice_instrumental import extract_voice_instrumental
from .extract_danceability import extract_danceability_features
from .extract_arousal_valence import extract_arousal_valence_features
from .load_audio import load_audio_file

# Define acceptable audio file extensions.
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')

def extract_all_features(audio_dict, 
                           tempo_method='tempocnn', 
                           tempo_model_file=None,
                           emb_discogs_model_file=None, 
                           emb_msd_model_file=None,
                           genre_model_file=None,
                           voice_model_file=None,
                           danceability_model_file=None,
                           emotion_model_file=None):
    """
    Extract all features using the pre-computed audio versions.

    Args:
        audio_dict (dict): Dictionary with keys:
                           'stereo_audio' (for loudness extraction),
                           'mono_audio' (for key extraction),
                           'mono_tempo' (for tempo extraction).
        tempo_method (str): 'tempocnn' or 'rhythm' for tempo extraction.
        tempo_model_file (str): Path to the TempoCNN model (if required).
        emb_discogs_model_file (str): Path to the Discogs-Effnet model for embeddings (if provided).
        emb_msd_model_file (str): Path to the MSD-MusicCNN model for embeddings (if provided).
        genre_model_file (str): Path to the Genre Discogs400 model (if genre predictions should be extracted).

    Returns:
        dict: Combined dictionary of extracted features.
    """
    features = {}

    # --- Tempo Extraction (using mono_tempo, which should be at 11025 Hz) ---
    tempo_feats = extract_tempo_features(audio_dict['mono_tempo'],
                                           method=tempo_method,
                                           model_file=tempo_model_file)
    for k, v in tempo_feats.items():
        features[f"tempo_{k}"] = v

    # --- Key Extraction (using mono_audio, which should be at 44100 Hz) ---
    key_feats = extract_key_features(audio_dict['mono_audio'])
    features.update(key_feats)

    # --- Loudness Extraction (using stereo_audio, at 44100 Hz) ---
    loudness_feats = extract_loudness_features(audio_dict['stereo_audio'],
                                               hopSize=1024/44100,
                                               sampleRate=44100,
                                               startAtZero=True)
    for k, v in loudness_feats.items():
        features[f"loudness_{k}"] = v

    # --- Embedding and Genre Extraction ---
    # The embedding models expect a mono audio signal at 16kHz.

    # --- Embedding and Genre & Voice/Dance Extraction ---
    if emb_discogs_model_file or emb_msd_model_file or genre_model_file or voice_model_file or danceability_model_file or emotion_model_file:
        mono_embeddings = es.Resample(inputSampleRate=audio_dict['sampleRate'], 
                                      outputSampleRate=16000)(audio_dict['mono_audio'])
        if emb_discogs_model_file:
            discogs_emb = extract_discogs_effnet_embeddings(mono_embeddings, model_file=emb_discogs_model_file)
            features["emb_discogs"] = discogs_emb.tolist()
            if genre_model_file:
                genre_predictions = extract_genre_features(discogs_emb, model_file=genre_model_file)
                features["genre_activations"] = genre_predictions.tolist()
            if voice_model_file:
                voice_result = extract_voice_instrumental(discogs_emb, model_file=voice_model_file)
                features["voice_instrumental"] = voice_result
            if danceability_model_file:
                dance_classifier = extract_danceability_features(discogs_emb, mode="classifier", model_file=danceability_model_file)
                features["danceability_classifier"] = dance_classifier
        if emb_msd_model_file:
            msd_emb = extract_msd_musicnn_embeddings(mono_embeddings, model_file=emb_msd_model_file)
            features["emb_msd"] = msd_emb.tolist()

        # --- Arousal/Valence Extraction ---
        if emotion_model_file:
            # In this case, we use MSD-MusicCNN embeddings.
            # Load or reuse msd embeddings; here, we'll use msd_emb if already computed.
            # Alternatively, resample the mono audio to 16kHz and compute the embedding again.
            # We'll assume that msd_emb is available; if not, compute it.
            if 'emb_msd' not in features:
                msd_emb = extract_msd_musicnn_embeddings(mono_embeddings, model_file=emb_msd_model_file)
            else:
                # Convert list back to numpy array.
                msd_emb = np.array(features["emb_msd"])
            emotion_predictions = extract_arousal_valence_features(mono_embeddings, 
                                                                    embedding_model_file=emb_msd_model_file, 
                                                                    regression_model_file=emotion_model_file)
            features["arousal_valence"] = emotion_predictions

    # --- Optionally, if no classifier model provided, use signal-based danceability extraction ---
    if danceability_model_file is None:
        dance_signal = extract_danceability_features(audio_dict['mono_audio'], mode="signal", sampleRate=44100)
        features["danceability_signal"] = dance_signal

    return features

def process_all_audio(raw_dir, output_csv, 
                      tempo_method='tempocnn', 
                      tempo_model_file=None,
                      emb_discogs_model_file=None, 
                      emb_msd_model_file=None,
                      genre_model_file=None):
    """
    Process all audio files in the raw directory, extract features, and save them in a CSV file.

    Args:
        raw_dir (str): Path to the directory with raw audio files.
        output_csv (str): Path to the CSV file where features will be saved.
        tempo_method (str): 'tempocnn' or 'rhythm' for tempo extraction.
        tempo_model_file (str): Path to the TempoCNN model (if required).
        emb_discogs_model_file (str): Path to the Discogs-Effnet model (if embeddings should be extracted).
        emb_msd_model_file (str): Path to the MSD-MusicCNN model (if embeddings should be extracted).
        genre_model_file (str): Path to the Genre Discogs400 model (if genre predictions should be extracted).
    """
    results = []  # List to hold feature dictionaries for each file.

    # Walk through the raw directory recursively.
    for root, _, files in os.walk(raw_dir):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.lower().endswith(AUDIO_EXTENSIONS):
                audio_path = os.path.join(root, file)
                try:
                    # Load the audio file once using our helper.
                    audio_dict = load_audio_file(audio_path,
                                                 targetMonoSampleRate=44100,
                                                 targetTempoSampleRate=11025)
                    # Extract features from the loaded audio.
                    features = extract_all_features(audio_dict,
                                                    tempo_method=tempo_method,
                                                    tempo_model_file=tempo_model_file,
                                                    emb_discogs_model_file=emb_discogs_model_file,
                                                    emb_msd_model_file=emb_msd_model_file,
                                                    genre_model_file=genre_model_file)
                    # Include the file path in the feature dictionary.
                    features['file'] = audio_path
                    results.append(features)
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    traceback.print_exc()

    # Write the results to a CSV file.
    if results:
        fieldnames = list(results[0].keys())
    else:
        fieldnames = []

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nFeature extraction complete. Results saved to {output_csv}")

if __name__ == '__main__':
    # Determine project root (assuming this file is in src/)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, '..'))
    raw_dir = os.path.join(project_root, 'data', 'raw')
    output_csv = os.path.join(project_root, 'data', 'processed', 'all_features.csv')

    # For tempo extraction via TempoCNN, specify the model file path.
    tempo_model_file = os.path.join(src_dir, 'deeptemp-k16-3.pb')
    tempo_method = 'tempocnn'

    # Specify embedding model files.
    emb_discogs_model_file = os.path.join(src_dir, 'discogs-effnet-bs64-1.pb')
    emb_msd_model_file = os.path.join(src_dir, 'msd-musicnn-1.pb')
    # Specify the genre model file.
    genre_model_file = os.path.join(src_dir, 'genre_discogs400-discogs-effnet-1.pb')
    # Specify the voice_instrumental model file.
    voice_model_file = os.path.join(src_dir, 'voice_instrumental-discogs-effnet-1.pb')
    # Specify the daceability model file.
    danceability_model_file = os.path.join(src_dir, 'danceability-discogs-effnet-1.pb')
    # Specify the emotion model file.
    emotion_model_file = os.path.join(src_dir, 'emomusic-msd-musicnn-2.pb')


    process_all_audio(raw_dir, output_csv,
                      tempo_method=tempo_method,
                      tempo_model_file=tempo_model_file,
                      emb_discogs_model_file=emb_discogs_model_file,
                      emb_msd_model_file=emb_msd_model_file,
                      genre_model_file=genre_model_file,
                      voice_model_file=voice_model_file,
                      danceability_model_file=danceability_model_file,
                      emotion_model_file=emotion_model_file)

