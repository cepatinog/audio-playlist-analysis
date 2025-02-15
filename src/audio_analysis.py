#!/usr/bin/env python3
"""
audio_analysis.py

This module processes an entire raw audio collection and extracts multiple features
using modular extractor functions. For each audio file, it loads the file only once
using the helper function in load_audio.py, then:
  - Uses the stereo audio for loudness extraction,
  - Uses the mono audio (at 44100 Hz) for key extraction,
  - Uses the resampled mono audio (at 11025 Hz) for tempo extraction, and
  - Optionally, uses a mono version resampled to 16000 Hz for embedding extraction 
    (Discogs-Effnet and MSD-MusicCNN), genre predictions, voice/instrumental classification,
    danceability (classifier mode), and arousal/valence (emotion).

Features for each audio file are saved as a separate JSON file in a checkpoint folder,
allowing the process to resume without reprocessing files already analyzed.
"""


import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled on GPU.")
    except RuntimeError as e:
        print(e)

import os
import json
import csv
import traceback
from tqdm import tqdm
import essentia.standard as es
import numpy as np


# Hide TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import configuration and our extractor functions.
from config import (RAW_DIR, PROCESSED_FEATURES_DIR, AUDIO_EXTENSIONS,
                    TEMPO_MODEL_FILE, EMB_DISCOGS_MODEL_FILE, EMB_MSD_MODEL_FILE,
                    GENRE_MODEL_FILE, VOICE_MODEL_FILE, DANCEABILITY_MODEL_FILE,
                    EMOTION_MODEL_FILE, TEMPO_METHOD)

from extraction import (extract_tempo_features,
                        extract_key_features,
                        extract_loudness_features,
                        extract_discogs_effnet_embeddings, extract_msd_musicnn_embeddings,
                        extract_genre_features,
                        extract_voice_instrumental,
                        extract_danceability_features,
                        extract_arousal_valence_features)

from utils import load_audio_file, convert_numpy


def extract_all_features(audio_dict, 
                           tempo_method=TEMPO_METHOD, 
                           tempo_model_file=TEMPO_MODEL_FILE,
                           emb_discogs_model_file=EMB_DISCOGS_MODEL_FILE, 
                           emb_msd_model_file=EMB_MSD_MODEL_FILE,
                           genre_model_file=GENRE_MODEL_FILE,
                           voice_model_file=VOICE_MODEL_FILE,
                           danceability_model_file=DANCEABILITY_MODEL_FILE,
                           emotion_model_file=EMOTION_MODEL_FILE):
    """
    Extract all features from the audio.
    Returns a dictionary with keys for tempo, key, loudness, embeddings, genre,
    voice/instrumental, danceability, and emotion.
    """
    features = {}

    # --- Tempo Extraction (mono_tempo at 11025 Hz) ---
    tempo_feats = extract_tempo_features(audio_dict['mono_tempo'],
                                           method=tempo_method,
                                           model_file=tempo_model_file)
    for k, v in tempo_feats.items():
        features[f"tempo_{k}"] = v

    # --- Key Extraction (mono_audio at 44100 Hz) ---
    key_feats = extract_key_features(audio_dict['mono_audio'])
    features.update(key_feats)

    # --- Loudness Extraction (stereo_audio at 44100 Hz) ---
    loudness_feats = extract_loudness_features(audio_dict['stereo_audio'],
                                               hopSize=1024/44100,
                                               sampleRate=44100,
                                               startAtZero=True)
    for k, v in loudness_feats.items():
        features[f"loudness_{k}"] = v

    # --- Embedding and Additional Extraction ---
    # All embedding models expect a mono signal at 16kHz.
    if any([emb_discogs_model_file, emb_msd_model_file, genre_model_file, 
            voice_model_file, danceability_model_file, emotion_model_file]):
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
            if "emb_msd" in features:
                msd_emb = np.array(features["emb_msd"])
            else:
                msd_emb = extract_msd_musicnn_embeddings(mono_embeddings, model_file=emb_msd_model_file)
            emotion_predictions = extract_arousal_valence_features(mono_embeddings, 
                                                                    embedding_model_file=emb_msd_model_file, 
                                                                    regression_model_file=emotion_model_file)
            features["arousal_valence"] = emotion_predictions

    # --- Signal-based Danceability Extraction  ---
    dance_signal = extract_danceability_features(audio_dict['mono_audio'], mode="signal", sampleRate=44100)
    features["danceability_signal"] = dance_signal

    return features

def process_all_audio_with_checkpoint(raw_dir, checkpoint_dir, 
                                        tempo_method=TEMPO_METHOD, 
                                        tempo_model_file=TEMPO_MODEL_FILE,
                                        emb_discogs_model_file=EMB_DISCOGS_MODEL_FILE, 
                                        emb_msd_model_file=EMB_MSD_MODEL_FILE,
                                        genre_model_file=GENRE_MODEL_FILE,
                                        voice_model_file=VOICE_MODEL_FILE,
                                        danceability_model_file=DANCEABILITY_MODEL_FILE,
                                        emotion_model_file=EMOTION_MODEL_FILE):
    """
    Process all audio files in raw_dir and save each file's extracted features as a JSON file
    in checkpoint_dir. Files already processed (whose JSON exists) are skipped.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Gather all audio file paths
    all_audio_files = []
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                all_audio_files.append(os.path.join(root, file))
                
    # Process with a unified progress bar
    for audio_path in tqdm(all_audio_files, desc="Processing audio files", unit="file"):
        rel_path = os.path.relpath(audio_path, raw_dir)
        safe_name = rel_path.replace(os.sep, "_")
        checkpoint_file = os.path.join(checkpoint_dir, safe_name + ".json")
        if os.path.exists(checkpoint_file):
            continue  # Skip already processed files.
        try:
            audio_dict = load_audio_file(audio_path, targetMonoSampleRate=44100, targetTempoSampleRate=11025)
            features = extract_all_features(audio_dict,
                                            tempo_method=tempo_method,
                                            tempo_model_file=tempo_model_file,
                                            emb_discogs_model_file=emb_discogs_model_file,
                                            emb_msd_model_file=emb_msd_model_file,
                                            genre_model_file=genre_model_file,
                                            voice_model_file=voice_model_file,
                                            danceability_model_file=danceability_model_file,
                                            emotion_model_file=emotion_model_file)
            features['file'] = audio_path
            features_converted = convert_numpy(features)
            with open(checkpoint_file, 'w') as f:
                json.dump(features_converted, f)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    process_all_audio_with_checkpoint(RAW_DIR, PROCESSED_FEATURES_DIR,
                                        tempo_method=TEMPO_METHOD,
                                        tempo_model_file=TEMPO_MODEL_FILE,
                                        emb_discogs_model_file=EMB_DISCOGS_MODEL_FILE,
                                        emb_msd_model_file=EMB_MSD_MODEL_FILE,
                                        genre_model_file=GENRE_MODEL_FILE,
                                        voice_model_file=VOICE_MODEL_FILE,
                                        danceability_model_file=DANCEABILITY_MODEL_FILE,
                                        emotion_model_file=EMOTION_MODEL_FILE)
