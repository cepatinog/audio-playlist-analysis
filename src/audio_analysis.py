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
# Hide unnecessary TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import json
import traceback
from tqdm import tqdm
import essentia.standard as es
import numpy as np

# Import our feature extractor functions using relative imports.
# from .extract_tempo import extract_tempo_features
# from .extract_key import extract_key_features
# from .extract_loudness import extract_loudness_features
# from .extract_embeddings import extract_discogs_effnet_embeddings, extract_msd_musicnn_embeddings
# from .extract_genre import extract_genre_features
# from .extract_voice_instrumental import extract_voice_instrumental
# from .extract_danceability import extract_danceability_features
# from .extract_arousal_valence import extract_arousal_valence_features
# from .load_audio import load_audio_file


from extract_tempo import extract_tempo_features
from extract_key import extract_key_features
from extract_loudness import extract_loudness_features
from extract_embeddings import extract_discogs_effnet_embeddings, extract_msd_musicnn_embeddings
from extract_genre import extract_genre_features
from extract_voice_instrumental import extract_voice_instrumental
from extract_danceability import extract_danceability_features
from extract_arousal_valence import extract_arousal_valence_features
from load_audio import load_audio_file

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
    Returns a dictionary with keys for tempo, key, loudness, embeddings, genre, voice/instrumental,
    danceability, and emotion.
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
            # For emotion extraction, we use MSD-MusicCNN embeddings.
            # If emb_msd was already computed, reuse it; otherwise, compute it.
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



def convert_numpy(obj):
    """
    Recursively convert NumPy arrays in a data structure to lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    else:
        return obj
    

def process_all_audio_with_checkpoint(raw_dir, checkpoint_dir, 
                                        tempo_method='tempocnn', 
                                        tempo_model_file=None,
                                        emb_discogs_model_file=None, 
                                        emb_msd_model_file=None,
                                        genre_model_file=None,
                                        voice_model_file=None,
                                        danceability_model_file=None,
                                        emotion_model_file=None):
    """
    Process all audio files in raw_dir and save each file's extracted features as a separate JSON file
    in checkpoint_dir. If a file has been processed already (i.e., its JSON exists), skip it.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for root, _, files in os.walk(raw_dir):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            # Skip files with unwanted substrings (e.g., ":Zone.Identifier")
            if ":Zone.Identifier" in file:
                continue
            if file.lower().endswith(AUDIO_EXTENSIONS):
                audio_path = os.path.join(root, file)
                # Create a safe checkpoint filename based on the relative path.
                rel_path = os.path.relpath(audio_path, raw_dir)
                safe_name = rel_path.replace(os.sep, "_")
                checkpoint_file = os.path.join(checkpoint_dir, safe_name + ".json")
                if os.path.exists(checkpoint_file):
                    continue  # Skip already processed file.
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
                    # Convert all NumPy arrays to lists so JSON can serialize them.
                    features_converted = convert_numpy(features)
                    with open(checkpoint_file, 'w') as f:
                        json.dump(features_converted, f)
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    traceback.print_exc()

if __name__ == '__main__':
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, '..'))
    raw_dir = os.path.join(project_root, 'data', 'raw')
    checkpoint_dir = os.path.join(project_root, 'data', 'processed', 'features')

    tempo_model_file = os.path.join(src_dir, 'deeptemp-k16-3.pb')
    tempo_method = 'tempocnn'
    emb_discogs_model_file = os.path.join(src_dir, 'discogs-effnet-bs64-1.pb')
    emb_msd_model_file = os.path.join(src_dir, 'msd-musicnn-1.pb')
    genre_model_file = os.path.join(src_dir, 'genre_discogs400-discogs-effnet-1.pb')
    voice_model_file = os.path.join(src_dir, 'voice_instrumental-discogs-effnet-1.pb')
    danceability_model_file = os.path.join(src_dir, 'danceability-discogs-effnet-1.pb')
    emotion_model_file = os.path.join(src_dir, 'emomusic-msd-musicnn-2.pb')

    process_all_audio_with_checkpoint(raw_dir, checkpoint_dir,
                                        tempo_method=tempo_method,
                                        tempo_model_file=tempo_model_file,
                                        emb_discogs_model_file=emb_discogs_model_file,
                                        emb_msd_model_file=emb_msd_model_file,
                                        genre_model_file=genre_model_file,
                                        voice_model_file=voice_model_file,
                                        danceability_model_file=danceability_model_file,
                                        emotion_model_file=emotion_model_file)
