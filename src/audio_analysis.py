#!/usr/bin/env python3
"""
audio_analysis.py

This module processes an entire raw audio collection and extracts multiple features
using modular extractor functions. For each audio file, it loads the file only once
using the helper function in load_audio.py, then:
  - Uses the stereo audio for loudness extraction,
  - Uses the mono audio (at 44100 Hz) for key extraction, and
  - Uses the resampled mono audio (at 11025 Hz) for tempo extraction.

The combined features are saved into a CSV file.
"""

import os
import csv
import traceback
from tqdm import tqdm
import essentia.standard as es

# Import our feature extractor functions
from .extract_tempo import extract_tempo_features
from .key_extraction import extract_key_features
from .extract_loudness import extract_loudness_features
from .load_audio import load_audio_file

# Import our audio loading helper function
#from load_audio import load_audio_file

# Define acceptable audio file extensions
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')

def extract_all_features(audio_dict, tempo_method='tempocnn', tempo_model_file=None):
    """
    Extract all features using the pre-computed audio versions.

    Args:
        audio_dict (dict): Dictionary with keys:
                           'stereo_audio' (for loudness extraction),
                           'mono_audio' (for key extraction),
                           'mono_tempo' (for tempo extraction).
        tempo_method (str): 'tempocnn' or 'rhythm' for tempo extraction.
        tempo_model_file (str): Path to the TempoCNN model (if required).

    Returns:
        dict: Combined dictionary of extracted features.
    """
    features = {}
    
    # --- Tempo Extraction (on mono_tempo) ---
    tempo_feats = extract_tempo_features(audio_dict['mono_tempo'],
                                           method=tempo_method,
                                           model_file=tempo_model_file)
    # Prefix tempo feature keys to avoid collisions.
    for k, v in tempo_feats.items():
        features[f"tempo_{k}"] = v

    # --- Key Extraction (on mono_audio) ---
    key_feats = extract_key_features(audio_dict['mono_audio'])
    features.update(key_feats)
    
    # --- Loudness Extraction (on stereo_audio) ---
    # Use a hopSize of 1024/44100 (which is ~0.0233 sec) here as in your example,
    # or adjust as needed.
    loudness_feats = extract_loudness_features(audio_dict['stereo_audio'],
                                               hopSize=1024/44100,
                                               sampleRate=44100,
                                               startAtZero=True)
    for k, v in loudness_feats.items():
        features[f"loudness_{k}"] = v

    return features

def process_all_audio(raw_dir, output_csv, tempo_method='tempocnn', tempo_model_file=None):
    """
    Process all audio files in the raw directory, extract features, and save them in a CSV file.

    Args:
        raw_dir (str): Path to the directory with raw audio files.
        output_csv (str): Path to the CSV file where features will be saved.
        tempo_method (str): 'tempocnn' or 'rhythm' for tempo extraction.
        tempo_model_file (str): Path to the TempoCNN model (if required).
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
                                                    tempo_model_file=tempo_model_file)
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
    
    process_all_audio(raw_dir, output_csv, tempo_method=tempo_method, tempo_model_file=tempo_model_file)
