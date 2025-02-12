#!/usr/bin/env python3
"""
audio_analysis.py

This module processes an entire raw audio collection and extracts multiple features
using modular extractor functions. It loads each audio file only once and passes the
loaded audio to each extractor, then writes the combined results to a CSV file.

Supported folder structure (relative to project root):
    data/raw        -> contains the audio files (possibly nested)
    data/processed  -> where the CSV file with extracted features will be saved

Extractors included in this version:
    - Tempo extraction (via extract_tempo.py)
    - Key extraction (via key_extraction.py)

You can later add more extractor functions to the pipeline.
"""

import os
import csv
import traceback
from tqdm import tqdm
import essentia.standard as es

# Import our feature extractor functions
from extract_tempo import extract_tempo_features
from key_extraction import extract_key_features  

# Define acceptable audio file extensions
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')

def extract_all_features(audio, tempo_method='tempocnn', tempo_model_file=None):
    """
    Extract all features from a loaded audio signal.

    Args:
        audio (np.array): Mono audio signal.
        tempo_method (str): 'tempocnn' or 'rhythm' for tempo extraction.
        tempo_model_file (str): Path to the TempoCNN model (if required).

    Returns:
        dict: Dictionary containing all extracted features.
    """
    features = {}
    
    # --- Tempo Extraction ---
    tempo_features = extract_tempo_features(audio, method=tempo_method, model_file=tempo_model_file)
    for k, v in tempo_features.items():
        features[f"tempo_{k}"] = v

    # --- Key Extraction ---
    key_features = extract_key_features(audio)
    features.update(key_features)

    return features

def process_all_audio(raw_dir, output_csv, tempo_method='tempocnn', tempo_model_file=None):
    """
    Process all audio files in the raw directory, extract features, and write them to a CSV file.

    Args:
        raw_dir (str): Path to the directory containing raw audio files.
        output_csv (str): Path to the CSV file for saving features.
        tempo_method (str): 'tempocnn' or 'rhythm' for tempo extraction.
        tempo_model_file (str): Path to the TempoCNN model (if required).
    """
    results = []
    
    for root, _, files in os.walk(raw_dir):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.lower().endswith(AUDIO_EXTENSIONS):
                audio_path = os.path.join(root, file)
                try:
                    # Load audio once at 44100 Hz for key extraction.
                    audio_full = es.MonoLoader(filename=audio_path, sampleRate=44100)()
                    
                    # Resample for tempo extraction if using TempoCNN.
                    if tempo_method == 'tempocnn':
                        audio_for_tempo = es.Resample(inputSampleRate=44100, outputSampleRate=11025)(audio_full)
                    else:
                        audio_for_tempo = audio_full
                    
                    features = extract_all_features(audio_for_tempo,
                                                    tempo_method=tempo_method,
                                                    tempo_model_file=tempo_model_file)
                    features['file'] = audio_path
                    results.append(features)
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    traceback.print_exc()
    
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
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, '..'))
    
    raw_dir = os.path.join(project_root, 'data', 'raw')
    output_csv = os.path.join(project_root, 'data', 'processed', 'all_features.csv')
    
    tempo_model_file = os.path.join(src_dir, 'deeptemp-k16-3.pb')
    tempo_method = 'tempocnn'
    
    process_all_audio(raw_dir, output_csv, tempo_method=tempo_method, tempo_model_file=tempo_model_file)
