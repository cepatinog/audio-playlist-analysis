#!/usr/bin/env python3
"""
extract_embeddings.py

This module provides functions to extract audio embeddings from pre-trained TensorFlow models.
We support two models:

1. Discogs-Effnet (trained on 400 Discogs music styles):
   - Uses TensorflowPredictEffnetDiscogs.
   - The model output (by default, "PartitionedCall:1") is expected to produce a 2D array of embeddings
     (one embedding per audio frame). We average these embeddings across time to get a single embedding.

2. MSD-MusicCNN (trained on 50 music tags from the Million Song Dataset):
   - Uses TensorflowPredictMusiCNN.
   - The model output (by default, "model/dense/BiasAdd") is similarly processed by averaging.

**Important:**  
The TensorFlow embedding models expect audio at 16000 Hz. Ensure your input audio is mono and resampled to 16000 Hz.

Example usage:
    from essentia.standard import MonoLoader
    from extract_embeddings import extract_discogs_effnet_embeddings, extract_msd_musicnn_embeddings

    # Load audio at 16000 Hz:
    audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
    discogs_embedding = extract_discogs_effnet_embeddings(audio, model_file="discogs-effnet-bs64-1.pb")
    musicnn_embedding = extract_msd_musicnn_embeddings(audio, model_file="msd-musicnn-1.pb")
    print("Discogs-Effnet embedding:", discogs_embedding)
    print("MSD-MusicCNN embedding:", musicnn_embedding)
"""

import numpy as np
import essentia.standard as es

def extract_discogs_effnet_embeddings(audio, model_file, output="PartitionedCall:1"):
    """
    Extracts embeddings using the Discogs-Effnet model.

    Args:
        audio (np.array): Mono audio signal, sampled at 16000 Hz.
        model_file (str): Path to the Discogs-Effnet model (.pb file).
        output (str): The name of the output tensor from the model. Default is "PartitionedCall:1".

    Returns:
        np.array: A 1D numpy array representing the average embedding across all frames.
    """
    # Instantiate the model wrapper.
    model = es.TensorflowPredictEffnetDiscogs(graphFilename=model_file, output=output)
    # Compute embeddings (output shape is [num_frames, embedding_dim]).
    embeddings = model(audio)
    # Average embeddings across frames (axis=0) to get a single embedding vector.
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def extract_msd_musicnn_embeddings(audio, model_file, output="model/dense/BiasAdd"):
    """
    Extracts embeddings using the MSD-MusicCNN model.

    Args:
        audio (np.array): Mono audio signal, sampled at 16000 Hz.
        model_file (str): Path to the MSD-MusicCNN model (.pb file).
        output (str): The name of the output tensor from the model. Default is "model/dense/BiasAdd".

    Returns:
        np.array: A 1D numpy array representing the average embedding across all frames.
    """
    model = es.TensorflowPredictMusiCNN(graphFilename=model_file, output=output)
    embeddings = model(audio)
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding
