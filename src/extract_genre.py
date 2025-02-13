#!/usr/bin/env python3
"""
extract_genre.py

This module provides a function to predict genre/style activations from 
discogs‐effnet embeddings using the Genre Discogs400 model.

The Genre Discogs400 model outputs activation values for 400 music styles.
It is designed to work with discogs‐effnet embeddings. Typically, you first
extract embeddings with TensorflowPredictEffnetDiscogs (see extract_embeddings.py),
and then feed those embeddings into this model.

Example usage:
    from essentia.standard import MonoLoader
    from extract_embeddings import extract_discogs_effnet_embeddings
    from extract_genre import extract_genre_features

    # Load audio at 16kHz:
    audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
    discogs_embedding = extract_discogs_effnet_embeddings(audio, model_file="discogs-effnet-bs64-1.pb")
    genre_predictions = extract_genre_features(discogs_embedding, model_file="genre_discogs400-discogs-effnet-1.pb")
    print("Genre predictions:", genre_predictions)
"""

import essentia.standard as es
import numpy as np

def extract_genre_features(embeddings, model_file, input_tensor="serving_default_model_Placeholder", output_tensor="PartitionedCall:0"):
    """
    Extract genre style predictions from discogs‐effnet embeddings using the Genre Discogs400 model.

    Args:
        embeddings (np.array): Discogs‐effnet embeddings. This can be a 2D array
                               (num_frames x embedding_dim) or a 1D averaged vector.
        model_file (str): Path to the Genre Discogs400 model (.pb file).
        input_tensor (str): Name of the model's input tensor. Default is "serving_default_model_Placeholder".
        output_tensor (str): Name of the model's output tensor. Default is "PartitionedCall:0".

    Returns:
        np.array: A 1D numpy array with 400 activation values representing the predicted
                  genre/style probabilities/activations.
    """
    # Instantiate the Genre Discogs400 model.
    # Ensure the embeddings are 2D
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
    
    model = es.TensorflowPredict2D(graphFilename=model_file, input=input_tensor, output=output_tensor)
    predictions = model(embeddings)

    # If predictions are provided per frame (2D array), average across time.
    if predictions.ndim == 2:
        predictions = np.mean(predictions, axis=0)
    return predictions
