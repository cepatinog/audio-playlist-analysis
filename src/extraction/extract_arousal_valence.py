#!/usr/bin/env python3
"""
extract_arousal_valence.py

This module provides a function to extract arousal and valence features
(music emotion) using a pre-trained regression model on the emoMusic dataset.
It uses the following two steps:

1. Embedding Extraction:
   - Uses the MSD-MusicCNN model to compute an embedding from a mono audio signal.
   - The model is instantiated via TensorflowPredictMusiCNN with the output tensor "model/dense/BiasAdd".
   - The output embedding is expected to be a 2D array with shape [num_frames, 200]. 
     We average over frames to get a single 200-dimensional vector.

2. Arousal-Valence Regression:
   - Uses the emoMusic regression model (e.g., "emomusic-msd-musicnn-2.pb") to predict a 
     2D output (valence, arousal). According to the metadata, the input tensor is named 
     "model/Placeholder" and is expected to have shape [200], and the output tensor is "model/Identity".
   - If necessary, the averaged embedding is reshaped to match the expected 2D shape.

The function returns a dictionary with keys "valence" and "arousal". The predicted values are 
in the range [1, 9].

Example usage:
    from essentia.standard import MonoLoader
    from extract_arousal_valence import extract_arousal_valence_features

    # Load audio at 16kHz (the models expect 16kHz mono)
    audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
    result = extract_arousal_valence_features(audio,
                                              embedding_model_file="msd-musicnn-1.pb",
                                              regression_model_file="emomusic-msd-musicnn-2.pb")
    print("Arousal/Valence:", result)
"""

import numpy as np
import essentia.standard as es

def extract_arousal_valence_features(audio, embedding_model_file, regression_model_file,
                                     input_tensor="model/Placeholder", output_tensor="model/Identity"):
    """
    Extracts arousal and valence features from a mono audio signal using the emoMusic regression model.
    
    Args:
        audio (np.array): Mono audio signal, sampled at 16000 Hz.
        embedding_model_file (str): Path to the MSD-MusicCNN model (.pb file) used for embeddings.
        regression_model_file (str): Path to the emoMusic regression model (.pb file).
        input_tensor (str): Name of the input tensor for the regression model (default "model/Placeholder").
        output_tensor (str): Name of the output tensor for the regression model (default "model/Identity").
        
    Returns:
        dict: A dictionary containing:
              - "valence": predicted valence (float),
              - "arousal": predicted arousal (float).
    """
    # --- Step 1: Extract embeddings using MSD-MusicCNN ---
    embedding_model = es.TensorflowPredictMusiCNN(graphFilename=embedding_model_file, output="model/dense/BiasAdd")
    embeddings = embedding_model(audio)
    # Average the frame-wise embeddings to obtain a single embedding vector.
    if embeddings.ndim == 2:
        avg_embedding = np.mean(embeddings, axis=0)
    else:
        avg_embedding = embeddings
    # Ensure the embedding is 2D (shape: [1, 200]) as required by the regression model.
    if avg_embedding.ndim == 1:
        avg_embedding = np.expand_dims(avg_embedding, axis=0)
    
    # --- Step 2: Predict arousal and valence using the regression model ---
    # Instantiate the regression model.
    regression_model = es.TensorflowPredict2D(graphFilename=regression_model_file,
                                              input=input_tensor,
                                              output=output_tensor)
    predictions = regression_model(avg_embedding)
    # If the regression model outputs multiple predictions (2D), average them.
    if predictions.ndim == 2:
        predictions = np.mean(predictions, axis=0)
    
    # According to metadata, the output shape is [2], corresponding to [valence, arousal].
    valence, arousal = predictions[0], predictions[1]
    
    return {"valence": float(valence), "arousal": float(arousal)}
