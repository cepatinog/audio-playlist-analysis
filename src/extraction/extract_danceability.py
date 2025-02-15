#!/usr/bin/env python3
"""
extract_danceability.py

This module provides a function to extract danceability features from an audio signal.
It supports two modes:

1. "signal" mode: Uses Essentia's signal processing Danceability algorithm.
   This algorithm takes a mono audio signal (typically at 44100 Hz) and returns:
      - danceability: a real value (usually in the range 0 to ~3), and
      - dfa: a vector representing the DFA exponent values.

2. "classifier" mode: Uses a pre-trained danceability classifier model (e.g., 
   "danceability-discogs-effnet-1.pb") that expects discogs-effnet embeddings as input.
   The classifier outputs a softmax probability vector over two classes.
   
   **Note on Metadata:**  
   The metadata for this model specifies:
      - Input tensor name: "model/Placeholder" with shape [1200]
      - Output tensor name: "model/Softmax" with shape [2]
      - Classes: ["danceable", "not_danceable"]

   To meet these requirements, if the provided embedding does not have length 1200,
   we trim it (if longer) or pad it with zeros (if shorter).

Example usage:
    # Signal mode:
    from essentia.standard import MonoLoader
    from extract_danceability import extract_danceability_features
    audio = MonoLoader(filename="audio.wav", sampleRate=44100)()
    result = extract_danceability_features(audio, mode="signal")
    print(result)
    
    # Classifier mode:
    # (Assuming you have already extracted discogs embeddings from the audio)
    from extract_danceability import extract_danceability_features
    result = extract_danceability_features(discogs_embedding, mode="classifier", model_file="danceability-discogs-effnet-1.pb")
    print(result)
"""

import essentia.standard as es
import numpy as np

def adjust_embedding_dimension(embedding, expected_dim=1280):
    """
    Adjust the embedding to have the expected dimension:
      - If embedding has more dimensions than expected, trim it.
      - If it's smaller, pad with zeros.
      
    Args:
        embedding (np.array): A 2D numpy array of shape (num_frames, embedding_dim)
                              or a 1D numpy array of shape (embedding_dim,).
        expected_dim (int): The desired dimensionality (default 1280).
        
    Returns:
        np.array: A 2D numpy array with shape (1, expected_dim).
    """
    # Ensure the embedding is 2D.
    if embedding.ndim == 1:
        embedding = np.expand_dims(embedding, axis=0)
    
    current_dim = embedding.shape[1]
    if current_dim > expected_dim:
        # Trim the embedding to the expected dimension.
        adjusted = embedding[:, :expected_dim]
    elif current_dim < expected_dim:
        # Pad with zeros to reach the expected dimension.
        pad_width = expected_dim - current_dim
        adjusted = np.pad(embedding, ((0, 0), (0, pad_width)), mode='constant')
    else:
        adjusted = embedding
    return adjusted

def extract_danceability_features(audio, mode="signal", model_file=None, output_tensor="model/Softmax",
                                  sampleRate=44100, maxTau=8800, minTau=310, tauMultiplier=1.1):
    """
    Extract danceability features from an audio signal.

    Args:
        audio (np.array): 
            - For "signal" mode: a mono audio signal (vector_real) at the given sampleRate.
            - For "classifier" mode: discogs-effnet embeddings. Can be a 1D averaged vector or a 2D array.
        mode (str): "signal" or "classifier".
        model_file (str): Path to the danceability classifier model (.pb file). Required if mode=="classifier".
        output_tensor (str): The name of the output tensor for the classifier (default "model/Softmax").
        sampleRate (int): Sampling rate for signal mode (default 44100 Hz).
        maxTau (float): Maximum segment length for DFA (default 8800 ms).
        minTau (float): Minimum segment length for DFA (default 310 ms).
        tauMultiplier (float): Tau multiplier (default 1.1).

    Returns:
        dict: If mode=="signal", returns:
                {"danceability": <danceability_value>, "dfa": <dfa_vector as list>}.
              If mode=="classifier", returns:
                {"danceability_classifier": <softmax probability vector as list>, "predicted_class": <"danceable" or "not_danceable">}.
    """
    if mode == "signal":
        # Use Essentia's Danceability algorithm.
        danceability_val, dfa = es.Danceability(maxTau=maxTau, minTau=minTau, sampleRate=sampleRate, tauMultiplier=tauMultiplier)(audio)
        return {"danceability": danceability_val, "dfa": dfa.tolist() if hasattr(dfa, "tolist") else dfa}
    
    elif mode == "classifier":
        if model_file is None:
            raise ValueError("For classifier mode, a model_file must be provided.")
        # Ensure the input embeddings are 2D.
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        # Adjust the embedding dimension to 1280, as expected by the model.
        adjusted_embedding = adjust_embedding_dimension(audio, expected_dim=1280)
        
        # Instantiate the classifier.
        # Use the input tensor name from metadata: "model/Placeholder"
        model = es.TensorflowPredict2D(graphFilename=model_file, input="model/Placeholder", output=output_tensor)
        predictions = model(adjusted_embedding)
        
        # If predictions come as multiple frames, average them.
        if predictions.ndim == 2:
            predictions = np.mean(predictions, axis=0)
        
        # According to the metadata, classes order is: ["danceable", "not_danceable"]
        labels = ["danceable", "not_danceable"]
        predicted_class = labels[np.argmax(predictions)]
        
        return {"danceability_classifier": predictions.tolist(), "predicted_class": predicted_class}
    
    else:
        raise ValueError("Unsupported mode. Choose 'signal' or 'classifier'.")
