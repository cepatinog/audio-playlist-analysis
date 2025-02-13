#!/usr/bin/env python3
"""
extract_voice_instrumental.py

This module provides a function to classify music as "voice" vs "instrumental"
using discogs‐effnet embeddings. It uses the pre-trained model
"voice_instrumental-discogs-effnet-1.pb", which outputs a softmax probability vector
over two classes.

Example usage:
    from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
    from extract_voice_instrumental import extract_voice_instrumental

    # Load audio at 16 kHz.
    audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(
                        graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    embeddings = embedding_model(audio)

    # Classify the track.
    result = extract_voice_instrumental(embeddings, model_file="voice_instrumental-discogs-effnet-1.pb")
    print(result)  # e.g., {"predictions": array([...]), "predicted_class": "voice"}
"""

import numpy as np
import essentia.standard as es

def extract_voice_instrumental(embeddings, model_file, input_tensor="model/Placeholder", output_tensor="model/Softmax"):
    """
    Classify music as instrumental or voice using discogs-effnet embeddings.

    Args:
        embeddings (np.array): Discogs-effnet embeddings. Can be a 1D averaged vector
                               or a 2D array (frames x embedding_dim).
        model_file (str): Path to the voice/instrumental model (.pb file).
        input_tensor (str): Name of the model's input tensor. Default: "model/Placeholder".
        output_tensor (str): Name of the output tensor. Default: "model/Softmax".

    Returns:
        dict: A dictionary containing:
              - "predictions": A 1D numpy array of softmax probabilities for the two classes.
              - "predicted_class": The class label with the highest probability ("instrumental" or "voice").
    """
    import numpy as np
    # Ensure embeddings are 2D.
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
    
    # Instantiate the voice/instrumental classifier.
    model = es.TensorflowPredict2D(graphFilename=model_file, input=input_tensor, output=output_tensor)
    
    # Compute predictions.
    predictions = model(embeddings)
    
    # If predictions come as multiple frames, average them.
    if predictions.ndim == 2:
        predictions = np.mean(predictions, axis=0)
    
    # Define the class labels. (Assumed order: [instrumental, voice])
    labels = ["instrumental", "voice"]
    predicted_class = labels[np.argmax(predictions)]
    
    return {"predictions": predictions, "predicted_class": predicted_class}

