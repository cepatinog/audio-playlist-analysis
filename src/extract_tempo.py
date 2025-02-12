#!/usr/bin/env python3
"""
extract_tempo.py

This module provides a function to extract tempo features from a given mono audio signal.
It supports two methods:
  - 'tempocnn': Uses Essentia’s TempoCNN (requires a TensorFlow model file).
  - 'rhythm'  : Uses the RhythmExtractor2013 algorithm.
  
The function returns a dictionary with tempo features.
"""

import essentia.standard as es
import numpy as np

def extract_tempo_features(audio, method='tempocnn', model_file=None):
    """
    Extract tempo features from a mono audio signal.

    Args:
        audio (np.array): Mono audio signal.
        method (str): 'tempocnn' or 'rhythm'. Default is 'tempocnn'.
        model_file (str): Required if method is 'tempocnn'; path to the TempoCNN model file.

    Returns:
        dict: Dictionary containing tempo features.
              - For 'tempocnn': keys 'global_bpm', 'local_bpms', 'local_probs'.
              - For 'rhythm': keys 'global_bpm', 'beats', 'beats_confidence'.
    """
    features = {}
    if method == 'tempocnn':
        if not model_file:
            raise ValueError("Model file must be provided for 'tempocnn' method.")
        # TempoCNN expects audio at 11025 Hz; ensure your audio is sampled appropriately.
        global_bpm, local_bpms, local_probs = es.TempoCNN(graphFilename=model_file)(audio)
        features['global_bpm'] = global_bpm
        features['local_bpms'] = local_bpms.tolist() if hasattr(local_bpms, 'tolist') else local_bpms
        features['local_probs'] = local_probs.tolist() if hasattr(local_probs, 'tolist') else local_probs
    elif method == 'rhythm':
        bpm, beats, beats_confidence, _, beat_intervals = es.RhythmExtractor2013(method="multifeature")(audio)
        features['global_bpm'] = bpm
        features['beats'] = beats
        features['beats_confidence'] = beats_confidence
    else:
        raise ValueError("Unsupported method: {}".format(method))
    return features
