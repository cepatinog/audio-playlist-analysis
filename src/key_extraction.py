#!/usr/bin/env python3
"""
key_extraction.py

This module provides a function to extract key features from a given mono audio signal
using Essentia’s KeyExtractor algorithm. It applies the algorithm three times—once for each
profile ("temperley", "krumhansl", and "edma")—and returns the results in a dictionary.

The returned dictionary contains:
  - "<profile>_key": the estimated musical key (e.g., "C", "G#", etc.)
  - "<profile>_scale": the scale detected ("major" or "minor")
  - "<profile>_strength": a confidence/strength value for the estimation

Example usage:
    import essentia.standard as es
    from key_extraction import extract_key_features

    audio = es.MonoLoader(filename='example.mp3')()
    features = extract_key_features(audio)
    print(features)
"""

import essentia.standard as es

def extract_key_features(audio):
    """
    Extract key-related features from a mono audio signal using three key estimation profiles.

    Args:
        audio (np.array): Mono audio signal.

    Returns:
        dict: A dictionary with key, scale, and strength for each profile. For example:
            {
              'temperley_key': 'C',
              'temperley_scale': 'major',
              'temperley_strength': 0.85,
              'krumhansl_key': 'C',
              'krumhansl_scale': 'major',
              'krumhansl_strength': 0.80,
              'edma_key': 'C',
              'edma_scale': 'major',
              'edma_strength': 0.78
            }
    """
    results = {}
    profiles = ["temperley", "krumhansl", "edma"]
    for profile in profiles:
        key_extractor = es.KeyExtractor(
            averageDetuningCorrection=True,
            frameSize=4096,
            hopSize=4096,
            hpcpSize=12,
            maxFrequency=3500,
            maximumSpectralPeaks=60,
            minFrequency=25,
            pcpThreshold=0.2,
            profileType=profile,
            sampleRate=44100,
            spectralPeaksThreshold=0.0001,
            tuningFrequency=440,
            weightType="cosine",
            windowType="hann"
        )
        key, scale, strength = key_extractor(audio)
        results[f"{profile}_key"] = key
        results[f"{profile}_scale"] = scale
        results[f"{profile}_strength"] = strength
    return results
    