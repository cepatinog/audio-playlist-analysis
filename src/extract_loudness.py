#!/usr/bin/env python3
"""
extract_loudness.py

This module provides a function to extract loudness features from a given stereo audio signal.
It uses Essentia's LoudnessEBUR128 algorithm to compute:
  - momentary loudness (over 400 ms) in LUFS,
  - short-term loudness (over 3 s) in LUFS,
  - integrated loudness (overall) in LUFS,
  - loudness range (LU).

The function returns a dictionary with these loudness features.
"""

import essentia.standard as es

def extract_loudness_features(audio_stereo, hopSize=0.1, sampleRate=44100, startAtZero=False):
    """
    Extract loudness features from a stereo audio signal using LoudnessEBUR128.

    Args:
        audio_stereo (np.array): Stereo audio signal.
        hopSize (float): The hop size (in seconds) for computing loudness (default 0.1).
        sampleRate (int): The sampling rate of the audio signal (default 44100).
        startAtZero (bool): If True, start loudness estimation at time 0 (default False).

    Returns:
        dict: A dictionary containing:
            - 'momentary_loudness': list of momentary loudness values (LUFS),
            - 'short_term_loudness': list of short-term loudness values (LUFS),
            - 'integrated_loudness': overall integrated loudness (LUFS),
            - 'loudness_range': loudness range (LU).
    """
    loudness_extractor = es.LoudnessEBUR128(
        hopSize=hopSize,
        sampleRate=sampleRate,
        startAtZero=startAtZero
    )
    
    momentary_loudness, short_term_loudness, integrated_loudness, loudness_range = loudness_extractor(audio_stereo)
    
    return {
        'momentary_loudness': momentary_loudness,
        'short_term_loudness': short_term_loudness,
        'integrated_loudness': integrated_loudness,
        'loudness_range': loudness_range
    }
