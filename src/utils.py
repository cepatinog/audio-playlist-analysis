#!/usr/bin/env python3
"""
utils.py

This module provides a helper function to load an audio file (using AudioLoader) only once,
and produce both stereo and mono signals suitable for different processing tasks.

- The stereo signal is needed for loudness extraction (LoudnessEBUR128 requires stereo).
- The mono signal is obtained using MonoMixer and is used for key extraction.
- For tempo extraction via TempoCNN, we further resample the mono signal to the target sample rate.
  
Assumptions:
  - If AudioLoader returns a tuple, the first element is the audio data (a VECTOR_STEREOSAMPLE),
    the second is the native sample rate, and the third is the number of channels.
  - If not, we assume a default sample rate (e.g., 44100 Hz) and 2 channels.
"""

import numpy as np   
import essentia.standard as es
import os

def load_audio_file(filename, targetMonoSampleRate=44100, targetTempoSampleRate=11025):
    """
    Load an audio file once and generate the required versions.

    Args:
        filename (str): Path to the audio file.
        targetMonoSampleRate (int): The sample rate for the mono audio (for key extraction). Default is 44100 Hz.
        targetTempoSampleRate (int): The sample rate for tempo extraction (default is 11025 Hz).

    Returns:
        dict: A dictionary containing:
            - 'stereo_audio': Stereo audio signal (VECTOR_STEREOSAMPLE) at its native sample rate
                              (or resampled to 44100 Hz if necessary).
            - 'mono_audio': Mono audio signal (VECTOR_REAL) at targetMonoSampleRate (for key extraction).
            - 'mono_tempo': Mono audio signal resampled to targetTempoSampleRate (for tempo extraction).
            - 'sampleRate': The sample rate used for the mono audio version.
            - 'numChannels': Number of channels in the original file.
    """
    # Load audio using AudioLoader.
    audio_data = es.AudioLoader(filename=filename)()
    if isinstance(audio_data, tuple):
        stereo_audio = audio_data[0]
        native_rate = audio_data[1]
        numChannels = audio_data[2]
    else:
        stereo_audio = audio_data
        native_rate = 44100
        numChannels = 2

    # For loudness extraction, ensure stereo audio is at 44100 Hz.
    if native_rate != 44100:
        stereo_audio = es.Resample(inputSampleRate=native_rate, outputSampleRate=44100)(stereo_audio)
        native_rate = 44100

    # Convert stereo to mono using MonoMixer.
    # MonoMixer expects:
    #   - A stereo signal as a 2D NumPy array (VECTOR_STEREOSAMPLE), and
    #   - The number of channels (integer) as the second argument.
    # Our stereo_audio is already a 2D NumPy array (shape: (num_samples, 2)), so we pass it directly.
    mono_audio = es.MonoMixer()(stereo_audio, numChannels)

    # Ensure the mono audio is at targetMonoSampleRate for key extraction.
    if native_rate != targetMonoSampleRate:
        mono_audio = es.Resample(inputSampleRate=native_rate, outputSampleRate=targetMonoSampleRate)(mono_audio)
        used_rate = targetMonoSampleRate
    else:
        used_rate = native_rate

    # For tempo extraction using TempoCNN, resample the mono audio to targetTempoSampleRate.
    if used_rate != targetTempoSampleRate:
        mono_tempo = es.Resample(inputSampleRate=used_rate, outputSampleRate=targetTempoSampleRate)(mono_audio)
    else:
        mono_tempo = mono_audio

    return {
        'stereo_audio': stereo_audio,
        'mono_audio': mono_audio,
        'mono_tempo': mono_tempo,
        'sampleRate': used_rate,
        'numChannels': numChannels
    }


def convert_numpy(obj):
    """
    Recursively convert NumPy arrays to lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    else:
        return obj