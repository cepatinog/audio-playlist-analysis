"""
extract_tempo.py

This script demonstrates how to extract tempo (BPM) from an audio file using two different methods:
1. RhythmExtractor2013 (signal processing based)
2. TempoCNN (machine learning model based)

Usage:
    python extract_tempo.py <audio_file> <method> [model_file_if_using_TempoCNN]

    - <audio_file>    : Path to the audio file to analyze (e.g., .wav or .mp3).
    - <method>        : 'rhythm' for RhythmExtractor2013 or 'tempocnn' for TempoCNN.
    - [model_file]    : (Required only for 'tempocnn') Path to the TempoCNN model file (e.g., deeptemp-k16-3.pb).
"""

import sys
import essentia
import essentia.standard as es
import numpy as np
import matplotlib.pyplot as plt

def extract_tempo_rhythm_extractor(audio_file):
    """
    Uses RhythmExtractor2013 to compute the BPM, beat positions, and confidence.
    Also computes and plots a BPM histogram.

    Decisions & Details:
    - MonoLoader is used to load audio as mono. Most beat and tempo algorithms expect mono input.
    - We choose the "multifeature" method in RhythmExtractor2013 for higher accuracy,
      as it combines multiple features.
    - BpmHistogramDescriptors is applied to the intervals between beats (beat_intervals)
      to analyze the distribution of BPM estimates.
    """
    # Load the audio file in mono format
    print(f"Loading audio file: {audio_file}")
    audio = es.MonoLoader(filename=audio_file)()

    # Instantiate the RhythmExtractor2013 with the multifeature method
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beat_intervals = rhythm_extractor(audio)

    # Output the main tempo estimation results
    print("\n--- RhythmExtractor2013 Results ---")
    print(f"Estimated BPM: {bpm:.2f}")
    print(f"Beat positions (seconds): {beats}")
    print(f"Beat estimation confidence: {beats_confidence:.4f}")

    # Compute a BPM histogram to analyze the distribution of BPM estimates
    (peak1_bpm, peak1_weight, peak1_spread,
     peak2_bpm, peak2_weight, peak2_spread,
     histogram) = es.BpmHistogramDescriptors()(beat_intervals)

    print("\nBPM Histogram:")
    print(f"Primary BPM peak: {peak1_bpm:.2f} bpm")
    print(f"Secondary BPM peak: {peak2_bpm:.2f} bpm")

    # Plot the BPM histogram using matplotlib
    plt.figure()
    plt.bar(range(len(histogram)), histogram, width=1, edgecolor='black')
    plt.xlabel('BPM bins')
    plt.ylabel('Frequency')
    plt.title('BPM Histogram')
    plt.show()

def extract_tempo_tempocnn(audio_file, model_file):
    """
    Uses TempoCNN to estimate the tempo (BPM) from the audio.
    Also visualizes a slice of the waveform with tempo grid markers.

    Decisions & Details:
    - The TempoCNN model expects audio at a specific sample rate (here, we use 11025 Hz).
    - The algorithm outputs a global BPM (via majority voting over local predictions)
      and also local BPM values with their confidence probabilities.
    - We plot a 5-second slice of the audio waveform and overlay a grid based on the estimated BPM
      to visually verify the beat placements.
    """
    sr = 11025  # Sample rate required by TempoCNN
    print(f"Loading audio file: {audio_file} at sample rate {sr} Hz")
    audio = es.MonoLoader(filename=audio_file, sampleRate=sr)()

    # Instantiate TempoCNN using the provided model file.
    # Ensure you have downloaded the model (e.g., deeptemp-k16-3.pb) beforehand.
    global_bpm, local_bpm, local_probs = es.TempoCNN(graphFilename=model_file)(audio)

    # Output the TempoCNN estimation results
    print("\n--- TempoCNN Results ---")
    print(f"Global BPM: {global_bpm:.2f}")
    print(f"Local BPM values: {local_bpm}")
    print(f"Local BPM probabilities: {local_probs}")

    # Visualize a slice of the audio with a tempo grid overlay.
    duration = 5  # seconds to visualize
    audio_slice = audio[:sr * duration]
    plt.figure()
    plt.plot(audio_slice, label="Audio waveform")
    # Calculate grid markers: spacing (in samples) is determined by BPM (converted to seconds)
    # BPM (beats per minute) -> seconds per beat = 60 / BPM -> samples per beat = (60 / BPM) * sr
    samples_per_beat = (60 / global_bpm) * sr
    markers = np.arange(0, len(audio_slice), samples_per_beat)
    for marker in markers:
        plt.axvline(x=marker, color='red', linestyle='--', alpha=0.7)
    plt.title("Audio Waveform with Tempo Grid Overlay")
    plt.xlabel("Samples")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Verify command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python extract_tempo.py <audio_file> <method> [model_file_if_using_TempoCNN]")
        print("  <method> must be 'rhythm' or 'tempocnn'")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    method = sys.argv[2].lower()

    if method == 'rhythm':
        extract_tempo_rhythm_extractor(audio_file)
    elif method == 'tempocnn':
        if len(sys.argv) < 4:
            print("Error: For 'tempocnn' method, please provide the model file (e.g., deeptemp-k16-3.pb)")
            sys.exit(1)
        model_file = sys.argv[3]
        extract_tempo_tempocnn(audio_file, model_file)
    else:
        print("Error: Unknown method. Please use 'rhythm' or 'tempocnn'.")
        sys.exit(1)
