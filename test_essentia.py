import essentia
import essentia.standard as es

def test_essentia(audio_file):
    # Load the audio file in mono
    print(f"Loading audio file: {audio_file}")
    loader = es.MonoLoader(filename=audio_file)
    audio = loader()
    print(f"Audio loaded successfully. Number of samples: {len(audio)}")

    # Compute RMS (Root Mean Square) energy
    rms_value = es.RMS()(audio)
    print(f"RMS Energy: {rms_value:.4f}")

    # Compute tempo using RhythmExtractor2013
    print("Extracting tempo using RhythmExtractor2013...")
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, estimates, _, _ = rhythm_extractor(audio)
    print(f"Estimated BPM: {bpm:.2f}")
    print(f"Number of beats detected: {len(beats)}")

if __name__ == '__main__':
        
    audio_file_path = "0BJPGg90E6p2Ve0D8EcZGF.mp3"
    test_essentia(audio_file_path)
