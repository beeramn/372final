import os
import librosa
import soundfile as sf
import numpy as np

MUSDB_DIR = "data/musdb" # old
OUTPUT_DIR = "data/musdb_2stem" # new

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_audio(path, sr=44100):
    """Load a WAV file and return waveform, sampling rate."""
    audio, _ = librosa.load(path, sr=sr, mono=False)
    return audio

def ensure_same_shape(a, b):
    """Pad or trim b to match shape of a."""
    if b.shape[-1] > a.shape[-1]:
        b = b[..., :a.shape[-1]]
    elif b.shape[-1] < a.shape[-1]:
        pad = a.shape[-1] - b.shape[-1]
        b = np.pad(b, ((0, 0), (0, pad)), mode='constant')
    return b

for track_name in os.listdir(MUSDB_DIR):
    track_path = os.path.join(MUSDB_DIR, track_name)
    if not os.path.isdir(track_path):
        continue

    print(f"Processing {track_name} ...")

    # filenames in MUSDB
    bass_path = os.path.join(track_path, "bass.wav")
    drums_path = os.path.join(track_path, "drums.wav")
    other_path = os.path.join(track_path, "other.wav")
    vocals_path = os.path.join(track_path, "vocals.wav")
    mix_path = os.path.join(track_path, "mixture.wav")

    # load audio files
    mix = load_audio(mix_path)
    vocals = load_audio(vocals_path)
    bass = load_audio(bass_path)
    drums = load_audio(drums_path)
    other = load_audio(other_path)

    # make sure shapes match
    drums = ensure_same_shape(mix, drums)
    bass = ensure_same_shape(mix, bass)
    other = ensure_same_shape(mix, other)
    vocals = ensure_same_shape(mix, vocals)

    # combine: instrumental = bass + drums + other
    instrumental = bass + drums + other

    # save into new folder
    out_track_dir = os.path.join(OUTPUT_DIR, track_name)
    os.makedirs(out_track_dir, exist_ok=True)

    sf.write(os.path.join(out_track_dir, "mixture.wav"), mix.T, 44100)
    sf.write(os.path.join(out_track_dir, "vocals.wav"), vocals.T, 44100)
    sf.write(os.path.join(out_track_dir, "instrumental.wav"), instrumental.T, 44100)

print("Done!")