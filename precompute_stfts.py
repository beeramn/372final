import os
import torch

from processing.preprocessing import load_audio, stft_mag_phase

ROOT = os.path.join("data", "musdb_2stem")

def process_song(song_path):
    """Compute STFT magnitude + phase for mixture/vocals/instrumental."""
    mix = load_audio(os.path.join(song_path, "mixture.wav"))
    voc = load_audio(os.path.join(song_path, "vocals.wav"))
    inst = load_audio(os.path.join(song_path, "instrumental.wav"))

    # Compute STFT
    mix_mag, mix_phase = stft_mag_phase(mix)
    voc_mag, _ = stft_mag_phase(voc)
    inst_mag, _ = stft_mag_phase(inst)

    # Save tensors
    torch.save(mix_mag, os.path.join(song_path, "mix_mag.pt"))
    torch.save(mix_phase, os.path.join(song_path, "mix_phase.pt"))
    torch.save(voc_mag, os.path.join(song_path, "voc_mag.pt"))
    torch.save(inst_mag, os.path.join(song_path, "inst_mag.pt"))


def main():
    songs = sorted(os.listdir(ROOT))

    for song in songs:
        song_path = os.path.join(ROOT, song)
        if not os.path.isdir(song_path):
            continue

        print(f"Processing: {song}")
        process_song(song_path)

    print("\nDone! STFT files saved in each song folder.")


if __name__ == "__main__":
    main()
