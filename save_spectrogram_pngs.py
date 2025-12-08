#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
# Directory containing processed .npz files
INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/spectrogram_images"

STEMS = ["mix", "vocals", "bass", "drums",
         "guitar", "piano", "strings", "synth_pad"]


def save_spectrogram_image(spec, out_path):
    """
    Save a log-mel spectrogram (spec) as a PNG image.
    spec is assumed to already be in dB scale.
    """

    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label="dB")
    plt.title(os.path.basename(out_path).replace(".png", ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def show_logmel_with_specshow(logmel, out_path): #what i was using at the start but easier to use imshow^^
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        logmel,
        sr=22050,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        fmin=0,
        fmax=11025
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(os.path.basename(out_path).replace(".png", ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def process_npz_file(npz_path, out_root):
    base = os.path.splitext(os.path.basename(npz_path))[0]
    track_out = os.path.join(out_root, base)
    os.makedirs(track_out, exist_ok=True)

    data = np.load(npz_path)

    print(f"Processing {base}...")

    for stem in STEMS:
        if stem not in data:
            continue

        spec = data[stem]  # Already a log-mel spectrogram (dB)
        out_path = os.path.join(track_out, f"{stem}.png")

        # show_logmel_with_specshow(spec, out_path)
        save_spectrogram_image(spec, out_path)

    print(f"  Saved images to {track_out}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    npz_files = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".npz")
    ])

    if not npz_files:
        print("No .npz files found in", INPUT_DIR)
        return

    print("Found", len(npz_files), "processed tracks.")

    for npz in npz_files:
        process_npz_file(npz, OUTPUT_DIR)

    print("\nAll spectrogram images generated!\n")


if __name__ == "__main__":
    main()
