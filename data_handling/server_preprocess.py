#!/usr/bin/env python

import os
import argparse
import json
import yaml
from glob import glob

import numpy as np
import librosa
import soundfile as sf


# ------------ CONFIG ------------
SR = 22050 #sample rate 
N_FFT = 2048 #
HOP_LENGTH = 512
N_MELS = 128 #ideal number of mel bands
FMIN = 0 #min f
FMAX = SR // 2 #max f

STEM_NAMES = ["vocals", "bass", "drums", "guitar", "piano", "strings", "synth_pad"]
# --------------------------------


def compute_log_mel(y, sr=SR):
    """Compute log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


# ---------- MUSDB PROCESSING ----------

def process_musdb_track(track_dir, out_dir):
    """
    Expected files from each song folder:
        mixture.wav
        vocals.wav
        bass.wav
        drums.wav
        other.wav
    """
    track_name = os.path.basename(track_dir.rstrip("/"))
    print(f"[MUSDB] Processing {track_name}")

    mix_path = os.path.join(track_dir, "mixture.wav")
    if not os.path.exists(mix_path):
        print(f"  Skipping {track_name}, no mixture.wav found")
        return

    # Load mixture at 44.1k then resample to 22.05k
    y_mix_44k, sr_in = librosa.load(mix_path, sr=None, mono=True)
    y_mix = librosa.resample(y_mix_44k, orig_sr=sr_in, target_sr=SR)

    # Initialize 7 target waveforms with zeros
    targets = {name: np.zeros_like(y_mix) for name in STEM_NAMES}

    # Vocals, bass, drums are present as separate stems
    for stem_name in ["vocals", "bass", "drums"]:
        stem_file = os.path.join(track_dir, f"{stem_name}.wav")
        if os.path.exists(stem_file):
            y_stem_44k, sr_in = librosa.load(stem_file, sr=None, mono=True)
            y_stem = librosa.resample(y_stem_44k, orig_sr=sr_in, target_sr=SR)
            # Pad or trim to same length as mix
            if len(y_stem) < len(y_mix):
                y_stem = librosa.util.fix_length(y_stem, len(y_mix))
            else:
                y_stem = y_stem[: len(y_mix)]
            targets[stem_name] = y_stem

    # Guitar / piano / strings / synth_pad = 0 for MUSDB
    # (We don't try to separate those from MUSDB's "other" stem.)

    # Ensure all have same length
    for name in STEM_NAMES:
        if len(targets[name]) < len(y_mix):
            targets[name] = librosa.util.fix_length(targets[name], len(y_mix))
        else:
            targets[name] = targets[name][: len(y_mix)]

    # Compute log-mel for mix + all stems
    mix_lms = compute_log_mel(y_mix)
    stems_lms = {name: compute_log_mel(targets[name]) for name in STEM_NAMES}

    # Save
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{track_name}.npz")
    np.savez_compressed(out_path, mix=mix_lms, **stems_lms)
    print(f"  Saved {out_path}")


# ---------- SLAKH PROCESSING ----------

def slakh_instrument_to_stem(inst_class: str, is_drum: bool):
    """Map Slakh instrument metadata to one of the 7 stem categories."""
    
    if is_drum:
        return "drums"

    if inst_class is None:
        return None
    
    name = inst_class.lower()

    if "bass" in name:
        return "bass"
    if "guitar" in name:
        return "guitar"
    if "piano" in name or "keyboard" in name or "keys" in name:
        return "piano"
    if "string" in name or "violin" in name or "cello" in name or "viola" in name:
        return "strings"
    if "pad" in name or "synth" in name or "lead" in name or "pluck" in name:
        return "synth_pad"

    return None


def process_slakh_track(track_dir, out_dir):
    """
    Process a Slakh track using the actual YAML structure:
    - metadata.yaml contains:
        stems:
            S00:
                inst_class: Guitar
                is_drum: false
                ...
    """

    track_name = os.path.basename(track_dir.rstrip("/"))
    print(f"[SLAKH] Processing {track_name}")

    mix_path = os.path.join(track_dir, "mix.wav")
    meta_path = os.path.join(track_dir, "metadata.yaml")
    stems_dir = os.path.join(track_dir, "stems")

    if not (os.path.exists(mix_path) and os.path.exists(meta_path) and os.path.isdir(stems_dir)):
        print(f"  Skipping {track_name}, missing mix/metadata/stems")
        return

    # Load mixture (Slakh is already at 22.05k)
    y_mix, sr_in = librosa.load(mix_path, sr=SR, mono=True)
    assert sr_in == SR or sr_in is None

    # Initialize target signals (no vocals in Slakh)
    targets = {name: np.zeros_like(y_mix) for name in STEM_NAMES}

    # Read YAML metadata
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    stems_meta = meta.get("stems", {})

    # --- Process each stem according to inst_class and is_drum ---
    for stem_id, stem_info in stems_meta.items():

        # inst_class = stem_info.get("inst_class", "")
        inst_class = (
        stem_info.get("inst_class")      # Slakh v2
        or stem_info.get("instrument")   # Slakh v1
        or stem_info.get("plugin_name")  # Sometimes used in subsets
        or ""
        )

        is_drum = stem_info.get("is_drum", False)


        # Updated instrument classifier
        target_stem = slakh_instrument_to_stem(inst_class, is_drum=is_drum)

        if target_stem is None:
            # Ignore non-target instruments (Organ, Sound Effects, etc.)
            continue

        # WAV file name should match stem_id, e.g., S00.wav
        stem_file = os.path.join(stems_dir, f"{stem_id}.wav")

        if not os.path.exists(stem_file):
            # Some datasets use uppercase names
            alt_file = os.path.join(stems_dir, f"{stem_id.upper()}.wav")
            if os.path.exists(alt_file):
                stem_file = alt_file
            else:
                print(f"  Missing stem file for {stem_id}, skipping")
                continue

        # Load actual audio
        y_stem, _ = librosa.load(stem_file, sr=SR, mono=True)

        # Match mix length
        if len(y_stem) < len(y_mix):
            y_stem = librosa.util.fix_length(y_stem, len(y_mix))
        else:
            y_stem = y_stem[:len(y_mix)]

        # Add to the appropriate target
        targets[target_stem] += y_stem

    # --- Ensure length consistency ---
    for name in STEM_NAMES:
        targets[name] = librosa.util.fix_length(targets[name], len(y_mix))

    # --- Compute log-mels ---
    mix_lms = compute_log_mel(y_mix)
    stems_lms = {name: compute_log_mel(targets[name]) for name in STEM_NAMES}

    # Save output
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{track_name}.npz")

    np.savez_compressed(out_path, mix=mix_lms, **stems_lms)
    print(f"  Saved {out_path}")

# ---------- DRIVER ----------

def find_subdirs(root):
    return sorted([d for d in glob(os.path.join(root, "*")) if os.path.isdir(d)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["slakh", "musdb"], required=True)
    parser.add_argument("--raw_root", required=True, help="Root folder for raw dataset")
    parser.add_argument("--out_root", required=True, help="Where to write processed .npz files")
    parser.add_argument("--max_tracks", type=int, default=None, help="Optional limit for debugging")
    args = parser.parse_args()

    # Each dataset may have different folder layout; adjust if needed.
    if args.dataset == "slakh":
        track_dirs = find_subdirs(args.raw_root)
        out_dir = os.path.join(args.out_root, "slakh_lms")
        os.makedirs(out_dir, exist_ok=True)

        for i, track_dir in enumerate(track_dirs):
            if args.max_tracks is not None and i >= args.max_tracks:
                break
            process_slakh_track(track_dir, out_dir)

    elif args.dataset == "musdb":
        # For MUSDB18, you might have raw_root/train and raw_root/test
        # You can choose train only for preprocessing:
        track_dirs = find_subdirs(args.raw_root)
        out_dir = os.path.join(args.out_root, "musdb_lms")
        os.makedirs(out_dir, exist_ok=True)

        for i, track_dir in enumerate(track_dirs):
            if args.max_tracks is not None and i >= args.max_tracks:
                break
            process_musdb_track(track_dir, out_dir)


if __name__ == "__main__":
    main()
