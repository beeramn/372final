#!/usr/bin/env python3
import os
import yaml
import numpy as np
import librosa


# =======================================
# CONFIGURATION
# =======================================

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 0
FMAX = SR // 2

STEM_NAMES = ["vocals", "bass", "drums", "guitar", "piano", "strings", "synth_pad"]

# Base directories (relative to project root)
DATA_ROOT = "../data"
MUSDB_ROOT = os.path.join(DATA_ROOT, "musdb")
SLAKH_ROOT = os.path.join(DATA_ROOT, "slakh")
OUT_ROOT = os.path.join(DATA_ROOT, "processed")


# =======================================
# MEL SPECTROGRAM
# =======================================

def compute_log_mel(y, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


# =======================================
# SLAKH INSTRUMENT MAPPER
# =======================================

def slakh_instrument_to_stem(inst_name: str, is_drum: bool):
    """Map Slakh metadata to one of the canonical stems."""
    
    if is_drum:
        return "drums"

    if inst_name is None:
        return None

    name = inst_name.lower()

    if "bass" in name:
        return "bass"
    if "guitar" in name:
        return "guitar"
    if "piano" in name or "keyboard" in name or "keys" in name:
        return "piano"
    if any(s in name for s in ["string", "violin", "cello", "viola"]):
        return "strings"
    if any(s in name for s in ["pad", "synth", "lead", "pluck"]):
        return "synth_pad"

    # Ignore brass, organ, fx, ambient, etc.
    return None


# =======================================
# MUSDB PROCESSOR
# =======================================

def process_musdb_track(track_dir, out_dir):
    track_name = os.path.basename(track_dir.rstrip("/"))
    print(f"[MUSDB] {track_name}")

    mix_path = os.path.join(track_dir, "mixture.wav")
    if not os.path.exists(mix_path):
        print(f"  Missing mixture.wav → skipping")
        return

    # Load and resample mixture
    y_mix_44k, sr_in = librosa.load(mix_path, sr=None, mono=True)
    y_mix = librosa.resample(y_mix_44k, orig_sr=sr_in, target_sr=SR)

    # Prepare target arrays
    targets = {name: np.zeros_like(y_mix) for name in STEM_NAMES}

    # MUSDB has only {vocals, bass, drums}. Others → stay zero.
    for stem_name in ["vocals", "bass", "drums"]:
        path = os.path.join(track_dir, f"{stem_name}.wav")
        if os.path.exists(path):
            y_s_44k, sr_s = librosa.load(path, sr=None, mono=True)
            y_s = librosa.resample(y_s_44k, orig_sr=sr_s, target_sr=SR)
            y_s = librosa.util.fix_length(data=y_s, size=len(y_mix))
            targets[stem_name] = y_s

    # Compute spectrograms
    mix_lms = compute_log_mel(y_mix)
    stems_lms = {stem: compute_log_mel(targets[stem]) for stem in STEM_NAMES}

    # Save .npz
    save_path = os.path.join(out_dir, f"musdb__{track_name}.npz")
    np.savez_compressed(save_path, mix=mix_lms, **stems_lms)

    print(f"  Saved → {save_path}")


# =======================================
# SLAKH PROCESSOR
# =======================================

def process_slakh_track(track_dir, out_dir):
    track_name = os.path.basename(track_dir.rstrip("/"))
    print(f"[SLAKH] {track_name}")

    mix_path = os.path.join(track_dir, "mix.wav")
    meta_path = os.path.join(track_dir, "metadata.yaml")
    stems_root = os.path.join(track_dir, "stems")

    if not (os.path.exists(mix_path) and os.path.exists(meta_path)):
        print("  Missing required files → skipping")
        return

    # Load mixture at SR=22050
    y_mix, _ = librosa.load(mix_path, sr=SR, mono=True)

    # Initialize target buffers
    targets = {name: np.zeros_like(y_mix) for name in STEM_NAMES}

    # Load metadata
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    for stem_id, info in meta.get("stems", {}).items():

        inst_name = (
            info.get("inst_class") or
            info.get("instrument") or
            info.get("plugin_name") or ""
        )
        is_drum = info.get("is_drum", False)

        category = slakh_instrument_to_stem(inst_name, is_drum)
        if category is None:
            continue

        wav_path = os.path.join(stems_root, f"{stem_id}.wav")
        if not os.path.exists(wav_path):
            continue

        y_s, _ = librosa.load(wav_path, sr=SR, mono=True)
        y_s = librosa.util.fix_length(data=y_s, size=len(y_mix))
        targets[category] += y_s

    # Compute spectrograms
    mix_lms = compute_log_mel(y_mix)
    stems_lms = {stem: compute_log_mel(targets[stem]) for stem in STEM_NAMES}

    # Save .npz
    save_path = os.path.join(out_dir, f"slakh__{track_name}.npz")
    np.savez_compressed(save_path, mix=mix_lms, **stems_lms)

    print(f"  Saved → {save_path}")


# =======================================
# MAIN
# =======================================

def main():

    print("\n=== Starting full preprocessing ===")

    os.makedirs(OUT_ROOT, exist_ok=True)

    # ---- Process MUSDB ----
    print("\n=== MUSDB ===")
    for song_dir in sorted(os.listdir(MUSDB_ROOT)):
        full_path = os.path.join(MUSDB_ROOT, song_dir)
        if os.path.isdir(full_path):
            process_musdb_track(full_path, OUT_ROOT)

    # ---- Process Slakh ----
    print("\n=== SLAKH ===")
    for track_dir in sorted(os.listdir(SLAKH_ROOT)):
        full_path = os.path.join(SLAKH_ROOT, track_dir)
        if os.path.isdir(full_path):
            process_slakh_track(full_path, OUT_ROOT)

    print("\n=== Finished preprocessing all data ===")


if __name__ == "__main__":
    main()
