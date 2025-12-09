import os
import torch
from torch.utils.data import Dataset
from processing.preprocessing import (
    load_audio, stft_mag_phase, normalize_spectrogram,
    compute_ratio_masks, apply_augmentations
)

# wraps MUSDB into a PyTorch Dataset with train-test split

class MUSDB2StemDataset(Dataset):
    """
    Dataset that loads mixture, vocals, and instrumentals for 2-stem separation.
    Produces normalized spectrograms + ratio masks.
    """

    def __init__(self, root_dir, split="train", 
                 segment_seconds=1.0,
                 augment=False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.segment_seconds = segment_seconds
        self.segment_length = int(segment_seconds * 44100)
        self.augment = augment

        # -------------------------
        # Build a list of track folders
        # -------------------------
        all_tracks = sorted(os.listdir(root_dir))
        N = len(all_tracks)

        # Documented splits:
        # 70% train, 15% val, 15% test
        train_end = int(0.7 * N)
        val_end = int(0.85 * N)

        if split == "train":
            self.tracks = all_tracks[:train_end]
        elif split == "val":
            self.tracks = all_tracks[train_end:val_end]
        else:
            self.tracks = all_tracks[val_end:]

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        path = os.path.join(self.root_dir, track)

        # -------- Load precomputed STFTs --------
        mix_mag = torch.load(os.path.join(path, "mix_mag.pt"))
        mix_phase = torch.load(os.path.join(path, "mix_phase.pt"))
        voc_mag = torch.load(os.path.join(path, "voc_mag.pt"))
        inst_mag = torch.load(os.path.join(path, "inst_mag.pt"))

        # -------- Normalize --------
        mix_mag_norm, mean, std = normalize_spectrogram(mix_mag)

        # -------- Compute ratio masks --------
        vocal_mask, inst_mask = compute_ratio_masks(voc_mag, inst_mag)

        return {
            "mix_mag": mix_mag_norm,
            "mean": mean,
            "std": std,
            "mix_phase": mix_phase,
            "vocal_mask": vocal_mask,
            "inst_mask": inst_mask
        }

    # def __getitem__(self, idx):
    #     track = self.tracks[idx]
    #     path = os.path.join(self.root_dir, track)

    #     # -------------------------
    #     # Load audio (C, T)
    #     # -------------------------
    #     mix = load_audio(os.path.join(path, "mixture.wav"))
    #     voc = load_audio(os.path.join(path, "vocals.wav"))
    #     inst = load_audio(os.path.join(path, "instrumental.wav"))

    #     # -------------------------
    #     # Augmentations
    #     # -------------------------
    #     if self.augment and self.split == "train":
    #         mix = apply_augmentations(mix)
    #         voc = apply_augmentations(voc)
    #         inst = apply_augmentations(inst)

    #     # -------------------------
    #     # Random segment selection
    #     # -------------------------
    #     if mix.shape[-1] > self.segment_length:
    #         start = torch.randint(0, mix.shape[-1] - self.segment_length, (1,))
    #         end = start + self.segment_length
    #         mix = mix[:, start:end]
    #         voc = voc[:, start:end]
    #         inst = inst[:, start:end]

    #     # -------------------------
    #     # STFT (C, F, T)
    #     # -------------------------
    #     mix_mag, mix_phase = stft_mag_phase(mix)
    #     voc_mag, _         = stft_mag_phase(voc)
    #     inst_mag, _        = stft_mag_phase(inst)

    #     # -------------------------
    #     # Normalize mixture magnitude
    #     # -------------------------
    #     mix_mag_norm, mean, std = normalize_spectrogram(mix_mag)

    #     # -------------------------
    #     # Compute ratio masks
    #     # -------------------------
    #     vocal_mask, inst_mask = compute_ratio_masks(voc_mag, inst_mag)

    #     # -------------------------
    #     # Force mono input & labels
    #     # -------------------------
    #     mix_mag_norm = mix_mag_norm.mean(dim=0, keepdim=True)   # (1, F, T)
    #     mix_phase    = mix_phase.mean(dim=0, keepdim=True)      # (1, F, T)

    #     vocal_mask = vocal_mask.mean(dim=0, keepdim=True)       # (1, F, T)
    #     inst_mask  = inst_mask.mean(dim=0, keepdim=True)        # (1, F, T)

    #     return {
    #         "mix_mag": mix_mag_norm, 
    #         "mean": mean,
    #         "std": std,
    #         "mix_phase": mix_phase,
    #         "vocal_mask": vocal_mask,
    #         "inst_mask": inst_mask
    #     }
