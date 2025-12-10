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

        # making the list of the track folders
        # 
        all_tracks = sorted(os.listdir(root_dir))
        N = len(all_tracks)

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

        #  load STFTs 
        mix_mag = torch.load(os.path.join(path, "mix_mag.pt"))
        mix_phase = torch.load(os.path.join(path, "mix_phase.pt"))
        voc_mag = torch.load(os.path.join(path, "voc_mag.pt"))
        inst_mag = torch.load(os.path.join(path, "inst_mag.pt"))

        #  normalize 
        mix_mag_norm, mean, std = normalize_spectrogram(mix_mag)

        #  compute ratio masks 
        vocal_mask, inst_mask = compute_ratio_masks(voc_mag, inst_mag)

        return {
            "mix_mag": mix_mag_norm,
            "mean": mean,
            "std": std,
            "mix_phase": mix_phase,
            "vocal_mask": vocal_mask,
            "inst_mask": inst_mask
        }

