import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


STEM_NAMES = ["vocals", "bass", "drums", "guitar", "piano", "strings", "synth_pad"]


class NPZStemDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        segment_frames: int | None = None,
        max_files: int | None = None,
        stem_names=None,
    ):
        """
        Args:
            root_dir: directory containing *.npz files
            segment_frames: if not None, randomly crop to this many frames along time
            max_files: limit how many files to use (for debugging)
            stem_names: list of stem keys in the npz (default: canonical 7)
        """
        self.root_dir = root_dir
        self.segment_frames = segment_frames
        self.stem_names = stem_names or STEM_NAMES

        pattern = os.path.join(root_dir, "*.npz")
        self.files = sorted(glob(pattern))
        if max_files is not None:
            self.files = self.files[:max_files]

        if not self.files:
            raise RuntimeError(f"No .npz files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def _random_crop(self, arr: np.ndarray) -> np.ndarray:
        """
        arr: (freq, time)
        """
        if self.segment_frames is None:
            return arr

        n_frames = arr.shape[1]
        if n_frames <= self.segment_frames:
            # If song is short, pad on right
            pad_width = self.segment_frames - n_frames
            if pad_width > 0:
                arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")
            return arr

        start = np.random.randint(0, n_frames - self.segment_frames + 1)
        end = start + self.segment_frames
        return arr[:, start:end]

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)

        mix = data["mix"]  # shape: (n_mels, T)

        # Stack stems: (num_stems, n_mels, T)
        stems = []
        for name in self.stem_names:
            stems.append(data[name])  # each (n_mels, T)
        stems = np.stack(stems, axis=0)

        # Optional cropping
        mix = self._random_crop(mix)
        stems = np.stack(
            [self._random_crop(stems[i]) for i in range(stems.shape[0])],
            axis=0,
        )

        # Add channel dim for mix: (1, n_mels, T)
        mix = mix[None, :, :]

        # Convert to tensors (float32)
        mix_tensor = torch.from_numpy(mix).float()
        stems_tensor = torch.from_numpy(stems).float()

        # (optional) simple normalization: scale dB to roughly [-1, 1]
        # assuming log-mels in ~[-80, 0] dB; adjust as needed
        mix_tensor = mix_tensor / 80.0
        stems_tensor = stems_tensor / 80.0

        return mix_tensor, stems_tensor, os.path.basename(path)
