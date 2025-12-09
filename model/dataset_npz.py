import os
import random
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


STEM_NAMES = ["vocals", "bass", "drums", "guitar", "piano", "strings", "synth_pad"]


class FixedNPZStemDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        segment_frames: int | None = None,
        max_files: int | None = None,
        stem_names=None,
        augment: bool = False,
        normalize: bool = True,
    ):
        self.root_dir = root_dir
        self.segment_frames = segment_frames
        self.stem_names = stem_names or STEM_NAMES
        self.augment = augment
        
        pattern = os.path.join(root_dir, "*.npz")
        self.files = sorted(glob(pattern))
        if max_files is not None:
            self.files = self.files[:max_files]
        
        if not self.files:
            raise RuntimeError(f"No .npz files found in {root_dir}")
        
        # Compute dataset statistics for normalization
        self.mean, self.std = self._compute_stats()
        print(f"Dataset stats: mean={self.mean:.4f}, std={self.std:.4f}")
        
        self.normalize = normalize
    
    def _compute_stats(self, num_samples=20):
        """Compute mean and std from a subset of the dataset"""
        all_values = []
        for i in range(min(num_samples, len(self.files))):
            path = self.files[i]
            data = np.load(path)
            
            mix = data["mix"].flatten()
            all_values.append(mix)
            
            for name in self.stem_names:
                stem = data[name].flatten()
                all_values.append(stem)
        
        all_values = np.concatenate(all_values)
        return np.mean(all_values), np.std(all_values)
    
    def _apply_augmentation(self, mix, stems):
        """Apply synchronized augmentation to mix and stems"""
        if not self.augment:
            return mix, stems
        
        # Frequency masking
        if random.random() < 0.3:
            freq_mask_width = random.randint(5, 15)
            freq_start = random.randint(0, mix.shape[0] - freq_mask_width)
            mix[freq_start:freq_start+freq_mask_width, :] = 0
            stems[:, freq_start:freq_start+freq_mask_width, :] = 0
        
        # Time masking
        if random.random() < 0.3:
            time_mask_width = random.randint(10, 25)
            time_start = random.randint(0, mix.shape[1] - time_mask_width)
            mix[:, time_start:time_start+time_mask_width] = 0
            stems[:, :, time_start:time_start+time_mask_width] = 0
        
        return mix, stems
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        
        mix = data["mix"]  # (n_mels, T)
        
        # Load stems
        stems = []
        for name in self.stem_names:
            stems.append(data[name])
        stems = np.stack(stems, axis=0)  # (7, n_mels, T)
        
        T = mix.shape[1]
        
        # Apply synchronized cropping
        if self.segment_frames and T > self.segment_frames:
            start = random.randint(0, T - self.segment_frames)
            end = start + self.segment_frames
            mix = mix[:, start:end]
            stems = stems[:, :, start:end]
        elif self.segment_frames and T < self.segment_frames:
            # Pad with zeros
            pad_width = self.segment_frames - T
            mix = np.pad(mix, ((0, 0), (0, pad_width)), mode="constant")
            stems = np.pad(stems, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
        
        # Apply augmentation
        mix, stems = self._apply_augmentation(mix, stems)
        
        # Add channel dimension to mix
        mix = mix[None, :, :]  # (1, n_mels, T)
        
        # Convert to tensors
        mix_tensor = torch.from_numpy(mix).float()
        stems_tensor = torch.from_numpy(stems).float()
        
        # Normalize
        if self.normalize:
            mix_tensor = (mix_tensor - self.mean) / (self.std + 1e-8)
            stems_tensor = (stems_tensor - self.mean) / (self.std + 1e-8)
        
        return mix_tensor, stems_tensor, os.path.basename(path)