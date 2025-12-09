import torch
import torchaudio
import soundfile as sf
import numpy as np
import random

# note: this file contains reusable functions for: 
# loading audio
# computing STFT magnitude + phase
# normalizing spectrograms
# computing ratio masks
# data augmentations

# hyperparameters for spectrogram
SR = 44100
N_FFT = 512
HOP = 128
WIN_LENGTH = 512

# -----------------------------
# Loading audio
# -----------------------------
def load_audio(path, sr=SR):
    """
    Load audio from disk as (C, T) float32 tensor at the target sample rate.
    Uses soundfile for I/O and torchaudio for resampling.
    """
    audio, file_sr = sf.read(path, always_2d=True)  # shape: (T, C)
    audio = torch.from_numpy(audio).float().transpose(0, 1)  # (C, T)

    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)

    return audio


# -----------------------------
# STFT → magnitude + phase
# -----------------------------
def stft_mag_phase(waveform: torch.Tensor):
    """
    Compute complex STFT and return magnitude and phase.
    waveform: (C, T)
    Returns:
        mag:   (C, F, T)
        phase: (C, F, T)
    """
    # Make sure window is on the same device as the input (CPU/GPU/MPS safe)
    window = torch.hann_window(WIN_LENGTH, device=waveform.device)

    spec = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
        center=True,
    )  # (C, F, T) complex

    mag = spec.abs()
    phase = torch.angle(spec)
    return mag, phase


# -----------------------------
# Normalization
# -----------------------------
def normalize_spectrogram(mag: torch.Tensor):
    """
    Normalize per-frequency-bin (channelwise).
    mag: (C, F, T)
    Returns:
        mag_norm: (C, F, T)
        mean:     (C, F, 1)
        std:      (C, F, 1)
    """
    mean = mag.mean(dim=-1, keepdim=True)
    std = mag.std(dim=-1, keepdim=True) + 1e-7
    mag_norm = (mag - mean) / std
    return mag_norm, mean, std


def denormalize(mag_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return mag_norm * std + mean


# -----------------------------
# Mask computation
# -----------------------------
def compute_ratio_masks(voc_mag: torch.Tensor, inst_mag: torch.Tensor, eps: float = 1e-8):
    """
    Compute soft ratio masks for vocals and instrumentals.
    voc_mag, inst_mag: (C, F, T)
    Returns:
        vocal_mask, inst_mask both in [0, 1], shape (C, F, T)
    """
    mix_mag = voc_mag + inst_mag  # mixture magnitude approximation

    denom = mix_mag + eps
    vocal_mask = voc_mag / denom
    inst_mask = inst_mag / denom

    # numerical safety: clamp to [0,1]
    vocal_mask = vocal_mask.clamp(0.0, 1.0)
    inst_mask = inst_mask.clamp(0.0, 1.0)

    return vocal_mask, inst_mask


# -----------------------------
# Simple data augmentations
# -----------------------------
def apply_augmentations(waveform: torch.Tensor):
    """
    Example augmentations:
    - small pitch shift
    - frequency masking

    waveform: (C, T)
    """
    # Pitch shift up/down up to 2 semitones
    if random.random() < 0.3:
        semitone_shift = random.uniform(-2, 2)
        waveform = torchaudio.functional.pitch_shift(
            waveform, SR, semitone_shift
        )

    # Frequency masking on the spectrogram-like representation
    if random.random() < 0.3:
        freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
        waveform = freq_mask(waveform)

    return waveform
