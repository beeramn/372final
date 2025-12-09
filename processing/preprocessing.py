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
N_FFT = 2048
HOP = 512
WIN_LENGTH = 2048

# transforms for efficiency
STFT = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    hop_length=HOP,
    win_length=WIN_LENGTH,
    power=None,   # complex STFT
)

ISTFT = torchaudio.transforms.InverseSpectrogram(
    n_fft=N_FFT,
    hop_length=HOP,
    win_length=WIN_LENGTH,
)

# -----------------------------
# Loading audio
# -----------------------------
def load_audio(path, sr=SR):
    audio, file_sr = sf.read(path, always_2d=True)  # shape: (T, C)
    audio = torch.from_numpy(audio).float().transpose(0, 1)  # (C, T)

    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)

    return audio

# -----------------------------
# STFT → magnitude + phase
# -----------------------------
def stft_mag_phase(waveform):
    """
    Returns magnitude and phase tensors.
    waveform: (C, T)
    """
    spec = STFT(waveform)  # (C, freq, time)
    mag = spec.abs()
    phase = spec.angle()
    return mag, phase

# -----------------------------
# Normalization
# -----------------------------
def normalize_spectrogram(mag):
    """
    Normalize per-frequency-bin (channelwise)
    mag: (C, F, T)
    """
    mean = mag.mean(dim=-1, keepdim=True)
    std = mag.std(dim=-1, keepdim=True) + 1e-7
    mag_norm = (mag - mean) / std
    return mag_norm, mean, std

def denormalize(mag_norm, mean, std):
    return mag_norm * std + mean

# -----------------------------
# Mask computation
# -----------------------------
def compute_ratio_masks(vocal_mag, instr_mag):
    eps = 1e-8
    mixture_mag = vocal_mag + instr_mag + eps

    vocal_mask = vocal_mag / mixture_mag
    instr_mask = instr_mag / mixture_mag

    return vocal_mask, instr_mask

# -----------------------------
# Simple data augmentations
# -----------------------------
def apply_augmentations(waveform):
    """
    Example augmentations:
    - small pitch shift
    - time stretching
    - frequency masking
    """
    if random.random() < 0.3:
        # Pitch shift up/down up to 2 semitones
        semitone_shift = random.uniform(-2, 2)
        waveform = torchaudio.functional.pitch_shift(
            waveform, SR, semitone_shift
        )

    if random.random() < 0.3:
        # Frequency masking
        freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
        waveform = freq_mask(waveform)

    return waveform
