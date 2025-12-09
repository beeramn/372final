import torch
import torchaudio
import soundfile as sf
import os
import numpy as np
import argparse

from models.unet import UNet
from train_unet import Config


# ----------------------------
# STFT hyperparameters
# ----------------------------
SR = 44100
N_FFT = 512
HOP = 128
WIN = 512


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # IMPORTANT: Stereo UNet (input 2ch, output 4ch)
    model = UNet(base_channels=8)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------
# Audio → STFT magnitude + phase
# --------------------------------------------------
def audio_to_mag_phase(audio, device):
    window = torch.hann_window(WIN, device=device)

    stft = torch.stft(
        audio.to(device),
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=window,
        return_complex=True,
        center=True,
    )

    mag = stft.abs()
    phase = torch.angle(stft)
    return mag, phase


# --------------------------------------------------
# ISTFT reconstruction
# --------------------------------------------------
def complex_to_audio(complex_stft, device):
    window = torch.hann_window(WIN, device=device)
    audio = torch.istft(
        complex_stft,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=window,
        center=True,
    )
    return audio


# --------------------------------------------------
# Wiener filtering
# --------------------------------------------------
def wiener_filter(V, I, mix_complex, eps=1e-10):
    V2 = V ** 2
    I2 = I ** 2
    denom = V2 + I2 + eps

    Vc = (V2 / denom) * mix_complex
    Ic = (I2 / denom) * mix_complex
    return Vc, Ic


# --------------------------------------------------
# Chunked inference
# --------------------------------------------------
def run_chunked_inference(model, mag, chunk_size=256, overlap=64, device="cpu"):
    C, F, T = mag.shape
    step = chunk_size - overlap

    vocal_out = torch.zeros((C, F, T), device=device)
    inst_out = torch.zeros((C, F, T), device=device)
    weight = torch.zeros((T,), device=device)

    for start in range(0, T, step):
        end = min(start + chunk_size, T)
        mag_chunk = mag[:, :, start:end].unsqueeze(0).to(device)

        with torch.no_grad():
            masks = model(mag_chunk)
            vocal_mask = masks[0, 0]
            inst_mask = masks[0, 1]

        vocal_out[:, :, start:end] += vocal_mask
        inst_out[:, :, start:end] += inst_mask
        weight[start:end] += 1

    weight = weight.clamp(min=1)
    vocal_out /= weight
    inst_out /= weight
    return vocal_out, inst_out


# --------------------------------------------------
# Full separation
# --------------------------------------------------
def separate(model, filepath, device):
    audio, sr = sf.read(filepath, always_2d=True)
    audio = torch.tensor(audio.T, dtype=torch.float32)

    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)

    mag, phase = audio_to_mag_phase(audio, device)
    mix_complex = torch.polar(mag, phase)

    vocal_mask, inst_mask = run_chunked_inference(model, mag, device=device)
    V = vocal_mask * mag
    I = inst_mask * mag

    Vc, Ic = wiener_filter(V, I, mix_complex)

    vocal_audio = complex_to_audio(Vc, device).cpu().numpy().T
    inst_audio = complex_to_audio(Ic, device).cpu().numpy().T

    return vocal_audio, inst_audio


# --------------------------------------------------
# Save outputs
# --------------------------------------------------
def save_outputs(vocal, inst, filepath):
    parent_dir = os.path.basename(os.path.dirname(filepath))
    out_dir = os.path.join("outputs", parent_dir)
    os.makedirs(out_dir, exist_ok=True)

    sf.write(os.path.join(out_dir, "vocals.wav"), vocal, SR)
    sf.write(os.path.join(out_dir, "instrumental.wav"), inst, SR)
    print(f"Saved separated stems in: {out_dir}/")


# --------------------------------------------------
# Command-line interface
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Stereo music source separation (vocal/instrumental)")
    parser.add_argument("filepath", type=str, help="Path to input audio file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/unet_best.pt",
                        help="Path to trained UNet checkpoint")
    return parser.parse_args()


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    print("Using device:", device)
    print("Loading model:", args.checkpoint)
    print("Input audio:", args.filepath)

    model = load_model(args.checkpoint, device)
    vocal, inst = separate(model, args.filepath, device)
    save_outputs(vocal, inst, args.filepath)
