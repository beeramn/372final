import torch
import torchaudio
import soundfile as sf
import os
import numpy as np

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
    """
    audio: (2, T) stereo
    returns:
        mag:   (2, F, T)
        phase: (2, F, T)
    """

    window = torch.hann_window(WIN, device=device)

    stft = torch.stft(
        audio.to(device),
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=window,
        return_complex=True,
        center=True,
    )  # (2, F, T)

    mag = stft.abs()
    phase = torch.angle(stft)
    return mag, phase


# --------------------------------------------------
# ISTFT reconstruction
# --------------------------------------------------
def complex_to_audio(complex_stft, device):
    """
    complex_stft: (2, F, T) complex tensor
    """
    window = torch.hann_window(WIN, device=device)
    audio = torch.istft(
        complex_stft,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=window,
        center=True,
    )
    return audio  # (2, T)


# --------------------------------------------------
# Wiener filtering refinement
# --------------------------------------------------
def wiener_filter(V, I, mix_complex, eps=1e-10):
    """
    V, I: magnitude estimates: (2, F, T)
    mix_complex: (2, F, T)
    """

    V2 = V ** 2
    I2 = I ** 2
    denom = V2 + I2 + eps

    Vc = (V2 / denom) * mix_complex
    Ic = (I2 / denom) * mix_complex

    return Vc, Ic


# --------------------------------------------------
# Chunked inference for long spectrograms
# --------------------------------------------------
def run_chunked_inference(model, mag, chunk_size=256, overlap=64, device="cpu"):
    """
    mag: (2, F, T)
    Returns:
        vocal_mask: (2, F, T)
        inst_mask:  (2, F, T)
    """

    C, F, T = mag.shape
    step = chunk_size - overlap

    vocal_out = torch.zeros((C, F, T), device=device)
    inst_out = torch.zeros((C, F, T), device=device)
    weight = torch.zeros((T,), device=device)

    for start in range(0, T, step):
        end = min(start + chunk_size, T)

        # shape: (1, 2, F, t)
        mag_chunk = mag[:, :, start:end].unsqueeze(0).to(device)

        with torch.no_grad():
            # Model output: (1, 2, 2, F, t)
            masks = model(mag_chunk)

            vocal_mask = masks[0, 0]   # (2, F, t)
            inst_mask = masks[0, 1]    # (2, F, t)

        vocal_out[:, :, start:end] += vocal_mask
        inst_out[:, :, start:end] += inst_mask
        weight[start:end] += 1

    weight = weight.clamp(min=1)
    vocal_out /= weight
    inst_out /= weight

    return vocal_out, inst_out


# --------------------------------------------------
# Full stereo separation with Wiener filtering
# --------------------------------------------------
def separate(model, filepath, device):

    # Load stereo audio
    audio, sr = sf.read(filepath, always_2d=True)  # shape (T,2)
    audio = torch.tensor(audio.T, dtype=torch.float32)  # (2, T)

    # Resample if needed
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)

    # STFT
    mag, phase = audio_to_mag_phase(audio, device)
    mix_complex = torch.polar(mag, phase)  # (2, F, T)

    # Predict masks
    vocal_mask, inst_mask = run_chunked_inference(
        model, mag, chunk_size=256, overlap=64, device=device
    )

    # Initial magnitude estimates
    V = vocal_mask * mag
    I = inst_mask * mag

    # Wiener filtering
    Vc, Ic = wiener_filter(V, I, mix_complex)

    # Reconstruct waveforms
    vocal_audio = complex_to_audio(Vc, device).cpu().numpy().T  # shape (T,2)
    inst_audio  = complex_to_audio(Ic, device).cpu().numpy().T

    return vocal_audio, inst_audio


# --------------------------------------------------
# Save WAV output
# --------------------------------------------------
def save_outputs(vocal, inst, filepath):
    """
    Save stems inside a folder named after the second-to-last directory
    in the input file path.
    Example:
        filepath = "data/track01/mixture.wav"
        -> output folder = "track01"
    """

    # Extract parent directory name
    parent_dir = os.path.basename(os.path.dirname(filepath))

    # Create output folder
    out_dir = os.path.join("outputs", parent_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Write files
    sf.write(os.path.join(out_dir, "vocals.wav"), vocal, SR)
    sf.write(os.path.join(out_dir, "instrumental.wav"), inst, SR)

    print(f"Saved separated stems in: {out_dir}/")


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    print("Using device:", device)

    checkpoint_path = "../checkpoints/unet_best.pt"
    model = load_model(checkpoint_path, device)

    
    input_song = "data/musdb/track/mixture.wav"
    # input_song = "WASTE_BROCKHAMPTON.wav"
    vocal, inst = separate(model, input_song, device)

    save_outputs(vocal, inst, input_song)
