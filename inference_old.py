import torch
import torchaudio
import soundfile as sf
import os

from models.unet import UNet
from train_unet import Config

# ----------------------------
# STFT hyperparameters
# (must match preprocessing)
# ----------------------------
SR = 44100
N_FFT = 1024
HOP = 256
WIN = 1024


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model using SAME ARCHITECTURE used in training
    model = UNet(base_channels=8)

    # IMPORTANT: Load ONLY the stored model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()
    return model


# --------------------------------------------------
# Audio → STFT magnitude + phase
# --------------------------------------------------
def audio_to_mag(audio, device):
    """
    audio: (1, T) mono
    returns:
        mag: (1, F, T)
        phase: (1, F, T)
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
    )

    mag = stft.abs()
    phase = torch.angle(stft)
    return mag, phase


# --------------------------------------------------
# ISTFT reconstruction (PATCHED)
# --------------------------------------------------
def magphase_to_audio(mag, phase, original_length, device):
    """
    mag, phase: (1, F, T)
    """

    complex_stft = torch.polar(mag, phase)
    window = torch.hann_window(WIN, device=device)

    audio = torch.istft(
        complex_stft,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=window,
        center=True,
        length=original_length,      # <<< FIXED: enforces correct output length
    )
    return audio


# --------------------------------------------------
# Chunked inference for long spectrograms
# --------------------------------------------------
def run_chunked_inference(model, mag, chunk_size=256, overlap=64, device="cpu"):

    _, F, T = mag.shape
    step = chunk_size - overlap

    vocal_out = torch.zeros((1, F, T), device=device)
    inst_out  = torch.zeros((1, F, T), device=device)
    weight    = torch.zeros(T, device=device)

    for start in range(0, T, step):
        end = min(start + chunk_size, T)
        mag_chunk = mag[:, :, start:end].unsqueeze(0).to(device)  # (1,1,F,t)

        with torch.no_grad():
            masks = model(mag_chunk)  # (1, 2, F, t)
            v = masks[0, 0]
            i = masks[0, 1]

        vocal_out[:, :, start:end] += v
        inst_out[:, :, start:end]  += i
        weight[start:end] += 1

    weight = weight.clamp(min=1)
    vocal_out /= weight
    inst_out  /= weight

    return vocal_out, inst_out


# --------------------------------------------------
# Full separation pipeline (PATCHED)
# --------------------------------------------------
def separate(model, filepath, device):

    # Load audio
    audio, sr = sf.read(filepath, always_2d=True)          # (T, C)
    audio = torch.tensor(audio.T, dtype=torch.float32)     # (C, T)

    # resample to SR if needed (just like training)
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
        sr = SR

    # stereo → mono
    audio = audio.mean(dim=0, keepdim=True)  # (1, T)
    original_length = audio.shape[-1]

    # STFT
    mag, phase = audio_to_mag(audio, device=device)

    # Run model
    vocal_mask, inst_mask = run_chunked_inference(
        model,
        mag,
        chunk_size=256,
        overlap=64,
        device=device
    )

    # Apply masks
    vocal_mag = vocal_mask * mag
    inst_mag  = inst_mask * mag

    # ISTFT
    vocal_audio = magphase_to_audio(vocal_mag, phase, original_length, device=device).cpu().numpy()
    inst_audio  = magphase_to_audio(inst_mag,  phase, original_length, device=device).cpu().numpy()

    return vocal_audio, inst_audio


# --------------------------------------------------
# Save WAV output
# --------------------------------------------------
def save_outputs(vocal, inst, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, "vocals.wav"), vocal.T, SR)
    sf.write(os.path.join(out_dir, "instrumental.wav"), inst.T, SR)
    print(f"Saved separated stems in: {out_dir}/")


# --------------------------------------------------
# STFT → ISTFT sanity check (PATCHED)
# --------------------------------------------------
def stft_istft_sanity(filepath, device):
    print("Running STFT → iSTFT sanity check...")

    audio, sr = sf.read(filepath, always_2d=True)
    audio = torch.tensor(audio.T, dtype=torch.float32)  # (C, T)

    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
        sr = SR

    # mono for sanity check
    audio = audio.mean(dim=0, keepdim=True)  # (1, T)
    original_length = audio.shape[-1]

    mag, phase = audio_to_mag(audio, device=device)
    recon = magphase_to_audio(mag, phase, original_length, device=device).cpu().numpy()

    os.makedirs("sanity", exist_ok=True)
    sf.write("sanity/reconstructed.wav", recon.T, SR)

    print("Saved sanity STFT→iSTFT reconstruction to sanity/reconstructed.wav")


# --------------------------------------------------
# Entry point (currently sanity check mode)
# --------------------------------------------------
if __name__ == "__main__":
    # Choose device
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    print("Using device:", device)

    # Load model
    checkpoint_path = "checkpoints/unet_best.pt"
    model = load_model(checkpoint_path, device)

    # Input song
    input_song = "WASTE_BROCKHAMPTON.wav" 
    # input_song = "data/musdb/Music Delta - Britpop/mixture.wav" 
    print(f"Separating: {input_song}")

    # Run separation
    vocal, inst = separate(model, input_song, device)

    # Save
    save_outputs(vocal, inst, out_dir="outputs")

    print("Done! Separated stems saved in 'outputs/'.")

# sanity check
# if __name__ == "__main__":
#     device = (
#         torch.device("cuda") if torch.cuda.is_available() else
#         torch.device("mps") if torch.backends.mps.is_available() else
#         torch.device("cpu")
#     )

#     print("Using device:", device)

#     test_file = "data/musdb/Music Delta - Britpop/mixture.wav"
#     stft_istft_sanity(test_file, device)
