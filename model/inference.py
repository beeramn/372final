import os
import numpy as np
import torch
import librosa
import soundfile as sf

from unet_model import UNet2D
from dataset_npz import STEM_NAMES

# --------- MATCH TRAINING PREPROCESSING ---------
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 0
FMAX = SR // 2


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


def invert_mel_to_audio(log_mel):
    """
    Invert log-mel spectrogram back to waveform using mel→stft→Griffin-Lim.
    """
    mel_power = librosa.db_to_power(log_mel)  # undo librosa.power_to_db

    # mel → linear-frequency STFT magnitude
    S = librosa.feature.inverse.mel_to_stft(
        M=mel_power,
        sr=SR,
        n_fft=N_FFT,
        fmin=FMIN, fmax=FMAX,
    )

    # Griffin–Lim to approximate phase
    audio = librosa.griffinlim(
        S,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        n_iter=32
    )

    return audio


def main():
    # ----- Paths -----
    checkpoint_path = "checkpoints/best_unet_epoch_20.pt"
    input_audio_path = "../WASTE_BROCKHAMPTON.wav"
    output_dir = "separated_stems"
    os.makedirs(output_dir, exist_ok=True)

    # ----- Load model -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(in_channels=1, out_channels=len(STEM_NAMES))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ----- Load mixture audio -----
    y, sr = librosa.load(input_audio_path, sr=SR)
    log_mel = compute_log_mel(y, SR)        # shape: (128, T)
    log_mel_norm = log_mel / 80.0           # match dataset normalization

    # Model input: (1, 1, 128, T)
    inp = torch.from_numpy(log_mel_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp)[0].cpu().numpy()   # (7, 128, T)

    # Undo normalization
    pred = pred * 80.0

    # ----- Convert each predicted log-mel → audio -----
    for i, stem_name in enumerate(STEM_NAMES):
        stem_log_mel = pred[i]
        audio = invert_mel_to_audio(stem_log_mel)

        out_path = os.path.join(output_dir, f"{stem_name}.wav")
        sf.write(out_path, audio, SR)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
