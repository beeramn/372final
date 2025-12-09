import os
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path

from unet_model import UNet2D
from dataset_npz import STEM_NAMES

# --------- MATCH TRAINING PREPROCESSING ---------
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 0
FMAX = SR // 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_log_mel(y, sr=SR):
    """Compute log-mel spectrogram matching training preprocessing"""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def stft_to_waveform(magnitude, iterations=32):
    """
    Reconstruct waveform from magnitude spectrogram using Griffin-Lim
    with phase initialization from mixture.
    """
    # Use mixture phase as initialization
    phase = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    
    for i in range(iterations):
        # Reconstruct complex STFT
        stft = magnitude * phase
        
        # Inverse STFT
        audio = librosa.istft(stft, hop_length=HOP_LENGTH, win_length=N_FFT)
        
        # Forward STFT to get updated phase
        if i < iterations - 1:
            stft_recon = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
            phase = np.exp(1j * np.angle(stft_recon))
    
    return audio


def mel_to_waveform(log_mel, mixture_audio=None):
    """
    Convert log-mel spectrogram back to waveform.
    If mixture_audio is provided, use its phase for better reconstruction.
    """
    # Convert back to linear scale
    mel_linear = librosa.db_to_power(log_mel)
    
    # Convert mel spectrogram back to STFT magnitude
    stft_mag = librosa.feature.inverse.mel_to_stft(
        mel_linear,
        sr=SR,
        n_fft=N_FFT,
        fmin=FMIN,
        fmax=FMAX,
    )
    
    # Use Griffin-Lim for phase reconstruction
    if mixture_audio is not None:
        # Get mixture STFT phase
        mixture_stft = librosa.stft(mixture_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mixture_phase = np.angle(mixture_stft)
        
        # Ensure shapes match
        min_freq = min(stft_mag.shape[0], mixture_phase.shape[0])
        min_time = min(stft_mag.shape[1], mixture_phase.shape[1])
        
        stft_mag = stft_mag[:min_freq, :min_time]
        mixture_phase = mixture_phase[:min_freq, :min_time]
        
        # Combine magnitude with mixture phase
        stft_complex = stft_mag * np.exp(1j * mixture_phase)
        audio = librosa.istft(stft_complex, hop_length=HOP_LENGTH, win_length=N_FFT)
    else:
        # Fallback to Griffin-Lim
        audio = librosa.griffinlim(
            stft_mag,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            n_iter=32
        )
    
    return audio


def process_long_audio(model, log_mel, chunk_size=512, overlap=128):
    """
    Process long audio in chunks with overlap-add to handle memory constraints.
    """
    n_mels, T = log_mel.shape
    pred_stems = []
    
    # Initialize output arrays for each stem
    for _ in range(len(STEM_NAMES)):
        pred_stems.append(np.zeros((n_mels, T)))
    
    # Create weighting array for overlap-add
    weight = np.ones(chunk_size)
    
    # Create window for overlap regions (Hanning window)
    # For the left overlap region (first 'overlap' samples)
    left_window = np.hanning(2 * overlap + 1)[:overlap]
    weight[:overlap] = left_window
    
    # For the right overlap region (last 'overlap' samples)
    right_window = np.hanning(2 * overlap + 1)[overlap + 1:]
    weight[-overlap:] = right_window
    
    # Calculate chunk indices with overlap
    step = chunk_size - overlap
    starts = list(range(0, T - chunk_size + 1, step))
    
    # Process last chunk if needed
    if T - starts[-1] > chunk_size:
        starts.append(T - chunk_size)
    
    print(f"Processing {len(starts)} chunks with step={step}, overlap={overlap}")
    
    for idx, start in enumerate(starts):
        end = start + chunk_size
        
        # Extract chunk
        chunk = log_mel[:, start:end]
        
        # Prepare input
        inp = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            pred = model(inp)[0].cpu().numpy()  # (7, n_mels, chunk_size)
        
        # Add to output with overlap weighting
        for i in range(len(STEM_NAMES)):
            pred_stems[i][:, start:end] += pred[i] * weight
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed chunk {idx + 1}/{len(starts)}")
    
    return pred_stems


def process_audio_simple(model, log_mel_norm):
    """
    Simple processing for any length audio by processing the entire spectrogram at once.
    This works if you have enough GPU memory.
    """
    n_mels, T = log_mel_norm.shape
    
    # Pad to make it divisible by 16 (for U-Net pooling)
    # U-Net with 4 pooling layers divides by 16
    if T % 16 != 0:
        pad_right = 16 - (T % 16)
        log_mel_padded = np.pad(log_mel_norm, ((0, 0), (0, pad_right)), mode='constant')
    else:
        log_mel_padded = log_mel_norm
        pad_right = 0
    
    # Prepare input
    inp = torch.from_numpy(log_mel_padded).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        pred = model(inp)[0].cpu().numpy()  # (7, n_mels, T_padded)
    
    # Remove padding if we added it
    if pad_right > 0:
        pred = pred[:, :, :-pad_right]
    
    return [pred[i] for i in range(len(STEM_NAMES))]


def main():
    # ----- Paths -----
    checkpoint_path = "best_model.pt"
    input_audio_path = "../WASTE_BROCKHAMPTON.wav"
    output_dir = "separated_stems"
    os.makedirs(output_dir, exist_ok=True)
    
    # ----- Load model -----
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Get normalization stats from checkpoint
    mean = checkpoint.get('mean', 0.0)
    std = checkpoint.get('std', 1.0)
    
    print(f"Using normalization: mean={mean:.4f}, std={std:.4f}")
    
    # Initialize model
    model = UNet2D(
        in_channels=1,
        out_channels=len(STEM_NAMES),
        base_channels=16  # Should match training
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    
    # ----- Load mixture audio -----
    print(f"Loading audio: {input_audio_path}")
    y, sr = librosa.load(input_audio_path, sr=SR)
    
    # Compute log-mel spectrogram
    print("Computing spectrogram...")
    log_mel = compute_log_mel(y, SR)  # shape: (128, T)
    
    # Normalize using dataset statistics
    log_mel_norm = (log_mel - mean) / (std + 1e-8)
    
    T = log_mel.shape[1]
    print(f"Audio length: {len(y)/SR:.2f}s, Spectrogram shape: {log_mel.shape}")
    
    # ----- Process audio -----
    print("Separating stems...")
    
    # Try simple processing first (if GPU memory allows)
    # For longer audio, we'll use chunked processing
    if T <= 1000:  # Roughly 23 seconds at hop_length=512
        print("Processing entire audio at once...")
        pred_stems = process_audio_simple(model, log_mel_norm)
    else:
        print(f"Audio is long ({T} frames), processing in chunks...")
        # Use smaller chunk size if memory is limited
        pred_stems = process_long_audio(model, log_mel_norm, chunk_size=256, overlap=64)
    
    # Unnormalize
    pred_stems = [stem * std + mean for stem in pred_stems]
    
    # ----- Convert stems to audio -----
    print("\nReconstructing waveforms...")
    
    # Save mixture for reference
    mixture_path = os.path.join(output_dir, "00_mixture.wav")
    sf.write(mixture_path, y, SR)
    print(f"Saved mixture: {mixture_path}")
    
    # Save each stem
    for i, stem_name in enumerate(STEM_NAMES):
        print(f"  Processing {stem_name}...")
        
        # Convert log-mel to waveform
        audio = mel_to_waveform(pred_stems[i], mixture_audio=y)
        
        # Normalize audio to prevent clipping
        if len(audio) > 0:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9
        else:
            print(f"Warning: {stem_name} audio is empty")
            audio = np.zeros_like(y)
        
        # Trim to match original length
        min_len = min(len(audio), len(y))
        audio = audio[:min_len]
        
        # Save
        out_path = os.path.join(output_dir, f"{i+1:02d}_{stem_name}.wav")
        sf.write(out_path, audio, SR)
        
        print(f"    Saved: {out_path} (length: {len(audio)/SR:.2f}s)")
    
    # ----- Create a composite to check quality -----
    print("\nCreating composite for verification...")
    
    # Sum all stems (should approximate mixture)
    all_stems_sum = np.zeros_like(y)
    
    for i, stem_name in enumerate(STEM_NAMES):
        stem_path = os.path.join(output_dir, f"{i+1:02d}_{stem_name}.wav")
        stem_audio, _ = librosa.load(stem_path, sr=SR)
        
        # Ensure same length
        min_len = min(len(all_stems_sum), len(stem_audio))
        all_stems_sum[:min_len] += stem_audio[:min_len]
    
    # Normalize
    if np.max(np.abs(all_stems_sum)) > 0:
        all_stems_sum = all_stems_sum / np.max(np.abs(all_stems_sum)) * 0.9
    
    # Save composite
    composite_path = os.path.join(output_dir, "99_composite_sum.wav")
    sf.write(composite_path, all_stems_sum, SR)
    
    print(f"Saved composite sum: {composite_path}")
    print("\nSeparation complete!")
    
    # ----- Calculate some metrics -----
    print("\nSeparation Quality Metrics:")
    print("-" * 40)
    
    # Load mixture for comparison
    mixture_audio = y
    
    for i, stem_name in enumerate(STEM_NAMES):
        stem_path = os.path.join(output_dir, f"{i+1:02d}_{stem_name}.wav")
        stem_audio, _ = librosa.load(stem_path, sr=SR)
        
        # Trim to common length
        min_len = min(len(mixture_audio), len(stem_audio))
        
        if min_len > 0:
            # Energy ratio
            stem_energy = np.sum(stem_audio[:min_len] ** 2)
            mix_energy = np.sum(mixture_audio[:min_len] ** 2)
            energy_ratio = stem_energy / (mix_energy + 1e-10)
            
            # Correlation with mixture
            correlation = np.corrcoef(mixture_audio[:min_len], stem_audio[:min_len])[0, 1]
            
            print(f"{stem_name:12s}: Energy ratio = {energy_ratio:.4f}, "
                  f"Correlation = {correlation:.4f}")
        else:
            print(f"{stem_name:12s}: No audio data")


if __name__ == "__main__":
    main()