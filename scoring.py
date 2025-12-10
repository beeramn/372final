import librosa
import numpy as np
import soundfile as sf
import os
import torch
from train_unet import Config

import matplotlib.pyplot as plt



# import inference helpers
from inference import load_model, separate, save_outputs, SR


# audio helpers
def load_audio(path, sr=22050):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio

def pad_to_match(a, b):
    m = min(len(a), len(b))
    return a[:m], b[:m]


# metrics
def mse(a, b):
    return np.mean((a - b) ** 2)

def mae(a, b):
    return np.mean(np.abs(a - b))

def mel_l1_distance(a, b, sr=22050, n_mels=128):
    S_a = librosa.feature.melspectrogram(y=a, sr=sr, n_mels=n_mels)
    S_b = librosa.feature.melspectrogram(y=b, sr=sr, n_mels=n_mels)

    S_a = librosa.power_to_db(S_a)
    S_b = librosa.power_to_db(S_b)

    t = min(S_a.shape[1], S_b.shape[1])
    return np.mean(np.abs(S_a[:, :t] - S_b[:, :t]))

def si_sdr(reference, estimation, eps=1e-8):
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)

    alpha = np.dot(estimation, reference) / (np.dot(reference, reference) + eps)
    projection = alpha * reference
    noise = estimation - projection

    ratio = (np.sum(projection**2) + eps) / (np.sum(noise**2) + eps)
    return 10 * np.log10(ratio + eps)



# evaluation
def evaluate_pair(pred_path, true_path):
    pred = load_audio(pred_path)
    true = load_audio(true_path)
    pred, true = pad_to_match(pred, true)

    return {
        "MSE": mse(pred, true),
        "MAE": mae(pred, true),
        "Mel_L1": mel_l1_distance(pred, true),
        "SI-SDR": si_sdr(true, pred),
    }


# run the inference and score for a single song
def evaluate_song(model, song_dir):

    mixture_path = os.path.join(song_dir, "mixture.wav")
    gt_vocals_path = os.path.join(song_dir, "vocals.wav")
    gt_inst_path = os.path.join(song_dir, "instrumental.wav")

    song_name = os.path.basename(song_dir)

    # output directory for predictions made by the model
    pred_dir = os.path.join("outputs", song_name)
    pred_vocals = os.path.join(pred_dir, "vocals.wav")
    pred_inst = os.path.join(pred_dir, "instrumental.wav")

    # check if the output is alr in the repository
    if os.path.exists(pred_vocals) and os.path.exists(pred_inst):
        print(f"\n=== Skipping inference for {song_name} (cached outputs found) ===")
    else:
        print(f"\n=== Running inference on: {song_name} ===")

        # Run inference
        vocal, inst = separate(model, mixture_path, device)

        # Save outputs in outputs/<song_name>/
        save_outputs(vocal, inst, mixture_path)

    # handle scoring
    vocal_scores = evaluate_pair(pred_vocals, gt_vocals_path)
    inst_scores = evaluate_pair(pred_inst, gt_inst_path)

    return vocal_scores, inst_scores


# plot scores 
def plot_scores(song_names, vocal_scores_list, inst_scores_list):
    """
    Makes a plot of per-song metrics.
    Each metric gets its own subplot, showing vocals + instrumental curves.
    """

    metrics = ["MSE", "MAE", "Mel_L1", "SI-SDR"]

    fig, axs = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    fig.suptitle("Per-Song Separation Metrics", fontsize=20, weight="bold")

    x = np.arange(len(song_names))

    for i, metric in enumerate(metrics):
        ax = axs[i]

        # get numbers for each metric
        vocal_vals = [scores[metric] for scores in vocal_scores_list]
        inst_vals  = [scores[metric] for scores in inst_scores_list]

        ax.plot(x, vocal_vals, marker='o', label="Vocals")
        ax.plot(x, inst_vals, marker='o', label="Instrumental")

        ax.set_ylabel(metric, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(song_names, rotation=45, ha='right', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# lets go through the entire directory
if __name__ == "__main__":

    musdb_root = "data/testingDB_2stem"

    # Pick device
    device = (
        # torch.device("cuda") if torch.cuda.is_available() else
        # torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu") #was timing out on my MPS, change if needed
    )
    print("Using device:", device)

    # load the model once
    model = load_model("checkpoints/unet_best.pt", device)

    # store for average metrics
    all_vocal_scores = []
    all_inst_scores = []
    song_names = []


    # loop through all song directories
    for song_name in os.listdir(musdb_root):
        song_dir = os.path.join(musdb_root, song_name)
        if not os.path.isdir(song_dir):
            continue  # skip files

        vocal_scores, inst_scores = evaluate_song(model, song_dir)
        song_names.append(song_name)


        all_vocal_scores.append(vocal_scores)
        all_inst_scores.append(inst_scores)

        print(f"Scores for {song_name}:")
        print("  Vocals:", vocal_scores)
        print("  Inst:  ", inst_scores)



    # calculate averages
    def average_dict(list_of_dicts):
        avg = {}
        keys = list_of_dicts[0].keys()
        for k in keys:
            avg[k] = np.mean([d[k] for d in list_of_dicts])
        return avg

    avg_vocals = average_dict(all_vocal_scores)
    avg_inst = average_dict(all_inst_scores)

    print("\n================ FINAL AVERAGE METRICS ================")
    print("Vocals Average:")
    for k, v in avg_vocals.items():
        print(f"  {k:10s}: {v:.4f}")

    print("\nInstrumental Average:")
    for k, v in avg_inst.items():
        print(f"  {k:10s}: {v:.4f}")

    print("========================================================\n")

    plot_scores(song_names, all_vocal_scores, all_inst_scores)

