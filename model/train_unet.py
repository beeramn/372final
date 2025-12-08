import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from unet_model import UNet2D
from dataset_npz import NPZStemDataset, STEM_NAMES


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for mix, stems, _ in loader:
        mix = mix.to(device)
        stems = stems.to(device)

        optimizer.zero_grad()
        pred = model(mix)

        loss = torch.nn.functional.l1_loss(pred, stems)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for mix, stems, _ in loader:
            mix = mix.to(device)
            stems = stems.to(device)

            pred = model(mix)
            loss = torch.nn.functional.l1_loss(pred, stems)

            running_loss += loss.item()

    return running_loss / len(loader)


def main():
    data_dir = "../data/processed"
    batch_size = 4
    num_epochs = 20
    segment_frames = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Load full dataset
    # ----------------------------
    full_dataset = NPZStemDataset(
        root_dir=data_dir,
        segment_frames=segment_frames,
        stem_names=STEM_NAMES
    )

    # ----------------------------
    # Train/Validation split (90/10)
    # ----------------------------
    val_ratio = 0.1
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    # ----------------------------
    # Model + optimizer
    # ----------------------------
    model = UNet2D(
        in_channels=1,
        out_channels=len(STEM_NAMES),
        base_channels=32
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(1, num_epochs + 1):

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        print(f"Epoch {epoch}:  Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        # Save if val improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / f"best_unet_epoch_{epoch}.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )

            print(f"🔥 New best model saved: {ckpt_path}")


if __name__ == "__main__":
    main()
