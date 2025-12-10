import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make sure repo root is on sys.path so imports work when run as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from processing.dataset import MUSDB2StemDataset
from models.unet import UNet

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# config for training
@dataclass
class Config:
    data_root: str = os.path.join("data", "musdb_2stem")
    batch_size: int = 1
    num_workers: int = 0
    num_epochs: int = 20

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5          # L2 regularization
    optimizer_name: str = "adamw"       # "adam", "adamw", "sgd"

    grad_clip: float = 1.0              # gradient clipping max norm
    use_amp: bool = True                # mixed precision

    patience: int = 7                   # early stopping patience (epochs)
    checkpoint_dir: str = "checkpoints"


# dataloaders
def get_dataloaders(cfg: Config):
    train_ds = MUSDB2StemDataset(
        root_dir=cfg.data_root,
        split="train",
        segment_seconds=2.0,
        augment=False
    )

    val_ds = MUSDB2StemDataset(
        root_dir=cfg.data_root,
        split="val",
        segment_seconds=2.0,
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False
    )

    return train_loader, val_loader


# optimizer and Scheduler
def get_optimizer(cfg: Config, model: nn.Module):
    if cfg.optimizer_name.lower() == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay  # L2 regularization
        )
    elif cfg.optimizer_name.lower() == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=cfg.weight_decay
        )
    else:  # default AdamW
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )

    # ReduceLROnPlateau scheduler (uses val loss)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=3
    )
    return opt, scheduler


# training and validation loops
def train_one_epoch(model, loader, optimizer, device, scaler=None, cfg: Config = None):

    model.train()
    total_loss = 0.0
    criterion = nn.L1Loss()  # mask regression

    for batch in loader:
        mix_mag = batch["mix_mag"].to(device)           # (B, C, F, T)
        vocal_mask = batch["vocal_mask"].to(device)     # (B, C, F, T)
        inst_mask = batch["inst_mask"].to(device)       # (B, C, F, T)

        # temp
        # print("mix_mag:", mix_mag.shape)
        # print("vocal_mask:", vocal_mask.shape)
        # print("inst_mask:", inst_mask.shape)
        # exit()

        # Convert stereo -> mono by averaging channels
        # mix_mag = mix_mag.mean(dim=1, keepdim=True) 
        # vocal_mask = vocal_mask.mean(dim=1, keepdim=True) 
        # inst_mask = inst_mask.mean(dim=1, keepdim=True) 

        target_masks = torch.stack([vocal_mask, inst_mask], dim=1)  # (B,2,2,F,T)

        optimizer.zero_grad()

        if cfg.use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                pred_masks = model(mix_mag)  # (B, 2, F, T)
                loss = criterion(pred_masks, target_masks)
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_masks = model(mix_mag)
            loss = criterion(pred_masks, target_masks)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        total_loss += loss.item() * mix_mag.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.L1Loss()

    for batch in loader:
        mix_mag = batch["mix_mag"].to(device)
        vocal_mask = batch["vocal_mask"].to(device)
        inst_mask = batch["inst_mask"].to(device)

        # mix_mag = mix_mag.mean(dim=1, keepdim=True)
        # vocal_mask = vocal_mask.mean(dim=1, keepdim=True)
        # inst_mask = inst_mask.mean(dim=1, keepdim=True)

        # target_masks = torch.cat([vocal_mask, inst_mask], dim=1)
        target_masks = torch.stack([vocal_mask, inst_mask], dim=1)

        pred_masks = model(mix_mag)
        loss = criterion(pred_masks, target_masks)

        total_loss += loss.item() * mix_mag.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


# main training driver
def main():
    cfg = Config()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # datasets and loaders
    train_loader, val_loader = get_dataloaders(cfg)

    # model
    model = UNet(base_channels=8).to(device)

    # optimizer and scheduler
    optimizer, scheduler = get_optimizer(cfg, model)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler() if (cfg.use_amp and device.type == "cuda") else None

    best_val_loss = float("inf")
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, cfg)
        val_loss = eval_one_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val   Loss: {val_loss:.6f}")

        # step the LR scheduler based on val loss
        scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # save best model
            best_path = os.path.join(cfg.checkpoint_dir, "unet_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    # remove config?
                    "config": cfg,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                },
                best_path,
            )
            print(f" New best model saved to {best_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s).")

        # Save last model each epoch
        last_path = os.path.join(cfg.checkpoint_dir, "unet_last.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            last_path,
        )

        # Early stopping check
        if epochs_no_improve >= cfg.patience:
            print(f" Early stopping triggered after {epoch} epochs.")
            break

    print(" Training complete.")


if __name__ == "__main__":
    main()