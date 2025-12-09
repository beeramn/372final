import os
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from unet_model import UNet2D
from dataset_npz import FixedNPZStemDataset, STEM_NAMES


def split_by_track(dataset, val_ratio=0.1):
    """Split dataset by track ID to avoid data leakage"""
    # Group indices by track (filename without extension)
    track_to_indices = {}
    for idx in range(len(dataset)):
        _, _, filename = dataset[idx]
        track = os.path.splitext(filename)[0]
        if track not in track_to_indices:
            track_to_indices[track] = []
        track_to_indices[track].append(idx)
    
    tracks = list(track_to_indices.keys())
    random.shuffle(tracks)
    
    val_size = int(len(tracks) * val_ratio)
    val_tracks = set(tracks[:val_size])
    
    train_indices = []
    val_indices = []
    
    for track, indices in track_to_indices.items():
        if track in val_tracks:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def spectral_convergence_loss(pred, target):
    """Better loss for audio reconstruction"""
    pred_mag = torch.abs(pred)
    target_mag = torch.abs(target)
    
    # L1 loss
    l1_loss = F.l1_loss(pred_mag, target_mag)
    
    # Spectral convergence
    sc_loss = torch.norm(target_mag - pred_mag, 'fro') / torch.norm(target_mag, 'fro')
    
    # Weighted combination
    return 0.7 * l1_loss + 0.3 * sc_loss


def main():
    data_dir = "../data/processed"
    batch_size = 4
    num_epochs = 100  # More epochs needed for small dataset
    segment_frames = 512
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset with fixes
    full_dataset = FixedNPZStemDataset(
        root_dir=data_dir,
        segment_frames=segment_frames,
        stem_names=STEM_NAMES,
        augment=True,  # Enable augmentation
        normalize=True
    )
    
    print(f"Total tracks: {len(full_dataset)}")
    print(f"Dataset mean: {full_dataset.mean:.4f}, std: {full_dataset.std:.4f}")
    
    # Split by track (not by random segments)
    train_dataset, val_dataset = split_by_track(full_dataset, val_ratio=0.1)
    
    print(f"Training tracks: {len(train_dataset)}")
    print(f"Validation tracks: {len(val_dataset)}")
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    # Smaller model for small dataset
    model = UNet2D(
        in_channels=1,
        out_channels=len(STEM_NAMES),
        base_channels=16  # Reduced from 32
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = float("inf")
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for mix, stems, _ in train_loader:
            mix, stems = mix.to(device), stems.to(device)
            
            optimizer.zero_grad()
            pred = model(mix)
            
            loss = spectral_convergence_loss(pred, stems)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mix, stems, _ in val_loader:
                mix, stems = mix.to(device), stems.to(device)
                pred = model(mix)
                val_loss += spectral_convergence_loss(pred, stems).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch:03d}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, "best_model.pt")
            print(f"  Saved best model (val loss: {val_loss:.4f})")


if __name__ == "__main__":
    main()