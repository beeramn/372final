import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# block: Conv → BatchNorm → ReLU → Dropout
# ------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),   # rubric: normalization layer
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),            # rubric: regularization
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------
# U-Net Architecture for Audio Spectrogram Masking
# Input: (B, 1, F, T)
# Output: (B, 2, F, T)  → two masks
# ------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, base_channels=64, dropout=0.1):
        super().__init__()

        # ---------------- Encoder ----------------
        self.enc1 = ConvBlock(1, base_channels, dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)

        # ---------------- Bottleneck ----------------
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16, dropout)

        # ---------------- Decoder ----------------
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8, dropout)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4, dropout)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2, dropout)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, dropout)

        # ---------------- Output ----------------
        # 2 channels = [vocals_mask, instrumental_mask]
        self.out_conv = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with skip connections
        u4 = self.up4(b)
        u4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        # Predict 2 masks
        masks = self.out_conv(d1)
        masks = torch.sigmoid(masks)  # keep masks in [0,1]

        return masks
