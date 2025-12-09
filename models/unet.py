import torch
import torch.nn as nn
import torch.nn.functional as F


# helper

def match_size(tensor, target):
    """
    Make `tensor` match the spatial size (H, W) of `target`.

    If tensor is smaller -> pad.
    If tensor is larger  -> center-crop.
    """
    import torch.nn.functional as F

    _, _, h, w = tensor.shape
    _, _, h_t, w_t = target.shape

    # --------- PAD IF TOO SMALL ----------
    pad_h = max(0, h_t - h)
    pad_w = max(0, w_t - w)

    if pad_h > 0 or pad_w > 0:
        # F.pad uses (left, right, top, bottom) for 4D tensors' last two dims
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        h += pad_h
        w += pad_w

    # --------- CROP IF TOO BIG ----------
    dh = h - h_t
    dw = w - w_t

    if dh > 0 or dw > 0:
        crop_top = dh // 2
        crop_left = dw // 2
        tensor = tensor[:, :, crop_top:crop_top + h_t, crop_left:crop_left + w_t]

    return tensor

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

        # temp debugging
        # print("x:", x.shape)
        # print("e1:", e1.shape)
        # print("e2:", e2.shape)
        # print("e3:", e3.shape)
        # print("e4:", e4.shape)
        # print("b:", b.shape)

        # # Decoder shapes
        # u4 = self.up4(b)
        # print("u4 (after up4):", u4.shape)

        # u4 = match_size(u4, e4)
        # print("u4 (matched to e4):", u4.shape)

        # u4_cat = torch.cat([u4, e4], dim=1)
        # print("u4_cat:", u4_cat.shape)

        # d4 = self.dec4(u4_cat)
        # u3 = self.up3(d4)
        # print("u3 (after up3):", u3.shape)

        # u3 = match_size(u3, e3)
        # print("u3 (matched to e3):", u3.shape)

        # u3_cat = torch.cat([u3, e3], dim=1)
        # print("u3_cat:", u3_cat.shape)

        # d3 = self.dec3(u3_cat)
        # u2 = self.up2(d3)
        # print("u2 (after up2):", u2.shape)   

        # u2 = match_size(u2, e2)
        # print("u2 (matched to e2):", u2.shape)   

        # exit()

        # Decoder with skip connections
        u4 = self.up4(b)
        u4 = match_size(u4, e4)
        u4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = match_size(u3, e3)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = match_size(u2, e2)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = match_size(u1, e1)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        # Predict 2 masks
        masks = self.out_conv(d1)
        masks = torch.sigmoid(masks)  # keep masks in [0,1]

        return masks
