import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------
# match_size helper
# -----------------------------------------------
def match_size(tensor, target):
    _, _, h, w = tensor.shape
    _, _, h_t, w_t = target.shape

    # pad if too small
    pad_h = max(0, h_t - h)
    pad_w = max(0, w_t - w)
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        h += pad_h
        w += pad_w

    # crop if too big
    dh = h - h_t
    dw = w - w_t
    if dh > 0 or dw > 0:
        tensor = tensor[:, :, dh // 2: dh // 2 + h_t, dw // 2: dw // 2 + w_t]

    return tensor


# -----------------------------------------------
# Conv block
# -----------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------
#                SUPER-LIGHTWEIGHT 2-LEVEL U-NET
# ------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, base_channels=16, dropout=0.1):
        """
        2-level U-Net for low-memory GPUs:
            Level 1:  base_channels
            Level 2:  base_channels * 2
            Bottleneck: base_channels * 4
        """
        super().__init__()

        # -------- Encoder --------
        self.enc1 = ConvBlock(1, base_channels, dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)

        # -------- Bottleneck --------
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4, dropout)

        # -------- Decoder --------
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2, dropout)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, dropout)

        # -------- Output mask --------
        self.out_conv = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder level 2
        u2 = match_size(self.up2(b), e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        # Decoder level 1
        u1 = match_size(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # Output masks
        masks = self.out_conv(d1)
        return torch.sigmoid(masks)
