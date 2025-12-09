import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        #let me add residual connection for same channel dimensions 
        self.residual = (in_channels == out_channels)
        if not self.residual:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        if self.residual:
            x = x + residual
        else:
            x = x + self.res_conv(residual)
        
        return F.relu(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle odd shapes due to pooling when T (time) is not divisible by 2^n
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [0, diff_x, 0, diff_y])

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 7,   # 7 canonical stems
        base_channels: int = 32,
    ):
        super().__init__()

        # Encoder
        self.down1 = DownBlock(in_channels, base_channels)          # 1 -> 32
        self.down2 = DownBlock(base_channels, base_channels * 2)    # 32 -> 64
        self.down3 = DownBlock(base_channels * 2, base_channels * 4) # 64 -> 128
        self.down4 = DownBlock(base_channels * 4, base_channels * 8) # 128 -> 256

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)  # 256 -> 512

        # Decoder
        self.up4 = UpBlock(base_channels * 16, base_channels * 8)   # 512 -> 256
        self.up3 = UpBlock(base_channels * 8, base_channels * 4)    # 256 -> 128
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)    # 128 -> 64
        self.up1 = UpBlock(base_channels * 2, base_channels)        # 64 -> 32

        # Final 1x1 conv to map to stem channels
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, 1, n_mels=128, T)
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        x = self.bottleneck(x)

        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        x = self.out_conv(x)
        # output: (B, 7, 128, T)
        return x
