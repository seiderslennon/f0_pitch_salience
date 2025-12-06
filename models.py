import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class PitchSalienceUnet(nn.Module):
    def __init__(self, in_channels: int = 5, base_channels: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

        # Decoder (upsample with interpolation to keep shapes aligned, then reduce channels)
        self.up_conv3 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up_conv2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up_conv1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)           # (B, base, T, F)
        x2 = self.enc2(self.pool1(x1))  # (B, 2*base, T/2, F/2)
        x3 = self.enc3(self.pool2(x2))  # (B, 4*base, T/4, F/4)

        # Bottleneck
        xb = self.bottleneck(self.pool3(x3))  # (B, 8*base, T/8, F/8)

        # Decoder
        u3 = F.interpolate(self.up_conv3(xb), size=x3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([x3, u3], dim=1))

        u2 = F.interpolate(self.up_conv2(d3), size=x2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([x2, u2], dim=1))

        u1 = F.interpolate(self.up_conv1(d2), size=x1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([x1, u1], dim=1))

        out = self.final_conv(d1)  # (B, 1, T, F)

        out = torch.transpose(out, 1, 2)  # (B, T, 1, F)
        out = torch.transpose(out, 2, 3)  # (B, T, F, 1)
        # BCEWithLogitsLoss expects logits, so no sigmoid here.
        return out

class PitchSalience(nn.Module):
    def __init__(self):
        super(PitchSalience, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, (3, 3), padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, (3, 3), padding="same")
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # input is (batch, channels, time, freq)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)  # (batch, 1, time, freq)

        x = torch.transpose(x, 1, 2)  # (batch, time, 1, freq)
        x = torch.transpose(x, 2, 3)  # (batch, time, freq, 1)
        # BCEWithLogitsLoss no sigmoid
        return x