import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, num_classes=23):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(),
            )

        self.enc1 = block(3, 64)
        self.enc2 = block(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.dec1 = block(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))

        x3 = nn.functional.interpolate(x2, scale_factor=2)
        x3 = self.dec1(x3)

        return self.out(x3)
