import torch
import torch.nn as nn


class MSCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 16, 1, 1, 0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(3, 16, 7, 1, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 7, 1, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(3, 16, 15, 1, 7),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 15, 1, 7),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Conv2d(64, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x3 = self.cnn3(x)
        x4 = self.cnn4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.output(x)
