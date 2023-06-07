import torch
import torch.nn as nn


class Down(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(inC, outC, 3, 1, 1),
            nn.BatchNorm2d(outC),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3, 1, 1),
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, return_indices=True)

    def forward(self, x):
        x = self.model(x)
        x_pool, ind = self.pool(x)
        return x, x_pool, ind


class Middle(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class Up(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(2 * inC, inC, 3, 1, 1),
            nn.BatchNorm2d(inC),
            nn.ReLU(),
            nn.ConvTranspose2d(inC, outC, 3, 1, 1),
            nn.BatchNorm2d(outC),
            nn.ReLU(),
        )

    def forward(self, x, ind, y):
        x = self.unpool(x, ind)
        x = torch.cat([x, y], dim=1)
        return self.model(x)


class Out(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(c, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down(3, 16)  # 16 64 64
        self.down2 = Down(16, 32)  # 32 32 32
        self.down3 = Down(32, 64)  # 64 16 16
        self.down4 = Down(64, 128)  # 128 8 8
        self.middle = Middle(128)
        self.up4 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.up1 = Up(16, 16)
        self.out = Out(16)

    def forward(self, x):
        x1, x1p, ind1 = self.down1(x)
        x2, x2p, ind2 = self.down2(x1p)
        x3, x3p, ind3 = self.down3(x2p)
        x4, x4p, ind4 = self.down4(x3p)
        xMid = self.middle(x4p)
        z4 = self.up4(xMid, ind4, x4)
        z3 = self.up3(z4, ind3, x3)
        z2 = self.up2(z3, ind2, x2)
        z1 = self.up1(z2, ind1, x1)
        z = self.out(z1)
        return z
