import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, ch, k=3, p=1):
        super().__init__()
        self.c1 = ConvBlock(ch, ch, k, 1, p, act=True)
        self.c2 = ConvBlock(ch, ch, k, 1, p, act=False)
    def forward(self, x):
        return x + self.c2(self.c1(x))

class Raw2HSI(nn.Module):
    """
    Input:  mosaic raw (N,1,H,W)  -- Bayer single-plane
    Output: HSI cube  (N,61,H,W)
    """
    def __init__(self, base_ch=64, n_blocks=8, out_bands=61):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)       # (N,1,H,W) -> (N,4,H/2,W/2)
        self.head = ConvBlock(4, base_ch, k=3, s=1, p=1, act=True)
        self.body = nn.Sequential(*[ResBlock(base_ch) for _ in range(n_blocks)])
        self.tail = ConvBlock(base_ch, base_ch, k=3, s=1, p=1, act=True)
        # predict (out_bands * 4) channels so that PixelShuffle(2) -> out_bands
        self.out_conv = nn.Conv2d(base_ch, out_bands*4, kernel_size=1, stride=1, padding=0)
        self.shuffle = nn.PixelShuffle(2)

        # global skip (helps stability)
        self.skip_conv = nn.Conv2d(4, out_bands*4, kernel_size=1, stride=1, padding=0)

    def forward(self, mosaic):
        # mosaic: (N,1,H,W)
        x = self.unshuffle(mosaic)                  # -> (N,4,H/2,W/2)
        s = self.skip_conv(x)                       # skip for stability
        y = self.head(x)
        y = self.body(y)
        y = self.tail(y)
        y = self.out_conv(y) + s                    # pred in packed space
        out = self.shuffle(y)                       # -> (N,61,H,W)
        out = torch.clamp(out, 0.0, 1.0)            # reflectance bound (optional)
        return out