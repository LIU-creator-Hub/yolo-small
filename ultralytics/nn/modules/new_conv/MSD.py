"""MSD: brand-new lightweight operators for steel surface defect detection.

This module deliberately avoids the three mechanisms already used in the
project (SDAConv = pooling high/low frequency split, P_SMD = partial
deformable conv, LYMA = multi-shape convolution attention).  The four blocks
below are new:

    DMF    - Differentiable Morphology Filter      (grayscale top-hat / black-hat)
    HPD    - High-frequency Preserving Downsample  (low/high-band gated downsample)
    GAF    - Gradient-Aware Fusion                 (gradient gated fusion + DMF refine)
    MBlock - CSP-style block whose unit is DMF

Motivation on NEU-DET (200x200 steel surface, 6 classes):
    crazing / scratches  -> bright thin structures -> top-hat (x - opening)
    pitted / inclusion   -> dark blobs              -> black-hat (closing - x)
    patches / rolled-in  -> large area texture      -> multi-scale morphology
"""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ("DMF", "GAF", "HPD", "MBlock")


def _odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1


def _dilate(x, k):
    """Grayscale morphological dilation with a flat k x k structuring element."""
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def _erode(x, k):
    """Grayscale morphological erosion with a flat k x k structuring element."""
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)


class DMF(nn.Module):
    """Differentiable Morphology Filter.

    Two-scale opening/closing produce top-hat (bright detail) and black-hat (dark detail) responses. They are
    concatenated with the raw feature and fused by a depthwise-separable bottleneck with a residual connection.
    """

    def __init__(self, c1, c2, k=3, e=0.5):
        super().__init__()
        k = _odd(k)
        self.k1, self.k2 = k, k + 2
        cm = max(8, int(c2 * e))
        self.alpha = nn.Parameter(torch.full((1, c1, 1, 1), 0.5))
        self.beta = nn.Parameter(torch.full((1, c1, 1, 1), 0.5))
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * c1, cm, 1, bias=False),
            nn.BatchNorm2d(cm),
            nn.SiLU(),
            nn.Conv2d(cm, cm, 3, padding=1, groups=cm, bias=False),
            nn.BatchNorm2d(cm),
            nn.SiLU(),
            nn.Conv2d(cm, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.act = nn.SiLU()
        self.add = c1 == c2

    def forward(self, x):
        opening = 0.5 * (_dilate(_erode(x, self.k1), self.k1) + _dilate(_erode(x, self.k2), self.k2))
        closing = 0.5 * (_erode(_dilate(x, self.k1), self.k1) + _erode(_dilate(x, self.k2), self.k2))
        tophat = (x - opening) * self.alpha
        blackhat = (closing - x) * self.beta
        z = self.fuse(torch.cat((x, tophat, blackhat), 1))
        return self.act(x + z) if self.add else self.act(z)


class HPD(nn.Module):
    """High-frequency Preserving Downsample.

    Low band : average pooling + 1x1 projection. High band : (x - local mean) through a depthwise stride convolution.
    The two bands are combined by a learned per-pixel gate, then projected.
    """

    def __init__(self, c1, c2, k=3, s=2, e=0.5):
        super().__init__()
        k, s = _odd(k), int(s)
        cm = max(8, int(c2 * e))
        self.s = s
        self.pool = nn.AvgPool2d(k, s, k // 2)
        self.low = nn.Conv2d(c1, cm, 1, bias=False)
        self.hp_dw = nn.Conv2d(c1, c1, k, s, k // 2, groups=c1, bias=False)
        self.hp_pw = nn.Conv2d(c1, cm, 1, bias=False)
        self.gate = nn.Conv2d(2 * cm, 2, 1, bias=True)
        self.proj = nn.Sequential(
            nn.Conv2d(cm, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.add = c1 == c2 and s == 1

    def forward(self, x):
        low = self.low(self.pool(x))
        hp = x - F.avg_pool2d(x, 3, 1, 1)
        high = self.hp_pw(self.hp_dw(hp))
        z = torch.cat((low, high), 1)
        w = self.gate(z).softmax(1)
        z = w[:, :1] * low + w[:, 1:] * high
        out = self.proj(z)
        return out + x if self.add else out


class GAF(nn.Module):
    """Gradient-Aware Fusion.

    1x1 compression -> Sobel gradient-energy gate (defect edges) -> DMF morphology refinement, with a residual. Used as
    a neck fusion node.
    """

    def __init__(self, c1, c2, e=0.25, gk=7):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        gk = _odd(gk)
        self.gate = nn.Sequential(nn.Conv2d(2, 1, gk, padding=gk // 2, bias=True), nn.Sigmoid())
        self.dmf = DMF(c2, c2, e=e)
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
        self.register_buffer("sobel_x", sx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sx.t().contiguous().view(1, 1, 3, 3), persistent=False)
        self.act = nn.SiLU()

    def forward(self, x):
        z = self.proj(x)
        m = z.mean(1, keepdim=True)
        kx = self.sobel_x.to(dtype=z.dtype)
        ky = self.sobel_y.to(dtype=z.dtype)
        gx = F.conv2d(m, kx, padding=1)
        gy = F.conv2d(m, ky, padding=1)
        mag = (gx * gx + gy * gy + 1e-6).sqrt()
        g = self.gate(torch.cat((m, mag), 1))
        z = z * (1.0 + g)
        return self.act(z + self.dmf(z))


class MBlock(nn.Module):
    """CSP-style morphology block: split -> n x DMF -> concat -> 1x1 fuse."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        n = max(int(n), 1)
        c = max(8, int(c2 * e))
        self.cv1 = nn.Sequential(nn.Conv2d(c1, 2 * c, 1, bias=False), nn.BatchNorm2d(2 * c), nn.SiLU())
        self.m = nn.ModuleList(DMF(c, c, e=e) for _ in range(n))
        self.cv2 = nn.Sequential(nn.Conv2d((2 + n) * c, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))
