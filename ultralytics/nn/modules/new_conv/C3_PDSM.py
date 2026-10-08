import torch
import torch.nn as nn
from.PDSM import *


# 确保这里导入了你写的 PDSM
# from ultralytics.nn.modules.Addmodules.PDSM import PDSM

class C3_PDSM(nn.Module):
    """
    原创极限轻量主干模块：C3_PDSM
    结合 CSP 结构与 PDSM (部分深度可分离混洗)，专门用于替换沉重的 C3k2，实现断崖式降参。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, bias=False)

        # 核心：使用你的 PDSM 替换笨重的 Bottleneck
        self.m = nn.Sequential(*(PDSM(c_, c_, shortcut=shortcut, ratio=0.5) for _ in range(n)))

    def forward(self, x):
        # CSP 分流与拼接逻辑
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))