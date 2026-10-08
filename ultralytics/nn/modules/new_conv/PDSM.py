import torch
from torch import nn


class PDSM(nn.Module):
    """部分深度可分离混洗模块 (Partial Depthwise Shuffle Module) 仅对部分通道做深度可分离卷积，其余直连，最后通道混洗。 轻量高效，适合小数据集防过拟合。.

    参数:
        c1: 输入通道（框架自动传入）
        c2: 输出通道（YAML 中指定）
        shortcut: 是否残差连接（c1 == c2 时生效）
        ratio: 参与深度卷积的通道比例（0~1），推荐 0.5
    """

    def __init__(self, c1, c2, shortcut=True, ratio=0.5):
        super().__init__()
        c1 = int(c1)
        c2 = int(c2)
        ratio = float(ratio)
        self.cp = max(1, int(c1 * ratio))  # 参与卷积的通道数
        self.c_rest = c1 - self.cp
        self.shortcut = shortcut and c1 == c2

        # 深度可分离卷积分支（仅作用于前 cp 个通道）
        self.dwconv = nn.Sequential(
            nn.Conv2d(self.cp, self.cp, 3, padding=1, groups=self.cp, bias=False),
            nn.BatchNorm2d(self.cp),
            nn.Conv2d(self.cp, self.cp, 1, bias=False),
            nn.BatchNorm2d(self.cp),
            nn.SiLU(),
        )
        # 1x1 投影到输出通道 c2
        self.project = nn.Sequential(nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())

    def forward(self, x):
        # 分割通道
        x_p = x[:, : self.cp, :, :]
        x_id = x[:, self.cp :, :, :]
        # 深度可分离处理
        out_p = self.dwconv(x_p)
        # 拼接
        out = torch.cat([out_p, x_id], dim=1)
        # 通道混洗
        out = self._channel_shuffle(out, 2)
        # 1x1 投影
        out = self.project(out)
        return x + out if self.shortcut else out

    @staticmethod
    def _channel_shuffle(x, groups):
        B, C, H, W = x.shape
        x = x.view(B, groups, C // groups, H, W)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return x
