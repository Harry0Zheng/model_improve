"""3. CBAM (通道与空间注意力机制)

CBAM同时关注通道和空间两个方面的特征，从而能够更精确地捕捉重要信息。

来源：B站 DT算法工程师前钰 模块合集，已拆入零件库便于按需引用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出形状为 (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 输出形状为 (B, C, 1, 1)

        # 通道注意力的MLP：先降维再升维，使用两个1x1卷积实现
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),  # (B, C, 1, 1) → (B, C//r, 1, 1)
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)   # (B, C//r, 1, 1) → (B, C, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()  # 输出通道注意力权重

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # (B, C, H, W) → (B, C, 1, 1) → MLP → (B, C, 1, 1)
        max_out = self.fc(self.max_pool(x))  # 同上，最大池化路径
        out = avg_out + max_out              # 融合两条路径 (逐元素相加)
        return self.sigmoid(out)             # 输出通道注意力权重，范围0~1，形状为 (B, C, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 用一个大卷积核来提取空间位置的重要性
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)  # 输入是2通道，输出是1通道
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, C, H, W) → (B, 1, H, W)，对通道求平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)，对通道取最大
        x = torch.cat([avg_out, max_out], dim=1)  # 拼接后形状为 (B, 2, H, W)
        x = self.conv(x)                          # 卷积后输出 (B, 1, H, W)
        return self.sigmoid(x)                    # 输出空间注意力权重

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)  # 通道注意力模块
        self.spatial_attention = SpatialAttention(kernel_size)                   # 空间注意力模块

    def forward(self, x):
        out = self.channel_attention(x) * x  # 注意力加权：通道注意力乘原始特征 (广播机制)，shape 不变 (B, C, H, W)
        out = self.spatial_attention(out) * out  # 空间注意力乘前一结果，shape 仍为 (B, C, H, W)
        return out

# 测试 CBAM 模块
if __name__ == "__main__":
    x = torch.randn(4, 64, 32, 32)  # 模拟输入：batch=4, 通道数=64, 尺寸32x32
    cbam = CBAM(64)
    out = cbam(x)
    print(out.shape)  # 输出形状应为 torch.Size([4, 64, 32, 32])
