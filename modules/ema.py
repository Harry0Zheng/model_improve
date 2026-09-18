"""6. EMA 注意力机制 (基于跨空间学习的高效多尺度注意力模块，ICASSP2023推出，效果优于ECA、CBAM、CA ，小目标涨点明显)

通道或空间注意力机制在产生更清晰的特征表示方面的显著有效性得到了证明，但是通过通道降维来建模跨通道关系可能会给提取深度视觉表示带来副作用，因此提出了一种新的高效的多尺度注意力(EMA)模块，以保留每个通道上的信息和降低计算开销为目标，将部分通道重塑为批量维度，并将通道维度分组为多个子特征，使空间语义特征在每个特征组中均匀分布。具体来说，除了对全局信息进行编码以重新校准每个并行分支中的通道权重外，还通过跨维度交互进一步聚合两个并行分支的输出特征，以捕获像素级成对关系。

来源：B站 DT算法工程师前钰 模块合集，已拆入零件库便于按需引用。
"""

import torch
from torch import nn

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()

        # 分组数，表示通道被划分成多少个 group
        self.groups = factor

        # 断言保证每组至少有一个通道
        assert channels // self.groups > 0
        # 通道维度方向上的 softmax，用于计算注意力权重
        self.softmax = nn.Softmax(-1)
        # 适应性池化层
        self.agp = nn.AdaptiveAvgPool2d((1, 1))        # 输出尺寸: (B, C, 1, 1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 只沿 H 方向平均, 输出: (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 只沿 W 方向平均, 输出: (B, C, 1, W)
        # 每组独立归一化（在每组通道内部）
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        # 用于方向注意力融合的 1×1 卷积（H方向 + W方向 → 融合）
        self.conv1x1 = nn.Conv2d(
            channels // self.groups, channels // self.groups,
            kernel_size=1, stride=1, padding=0
        )
        # 空间特征提取用的 3×3 卷积
        self.conv3x3 = nn.Conv2d(
            channels // self.groups, channels // self.groups,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # 获取输入维度: B=batch size, C=通道数, H=高度, W=宽度
        b, c, h, w = x.size()
        # 把输入 x 按通道数分为 G 组，每组大小为 C/G
        # 输出尺寸: (B*G, C/G, H, W)
        group_x = x.reshape(b * self.groups, -1, h, w)

        ### 方向注意力通道增强 ###

        # 沿高度方向平均池化（竖向 Y 方向）
        # 输出尺寸: (B*G, C/G, H, 1)
        x_h = self.pool_h(group_x)

        # 沿宽度方向平均池化（横向 X 方向）
        # 输出尺寸: (B*G, C/G, 1, W)，但要转置成 (B*G, C/G, W, 1) 才能拼接
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # 拼接两个方向的池化结果，在 H 维度拼接 → (B*G, C/G, H+W, 1)
        concat_hw = torch.cat([x_h, x_w], dim=2)

        # 融合方向信息：经过 1×1 卷积（通道不变），输出尺寸: (B*G, C/G, H+W, 1)
        hw = self.conv1x1(concat_hw)

        # 把融合后的特征切分回 x_h 和 x_w 两部分
        # x_h: (B*G, C/G, H, 1), x_w: (B*G, C/G, W, 1)
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # 注意 x_w 要转置回 (B*G, C/G, 1, W)，以与 group_x 对应
        x_w = x_w.permute(0, 1, 3, 2)

        # 使用 sigmoid 激活生成方向注意力图，乘上原始特征后送入 GroupNorm
        # 输出尺寸: (B*G, C/G, H, W)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.sigmoid())

        ### 空间注意力建模（Cross-Spatial Learning） ###

        # 对 x1 做全局池化：输出 (B*G, C/G, 1, 1)，再 reshape 为 (B*G, C/G)
        # 再 reshape → (B*G, 1, C/G) → 做 softmax 得到注意力权重
        x11 = self.softmax(
            self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )  # Shape: [B*G, 1, C/G]

        # 对 x2 做特征提取，用于 spatial 路径的 value 分支
        # 输出: (B*G, C/G, H, W)
        x2 = self.conv3x3(group_x)

        # 把 x2 reshape 为 (B*G, C/G, H*W)，作为注意力 value
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # [B*G, C/G, H*W]

        # 对 x2 做全局池化，输出 (B*G, C/G, 1, 1) → reshape → softmax
        x21 = self.softmax(
            self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )  # Shape: [B*G, 1, C/G]

        # 对 x1 reshape 为 (B*G, C/G, H*W)，作为另一路的 value
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        # 两路注意力（x11 @ x12）和（x21 @ x22），都生成 spatial 权重图，尺寸: (B*G, 1, H*W)
        # 相加后 reshape 为: (B*G, 1, H, W)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )

        ### 最终特征融合 ###
        # 用 sigmoid 激活权重，然后与原始特征 group_x 相乘（做 re-weight）
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)

        return out

# 测试代码
if __name__ == '__main__':
    block = EMA(64).cuda()  # 定义一个 EMA 模块
    input = torch.rand(1, 64, 64, 64).cuda()  # 随机输入
    output = block(input)
    print(input.size(), output.size())  # 应输出一致尺寸: torch.Size([1, 64, 64, 64])
