"""22. MCA Attention(2023多维协同注意力模块)

通过三个平行分支同时建模通道、高度和宽度维度的注意力、实现多维度的协同注意力；引入了挤压变换和激励变换组件，通过自适应的方式聚合特征和捕捉局部特征交互，提高了网络的表征能力；轻量化的模块，在提升准确率和效率的同时几乎没有额外的计算负担。

来源：B站 DT算法工程师前钰 模块合集，已拆入零件库便于按需引用。
"""


import torch
from torch import nn
import math

# 定义当前模块中对外暴露的类
__all__ = ['MCALayer', 'MCAGate']

# 标准差池化层（StdPool）
class StdPool(nn.Module):
    def __init__(self):
        # 调用父类nn.Module的初始化函数
        super(StdPool, self).__init__()

    # 前向传播函数
    def forward(self, x):
        # 获取输入张量的大小：batch 大小 b，通道数 c，高和宽均忽略（使用_占位符）
        b, c, _, _ = x.size()

        # 将输入张量展平到最后一个维度，然后计算标准差，沿着维度2（即空间维度）进行
        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        # 重新调整标准差张量的形状为 (b, c, 1, 1)
        std = std.reshape(b, c, 1, 1)

        # 返回标准差池化后的结果
        return std

# 多尺度自适应门控机制（MCAGate）
class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """
        初始化 MCAGate 模块
        Args:
            k_size: 卷积核的大小
            pool_types: 池化类型。可以是 'avg'（平均池化）、'max'（最大池化）或 'std'（标准差池化）
        """
        super(MCAGate, self).__init__()

        # 创建一个用于存放不同池化操作的列表
        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            # 根据池化类型将对应的池化层加入列表
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))  # 自适应平均池化到1x1
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))  # 自适应最大池化到1x1
            elif pool_type == 'std':
                self.pools.append(StdPool())  # 标准差池化
            else:
                raise NotImplementedError  # 未实现的池化类型将抛出错误

        # 定义一个卷积层，1xk_size 的卷积，padding 确保输出与输入相同
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        # 使用 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

        # 定义可学习的权重参数，初始化为随机值，用于控制池化后的特征组合
        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        # 对输入 x 进行池化操作，得到不同的池化特征
        feats = [pool(x) for pool in self.pools]

        # 如果只有一种池化方式，直接返回池化结果
        if len(feats) == 1:
            out = feats[0]
        # 如果有两种池化方式，按照一定的权重进行加权组合
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)  # 使用 Sigmoid 对权重归一化
            # 组合两种特征的权重计算方式
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            # 如果池化特征不符合预期，抛出错误
            assert False, "Feature Extraction Exception!"

        # 改变维度顺序，便于后续卷积操作
        out = out.permute(0, 3, 2, 1).contiguous()
        # 对变换后的特征进行卷积操作
        out = self.conv(out)
        # 恢复维度顺序
        out = out.permute(0, 3, 2, 1).contiguous()

        # 使用 Sigmoid 激活函数将输出限制在0到1之间
        out = self.sigmoid(out)
        # 将输出扩展到与输入 x 的形状相同
        out = out.expand_as(x)

        # 返回与输入 x 元素逐点相乘的结果，进行通道注意力调整
        return x * out

# 多尺度自适应层（MCALayer）
class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=False):
        """
        初始化 MCA 模块
        Args:
            inp: 输入特征的通道数
            no_spatial: 是否只进行通道维度的交互，而不考虑空间维度
        """
        super(MCALayer, self).__init__()

        # 定义参数 lambd 和 gamma，用于确定卷积核大小
        lambd = 1.5
        gamma = 1
        # 根据输入通道数计算卷积核大小
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1  # 确保卷积核大小为奇数

        # 定义三个方向的 MCAGate（通道-高度，宽度-通道，通道-宽度）
        self.h_cw = MCAGate(3)  # 通道与高度交互
        self.w_hc = MCAGate(3)  # 宽度与通道交互
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)  # 通道与宽度交互（如果不忽略空间维度）

    def forward(self, x):
        # 首先进行通道-高度方向的注意力操作
        x_h = x.permute(0, 2, 1, 3).contiguous()  # 交换通道和高度的维度
        x_h = self.h_cw(x_h)  # 通过 MCAGate 模块
        x_h = x_h.permute(0, 2, 1, 3).contiguous()  # 恢复原始维度顺序

        # 接着进行宽度-通道方向的注意力操作
        x_w = x.permute(0, 3, 2, 1).contiguous()  # 交换通道和宽度的维度
        x_w = self.w_hc(x_w)  # 通过 MCAGate 模块
        x_w = x_w.permute(0, 3, 2, 1).contiguous()  # 恢复原始维度顺序

        # 如果不忽略空间维度，还要进行通道-宽度方向的注意力操作
        if not self.no_spatial:
            x_c = self.c_hw(x)
            # 最终输出为三个方向特征的平均值
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            # 如果忽略空间维度，则只输出两个方向特征的平均值
            x_out = 1 / 2 * (x_h + x_w)

        # 返回多尺度注意力调整后的输出
        return x_out

# 示例用法
if __name__ == "__main__":
    # 生成一个随机张量，模拟输入：batch size = 4, channels = 64, height = width = 32
    input = torch.randn(4, 64, 32, 32)
    # 创建一个 MCA 模块实例，输入通道数为 64
    mca_layer = MCALayer(inp=64, no_spatial=False)
    # 通过 MCA 模块调整输入特征
    output = mca_layer(input)
    # 打印输出张量的形状，应该与输入相同
    print(output.shape)  # 输出: torch.Size([4, 64, 32, 32])
