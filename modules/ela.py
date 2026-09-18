"""17. ELA(2024最新高效的局部注意力机制)

一种轻量级且高效的局部注意力（ELA）模块。这个模块帮助深度CNN更准确地定位感兴趣的目标，在仅增加少量参数的情况下显著提高了CNN的整体性能。在包括ImageNet、MS COCO和Pascal VOC在内的流行数据集上的大量实验结果表明，提出的（ELA）模块在性能上超越了当前的最新注意力方法，同时保持了有竞争力的模型复杂度。

来源：B站 DT算法工程师前钰 模块合集，已拆入零件库便于按需引用。
"""

import torch
import torch.nn as nn
 
class ELA(nn.Module):
    def __init__(self, in_channels, phi):
        super(ELA, self).__init__()
        """
        ELA-T 和 ELA-B 设计为轻量级，非常适合网络层数较少或轻量级网络的 CNN 架构
        ELA-B 和 ELA-S 在具有更深结构的网络上表现最佳
        ELA-L 特别适合大型网络。
        
        参数:
        - in_channels (int): 输入特征图的通道数
        - phi (str): 表示卷积核大小和组数的选择，'T', 'B', 'S', 'L'中的一个
        """
        # 根据 phi 参数选择不同的卷积核大小
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        # 根据 phi 参数选择不同的卷积组数
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]
        # 根据 phi 参数选择不同的归一化组数
        num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi] 
        # 计算填充大小以保持卷积后尺寸不变
        pad = Kernel_size // 2
        # 1D 卷积层，使用分组卷积，卷积核大小为 Kernel_size
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        # 组归一化层
        self.GN = nn.GroupNorm(num_groups, in_channels)
        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, input):
        """
        前向传播函数。
        参数:
        - input (torch.Tensor): 输入特征图，形状为 (batch_size, channels, height, width)
        返回:
        - torch.Tensor: 应用边缘注意力后的特征图
        """
        b, c, h, w = input.size()  # 获取输入特征图的形状
        # 在宽度方向上进行平均池化
        x_h = torch.mean(input, dim=3, keepdim=True).view(b, c, h)
        # 在高度方向上进行平均池化
        x_w = torch.mean(input, dim=2, keepdim=True).view(b, c, w)
        # 对池化后的特征图应用 1D 卷积
        x_h = self.con1(x_h)  # [b, c, h]
        x_w = self.con1(x_w)  # [b, c, w]
        # 对卷积后的特征图进行归一化和激活，并 reshape 回来
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)  # [b, c, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)  # [b, c, 1, w]
        # 将输入特征图、x_h 和 x_w 按元素相乘，得到最终的输出特征图
        return x_h * x_w * input

# 调试部分
if __name__ == "__main__":
    # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
    input = torch.randn(1, 32, 256, 256)
    ela = ELA(in_channels=32, phi='T')
    output = ela(input)
    print(output.size())
