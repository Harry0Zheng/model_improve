"""23. ASPP(空洞空间金字塔池化)

多尺度空洞卷积 + 全局池化。输出通道由 out_channels 决定，不一定等于输入通道。
默认实现是 2D。膨胀率过大时注意 Grid Effect 和 padding。

来源：B站 DT算法工程师前钰 模块合集，已拆入零件库便于按需引用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18]):
        """
        空洞空间金字塔池化 (ASPP) 模块。

        参数:
        - in_channels: 输入特征图的通道数
        - out_channels: ASPP模块输出特征图的通道数
        - dilations: 空洞卷积的扩张率列表，默认为[1, 6, 12, 18]
        """
        super(ASPP, self).__init__()
        # 1x1卷积，保持语义信息
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        # 空洞卷积的扩张率，dilation不同，感受野也会不同
        self.conv_3x3_dil1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[0], dilation=dilations[0])
        self.conv_3x3_dil2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1])
        self.conv_3x3_dil3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2])
        self.conv_3x3_dil4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3])
        # 全局平均池化层，用于捕捉全局上下文
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应全局平均池化，输出特征图大小为1x1
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # 1x1卷积
            nn.ReLU(inplace=True)
        )
        # 最终输出卷积层，汇聚ASPP不同分支的特征
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x: 输入特征图，形状为 (batch_size, in_channels, height, width)

        返回:
        - 输出特征图，形状为 (batch_size, out_channels, height, width)
        """
        # ASPP分支1：1x1卷积
        x1 = self.relu(self.conv_1x1(x))
        # ASPP分支2：3x3卷积，dilation=1
        x2 = self.relu(self.conv_3x3_dil1(x))
        # ASPP分支3：3x3卷积，dilation=6
        x3 = self.relu(self.conv_3x3_dil2(x))
        # ASPP分支4：3x3卷积，dilation=12
        x4 = self.relu(self.conv_3x3_dil3(x))
        # ASPP分支5：3x3卷积，dilation=18
        x5 = self.relu(self.conv_3x3_dil4(x))
        # 将5个分支的特征图拼接在一起（注意这里是5个分支）
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        # 最终输出卷积，进一步融合特征
        x = self.conv_out(x)

        return x

# 测试ASPP模块
if __name__ == "__main__":
    model = ASPP(in_channels=512, out_channels=256)
    input_tensor = torch.randn(1, 512, 64, 64)  # 假设输入是一个 (1, 512, 64, 64) 大小的特征图
    output = model(input_tensor)
    print(output.shape)  # 输出大小应该是 (1, 256, 64, 64)
