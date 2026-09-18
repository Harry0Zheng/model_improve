"""24. DSConv(深度可分离卷积)

轻量化替换普通 Conv2d。输出通道可以与输入不同。
原示例注释里的输出通道与代码不一致，以 out_channels 为准。

来源：B站 DT算法工程师前钰 模块合集，已拆入零件库便于按需引用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Step 1: Depthwise convolution（每个输入通道单独卷积）
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        
        # Step 2: Pointwise convolution（1x1卷积，用于通道之间的信息融合）
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)   # [B, C, H, W]
        out = self.pointwise(out) # [B, C_out, H, W]
        out = self.bn(out)
        out = self.relu(out)
        return out
    
model = DepthwiseSeparableConv(in_channels=32, out_channels=32)
input_tensor = torch.randn(1, 32, 128, 128)  # batch size 1, 32通道，128x128图像
output = model(input_tensor)
print(output.shape)  # 输出: torch.Size([1, 64, 128, 128])
