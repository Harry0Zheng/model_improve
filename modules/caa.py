"""19. CAA(CVPR2024最新上下文锚点注意力机制)

CAA模块通过引入特定的上下文锚点来计算注意力权重。与传统的注意力机制不同，这里使用了水平和垂直方向的分离卷积操作来捕捉局部特征，通过这种方式，它可以更好地关注输入特征图的不同部分。这种方法能够增强网络对特征图中关键区域的关注，有助于提升模型在复杂任务中的表现，同时减少计算量和参数量。

来源：B站 DT算法工程师前钰 模块合集，已拆入零件库便于按需引用。
"""

# 使用前需要先安装mmcv，建议3.9版本python，
# 安装方法：1.终端输入pip install -U openmim   2.终端输入mim install "mmcv>=2.0.0rc1" 安装完成
from typing import Optional  # 引入Optional类型提示，用于表示某个参数可以是None。
import torch  # 引入PyTorch库，用于深度学习的张量操作。
import torch.nn as nn  # 引入PyTorch的神经网络模块，用于构建神经网络层。
from mmcv.cnn import ConvModule  # 引入mmcv中的ConvModule模块，用于方便地创建卷积层。
from mmengine.model import BaseModule  # 引入mmengine中的BaseModule基类，用于自定义模块。

class CAA(BaseModule):  # 定义一个名为CAA的类，继承自BaseModule，用于实现“上下文锚点注意力”机制。
    """Context Anchor Attention"""  # 简要描述该模块的作用。
    def __init__(
            self,
            channels: int,  # 输入特征图的通道数。
            h_kernel_size: int = 11,  # 水平方向卷积核的大小，默认为11。
            v_kernel_size: int = 11,  # 垂直方向卷积核的大小，默认为11。
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),  # 归一化配置，默认使用批归一化（BN）。
            act_cfg: Optional[dict] = dict(type='SiLU'),  # 激活函数配置，默认使用SiLU激活函数。
            init_cfg: Optional[dict] = None,  # 初始化配置，默认为None。
    ):
        super().__init__(init_cfg)  # 调用父类的初始化方法，传入初始化配置。
        # 定义平均池化层，核大小为7，步幅为1，填充为3（保持特征图尺寸不变）。
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        # 定义第一个卷积模块，1x1卷积，用于调整通道数和进行非线性变换。
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        # 定义水平卷积模块，卷积核为(1, h_kernel_size)，按通道分组卷积，保持通道数不变。
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        # 定义垂直卷积模块，卷积核为(v_kernel_size, 1)，按通道分组卷积，保持通道数不变。
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        # 定义第二个卷积模块，再次使用1x1卷积，用于进一步调整通道数和进行非线性变换。
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        # 定义Sigmoid激活函数，用于生成注意力权重。
        self.act = nn.Sigmoid()

    def forward(self, x):  # 定义前向传播函数。
        # 输入经过平均池化、卷积模块和激活函数，最终生成注意力因子。
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor    # 将注意力因子与原始输入特征图逐元素相乘，调整特征图的权重，突出重要特征。
    
# 测试CAA模块
if __name__ == "__main__":  # 如果此脚本作为主程序运行，则执行以下代码。
    x = torch.randn(4, 64, 32, 32)  # 生成一个随机张量作为输入，形状为(4, 64, 32, 32)。
    caa = CAA(64)  # 创建CAA模块实例，输入通道数为64。
    out = caa(x)  # 将输入张量通过CAA模块。
    print(out.shape)  # 打印输出张量的形状，应该为torch.Size([4, 64, 32, 32])。
