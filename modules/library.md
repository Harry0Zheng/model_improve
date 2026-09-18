# 1. SENet (通道注意力机制)

## 介绍
SENet通过引入通道注意力机制，使模型能够更好地捕捉不同通道的重要性。传统卷积网络会平等对待所有通道，而SENet通过学习权重来强调信息量较大的通道，并抑制不重要的通道。

## 代码实现

```python
import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        初始化SELayer类。
        参数:
        channel (int): 输入特征图的通道数。
        reduction (int): 用于减少通道数的缩减率，默认为16。它用于在全连接层中压缩特征的维度。
        """
        super(SELayer, self).__init__()
        # 自适应平均池化层，将每个通道的空间维度（H, W）压缩到1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层序列，包含两个线性变换和中间的ReLU激活函数
        self.fc = nn.Sequential(
            # 第一个线性层，将通道数从 'channel' 缩减到 'channel // reduction'
            nn.Linear(channel, channel // reduction, bias=False),
            # ReLU激活函数，用于引入非线性
            nn.ReLU(inplace=True),
            # 第二个线性层，将通道数从 'channel // reduction' 恢复到 'channel'
            nn.Linear(channel // reduction, channel, bias=False),
            # Sigmoid激活函数，将输出限制在(0, 1)之间
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        前向传播函数。
        参数:
        x (Tensor): 输入张量，形状为 (batch_size, channel, height, width)。
        返回:
        Tensor: 经过通道注意力调整后的输出张量，形状与输入相同。
        """
        # 获取输入张量的形状
        b, c, h, w = x.size()
        # Squeeze：通过全局平均池化层，将每个通道的空间维度（H, W）压缩到1x1
        y = self.avg_pool(x).view(b, c)
        # Excitation：通过全连接层序列，对压缩后的特征进行处理
        y = self.fc(y).view(b, c, 1, 1)
        # 通过扩展后的注意力权重 y 调整输入张量 x 的每个通道
        return x * y.expand_as(x)
        
# 示例用法
if __name__ == "__main__":
    # 生成一个随机张量，模拟输入：batch size = 4, channels = 64, height = width = 32
    x = torch.randn(4, 64, 32, 32)
    # 创建一个SELayer实例，通道数为64
    se_layer = SELayer(channel=64)
    # 通过SELayer调整输入特征
    y = se_layer(x)
    # 打印输出张量的形状，应该与输入相同
    print(y.shape)  # 输出: torch.Size([4, 64, 32, 32])

```

---

# 2. ECA (高效通道注意力机制)

## 介绍
ECA是一种改进的通道注意力机制，在不显著增加计算开销的前提下，通过更简洁的方式捕捉通道间的依赖关系，从而提升卷积神经网络的性能。

## 代码实现

```python
import torch
from torch import nn

class ECA_layer(nn.Module):
    """构建一个 ECA 模块。

    参数:
        channel: 输入特征图的通道数
        k_size: 自适应选择的一维卷积核大小
    """
    def __init__(self, channel, k_size=3):
        super(ECA_layer, self).__init__()
        
        # 全局平均池化层，用于将每个通道的空间信息压缩成一个值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 一维卷积层，用于捕捉通道之间的交互信息
        # 1. 输入通道数为1，因为经过全局平均池化后，每个特征图都变成了1x1
        # 2. 输出通道数为1，因为我们不想改变通道数量，只是调整权重
        # 3. kernel_size=k_size，指定卷积核的大小
        # 4. padding=(k_size - 1) // 2，用于保持卷积后的张量长度与输入一致
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        
        # Sigmoid激活函数，将输出的范围限制在(0, 1)之间
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数，定义数据流经过该模块的处理步骤。

        参数:
        x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。

        返回:
        Tensor: 经过ECA模块处理后的输出张量。
        """
        
        # 使用全局平均池化将每个通道的空间维度 (H, W) 压缩到 1x1
        # 输出张量的形状将变为 [batch_size, channels, 1, 1]
        y = self.avg_pool(x)
        
        # 去掉最后一个维度，并交换第二个和第三个维度，方便后续用1D卷积在通道维度上进行注意力建模
        # y.squeeze(-1) 的形状是 [batch_size, channels, 1]
        # y.transpose(-1, -2) 交换后的形状是 [batch_size, 1, channels]
        y = y.squeeze(-1).transpose(-1, -2)
        
        # 通过一维卷积处理，卷积核大小是 k_size
        # 形状保持 [batch_size, 1, channels]，内容经过一维卷积核处理
        y = self.conv(y)
        
        # 再次交换维度，恢复原始的通道顺序
        # y.transpose(-1, -2) 将形状从 [batch_size, 1, channels] 变为 [batch_size, channels, 1]
        y = y.transpose(-1, -2)
        # 与原始输入x的 shape 对齐，方便做通道加权
        # 恢复被去掉的维度，将形状从 [batch_size, channels, 1] 变为 [batch_size, channels, 1, 1]
        y = y.unsqueeze(-1)
        
        # 使用 Sigmoid 激活函数将输出限制在 (0, 1) 之间
        y = self.sigmoid(y)
        
        # 将输入张量 x 与处理后的权重 y 相乘，进行通道加权
        # expand_as 确保 y 的形状与 x 匹配，以便逐元素相乘
        return x * y.expand_as(x)

# 示例用法
if __name__ == "__main__":
    # 生成一个随机张量，模拟输入：batch size = 4, channels = 64, height = width = 32
    x = torch.randn(4, 64, 32, 32)
    # 创建一个 ECA 模块实例，通道数为 64
    eca = ECA_layer(channel=64)
    # 通过 ECA 模块调整输入特征
    y = eca(x)
    # 打印输出张量的形状，应该与输入相同
    print(y.shape)  # 输出: torch.Size([4, 64, 32, 32])

```

---

# 3. CBAM (通道与空间注意力机制)

## 介绍
CBAM同时关注通道和空间两个方面的特征，从而能够更精确地捕捉重要信息。

## 代码实现

```python
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


```

---

# 4. KAN (可学习的激活函数，平替MLP)

## 介绍
KAN网络是一个替代MLP的即插即用模块，用于提升模型性能，减少参数量。全领域通用！

## 代码实现

```python
import torch
import torch.nn.functional as F
import math

"""
1.内存效率提升：原始实现需要扩展所有中间变量来执行不同的激活函数，而此代码中将计算重新制定为使用不同的基函数激活输入，
  然后线性组合它们。这种重新制定可以显著降低内存成本，并将计算变得更加高效。

2.正则化方法的改变：原始实现中使用的L1正则化需要对张量进行非线性操作，与重新制定的计算不兼容。
  因此，此代码中将L1正则化改为对权重的L1正则化，这更符合神经网络中常见的正则化方法，并且与重新制定的计算兼容。

3.激活函数缩放选项：原始实现中包括了每个激活函数的可学习缩放，但这个库提供了一个选项来禁用这个特性。
  禁用缩放可以使模型更加高效，但可能会影响结果。

4.参数初始化的改变：为了解决在MNIST数据集上的性能问题，此代码修改了参数的初始化方式，使用kaiming初始化。
"""

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,  # 网格大小，默认为 5
        spline_order=3, # 分段多项式的阶数，默认为 3
        scale_noise=0.1,  # 缩放噪声，默认为 0.1
        scale_base=1.0,   # 基础缩放，默认为 1.0
        scale_spline=1.0,    # 分段多项式的缩放，默认为 1.0
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,  # 基础激活函数，默认为 SiLU（Sigmoid Linear Unit）
        grid_eps=0.02,
        grid_range=[-1, 1],  # 网格范围，默认为 [-1, 1]
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size # 设置网格大小和分段多项式的阶数
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size   # 计算网格步长
        grid = ( # 生成网格
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0] 
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 将网格作为缓冲区注册

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features)) # 初始化基础权重和分段多项式权重
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则初始化分段多项式缩放参数
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise # 保存缩放噪声、基础缩放、分段多项式的缩放、是否启用独立的分段多项式缩放、基础激活函数和网格范围的容差
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)# 使用 Kaiming 均匀初始化基础权重
        with torch.no_grad():
            noise = (# 生成缩放噪声
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_( # 计算分段多项式权重
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则使用 Kaiming 均匀初始化分段多项式缩放参数
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        """
        计算给定输入张量的 B-样条基函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: B-样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = ( # 形状为 (in_features, grid_size + 2 * spline_order + 1)
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        """
        计算插值给定点的曲线的系数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。
        返回:
        torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        # 计算 B-样条基函数
        A = self.b_splines(x).transpose(
            0, 1 # 形状为 (in_features, batch_size, grid_size + spline_order)
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features) # 形状为 (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(   # 使用最小二乘法求解线性方程组
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)  # 形状为 (in_features, grid_size + spline_order, out_features)
        result = solution.permute( # 调整结果的维度顺序
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        获取缩放后的分段多项式权重。

        返回:
        torch.Tensor: 缩放后的分段多项式权重张量，形状与 self.spline_weight 相同。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor): # 将输入数据通过模型的各个层，经过线性变换和激活函数处理，最终得到模型的输出结果
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight) # 计算基础线性层的输出
        spline_output = F.linear( # 计算分段多项式线性层的输出
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output  # 返回基础线性层输出和分段多项式线性层输出的和

    @torch.no_grad()
    # 更新网格。
    # 参数:
    # x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
    # margin (float): 网格边缘空白的大小。默认为 0.01。
    # 根据输入数据 x 的分布情况来动态更新模型的网格,使得模型能够更好地适应输入数据的分布特点，从而提高模型的表达能力和泛化能力。
    def update_grid(self, x: torch.Tensor, margin=0.01): 
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)  # 计算 B-样条基函数
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)  # 调整维度顺序为 (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)  # 调整维度顺序为 (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0] # 对每个通道单独排序以收集数据分布
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)   # 更新网格和分段多项式权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # 计算正则化损失，用于约束模型的参数，防止过拟合
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        """
        计算正则化损失。

        这是对原始 L1 正则化的简单模拟，因为原始方法需要从扩展的（batch, in_features, out_features）中间张量计算绝对值和熵，
        而这个中间张量被 F.linear 函数隐藏起来，如果我们想要一个内存高效的实现。

        现在的 L1 正则化是计算分段多项式权重的平均绝对值。作者的实现也包括这一项，除了基于样本的正则化。

        参数:
        regularize_activation (float): 正则化激活项的权重，默认为 1.0。
        regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
        torch.Tensor: 正则化损失。
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module): # 封装了一个KAN神经网络模型，可以用于对数据进行拟合和预测。
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """
        初始化 KAN 模型。

        参数:
            layers_hidden (list): 包含每个隐藏层输入特征数量的列表。
            grid_size (int): 网格大小，默认为 5。
            spline_order (int): 分段多项式的阶数，默认为 3。
            scale_noise (float): 缩放噪声，默认为 0.1。
            scale_base (float): 基础缩放，默认为 1.0。
            scale_spline (float): 分段多项式的缩放，默认为 1.0。
            base_activation (torch.nn.Module): 基础激活函数，默认为 SiLU。
            grid_eps (float): 网格调整参数，默认为 0.02。
            grid_range (list): 网格范围，默认为 [-1, 1]。
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False): # 调用每个KANLinear层的forward方法，对输入数据进行前向传播计算输出。
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否更新网格。默认为 False。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):#计算正则化损失的方法，用于约束模型的参数，防止过拟合。
        """
        计算正则化损失。

        参数:
            regularize_activation (float): 正则化激活项的权重，默认为 1.0。
            regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
            torch.Tensor: 正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
```

---

# 5. KAN卷积 (可学习的卷积核参数)

## 介绍
将卷积的经典线性变换更改为每个像素中可学习的非线性激活函数，KAN卷积与卷积非常相似，但不是在内核和图像中相应像素之间应用点积，而是对每个元素应用可学习的非线性激活函数，然后将它们相加。

## 代码实现

```python
import torch
import numpy as np
from typing import List, Tuple, Union

def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size,n_channels,n, m = matrix.shape
    h_out = np.floor((n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    b = [kernel_side // 2, kernel_side// 2]
    return h_out,w_out,batch_size,n_channels

def kan_conv2d(matrix: Union[List[List[float]], np.ndarray], #but as torch tensors. Kernel side asume q el kernel es cuadrado
             kernel, 
             kernel_side,
             stride= (1, 1), 
             dilation= (1, 1), 
             padding= (0, 0),
             device= "cuda"
             ) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out,batch_size,n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    
    matrix_out = torch.zeros((batch_size,n_channels,h_out,w_out)).to(device)#estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold((kernel_side,kernel_side), dilation=dilation, padding=padding, stride=stride)


    for channel in range(n_channels):
        #print(matrix[:,channel,:,:].unsqueeze(1).shape)
        conv_groups = unfold(matrix[:,channel,:,:].unsqueeze(1)).transpose(1, 2)
        #print("conv",conv_groups.shape)
        for k in range(batch_size):
            matrix_out[k,channel,:,:] = kernel.forward(conv_groups[k,:,:]).reshape((h_out,w_out))
    return matrix_out

def multiple_convs_kan_conv2d(matrix, #but as torch tensors. Kernel side asume q el kernel es cuadrado
             kernels, 
             kernel_side,
             stride= (1, 1), 
             dilation= (1, 1), 
             padding= (0, 0),
             device= "cuda"
             ) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out,batch_size,n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    n_convs = len(kernels)
    matrix_out = torch.zeros((batch_size,n_channels*n_convs,h_out,w_out)).to(device)#estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold((kernel_side,kernel_side), dilation=dilation, padding=padding, stride=stride)
    conv_groups = unfold(matrix[:,:,:,:]).view(batch_size, n_channels,  kernel_side*kernel_side, h_out*w_out).transpose(2, 3)#reshape((batch_size,n_channels,h_out,w_out))
    for channel in range(n_channels):
        for kern in range(n_convs):
            matrix_out[:,kern  + channel*n_convs,:,:] = kernels[kern].conv.forward(conv_groups[:,channel,:,:].flatten(0,1)).reshape((batch_size,h_out,w_out))
    return matrix_out


def add_padding(matrix: np.ndarray, 
                padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix. 

    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix

    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding
    
    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix
    
    return padded_matrix

import torch
import math
from KAN import KANLinear

#Script que contiene la implementación del kernel con funciones de activación.
class KAN_Convolutional_Layer(torch.nn.Module):
    def __init__(
            self,
            n_convs: int = 1,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order:int = 3,
            scale_noise:float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device: str = "cpu"
        ):
        """
        KAN 卷积层，支持多个卷积操作

        参数:
            n_convs (int): 卷积层的数量
            kernel_size (tuple): 卷积核的大小
            stride (tuple): 卷积操作的步幅
            padding (tuple): 卷积操作的填充
            dilation (tuple): 卷积核元素之间的间距
            grid_size (int): 网格的大小
            spline_order (int): 样条的阶数
            scale_noise (float): 噪声的比例
            scale_base (float): 基础尺度
            scale_spline (float): 样条的尺度
            base_activation (torch.nn.Module): 基础激活函数
            grid_eps (float): 网格的 epsilon 值
            grid_range (tuple): 网格的范围
            device (str): 使用的设备
        """


        super(KAN_Convolutional_Layer, self).__init__()  # 调用父类的初始化方法
        self.grid_size = grid_size # 网格大小，用于定义卷积操作的网格
        self.spline_order = spline_order # 样条顺序，用于定义卷积核的阶数
        self.kernel_size = kernel_size  # 卷积核的大小
        self.device = device # 设备（如 'cuda' 或 'cpu'），用于模型计算
        self.dilation = dilation  # 膨胀系数，用于膨胀卷积操作
        self.padding = padding # 填充大小，用于卷积操作的边界填充
        self.convs = torch.nn.ModuleList() # 卷积层列表，用于存储多个卷积操作
        self.n_convs = n_convs  # 卷积层的数量
        self.stride = stride # 步幅，用于卷积操作的步长


        # 创建 n_convs 个 KAN_Convolution 对象，并将它们添加到卷积层列表中
        for _ in range(n_convs):
            self.convs.append(
                KAN_Convolution(
                    kernel_size= kernel_size,  # 卷积核的大小
                    stride = stride,  # 卷积操作的步幅
                    padding=padding,  # 卷积操作的边界填充
                    dilation = dilation,  # 卷积操作的膨胀系数
                    grid_size=grid_size, # 用于定义卷积操作的网格大小
                    spline_order=spline_order,  
                    scale_noise=scale_noise,  # 噪声尺度
                    scale_base=scale_base, # 基础尺度
                    scale_spline=scale_spline, # 样条尺度
                    base_activation=base_activation, # 基础激活函数
                    grid_eps=grid_eps,  # 网格的误差
                    grid_range=grid_range,  # 网格范围
                    device = device  # 用于模型计算的设备（如 'cuda' 或 'cpu'）
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        # 如果有多个卷积层，应用所有卷积层
        if self.n_convs>1:
            # 使用 multiple_convs_kan_conv2d 函数进行多个卷积操作
            return multiple_convs_kan_conv2d(x, self.convs,self.kernel_size[0],self.stride,self.dilation,self.padding,self.device)
         # 如果只有一个卷积层，应用单个卷积层 
        return self.convs[0].forward(x)  # 使用第一个（也是唯一一个）卷积层进行卷积操作
        

class KAN_Convolution(torch.nn.Module):
    def __init__(
            self,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device = "cpu"
        ):
        """
        Args
        """
        super(KAN_Convolution, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.device = device
        self.conv = KANLinear(
            in_features = math.prod(kernel_size),
            out_features = 1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

    def forward(self, x: torch.Tensor, update_grid=False):
        return kan_conv2d(x, self.conv,self.kernel_size[0],self.stride,self.dilation,self.padding,self.device)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum( layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)

```

---

# 6. EMA 注意力机制 (基于跨空间学习的高效多尺度注意力模块，ICASSP2023推出，效果优于ECA、CBAM、CA ，小目标涨点明显)

## 介绍
通道或空间注意力机制在产生更清晰的特征表示方面的显著有效性得到了证明，但是通过通道降维来建模跨通道关系可能会给提取深度视觉表示带来副作用，因此提出了一种新的高效的多尺度注意力(EMA)模块，以保留每个通道上的信息和降低计算开销为目标，将部分通道重塑为批量维度，并将通道维度分组为多个子特征，使空间语义特征在每个特征组中均匀分布。具体来说，除了对全局信息进行编码以重新校准每个并行分支中的通道权重外，还通过跨维度交互进一步聚合两个并行分支的输出特征，以捕获像素级成对关系。

## 代码实现

```python
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


```

# 7. CA注意力机制(CVPR2021，帮助轻量级网络涨点、适用于分类、检测、分割等，即插即用)

## 介绍
CA注意力机制不仅考虑了通道信息，还考虑了方向相关的位置信息，足够的灵活和轻量，能够简单的插入到轻量级网络的核心模块中。与通过2维全局池化将特征张量转换为单个特征向量的通道注意力不同，coordinate注意力将通道注意力分解为两个1维特征编码过程，分别沿2个空间方向聚合特征。这样，可以沿一个空间方向捕获远程依赖关系，同时可以沿另一空间方向保留精确的位置信息。然后将生成的特征图分别编码为一对方向感知和位置敏感的attention map，可以将其互补地应用于输入特征图，以增强关注对象的表示。

## 代码实现

```python
import torch
import torch.nn as nn

# 定义一个 h_sigmoid 类，继承自 nn.Module。
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        # 调用父类的构造函数。
        super(h_sigmoid, self).__init__()
        # 使用 ReLU6 作为激活函数，inplace 参数用于节省内存。
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        # 实现 h-sigmoid 激活函数：f(x) = ReLU6(x + 3) / 6
        return self.relu(x + 3) / 6

# 定义一个 h_swish 类，继承自 nn.Module。
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        # 调用父类的构造函数。
        super(h_swish, self).__init__()
        # 使用 h_sigmoid 作为 h-swish 的一部分。
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        # 实现 h-swish 激活函数：f(x) = x * h_sigmoid(x)
        return x * self.sigmoid(x)

# 定义一个 CoordAtt 类，继承自 nn.Module。
# 该类实现了一种结合坐标信息的注意力机制。
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        # 调用父类的构造函数。
        super(CoordAtt, self).__init__()
        # 使用自适应平均池化提取高度方向和宽度方向的特征。
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 高度方向池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 宽度方向池化
        # 计算通道缩减后的通道数 mip，并保证最小值为 8。
        mip = max(8, inp // reduction)
        # 定义 1x1 卷积层，用于减少通道数。
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # 定义批归一化层，跟随卷积层。
        self.bn1 = nn.BatchNorm2d(mip)
        # 定义 h-swish 激活函数。
        self.act = h_swish()
        # 定义 1x1 卷积层，用于从压缩的通道数恢复到目标输出通道数。
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 高度方向
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 宽度方向

    def forward(self, x):
        # 保存输入的张量，用于与注意力权重相乘。
        identity = x
        # 获取输入张量的尺寸 (batch_size, channels, height, width)。
        n, c, h, w = x.size()
        # 沿高度方向进行自适应平均池化，输出尺寸为 (batch_size, channels, height, 1)。
        x_h = self.pool_h(x)
        # 沿宽度方向进行自适应平均池化，并对宽度和高度进行转置，输出尺寸为 (batch_size, channels, width, 1)。
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # 将两个池化后的张量在宽度方向连接，形成一个新的张量。
        y = torch.cat([x_h, x_w], dim=2)
        # 对连接后的张量进行 1x1 卷积，减少通道数。
        y = self.conv1(y)
        # 进行批归一化。
        y = self.bn1(y)
        # 应用 h-swish 激活函数。
        y = self.act(y)
        # 将卷积后的张量重新按高度和宽度方向切分。
        x_h, x_w = torch.split(y, [h, w], dim=2)
        # 对宽度方向的张量进行转置，恢复原来的维度顺序。
        x_w = x_w.permute(0, 1, 3, 2)
        # 使用 sigmoid 函数对高度和宽度方向的张量进行注意力权重计算。
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        # 将注意力权重应用到原始输入的张量上，进行逐元素相乘。
        out = identity * a_w * a_h
        # 返回输出张量，大小与输入张量相同。
        return out


# 输入 N C H W, 输出 N C H W
if __name__ == '__main__':
    # 实例化 CoordAtt 模块，输入和输出通道数为 64。
    block = CoordAtt(64, 64)
    # 生成一个随机输入张量，大小为 (1, 64, 64, 64)。
    input = torch.rand(1, 64, 64, 64)
    # 将输入张量传入 CoordAtt 模块，计算输出。
    output = block(input)
    # 打印输入和输出张量的大小。
    print(input.size(), output.size())

```

# 8. LSK注意力机制(选择性注意力机制，ICCV2023，适用于遥感目标检测)

## 介绍
LSK注意力机制通过在特征提取模块动态调整感受野，更有效地处理了不同目标所需的背景信息差异。其中，动态感受野由一个空间选择机制实现，该机制对一大串Depth-wise 卷积核所处理的特征进行有效加权和空间融合。这些卷积核的权重根据输入动态确定，同时允许模型针对空间上的不同目标自适应地选择不同大小的核并调整感受野。

## 代码实现

```python
import torch
import torch.nn as nn

class LSKblock(nn.Module):
    def __init__(self, dim):
        """
        LSKblock 初始化函数
        Args:
            dim (int): 输入通道的数量。
        """
        super().__init__()
        # 第一个卷积层，使用 5x5 的卷积核，通道数与输入一致，采用深度可分离卷积（groups=dim）。
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 第二个卷积层，使用 7x7 的卷积核，采用深度可分离卷积，并设置扩展卷积（dilation=3）和较大的填充（padding=9）来扩大感受野。
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 第三个卷积层，使用 1x1 的卷积核，将通道数减少一半（dim -> dim//2）。
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        # 第四个卷积层，使用 1x1 的卷积核，将通道数减少一半（dim -> dim//2）。
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        # 用于通道间注意力机制的卷积层，接收 2 个通道的数据并输出 2 个通道，卷积核大小为 7x7。
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        # 最后一个卷积层，使用 1x1 的卷积核，将通道数恢复到原始输入的维度（dim//2 -> dim）。
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
        Returns:
            Tensor: 输出张量，形状与输入相同。
        """
        # 通过第一个卷积层，获取局部特征（5x5 卷积）。
        attn1 = self.conv0(x)
        # 通过第二个卷积层，获取更大范围的空间特征（7x7 扩展卷积）。
        attn2 = self.conv_spatial(attn1)
        # 通过第三个卷积层，减少通道数（1x1 卷积，dim -> dim//2）。
        attn1 = self.conv1(attn1)
        # 通过第四个卷积层，减少通道数（1x1 卷积，dim -> dim//2）。
        attn2 = self.conv2(attn2)
        # 将两个不同感受野的特征在通道维度上进行拼接，形成一个新的张量。
        attn = torch.cat([attn1, attn2], dim=1)
        # 对拼接后的特征图计算通道维度的平均值特征（avg_attn）。
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        # 对拼接后的特征图计算通道维度的最大值特征（max_attn）。
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # 将平均值特征和最大值特征在通道维度上进行拼接。
        agg = torch.cat([avg_attn, max_attn], dim=1)
        # 通过注意力机制卷积层，生成两个通道的注意力权重。
        sig = self.conv_squeeze(agg).sigmoid()
        # 将两个不同感受野的特征分别乘以相应的注意力权重。
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        # 通过最后一个卷积层，将通道数恢复到原始输入的维度。
        attn = self.conv(attn)
        # 将原始输入与注意力加权后的特征相乘，得到增强后的输出。
        return x * attn

# 示例运行，输入张量形状为 (1, 64, 64, 64)，输出形状与输入相同。
if __name__ == '__main__':
    block = LSKblock(64).cuda()  # 初始化 LSKblock 模块，并将其移动到 GPU 上。
    input = torch.rand(1, 64, 64, 64).cuda()  # 创建一个随机输入张量，并移动到 GPU 上。
    output = block(input)  # 通过 LSKblock 模块进行前向传播。
    print(input.size(), output.size())  # 打印输入和输出的张量形状，确认它们相同。

```

# 9.SK注意力机制(CVPR2019)

## 介绍
SKNet的核心创新在于Selective Kernel (SK)模块，允许网络在多个卷积核之间选择，提供了自适应感受野和增强特征表示的优势，在不增加网络复杂性的情况下提升了特征提取的能力

## 代码实现

```python
import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        """
        SKAttention 初始化函数
        Args:
            channel (int): 输入和输出通道数。
            kernels (list): 多尺度卷积核的大小列表。
            reduction (int): 通道数缩减的比例因子。
            group (int): 深度卷积的组数。
            L (int): 计算降维的最小通道数。
        """
        super().__init__()

        # 计算缩减后的通道数，保证其不小于 L。
        self.d = max(L, channel // reduction)
        
        # 初始化多个卷积操作，每个卷积操作的卷积核大小由 kernels 列表决定。
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),  # 深度卷积
                    ('bn', nn.BatchNorm2d(channel)),  # 批归一化
                    ('relu', nn.ReLU())  # ReLU 激活函数
                ]))
            )

        # 线性层，用于将通道数降维为 d。
        self.fc = nn.Linear(channel, self.d)

        # 初始化多个线性层，用于将降维后的特征映射回原通道数。
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))

        # softmax 层，用于计算不同尺度特征的注意力权重。
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
        Returns:
            Tensor: 输出张量，形状与输入相同。
        """
        bs, c, _, _ = x.size()  # 获取输入张量的形状信息
        conv_outs = []

        ### 多尺度特征提取
        for conv in self.convs:
            conv_outs.append(conv(x))  # 使用不同的卷积核对输入进行卷积操作
        feats = torch.stack(conv_outs, 0)  # 将不同卷积核的输出在第一个维度上堆叠，形状为 (k, bs, channel, h, w)

        ### 特征融合
        U = sum(conv_outs)  # 将所有尺度的特征进行相加，形状为 (bs, c, h, w)

        ### 通道数缩减
        S = U.mean(-1).mean(-1)  # 对空间维度进行平均，得到形状为 (bs, c) 的张量
        Z = self.fc(S)  # 通过全连接层进行通道数缩减，得到形状为 (bs, d) 的张量

        ### 计算注意力权重
        weights = []
        for fc in self.fcs:
            weight = fc(Z)  # 通过线性层将降维后的特征映射回原通道数，形状为 (bs, c)
            weights.append(weight.view(bs, c, 1, 1))  # 调整形状为 (bs, channel, 1, 1)
        attention_weights = torch.stack(weights, 0)  # 将所有的注意力权重在第一个维度上堆叠，形状为 (k, bs, channel, 1, 1)
        attention_weights = self.softmax(attention_weights)  # 使用 softmax 进行归一化，得到最终的注意力权重

        ### 加权融合特征
        V = (attention_weights * feats).sum(0)  # 将注意力权重与对应的多尺度特征相乘并相加，得到最终的加权特征
        return V

# 示例运行
if __name__ == '__main__':
    input = torch.randn(1, 64, 64, 64)  # 创建一个随机输入张量，形状为 (1, 64, 64, 64)
    sk_attention = SKAttention(channel=64, reduction=8)  # 初始化 SKAttention 模块
    output = sk_attention(input)  # 通过 SKAttention 模块进行前向传播
    print(output.shape)  # 打印输出张量的形状，确认其与输入形状相同

```

# 10.MLCA注意力机制(混合局部通道注意力)

## 介绍
轻量级的 Mixed Local Channel Attention (MLCA) 模块，该模块同时考虑通道信息和空间信息，并结合局部信息和全局信息以提高网络的表达效果，主要应用于计算机视觉，更适于目标检测、实例分割

## 代码实现

```python
import math, torch
from torch import nn
import torch.nn.functional as F

# 定义多尺度局部上下文注意力模块 (MLCA)
class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super(MLCA, self).__init__()

        # 初始化参数
        # in_size: 输入的通道数
        # local_size: 局部池化尺寸，默认为5
        # gamma 和 b 用于计算卷积核的大小
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        # 根据 ECA 方法计算卷积核大小 k
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1  # 保证 k 是奇数，以便对称填充
        # 定义两个 1D 卷积，用于全局和局部的注意力计算
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        # 局部和全局注意力的加权参数
        self.local_weight = local_weight
        # 定义自适应平均池化，用于局部和全局特征提取
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 进行局部和全局的自适应平均池化
        local_arv = self.local_arv_pool(x)  # 局部特征池化
        global_arv = self.global_arv_pool(local_arv)  # 从局部特征中进一步提取全局特征
        # 获取输入和池化后的特征的形状
        b, c, m, n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape
        # 将局部特征重新排列为 (b, 1, local_size*local_size*c) 以便于通过 1D 卷积
        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        # 将全局特征重新排列为 (b, 1, c)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)
        # 通过局部卷积计算局部注意力
        y_local = self.conv_local(temp_local)
        # 通过全局卷积计算全局注意力
        y_global = self.conv(temp_global)
        # 将局部注意力重新排列回原始形状 (b, c, local_size, local_size)
        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c, self.local_size, self.local_size)
        # 将全局注意力重新排列回 (b, c, 1, 1)
        y_global_transpose = y_global.transpose(-1, -2).unsqueeze(-1)
        # 应用 sigmoid 激活函数，将注意力权重映射到 (0, 1) 区间
        att_local = y_local_transpose.sigmoid()
        # 将全局注意力池化到局部特征的大小
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])
        # 根据局部和全局的加权参数，融合两种注意力，调整到输入的空间维度
        att_all = F.adaptive_avg_pool2d(att_global * (1 - self.local_weight) + (att_local * self.local_weight), [m, n])
        # 将输入特征与注意力权重相乘，得到加权后的输出
        x = x * att_all
        return x

# 测试代码块
if __name__ == '__main__':
    # 创建 MLCA 模块实例，输入通道数为 256
    attention = MLCA(in_size=256)
    # 随机生成输入张量，形状为 (2, 256, 16, 16)
    inputs = torch.randn((2, 256, 16, 16))
    # 将输入张量传入 MLCA 模块，计算输出
    result = attention(inputs)
    # 打印输出张量的形状
    print(result.size())

```


# 11.A2Attention(双注意力块)

## 介绍
将整个空间的关键特征收集到一个紧凑的集合中，然后自适应地将其分布到每个位置，这样后续的卷积层即使没有很大的接收域也可以感知整个空间的特征。第一级的注意力集中操作有选择地从整个空间中收集关键特征，而第二级的注意力集中操作采用另一种注意力机制，自适应地分配关键特征的子集，这些特征有助于补充高级任务的每个时空位置。

## 代码实现

```python
import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库，用于深度学习和张量操作
from torch import nn  # 从 PyTorch 中导入神经网络模块
from torch.nn import init  # 从 PyTorch 中导入初始化模块，用于权重初始化
from torch.nn import functional as F  # 导入 PyTorch 的函数式 API，用于在神经网络中应用各种功能

# 定义双重注意力模块 (DoubleAttention)
class DoubleAttention(nn.Module):

    def __init__(self, in_channels, c_m=128, c_n=128, reconstruct=True):
        super().__init__()

        # 初始化输入参数
        # in_channels: 输入的通道数
        # c_m: 第一个特征映射的通道数，默认为 128
        # c_n: 第二个特征映射的通道数，默认为 128
        # reconstruct: 是否重新构建输出的通道数，默认为 True
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        # 定义三个 1x1 的卷积层
        self.convA = nn.Conv2d(in_channels, c_m, 1)  # 用于计算特征 A
        self.convB = nn.Conv2d(in_channels, c_n, 1)  # 用于计算注意力映射 B
        self.convV = nn.Conv2d(in_channels, c_n, 1)  # 用于计算注意力向量 V

        # 如果需要重新构建输出通道数，定义一个 1x1 的卷积层
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        # 初始化权重
        self.init_weights()

    # 定义权重初始化函数
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 对每个卷积层应用 He 正态分布初始化
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:  # 偏置初始化为 0
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # 对批量归一化层初始化
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 对全连接层应用正态分布初始化
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    # 定义前向传播函数
    def forward(self, x):
        # 获取输入的形状
        b, c, h, w = x.shape
        assert c == self.in_channels  # 确保输入的通道数与定义的通道数一致
        # 通过三个卷积层计算特征 A、注意力映射 B 和注意力向量 V
        A = self.convA(x)  # 特征 A 的形状为 (b, c_m, h, w)
        B = self.convB(x)  # 注意力映射 B 的形状为 (b, c_n, h, w)
        V = self.convV(x)  # 注意力向量 V 的形状为 (b, c_n, h, w)
        # 重塑特征 A 为 (b, c_m, h*w)
        tmpA = A.view(b, self.c_m, -1)
        # 重塑并应用 softmax 到注意力映射 B，得到注意力权重，形状为 (b, c_n, h*w)
        attention_maps = F.softmax(B.view(b, self.c_n, -1), dim=-1)
        # 重塑并应用 softmax 到注意力向量 V，得到注意力权重，形状为 (b, c_n, h*w)
        attention_vectors = F.softmax(V.view(b, self.c_n, -1), dim=-1)
        # 第一步：特征门控
        # 计算特征 A 与注意力映射 B 的批量矩阵乘法，得到全局描述符，形状为 (b, c_m, c_n)
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))
        # 第二步：特征分布
        # 将全局描述符与注意力向量 V 相乘，得到新的特征映射 Z，形状为 (b, c_m, h*w)
        tmpZ = global_descriptors.matmul(attention_vectors)
        # 重塑 Z 为 (b, c_m, h, w)
        tmpZ = tmpZ.view(b, self.c_m, h, w)

        # 如果需要重新构建输出通道数，应用卷积层
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)
        # 返回计算后的输出
        return tmpZ

# 测试代码块
if __name__ == '__main__':
    # 创建 DoubleAttention 模块实例，输入通道数为 32
    input = torch.randn(64, 32, 7, 7)  # 随机生成输入张量，形状为 (64, 32, 7, 7)
    a2 = DoubleAttention(32)
    output = a2(input)
    # 打印输出张量的形状，验证输出是否正确
    print(output.shape)


```


# 12.BAM(瓶颈注意力模块)

## 介绍
瓶颈注意模块(BAM)，可以与任何前馈卷积神经网络集成，沿着两个不同的路径（通道和空间）推断注意力映射。 将模块放在模型的每个瓶颈处（特征映射产生降采样），构建一个具有多个参数的分层注意，可以与任何前馈模型以端到端方式进行训练。

## 代码实现

```python
import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库，用于深度学习和张量操作
from torch import nn  # 从 PyTorch 中导入神经网络模块
from torch.nn import init  # 从 PyTorch 中导入初始化模块，用于权重初始化

# 自动填充函数，用于根据内核大小、填充和扩张比率自动计算填充值
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        # 如果扩张比率大于1，根据扩张比率调整内核大小
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # 如果没有指定填充值，自动计算填充值为内核大小的一半
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

# 定义一个展平层，将输入的多维张量展平为二维张量
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# 定义通道注意力模块 (Channel Attention)
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3):
        super().__init__()
        # 自适应平均池化，将输入特征图的空间维度缩小到 1x1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 定义全连接层的通道数
        gate_channels = [channel]  # 输入通道数
        gate_channels += [channel // reduction] * num_layers  # 缩减后的通道数
        gate_channels += [channel]  # 恢复到原始通道数

        # 使用 nn.Sequential 定义通道注意力模块的层次
        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())  # 展平层
        for i in range(len(gate_channels) - 2):
            # 添加全连接层、批量归一化层和激活函数
            self.ca.add_module('fc%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module('bn%d' % i, nn.BatchNorm1d(gate_channels[i + 1]))
            self.ca.add_module('relu%d' % i, nn.ReLU())
        # 添加最后的全连接层，将通道恢复到原始大小
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, channels, height, width)
        # 先通过自适应平均池化，输出形状为 (batch_size, channels, 1, 1)
        res = self.avgpool(x)
        # 通过通道注意力网络，输出形状为 (batch_size, channels)
        res = self.ca(res)
        # 调整维度以匹配输入特征图的形状，并在空间维度上扩展
        res = res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res

# 定义空间注意力模块 (Spatial Attention)
class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3, dia_val=2):
        super().__init__()
        # 使用 nn.Sequential 定义空间注意力模块的层次
        self.sa = nn.Sequential()
        # 第一个 1x1 卷积层用于减少通道数
        self.sa.add_module('conv_reduce1',
                           nn.Conv2d(kernel_size=1, in_channels=channel, out_channels=channel // reduction))
        self.sa.add_module('bn_reduce1', nn.BatchNorm2d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())
        for i in range(num_layers):
            # 3x3 卷积层，使用扩张卷积，并附加批量归一化和激活函数
            self.sa.add_module('conv_%d' % i, nn.Conv2d(kernel_size=3, in_channels=channel // reduction,
                                                        out_channels=channel // reduction, 
                                                        padding=autopad(3, None, dia_val), dilation=dia_val))
            self.sa.add_module('bn_%d' % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.ReLU())
        # 最后的 1x1 卷积层将输出通道数减少为 1
        self.sa.add_module('last_conv', nn.Conv2d(channel // reduction, 1, kernel_size=1))

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, channels, height, width)
        # 通过空间注意力网络，输出形状为 (batch_size, 1, height, width)
        res = self.sa(x)
        # 扩展为与输入相同的形状
        res = res.expand_as(x)
        return res

# 定义 BAMBlock 模块，结合通道和空间注意力
class BAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, dia_val=2):
        super().__init__()
        # 初始化通道注意力和空间注意力模块
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(channel=channel, reduction=reduction, dia_val=dia_val)
        self.sigmoid = nn.Sigmoid()  # 使用 Sigmoid 激活函数来计算注意力权重

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 对卷积层使用 He 初始化
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为 0
            elif isinstance(m, nn.BatchNorm2d):  # 对批量归一化层初始化
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 对全连接层使用正态分布初始化
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, channels, height, width)
        b, c, _, _ = x.size()
        # 计算空间注意力输出
        sa_out = self.sa(x)
        # 计算通道注意力输出
        ca_out = self.ca(x)
        # 将空间和通道注意力相加并通过 Sigmoid 激活函数计算注意力权重
        weight = self.sigmoid(sa_out + ca_out)
        # 结合输入和注意力权重
        out = (1 + weight) * x
        return out

# 测试代码块
if __name__ == '__main__':
    # 创建 BAMBlock 实例，输入通道数为 512
    input = torch.randn(32, 512, 7, 7)  # 随机生成输入张量，形状为 (32, 512, 7, 7)
    bam = BAMBlock(channel=512, reduction=16, dia_val=2)
    output = bam(input)
    # 打印输出张量的形状，验证输出是否正确
    print(output.shape)


```


# 13.GAM(全局注意力机制)

## 介绍
在减少信息弥散的情况下也能放大全局维交互特征，采用序贯的通道-空间注意力机制并重新设计了CBAM子模块，能够捕捉所有三个维度(通道、空间宽度和空间高度)的显著特征的注意机制

## 代码实现

```python
import torch.nn as nn  # 从 PyTorch 导入神经网络模块，简称 nn
import torch  # 导入 PyTorch 库，用于深度学习和张量操作

# 定义 GAM_Attention 类，该类继承自 nn.Module
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention, self).__init__()
        # 初始化 GAM_Attention 模块
        # in_channels: 输入特征图的通道数
        # rate: 缩减比率，用于通道和空间注意力的通道缩减
        
        # 定义通道注意力模块 (Channel Attention)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),  # 线性层，将通道数缩减为 1/rate
            nn.ReLU(inplace=True),  # 激活函数，ReLU
            nn.Linear(int(in_channels / rate), in_channels)  # 线性层，将通道数恢复为原始通道数
        )
        
        # 定义空间注意力模块 (Spatial Attention)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),  # 卷积层，卷积核大小为 7x7，填充为 3
            nn.BatchNorm2d(int(in_channels / rate)),  # 批量归一化
            nn.ReLU(inplace=True),  # 激活函数，ReLU
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),  # 卷积层，将通道数恢复为原始通道数
            nn.BatchNorm2d(in_channels)  # 批量归一化
        )

    def forward(self, x):
        # 前向传播函数，定义了数据通过网络时的计算过程
        b, c, h, w = x.shape  # 获取输入 x 的形状 (batch_size, channels, height, width)
        
        # 计算通道注意力
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  # 将输入 x 的维度从 (b, c, h, w) 变换为 (b, h*w, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)  # 通过通道注意力模块，输出形状为 (b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()  # 变换回原始形状 (b, c, h, w)，并通过 Sigmoid 函数
        
        # 应用通道注意力
        x = x * x_channel_att  # 将原始输入 x 与通道注意力权重相乘
        
        # 计算空间注意力
        x_spatial_att = self.spatial_attention(x).sigmoid()  # 通过空间注意力模块，输出形状为 (b, c, h, w)，并通过 Sigmoid 函数
        out = x * x_spatial_att  # 将通道加权后的输入与空间注意力权重相乘
        
        return out  # 返回最终的注意力加权输出

# 测试代码块
if __name__ == '__main__':
    x = torch.randn(1,64,7,7)  # 生成一个随机张量，形状为 (1, 64, 7, 7)
    b, c, h, w = x.shape  # 获取输入张量的形状
    net = GAM_Attention(in_channels=c)  # 创建 GAM_Attention 实例，输入通道数为 64
    y = net(x)  # 通过 GAM_Attention 模块进行前向传播
    print(y.size())  # 打印输出张量的形状，验证输出是否正确


```


# 14.MobileVITattention(轻量级)

## 介绍
使用轻量级的注意力机制来提取特征，在保证较高精度的同时，具有更快的推理速度和更小的模型体积

## 代码实现

```python
from torch import nn  # 导入 PyTorch 的神经网络模块
import torch  # 导入 PyTorch
from einops import rearrange  # 从 einops 导入 rearrange 函数，用于重排列张量的维度

# 定义一个带有 LayerNorm 的预处理模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)  # 定义 LayerNorm 层
        self.fn = fn  # 传入的函数，可以是注意力或前馈网络

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)  # 对输入先进行 LayerNorm，再通过传入的函数处理

# 定义一个前馈神经网络（MLP）模块
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),  # 第一个线性变换，输入维度为 dim，输出维度为 mlp_dim
            nn.SiLU(),  # 使用 SiLU 激活函数
            nn.Dropout(dropout),  # 添加 Dropout 以防止过拟合
            nn.Linear(mlp_dim, dim),  # 第二个线性变换，输入维度为 mlp_dim，输出维度为 dim
            nn.Dropout(dropout)  # 再次添加 Dropout
        )

    def forward(self, x):
        return self.net(x)  # 通过顺序模块处理输入

# 定义一个注意力机制模块
class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        inner_dim = heads * head_dim  # 内部维度为头数乘以每个头的维度
        project_out = not (heads == 1 and head_dim == dim)  # 判断是否需要输出投影

        self.heads = heads  # 注意力头的数量
        self.scale = head_dim ** -0.5  # 缩放因子，用于缩放点积结果

        self.attend = nn.Softmax(dim=-1)  # 使用 Softmax 函数来计算注意力权重
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 线性变换，生成查询、键和值

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 输出投影，将内部维度变换回原始维度
            nn.Dropout(dropout)  # 添加 Dropout
        ) if project_out else nn.Identity()  # 如果不需要输出投影，则使用 Identity 模块

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 计算查询、键和值，并沿着最后一个维度分割为三部分
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)  # 重排列查询、键和值的维度
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 计算点积并进行缩放
        attn = self.attend(dots)  # 计算注意力权重
        out = torch.matmul(attn, v)  # 根据注意力权重加权求和值
        out = rearrange(out, 'b p h n d -> b p n (h d)')  # 重排列输出的维度
        return self.to_out(out)  # 进行输出投影并返回

# 定义一个 Transformer 模块
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # 使用 ModuleList 存储多层 Transformer
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, head_dim, dropout)),  # 添加带有 LayerNorm 的注意力层
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))  # 添加带有 LayerNorm 的前馈层
            ]))

    def forward(self, x):
        out = x  # 初始化输出为输入
        for att, ffn in self.layers:  # 遍历每一层
            out = out + att(out)  # 残差连接注意力层的输出
            out = out + ffn(out)  # 残差连接前馈层的输出
        return out  # 返回最终输出

# 定义 MobileViTAttention 模块
class MobileViTAttention(nn.Module):
    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=7):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size  # 设置 patch 的高和宽
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)  # 第一层卷积
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)  # 第二层卷积，用于通道数变换

        self.trans = Transformer(dim=dim, depth=3, heads=8, head_dim=64, mlp_dim=1024)  # Transformer 模块，进行全局特征的提取

        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)  # 第三层卷积，用于通道数还原
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)  # 第四层卷积，用于特征融合

    def forward(self, x):
        y = x.clone()  # 复制输入张量，保持原始数据不变

        ## 局部表示 Local Representation
        y = self.conv2(self.conv1(x))  # 通过前两层卷积提取局部特征

        ## 全局表示 Global Representation
        _, _, h, w = y.shape  # 获取特征图的形状
        # 重新排列特征图，将其拆分成多个 patch
        y = rearrange(y, 'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim', ph=self.ph, pw=self.pw)
        y = self.trans(y)  # 通过 Transformer 处理提取全局特征
        # 将特征图重排列回原来的形状
        y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)', ph=self.ph, pw=self.pw, nh=h // self.ph, nw=w // self.pw)

        ## 融合 Fusion
        y = self.conv3(y)  # 通过第三层卷积将通道数还原
        y = torch.cat([x, y], 1)  # 将原始输入和全局特征连接在一起
        y = self.conv4(y)  # 通过第四层卷积融合特征

        return y  # 返回最终输出

# 测试代码块
if __name__ == '__main__':
    m = MobileViTAttention(in_channel=512)  # 创建 MobileViTAttention 实例，输入通道数为 512
    input = torch.randn(1, 512, 49, 49)  # 生成一个随机张量，形状为 (1, 512, 49, 49)
    output = m(input)  # 通过 MobileViTAttention 模块进行前向传播
    print(output.shape)  # 打印输出张量的形状，验证输出是否正确


```

# 15.SIMAM(无参数注意力模块，ICML2021)

## 介绍
与现有的通道式和空间式注意力模块不同，该模块在不增加原网络参数的情况下，为每一层的特征图推导出3D注意力权重。具体来说，基于一些广为人知的神经科学理论，提出优化一个能量函数来找出每个神经元的重要性。

## 代码实现

```python
import torch  # 导入 PyTorch
import torch.nn as nn  # 从 PyTorch 导入神经网络模块

# 定义 SimAM 模块类
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activation = nn.Sigmoid()  # 使用 Sigmoid 激活函数
        self.e_lambda = e_lambda  # 设置正则化项的系数

    # 重写 __repr__ 方法，提供模块的可读表示
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)  # 显示 e_lambda 的值
        return s

    # 静态方法，返回模块的名称
    @staticmethod
    def get_module_name():
        return "simam"

    # 定义前向传播方法
    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入张量的形状

        n = w * h - 1  # 计算每个通道的像素数减一，用于归一化

        # 计算每个像素点减去均值的平方差
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # 计算 y，公式为 x 减去均值的平方差，除以方差的 4 倍加上正则项，然后加上 0.5
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        # 使用 Sigmoid 函数激活 y，并与输入 x 相乘
        return x * self.activation(y)

# 测试代码块
if __name__ == '__main__':
    input = torch.randn(3, 64, 7, 7)  # 生成一个随机输入张量，形状为 (3, 64, 7, 7)
    model = SimAM()  # 创建 SimAM 模块实例
    outputs = model(input)  # 通过 SimAM 模块进行前向传播
    print(outputs.shape)  # 打印输出张量的形状，验证输出是否正确


```


# 16.GCNet(全局上下文注意力机制)

## 介绍
GCNet 引入了全局上下文注意力机制，允许网络在处理图像时动态地调整不同位置的权重，以更好地捕捉全局信息。GCNet 包括一个全局上下文模块，用于从整个图像中提取全局信息。这个模块通常由全局平均池化层（Global Average Pooling）组成，用于将整个特征图压缩成一个全局特征向量。


## 代码实现

```python
import torch  # 导入 PyTorch
from torch import nn  # 从 PyTorch 导入神经网络模块

# 定义 ContextBlock 类，继承自 nn.Module
class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        
        # 验证 fusion_types 是否有效
        valid_fusion_types = ['channel_add', 'channel_mul']

        # 检查 pooling_type 是否在有效选项 ['avg', 'att'] 中
        assert pooling_type in ['avg', 'att']
        # 确认 fusion_types 是列表或元组
        assert isinstance(fusion_types, (list, tuple))
        # 确认 fusion_types 中的所有元素都在 valid_fusion_types 中
        assert all([f in valid_fusion_types for f in fusion_types])
        # 确保至少有一个 fusion 类型被指定
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes  # 输入通道数
        self.ratio = ratio  # 缩减比例
        self.planes = int(inplanes * ratio)  # 缩减后的通道数
        self.pooling_type = pooling_type  # 池化类型（'avg' 或 'att'）
        self.fusion_types = fusion_types  # 融合类型

        # 如果池化类型为 'att'，定义注意力池化的卷积层和 softmax
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)  # 在维度 2 上做 softmax
        else:
            # 如果池化类型为 'avg'，定义自适应平均池化层
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 定义 'channel_add' 融合类型的卷积层序列
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),  # 1x1 卷积
                nn.LayerNorm([self.planes, 1, 1]),  # 层归一化
                nn.ReLU(inplace=True),  # 激活函数 ReLU
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)  # 1x1 卷积，恢复到原通道数
            )
        else:
            self.channel_add_conv = None

        # 定义 'channel_mul' 融合类型的卷积层序列
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),  # 1x1 卷积
                nn.LayerNorm([self.planes, 1, 1]),  # 层归一化
                nn.ReLU(inplace=True),  # 激活函数 ReLU
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)  # 1x1 卷积，恢复到原通道数
            )
        else:
            self.channel_mul_conv = None

    # 定义空间池化方法
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()  # 获取输入张量的形状
        if self.pooling_type == 'att':  # 如果池化类型为 'att'
            input_x = x.view(batch, channel, height * width)  # 展平 H 和 W
            input_x = input_x.unsqueeze(1)  # 增加一个维度
            context_mask = self.conv_mask(x)  # 应用 1x1 卷积层
            context_mask = context_mask.view(batch, 1, height * width)  # 展平 H 和 W
            context_mask = self.softmax(context_mask)  # 在 H * W 上应用 softmax
            context_mask = context_mask.unsqueeze(-1)  # 增加一个维度
            context = torch.matmul(input_x, context_mask)  # 计算加权和
            context = context.view(batch, channel, 1, 1)  # 恢复形状
        else:
            context = self.avg_pool(x)  # 如果池化类型为 'avg'，直接应用平均池化

        return context  # 返回上下文张量

    # 定义前向传播方法
    def forward(self, x):
        context = self.spatial_pool(x)  # 获取上下文信息
        out = x  # 初始化输出
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))  # 应用通道乘法融合
            out = out * channel_mul_term  # 输出乘以融合结果

        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)  # 应用通道加法融合
            out = out + channel_add_term  # 输出加上融合结果

        return out  # 返回最终输出

# 测试代码块
if __name__ == "__main__":
    in_tensor = torch.ones((1, 64, 128, 128))  # 创建一个全1的输入张量，形状为 (1, 64, 128, 128)
    cb = ContextBlock(inplanes=64, ratio=0.25, pooling_type='att')  # 创建 ContextBlock 实例
    out_tensor = cb(in_tensor)  # 传递输入张量进行前向传播
    print(in_tensor.shape)  # 打印输入张量的形状
    print(out_tensor.shape)  # 打印输出张量的形状


```


# 17.ELA(2024最新高效的局部注意力机制)

## 介绍
一种轻量级且高效的局部注意力（ELA）模块。这个模块帮助深度CNN更准确地定位感兴趣的目标，在仅增加少量参数的情况下显著提高了CNN的整体性能。在包括ImageNet、MS COCO和Pascal VOC在内的流行数据集上的大量实验结果表明，提出的（ELA）模块在性能上超越了当前的最新注意力方法，同时保持了有竞争力的模型复杂度。

## 代码实现

```python
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

```

# 18.Biformer(CVPR2023双层路由注意力机制)

## 介绍
通过双层路由提出了一种新颖的动态稀疏注意力，以实现更灵活的计算分配和内容感知，使其具备动态的查询感知稀疏性。关键思想为在粗糙区域级别过滤掉大部分不相关的键值对，以便只保留一小部分路由区域，然后在路由区域的联合中应用细粒度的token-to-token注意力。具有良好的性能和高计算效率，在图像分类、目标检测和语义分割等多项计算机视觉任务都证明了其有效性。


## 代码实现

```python
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)
    
    def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key) # per-window pooling -> (n, p^2, c) 
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1) # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1) # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit) # (n, p^2, k)
        
        return r_weight, topk_index
        

class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel? 
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                dim=2,
                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv) # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv

class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)
    
    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v

class BiLevelRoutingAttention(nn.Module):
    """
    BiLevelRoutingAttention 类的定义，用于实现分层路由注意力机制。
    参数：
    - dim: 输入张量的通道数。
    - num_heads: 注意力头的数量。
    - n_win: 每侧窗口的数量（总的窗口数量是 n_win*n_win）。
    - qk_dim: q 和 k 的维度。
    - qk_scale: qk 的缩放因子。
    - kv_per_win: kv_downsample_mode='ada_xxxpool' 时，每个窗口的 key/value 数量。类似于 n_win，总数量是 kv_per_win*kv_per_win。
    - kv_downsample_ratio: key/value 的下采样比例。
    - kv_downsample_kernel: key/value 下采样的卷积核大小。
    - kv_downsample_mode: key/value 下采样模式。
    - topk: 窗口过滤的 topk 值。
    - param_attention: 'qkvo' 表示 q, k, v 和 o 的线性层；'none' 表示无参数注意力。
    - param_routing: 路由的额外线性层。
    - diff_routing: 是否设置路由为可微分。
    - soft_routing: 是否使用软路由权重。
    - side_dwconv: 边深度卷积核大小。
    - auto_pad: 是否自动填充。
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=True):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads==0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5


        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        
        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing) # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing: # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing: # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')
        
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity': # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v 
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad=auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
        x = x.permute(0, 2, 3, 1) #  维度：N C H W -> N H W C 
         # NOTE: use padding for semantic segmentation
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, # dim=-1
                          pad_l, pad_r, # dim=-2
                          pad_t, pad_b)) # dim=-3
            _, H, W, _ = x.size() # padded size
        else:
            N, H, W, C = x.size()
            assert H%self.n_win == 0 and W%self.n_win == 0 #
        ###################################################


        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x) 

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3]) # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win) # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)
        
        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads) # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (q_pix * self.scale) @ k_pix_sel # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H//self.n_win, w=W//self.n_win)

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out.permute(0, 3, 1, 2) # N H W C -> N C H W
        
# 调试部分
if __name__ == "__main__":
    x = torch.randn(4, 64, 32, 32)  # 一个示例输入，形状为 (batch_size, channels，height, width)
    bilevel_attention = BiLevelRoutingAttention(64)
    out = bilevel_attention(x)
    print(out.shape)  # 应该输出 torch.Size([4, 64, 32, 32])

```

# 19.CAA(CVPR2024最新上下文锚点注意力机制)

## 介绍
CAA模块通过引入特定的上下文锚点来计算注意力权重。与传统的注意力机制不同，这里使用了水平和垂直方向的分离卷积操作来捕捉局部特征，通过这种方式，它可以更好地关注输入特征图的不同部分。这种方法能够增强网络对特征图中关键区域的关注，有助于提升模型在复杂任务中的表现，同时减少计算量和参数量。

## 代码实现

```python
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


```

# 20.Agent Attention(ECCV2024最新代理注意力机制)

## 介绍
Agent注意力机制，softmax注意力与线性注意力的完美融合,提出了一种新颖的注意力范式，称为代理注意力（Agent Attention），在传统的注意力三元组 (Q,K,V) 中引入了一组额外的代理向量 A，定义了一种新的四元注意力机制 (Q, A, K, V)。其中，代理向量 A 首先作为查询向量 Q 的代理，从 K 和 V 中聚合信息，然后将信息广播回 Q。由于代理向量的数量可以设计得比查询向量的数量小得多，代理注意力能够以很低的计算成本实现全局信息的建模。

## 代码实现

```python
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

# 定义了一个自定义的注意力机制模块，名为AgentAttention，它基于窗口的多头自注意力，支持平移和非平移窗口操作
class AgentAttention(nn.Module):
    r""" 基于窗口的多头自注意力(W-MSA)模块，带有相对位置偏置
    支持平移和非平移窗口操作。

    参数说明:
        dim (int): 输入通道数
        num_heads (int): 注意力头的数量
        qkv_bias (bool, optional): 如果为True，添加可学习的偏置到query, key, value中，默认值为True
        qk_scale (float | None, optional): 如果设置，则覆盖默认的缩放比例 head_dim ** -0.5
        attn_drop (float, optional): 注意力权重的dropout比率，默认值为0.0
        proj_drop (float, optional): 输出的dropout比率，默认值为0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 shift_size=0, agent_num=49, **kwargs):
        super().__init__()  # 调用父类的初始化方法
        self.dim = dim  # 输入通道数
        self.window_size = window_size  # 窗口的高和宽
        self.num_heads = num_heads  # 注意力头的数量
        head_dim = dim // num_heads  # 每个注意力头处理的通道数
        self.scale = head_dim ** -0.5  # 缩放比例
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 线性变换生成query、key、value
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力dropout
        self.proj = nn.Linear(dim, dim)  # 输出的线性投影层
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的dropout
        self.softmax = nn.Softmax(dim=-1)  # softmax用于计算注意力分布
        self.shift_size = shift_size  # 窗口平移大小

        self.agent_num = agent_num  # Agent数量
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)  # 深度可分离卷积
        # 初始化多个相对位置偏置参数
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))  # agent到窗口的偏置
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))  # 窗口到agent的偏置
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))  # 高度的agent偏置
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))  # 宽度的agent偏置
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))  # 窗口高度到agent的偏置
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))  # 窗口宽度到agent的偏置
        # 对位置偏置进行初始化
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        # 自适应池化，用于调整输入到agent的大小
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    # 前向传播逻辑，输入x形状为(num_windows*B, N, C)
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入特征，形状为 (num_windows*B, N, C)
            mask: 掩码 (0/-inf)，形状为 (num_windows, Wh*Ww, Wh*Ww)，可以为空
        """
        # 获取输入x的batch大小(b)，token数量(n)和通道数(c)
        # b, n, c = x.shape
        # h = int(n ** 0.5) # 计算窗口的高度
        # w = int(n ** 0.5)  # 计算窗口的宽度

        # 若输入为三维张量，将以下五行注释
        b,c,h,w = x.shape
        h = int(h)
        w = int(w)
        n = h*w
        x = x.view(b,h*w,c)

        num_heads = self.num_heads  # 获取注意力头的数量
        head_dim = c // num_heads  # 每个注意力头处理的通道数
        # 线性变换生成query, key, value，形状为(b, n, 3, c)
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)  
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别获取q, k, v
        # q, k, v: b, n, c
        # 对q进行池化操作，生成agent tokens，并调整形状
        agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        # 将q, k, v调整为多头注意力的形状
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        # 计算位置偏置并插值调整大小
        position_bias1 = nn.functional.interpolate(self.an_bias, size=self.window_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        # 计算agent注意力分布
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn) # 添加dropout
        agent_v = agent_attn @ v # 获取agent经过注意力后的输出
        # 计算agent到query的偏置，并进行插值调整
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        # 计算q的注意力分布
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v
        # print(x.shape)
        x = x.transpose(1, 2).reshape(b, n, c)
        # 将v重塑为卷积输入的形状
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # 使用深度可分离卷积更新x
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)
        # print(x.shape)  # torch.Size([4, 1024, 64])
        # x = x.permute(0,2,1).view(b,c,h,w)
        # 线性变换生成最终输出
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(x.shape) # torch.Size([4, 1024, 64])
        x = x.permute(0,2,1).view(b,c,h,w) # 若输入为三维张量，将此行注释
        return x # 返回最终的输出
    

# 测试代码块
if __name__ == "__main__":  # 如果此脚本作为主程序运行，则执行以下代码。
    # x = torch.randn(4,32*32,64) # batch为4 h*w=32*32 channel 64
    # agent = AgentAttention(64,[32,32],8)

    x = torch.randn(4,64,32,32)  # 生成一个随机张量作为输入，形状为(4, 64, 32, 32)。
    agent = AgentAttention(64,[32,32],8)  # 创建agent模块实例，输入通道数为64。

    out = agent(x)  # 将输入张量通过agent模块。
    print(out.shape)  # 打印输出张量的形状，应该为torch.Size([4, 64, 32, 32])。

```

# 21.SLA Attention(ICML2024最新线性注意力机制)

## 介绍
简化版本的线性注意力机制模块，该模块利用ReLU作为核函数，结合深度可分卷积来进行局部特征增强，这种注意力机制比之前的线性注意力更高效。在图像分类、目标检测和语言建模等任务中提供与现有模型相当的性能，同时降低延迟，提高推理速度，更适合实时应用

## 代码实现

```python
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

# 定义了一个自定义的注意力机制模块，名为AgentAttention，它基于窗口的多头自注意力，支持平移和非平移窗口操作
class AgentAttention(nn.Module):
    r""" 基于窗口的多头自注意力(W-MSA)模块，带有相对位置偏置
    支持平移和非平移窗口操作。

    参数说明:
        dim (int): 输入通道数
        num_heads (int): 注意力头的数量
        qkv_bias (bool, optional): 如果为True，添加可学习的偏置到query, key, value中，默认值为True
        qk_scale (float | None, optional): 如果设置，则覆盖默认的缩放比例 head_dim ** -0.5
        attn_drop (float, optional): 注意力权重的dropout比率，默认值为0.0
        proj_drop (float, optional): 输出的dropout比率，默认值为0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 shift_size=0, agent_num=49, **kwargs):
        super().__init__()  # 调用父类的初始化方法
        self.dim = dim  # 输入通道数
        self.window_size = window_size  # 窗口的高和宽
        self.num_heads = num_heads  # 注意力头的数量
        head_dim = dim // num_heads  # 每个注意力头处理的通道数
        self.scale = head_dim ** -0.5  # 缩放比例
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 线性变换生成query、key、value
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力dropout
        self.proj = nn.Linear(dim, dim)  # 输出的线性投影层
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的dropout
        self.softmax = nn.Softmax(dim=-1)  # softmax用于计算注意力分布
        self.shift_size = shift_size  # 窗口平移大小

        self.agent_num = agent_num  # Agent数量
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)  # 深度可分离卷积
        # 初始化多个相对位置偏置参数
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))  # agent到窗口的偏置
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))  # 窗口到agent的偏置
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))  # 高度的agent偏置
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))  # 宽度的agent偏置
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))  # 窗口高度到agent的偏置
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))  # 窗口宽度到agent的偏置
        # 对位置偏置进行初始化
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        # 自适应池化，用于调整输入到agent的大小
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    # 前向传播逻辑，输入x形状为(num_windows*B, N, C)
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入特征，形状为 (num_windows*B, N, C)
            mask: 掩码 (0/-inf)，形状为 (num_windows, Wh*Ww, Wh*Ww)，可以为空
        """
        # 获取输入x的batch大小(b)，token数量(n)和通道数(c)
        # b, n, c = x.shape
        # h = int(n ** 0.5) # 计算窗口的高度
        # w = int(n ** 0.5)  # 计算窗口的宽度

        # 若输入为三维张量，将以下五行注释
        b,c,h,w = x.shape
        h = int(h)
        w = int(w)
        n = h*w
        x = x.view(b,h*w,c)

        num_heads = self.num_heads  # 获取注意力头的数量
        head_dim = c // num_heads  # 每个注意力头处理的通道数
        # 线性变换生成query, key, value，形状为(b, n, 3, c)
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)  
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别获取q, k, v
        # q, k, v: b, n, c
        # 对q进行池化操作，生成agent tokens，并调整形状
        agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        # 将q, k, v调整为多头注意力的形状
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        # 计算位置偏置并插值调整大小
        position_bias1 = nn.functional.interpolate(self.an_bias, size=self.window_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        # 计算agent注意力分布
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn) # 添加dropout
        agent_v = agent_attn @ v # 获取agent经过注意力后的输出
        # 计算agent到query的偏置，并进行插值调整
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        # 计算q的注意力分布
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v
        # print(x.shape)
        x = x.transpose(1, 2).reshape(b, n, c)
        # 将v重塑为卷积输入的形状
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # 使用深度可分离卷积更新x
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)
        # print(x.shape)  # torch.Size([4, 1024, 64])
        # x = x.permute(0,2,1).view(b,c,h,w)
        # 线性变换生成最终输出
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(x.shape) # torch.Size([4, 1024, 64])
        x = x.permute(0,2,1).view(b,c,h,w) # 若输入为三维张量，将此行注释
        return x # 返回最终的输出
    

# 测试代码块
if __name__ == "__main__":  # 如果此脚本作为主程序运行，则执行以下代码。
    # x = torch.randn(4,32*32,64) # batch为4 h*w=32*32 channel 64
    # agent = AgentAttention(64,[32,32],8)

    x = torch.randn(4,64,32,32)  # 生成一个随机张量作为输入，形状为(4, 64, 32, 32)。
    agent = AgentAttention(64,[32,32],8)  # 创建agent模块实例，输入通道数为64。

    out = agent(x)  # 将输入张量通过agent模块。
    print(out.shape)  # 打印输出张量的形状，应该为torch.Size([4, 64, 32, 32])。

```

# 22.MCA Attention(2023多维协同注意力模块)

## 介绍
通过三个平行分支同时建模通道、高度和宽度维度的注意力、实现多维度的协同注意力；引入了挤压变换和激励变换组件，通过自适应的方式聚合特征和捕捉局部特征交互，提高了网络的表征能力；轻量化的模块，在提升准确率和效率的同时几乎没有额外的计算负担。

## 代码实现

```python

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


```

# 23.ASPP(空洞空间金字塔池化 (ASPP) 模块)
```python
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
```

# 24.DSConv(深度可分离卷积)
```python
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


```

# 25.MALA(ICCV2025幅值感知注意力机制)
Magnitude-Aware Linear Attention（MALA，能感知 Query 向量大小的线性注意力）是对线性注意力的改进。它通过引入 Query 的大小信息，使注意力分布随 Query 放大或缩小动态变化，更接近 Softmax Attention 的行为(即重要信息得到更多关注，不重要信息被弱化)，同时保持线性复杂度。MALA 提升了局部感知能力和全局建模效果，让线性注意力既快速又精准，在视觉、NLP、语音和生成任务中的表现均很好，是当前线性注意力的标杆方法。

```python
import torch
import torch.nn as nn
from einops import rearrange

# ==================== RoPE(旋转位置编码增强空间信息) 工具函数 ====================
def rotate_every_two(x): 
    """
    对最后一维两两旋转，用于 RoPE
    x: (..., d)
    返回: (..., d)
    """
    x1 = x[:, :, :, ::2]        # 取偶数索引维度
    x2 = x[:, :, :, 1::2]       # 取奇数索引维度
    x = torch.stack([-x2, x1], dim=-1)  # 两两组合成最后一维
    return x.flatten(-2)         # 将最后两维 flatten 回原来的 d

def theta_shift(x, sin, cos):
    """
    应用旋转位置编码 RoPE
    x: (num_heads, B, seq_len, head_dim)
    sin, cos: (1,1,seq_len,head_dim) 可广播
    """
    return (x * cos) + (rotate_every_two(x) * sin)

def build_rope_table(seq_len, dim, device=None):
    """
    构建 RoPE 的 sin/cos 表
    seq_len: 序列长度 H*W
    dim: head_dim
    返回: sin, cos shape (1,1,seq_len,head_dim)
    """
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # (L,1) 位置索引
    dim_t = torch.arange(dim // 2, dtype=torch.float32, device=device)                  # (d/2)
    inv_freq = 1.0 / (10000 ** (dim_t / (dim // 2)))                                   # 计算衰减因子
    freqs = position * inv_freq.unsqueeze(0)                                           # (L, d/2) 每个位置每个维度的频率

    sin = torch.sin(freqs)   # 正弦
    cos = torch.cos(freqs)   # 余弦

    # 重复一遍扩展到 head_dim，和 rotate_every_two 输出一致
    sin = sin.repeat(1, 2)
    cos = cos.repeat(1, 2)

    # 扩展 batch/head 维度，方便广播
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1,1,L,d)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,L,d)
    return sin, cos


# ==================== MALAAttention ====================
class MALAAttention(nn.Module):
    def __init__(self, dim, num_heads):
        """
        dim: 输入通道数
        num_heads: 注意力头数
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个 head 的维度

        # Q,K,V,O 投影卷积 (1x1卷积实现线性变换)
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)

        # LEPE: 局部位置编码卷积 (depthwise conv)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)

        # 输出线性投影
        self.proj = nn.Conv2d(dim, dim, 1)

        # 缩放因子
        self.scale = self.head_dim ** -0.5
        self.elu = nn.ELU()  # 激活函数 ELU

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 1x1 卷积生成 Q,K,V,O
        qkvo = self.qkvo(x)   # (B, 4C, H, W)

        # 拆分前 3C 为 Q,K,V
        qkv = qkvo[:, :3*self.dim, :, :]
        # 剩下的 C 作为 O
        o   = qkvo[:, 3*self.dim:, :, :]

        # 局部位置编码 (只作用在 V)
        lepe = self.lepe(qkv[:, 2*self.dim:, :, :])   # (B, C, H, W)

        # 拆分 Q,K,V 成多个 head
        # rearrange: (B, 3C, H*W) -> (3, B, num_heads, H*W, head_dim)
        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d',
                            m=3, n=self.num_heads)

        # 激活函数 +1 保证非负
        q = self.elu(q) + 1
        k = self.elu(k) + 1

        # MALA 幅值修正：q @ k.mean 保留 Query 的幅值信息
        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        # ========== RoPE 位置编码 ==========
        seq_len = H * W
        sin, cos = build_rope_table(seq_len, q.shape[-1], device=q.device)  # shape (1,1,L,d)
        q = theta_shift(q, sin, cos)  # Q + RoPE
        k = theta_shift(k, sin, cos)  # K + RoPE

        # 线性注意力核化计算
        kv = (k.transpose(-2, -1) * (self.scale / (H*W))**0.5) @ \
             (v * (self.scale / (H*W))**0.5)

        # MALA 注意力公式修正
        res = q @ kv * (1 + 1/(z + 1e-6)) - z * v.mean(dim=2, keepdim=True)

        # 还原回 (B, C, H, W)
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)

        # 加上 LEPE
        res = res + lepe

        # 输出投影
        return self.proj(res * o)


if __name__ == "__main__":
    x = torch.randn(4, 64, 32, 32)  # batch=4, 通道=64, 尺寸32x32
    mala = MALAAttention(dim=64, num_heads=8)

    out = mala(x)
    print("输出 shape:", out.shape)  # torch.Size([4, 64, 32, 32])

```


# 26.EPA(IEEE TMI-2024高效成对注意力机制)
Efficient Paired Attention（EPA）是 UNETR++ 提出的核心模块，专为 3D 医学图像分割设计。它通过“空间注意力 + 通道注意力”双分支结构，同时建模空间依赖和通道依赖关系，并采用线性化空间注意力降低计算复杂度，使整体复杂度接近 O(N)。
EPA 的关键创新在于：共享 Query 和 Key 权重（Paired 机制），增强两个分支的信息协同，同时减少参数量；将传统 N×N 的空间自注意力改为 N×P 的线性投影形式，大幅降低 FLOPs；并行计算空间与通道关系，再进行融合，提高特征表达能力

相比传统 Transformer 自注意力，EPA 在保证精度的同时显著减少参数和计算量，特别适合 token 数量巨大的任务。
它实现了：更低的显存占用、更快的推理速度、更高的分割精度
是目前 3D 医学图像中兼顾效率与性能的代表性注意力设计之一。

```python
class EPA(nn.Module):
    """
    Efficient Paired Attention (EPA) 模块
    来自论文：
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation

    核心思想：
    1. 共享 Q 和 K（Paired Attention）
    2. 同时做 通道注意力 + 空间注意力
    3. 空间注意力通过投影降低复杂度（避免 N×N）
    """

    def __init__(
        self,
        input_size,     # token 数量 N = H×W×D
        hidden_size,    # 特征维度 C
        proj_size,      # 空间注意力投影维度 P（P << N）
        num_heads=4,
        qkv_bias=False,
        channel_attn_drop=0.1,
        spatial_attn_drop=0.1
    ):
        super().__init__()

        self.num_heads = num_heads

        # 每个 head 的缩放因子（类似 scaled dot-product attention）
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 一次线性层生成 4 组向量：
        # Q_shared, K_shared, V_channel, V_spatial
        # 输出维度 = 4 × hidden_size
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # 空间注意力里的投影矩阵
        # 把 N 维投影到 P 维
        # 目的是降低空间注意力复杂度
        self.E = self.F = nn.Linear(input_size, proj_size)

        # dropout
        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        # 输出投影（分别处理 SA 和 CA）
        self.out_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.out_proj2 = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x):
        """
        x: (B, N, C)
        B = batch size
        N = token 数（H×W×D）
        C = 通道数
        """

        B, N, C = x.shape

        # 生成 Q, K, V_channel, V_spatial
        # (B, N, 4C)
        qkvv = self.qkvv(x)

        # reshape 成多头形式
        # (B, N, 4, heads, C_head)
        qkvv = qkvv.reshape(B, N, 4, self.num_heads, C // self.num_heads)

        # 调整维度
        # (4, B, heads, N, C_head)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        # 拆分
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        # 转置，方便后续矩阵乘法
        # (B, heads, C_head, N)
        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        # --------------------------
        #  通道注意力 Channel Attention
        # --------------------------

        # 归一化（提高稳定性）
        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        # 计算通道相关性
        # (C_head × N) × (N × C_head)
        # → (C_head × C_head)
        # 注意：这是 C×C，不是 N×N
        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        # 加权求和
        x_CA = (attn_CA @ v_CA)

        # 恢复形状
        # (B, N, C)
        x_CA = x_CA.permute(0, 3, 1, 2).reshape(B, N, C)

        # --------------------------
        #  空间注意力 Spatial Attention
        # --------------------------

        # 关键步骤：投影压缩 N 维
        # 把 k_shared 从 N 维 → P 维
        k_shared_projected = self.E(k_shared)

        # 把 v_SA 从 N 维 → P 维
        v_SA_projected = self.F(v_SA)

        # 计算空间注意力
        # (N × C) × (C × P)
        # → (N × P)
        attn_SA = (
            q_shared.permute(0, 1, 3, 2) @ k_shared_projected
        ) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        # 加权求和
        x_SA = attn_SA @ v_SA_projected.transpose(-2, -1)

        # 恢复形状
        x_SA = x_SA.permute(0, 3, 1, 2).reshape(B, N, C)

        # --------------------------
        #  融合
        # --------------------------

        # 分别降维
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)

        # 拼接回原始维度
        x = torch.cat((x_SA, x_CA), dim=-1)

        return x
```

# 本文档由B站up主DT算法工程师前钰所出
附详细讲解链接：https://www.bilibili.com/video/BV18s421M7pZ/?spm_id_from=333.999.0.0
