"""21. SLA Attention（当前文件不可直接使用）

合集原文标题是 SLA，但代码整段复制自 Agent Attention，类名仍是 AgentAttention。
筛选模块时跳过本文件，等补上真正的 SLA 实现后再启用。

来源：B站 DT算法工程师前钰 模块合集，已拆入零件库便于按需引用。
"""

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
