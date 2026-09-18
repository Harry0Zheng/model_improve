"""26. EPA(IEEE TMI-2024高效成对注意力机制)

Efficient Paired Attention（EPA）是 UNETR++ 提出的核心模块，专为 3D 医学图像分割设计。
接口按 token 序列工作：input_size = H*W*D，hidden_size = C。不是即插即用的 NCHW 注意力。

来源：B站 DT算法工程师前钰 模块合集，已拆入零件库便于按需引用。
"""

import torch
import torch.nn as nn


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
