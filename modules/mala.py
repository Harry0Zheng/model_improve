"""25. MALA(幅值感知线性注意力)

对线性注意力的改进：让注意力分布随 Query 幅度变化。依赖 einops。
插入前确认 dim 能被 num_heads 整除，并做 Forward Test。

来源：B站 DT算法工程师前钰 模块合集，已拆入零件库便于按需引用。
"""

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
