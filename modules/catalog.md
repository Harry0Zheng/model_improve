# 候选模块索引

先读本表，再打开对应 `.py`。不要一次加载全部实现。

来源说明：初始 26 个条目拆自 B 站「DT算法工程师前钰」合集，只作为实现零件，不构成研究贡献。后续自增模块在表末追加。

| id | 文件 | 类型 | 同shape | 维数 | 依赖 | 优先插入 | 状态 | 备注 |
|----|------|------|---------|------|------|----------|------|------|
| se | [se.py](se.py) | Channel | 是 | 2D | torch | Encoder 各 stage / 残差块后 | 可用 | 与 ECA 同类，不要叠两个纯通道注意力 |
| eca | [eca.py](eca.py) | Channel | 是 | 2D | torch | 同上 | 可用 | 比 SE 更轻；`k_size` 需按通道数考虑 |
| cbam | [cbam.py](cbam.py) | Channel+Spatial | 是 | 2D | torch | Encoder / Skip | 可用 | 与 SE+空间门控功能重叠 |
| kan | [kan.py](kan.py) | MLP替代 | 视接口 | 2D/向量 | torch | 分类头 / 小 MLP | 需核对 | 不是注意力；体积大，不适合当即插头 |
| kan_conv | [kan_conv.py](kan_conv.py) | Conv替代 | 视接口 | 2D | torch | 替换少量 Conv | 需核对 | 计算和实现都重 |
| ema | [ema.py](ema.py) | Multi-scale+Channel | 是 | 2D | torch | 深层 Skip / Bottleneck | 可用 | 常用于小目标；先确认分组能整除通道 |
| ca | [ca.py](ca.py) | Channel+位置 | 是 | 2D | torch | 轻量骨干 | 可用 | 沿 H/W 编码，适合位置敏感任务 |
| lsk | [lsk.py](lsk.py) | Spatial+多核 | 需核对 | 2D | torch | 遥感检测骨干 | 需核对 | 核选择会改特征通路，插入前对一下 Shape |
| sk | [sk.py](sk.py) | 自适应感受野 | 需核对 | 2D | torch | Encoder | 需核对 | 与 LSK 同类思路，不要同时加 |
| mlca | [mlca.py](mlca.py) | Channel+Spatial | 需核对 | 2D | torch | 检测/分割骨干 | 需核对 | |
| a2_attention | [a2_attention.py](a2_attention.py) | Context | 需核对 | 2D | torch | 深层 | 需核对 | 双注意力，成本高于 SE/ECA |
| bam | [bam.py](bam.py) | Channel+Spatial | 需核对 | 2D | torch | 下采样瓶颈 | 需核对 | 原论文放 bottleneck |
| gam | [gam.py](gam.py) | Channel+Spatial | 是 | 2D | torch | 中深层 | 需核对 | 三维重排，显存敏感 |
| mobilevit_attention | [mobilevit_attention.py](mobilevit_attention.py) | Token-attn | 需核对 | 2D | torch | 轻量骨干中段 | 需核对 | 不是 NCHW 即插头 |
| simam | [simam.py](simam.py) | 3D能量注意力 | 是 | 2D | torch | 任意卷积后 | 可用 | 无额外参数，适合做便宜对照 |
| gcnet | [gcnet.py](gcnet.py) | Context | 是 | 2D | torch | 深层 | 可用 | 全局上下文，和 SE 部分重叠 |
| ela | [ela.py](ela.py) | Spatial局部 | 是 | 2D | torch | Encoder | 需核对 | |
| biformer | [biformer.py](biformer.py) | Token-attn | 否 | 2D | torch | 主干替换 | 需核对 | 重，不适合「加一块就走」 |
| caa | [caa.py](caa.py) | Spatial | 需核对 | 2D | mmcv | 局部增强 | 需核对 | 无 mmcv 则不要选 |
| agent_attention | [agent_attention.py](agent_attention.py) | Token-attn | 否 | 2D窗口 | timm | 窗口 Transformer | 需核对 | 依赖窗口划分，不是卷积即插头 |
| sla_attention | [sla_attention.py](sla_attention.py) | — | — | — | — | — | 不可用 | 正文是 Agent Attention 的复制，缺 SLA 实现 |
| mca | [mca.py](mca.py) | Channel+H+W | 需核对 | 2D | torch | 轻量骨干 | 需核对 | |
| aspp | [aspp.py](aspp.py) | Multi-scale | 否（通道可变） | 2D | torch | Bottleneck | 可用 | 分辨率保持；须指定 out_channels；注意膨胀率 |
| dsconv | [dsconv.py](dsconv.py) | Lightweight | 通道可变 | 2D | torch | 替换 Conv | 可用 | 轻量化，不是涨点注意力 |
| mala | [mala.py](mala.py) | Token-attn | 需核对 | 2D | einops | 注意力骨干 | 需核对 | 缺 einops 则不要选 |
| epa | [epa.py](epa.py) | Channel+Spatial | 否 | 3D token | torch | 3D 医学 Transformer 块 | 需核对 | 要 `N=H*W*D` 与 `hidden_size`，不是 NCHW 即插头 |

## 按问题的速查

| 现象 | 优先看 |
|------|--------|
| 小目标 / 多尺度不稳定 | ema, aspp, lsk, sk |
| 通道建模不足、想最小改动 | se, eca, simam |
| 边界 / 位置 | ca, ela, cbam |
| 全局上下文不够 | gcnet, a2_attention |
| 参数量和速度 | dsconv, simam, eca |
| 3D 医学 | epa（先核对 token 接口） |
| 纯凑模块、接口不清 | sla_attention、未标「可用」的条目 |

筛选时默认只从「可用」里挑；「需核对」必须先做 Forward 并写清接口再进入实验。
