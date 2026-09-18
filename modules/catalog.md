# 候选模块索引

先读本表，再只打开本地 `e:\Codex模型改进\模块.md` 里对应的那一节。不要整份零件库读进上下文，也不要把该文件放进本仓库。

实现以那份 Markdown 为准。改实验代码时，按该节写成项目里的独立 `nn.Module`。

初始 26 条来自 B 站合集，只作零件，不构成研究贡献。后续自增模块追加在 `模块.md` 文末，并在本表加一行。

| id | 节 | 类型 | 同shape | 维数 | 依赖 | 优先插入 | 状态 | 备注 |
|----|----|------|---------|------|------|----------|------|------|
| se | 1 | Channel | 是 | 2D | torch | Encoder 各 stage / 残差块后 | 可用 | 与 ECA 同类，不要叠两个纯通道注意力 |
| eca | 2 | Channel | 是 | 2D | torch | 同上 | 可用 | 比 SE 更轻；`k_size` 需按通道数考虑 |
| cbam | 3 | Channel+Spatial | 是 | 2D | torch | Encoder / Skip | 可用 | 与 SE+空间门控功能重叠 |
| kan | 4 | MLP替代 | 视接口 | 2D/向量 | torch | 分类头 / 小 MLP | 需核对 | 不是注意力；体积大，不适合当即插头 |
| kan_conv | 5 | Conv替代 | 视接口 | 2D | torch | 替换少量 Conv | 需核对 | 计算和实现都重 |
| ema | 6 | Multi-scale+Channel | 是 | 2D | torch | 深层 Skip / Bottleneck | 可用 | 常用于小目标；先确认分组能整除通道 |
| ca | 7 | Channel+位置 | 是 | 2D | torch | 轻量骨干 | 可用 | 沿 H/W 编码，适合位置敏感任务 |
| lsk | 8 | Spatial+多核 | 需核对 | 2D | torch | 遥感检测骨干 | 需核对 | 核选择会改特征通路，插入前对一下 Shape |
| sk | 9 | 自适应感受野 | 需核对 | 2D | torch | Encoder | 需核对 | 与 LSK 同类思路，不要同时加 |
| mlca | 10 | Channel+Spatial | 需核对 | 2D | torch | 检测/分割骨干 | 需核对 | |
| a2_attention | 11 | Context | 需核对 | 2D | torch | 深层 | 需核对 | 双注意力，成本高于 SE/ECA |
| bam | 12 | Channel+Spatial | 需核对 | 2D | torch | 下采样瓶颈 | 需核对 | 原论文放 bottleneck |
| gam | 13 | Channel+Spatial | 是 | 2D | torch | 中深层 | 需核对 | 三维重排，显存敏感 |
| mobilevit_attention | 14 | Token-attn | 需核对 | 2D | torch | 轻量骨干中段 | 需核对 | 不是 NCHW 即插头 |
| simam | 15 | 3D能量注意力 | 是 | 2D | torch | 任意卷积后 | 可用 | 无额外参数，适合做便宜对照 |
| gcnet | 16 | Context | 是 | 2D | torch | 深层 | 可用 | 全局上下文，和 SE 部分重叠 |
| ela | 17 | Spatial局部 | 是 | 2D | torch | Encoder | 需核对 | |
| biformer | 18 | Token-attn | 否 | 2D | torch | 主干替换 | 需核对 | 重，不适合「加一块就走」 |
| caa | 19 | Spatial | 需核对 | 2D | mmcv | 局部增强 | 需核对 | 无 mmcv 则不要选 |
| agent_attention | 20 | Token-attn | 否 | 2D窗口 | timm | 窗口 Transformer | 需核对 | 依赖窗口划分，不是卷积即插头 |
| sla_attention | 21 | — | — | — | — | — | 不可用 | 标题是 SLA，正文仍是 Agent Attention 复制 |
| mca | 22 | Channel+H+W | 需核对 | 2D | torch | 轻量骨干 | 需核对 | |
| aspp | 23 | Multi-scale | 否（通道可变） | 2D | torch | Bottleneck | 可用 | 分辨率保持；须指定 out_channels；注意膨胀率 |
| dsconv | 24 | Lightweight | 通道可变 | 2D | torch | 替换 Conv | 可用 | 轻量化，不是涨点注意力 |
| mala | 25 | Token-attn | 需核对 | 2D | einops | 注意力骨干 | 需核对 | 缺 einops 则不要选 |
| epa | 26 | Channel+Spatial | 否 | 3D token | torch | 3D 医学 Transformer 块 | 需核对 | 要 `N=H*W*D` 与 `hidden_size`，不是 NCHW 即插头 |

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

筛选时默认只从「可用」里挑；「需核对」必须先对照 `模块.md` 该节做 Forward，并写清接口再进入实验。
