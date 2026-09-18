# 分阶段工作流

一次只推进当前阶段。不要在分析轮改代码。

## 1. 项目整体分析

找出模型定义、训练入口、Dataset、Loss、Optimizer、Scheduler、Metrics、配置文件。梳理输入到输出的数据流。标出以后最可能改的文件。本轮不改代码。

## 2. Baseline 结构

按 Encoder / Bottleneck / Decoder / Skip / Head 说明各 Stage 的 Shape、Channel、分辨率。标出已有 Attention / 多尺度 / 融合。画出简化数据流。本轮不改代码。

## 3. 问题诊断

用户给出的现象（小目标、边界、假阳性、参数量、速度、融合不足等）映射到结构位置。区分「结构问题」和「数据 / Loss / 训练问题」。给出 3～5 个按优先级排序的方向，每个方向对应一个假设。

## 4. 改进方向

每个方向写清：问题 → 思路 → 修改位置 → 预期作用 → 计算成本 → 风险。禁止只罗列模块名。

## 5. 候选模块筛选

结合 [modules/catalog.md](modules/catalog.md) 与用户新模块。对每个候选写：核心作用、类型（Channel / Spatial / Multi-scale / Context / Fusion / Lightweight）、插入位置、是否适合当前任务、Params/FLOPs、是否与现有模块重复、训练风险。只推荐 1～3 个首先实验的模块。

## 6. 插入位置

对浅层 Encoder、深层 Encoder、Bottleneck、Skip、Decoder、Head 分别评估。推荐 1～2 个位置。不要为了用模块而硬插。

## 7. 改代码（用户确认后）

- 单模块：独立类、中文注释、检查 Channel/Shape、不改训练配置、Forward Test。
- 多模块：先判断功能是否重复；必须能分别关闭，支持 Baseline / +A / +B / +A+B。
- 替换：先比接口、分辨率、后续结构是否被带崩。

## 8. 验证与排错

Forward：`eval()` + `no_grad()` + 随机输入。逐层记录 Concat/Add/Residual/Skip/Up/Down 的 Shape。报错时不要靠删模块或改输入尺寸蒙混，先给期望 Shape vs 实际 Shape。能统计则对比 Params / FLOPs。

## 9. 消融与公平对比

默认组合：Baseline、+A、+B、+A+B。组合过多时取最有解释力的最小集合。逐项核对划分、增强、输入尺寸、优化器、Loss、预训练、Seed 是否一致。发现不一致只列出，不擅自改。

## 10. 根据结果再迭代

不要根据平均指标立刻再堆模块。看 per-class、尺度、FP/FN、边界、稳定性、收益是否配得上计算量。下一轮最多 3 个实验，每个仍是：假设 → 最小修改 → 若成立应看到什么。
