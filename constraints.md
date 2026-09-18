# 最小修改与公平对比

把下面整段接到任何「开始改结构」的指令后面。

```text
【重要限制】

当前 Baseline 已经能正常运行。本轮只研究用户指定的结构因素。

除非新模块无法接上，否则禁止修改：
- Dataset / DataLoader / 数据划分
- Preprocessing / Data Augmentation
- Loss / Optimizer / Learning Rate / Scheduler
- Epoch / Batch Size / Metrics / Random Seed / Post-processing

禁止：
1. 为了让代码运行而删除原有关键结构
2. 大范围重构与本实验无关的文件
3. 顺手「优化」其他代码
4. 同时引入未经要求的其他模块
5. 改实验配置，导致无法公平判断结构是否有效

新模块写成独立 nn.Module，并保留开关，至少能跑：
Baseline
Baseline + 本模块

发现必须改其他部分时，先说明原因，等确认后再改。
修改完成后用符合项目输入格式的随机 Tensor 做 Forward Test，
打印 Input/Output Shape，检查 NaN/Inf。不要只口头说「改完了」。
```

## 3D / 时序额外检查

- 3D：禁止把 5D 张量随意压成 2D；核对 Conv/BN/Pool 是否误用 2D。
- 时序：先确认输入是 `[B,T,C]` 还是 `[B,C,T]`，再插入模块。
- 多模态：先判断 Early / Middle / Late fusion 是否已经存在，避免再叠一套重复融合。
