# 给 Codex 用的短模板

在 Cursor 里直接按 SKILL 执行即可。在 Codex CLI（含 WSL）里，把对应模板填空后粘贴，并附上 `constraints.md` 里的限制段。

## 编写公式

```text
任务 + Baseline + 当前问题 + 研究假设
+ 准备改的位置 + 准备用的模块
+ 允许改的范围 + 禁止改的范围
+ Shape/Channel 检查 + Forward Test
+ 消融方案 + 结果总结
```

不要只写：`帮我给 U-Net 加一个注意力机制。`

## 分析（不改代码）

```text
请先完整分析当前项目，不要修改代码。
找出模型定义、训练入口、Dataset、Loss、配置文件。
梳理 Baseline 数据流和各 Stage 的 Tensor Shape。
当前问题：【填写】
请给出 3～5 个对应假设的改进方向，不要堆模块名。
```

## 筛选零件

```text
当前任务：【】
Baseline：【】
主要问题：【】
候选模块：【从零件库 catalog 列出】
请比较作用类型、插入位置、计算量、功能是否重复。
只推荐 1～3 个先做，并给出最合理的 1～2 个插入位置。
暂时不要改代码。
```

## 加入单个模块

```text
当前 Baseline 可运行。准备加入：【模块】
目的：【问题/假设】
计划位置：【】
请做成独立 nn.Module，保留开关，添加中文注释。
自动检查 Channel 与 Shape，做随机 Tensor Forward Test。
不修改 Dataset、划分、Loss、Optimizer、训练策略。
不要覆盖原 Baseline。
最后列出修改文件、Shape、新增参数量、Forward 是否通过、消融方案。
```

## 完整工作流（仅当用户明确要求一次做完）

仍须分阶段：先分析并给出方案，等确认后再改代码。阶段内容与 `workflow.md` 一致：分析 → 设计 → 最小修改 → Forward Test → 消融设计 → 总结。
