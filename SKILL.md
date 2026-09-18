---
name: model_improve
description: >-
  Guides deep-learning model structure experiments: diagnose a baseline, pick
  1–3 candidate modules, insert them with minimal edits, run forward tests,
  and design fair ablations. Use when improving CNN/Transformer/Mamba/U-Net
  models, adding attention or multi-scale blocks, replacing encoder/decoder
  parts, or when the user mentions model_improve, Codex 模型改进, 加模块, 消融实验, Baseline 改进.
---

# Codex 模型改进

协助把「研究假设」变成可验证的结构改动。不要为了创新而堆模块。

零件库在 [modules/catalog.md](modules/catalog.md)。完整 Prompt 模板在 [prompts.md](prompts.md)。硬约束在 [constraints.md](constraints.md)。分阶段细节在 [workflow.md](workflow.md)。新增模块写法见 [modules/ADDING.md](modules/ADDING.md)。

## 硬原则

1. 先分析，后改代码。用户没确认方案前，只输出分析。
2. 每一轮只验证一个主要结构因素。多个模块必须能单独开关。
3. 每个改动对应一个问题或假设，写清插入位置、预期现象、风险。
4. 默认不改 Dataset、划分、增强、Loss、Optimizer、Scheduler、Epoch、Batch Size、Seed、Metrics、后处理。
5. 保留原 Baseline，新结构用独立模块类 / 配置 / 模型名。
6. 改完必须做随机输入 Forward Test，再谈训练。
7. 零件库是候选实现，不是论文贡献。选用前对照 catalog 的适用场景与已知问题。

## 路由

按用户当前意图只做对应阶段，不要一次跑完全流程。

| 用户在说 | 进入 |
|---------|------|
| 刚丢来一个项目 / 还不懂结构 | workflow 第 1–2 步 |
| 效果差、漏检、边界糊、参数太大 | 第 3–4 步（诊断 + 假设） |
| 有一堆模块不知加哪个 | 读 catalog，第 5–6 步 |
| 确认加某一个模块 | 第 7 步 + constraints |
| shape 报错 / 改完不能跑 | 第 8 步 |
| 要消融 / 怕对比不公平 | 第 9 步 |
| 贴了指标，问下一步 | 第 10 步 |
| 自己要写给 Codex 的指令 | 用 prompts.md 填空，不要空喊「帮我创新」 |
| 点名 model_improve | 按本 Skill 执行 |

## 必做输出

分析轮：

```text
项目任务：
Baseline：
输入 / 输出：
模型文件 / 训练入口：
数据流（简图）：
当前问题 → 结构性假设：
候选模块（最多 3 个）与插入位置：
不建议加入的原因：
本轮是否改代码：否
```

改代码轮（用户确认后）：

```text
研究假设：
修改文件：
修改位置：
模块类名 / 开关：
输入 Shape → 输出 Shape：
新增 Params（能统计则统计）：
Forward Test：通过 / 失败原因
消融组合：
明确没改的训练配置：
```

## 零件库用法

1. 先读 [modules/catalog.md](modules/catalog.md)，不要整份 [modules/library.md](modules/library.md) 读进上下文。
2. 只按 catalog 的节号，打开 `library.md` 里准备实验的那 1 节（最多 3 节）。
3. 按该节 Markdown 里的代码，在用户项目中写成独立 `nn.Module`。
4. 优先选 **同 shape、2D、无额外重依赖** 的即插模块。
5. catalog 标注「不可用 / 需核对」的条目，禁止直接插入。
6. 用户新看到的模块：按 [modules/ADDING.md](modules/ADDING.md) 追加到 `library.md` 后再筛选。

## 给 Codex / WSL 的用法

本 Skill 约束的是 Cursor Agent。在 WSL 里跑训练不需要把 PyTorch 环境装进 Skill。

- 在 Cursor 里改结构：安装本仓库到 `~/.cursor/skills/model_improve`（Windows 与 WSL 各装一次，见 README）。
- 在 Codex CLI 里改结构：把 [prompts.md](prompts.md) 里对应模板贴进对话，并附上 [constraints.md](constraints.md) 的限制段。
- 实验代码放在用户项目里；需要某个零件时，只根据 `library.md` 对应节来写，不要把整份零件库拷进模型。
