# Codex 模型改进 Skill + 零件库

把「分析 Baseline → 选 1～3 个模块 → 最小改动 → Forward Test → 公平消融」写成 Cursor Skill。`modules/` 是可继续追加的零件库，不是实验方法文档。

## Windows / WSL 分别要什么

训练代码在 WSL 里跑，**不必**把这个仓库装进 conda 环境。Skill 只给 Agent 读。

| 你在哪边开 Cursor / Codex | 需要做的 |
|---------------------------|----------|
| Windows 上的 Cursor | 把本仓库放到 `C:\Users\<你>\.cursor\skills\codex-model-improve` |
| WSL 里的 Cursor 或 Codex CLI | 再 clone 一份到 Linux 的 `~/.cursor/skills/codex-model-improve` |
| 只在 WSL 里 `python train.py` | 不用装 Skill；需要某个模块时，把对应 `modules/*.py` 拷进实验项目 |

Windows 与 WSL 的 `~` 不是同一个目录，所以要装两次；内容用同一个 GitHub 仓库同步即可。

## 安装（个人 Skill）

```bash
# Windows PowerShell（示例路径）
git clone <你的仓库 URL> "$env:USERPROFILE\.cursor\skills\codex-model-improve"

# WSL / Linux
git clone <你的仓库 URL> ~/.cursor/skills/codex-model-improve
```

如果仓库已经在 `e:\Codex模型改进\codex-model-improve`，Windows 也可以建目录联接，避免两份拷贝：

```powershell
New-Item -ItemType Junction -Path "$env:USERPROFILE\.cursor\skills\codex-model-improve" -Target "e:\Codex模型改进\codex-model-improve"
```

安装后新开一轮对话，提到「加模块 / 改进 Baseline / 消融」时应加载本 Skill。也可在提示里写：`按 codex-model-improve 做。`

## 给 Codex CLI 用

Codex 不会自动读 Cursor Skill。把 `prompts.md` 里的模板填空，并附上 `constraints.md` 的限制段。需要零件时只贴 `catalog.md` 里选中的那一个 `.py`，不要整库粘贴。

## 目录

```text
SKILL.md           入口：原则、路由、输出格式
constraints.md     禁止改训练配置、公平对比
workflow.md        10 个阶段
prompts.md         给 Codex 复制的短模板
modules/catalog.md 零件索引（先读这个）
modules/ADDING.md  以后加新模块的规范
modules/*.py       实现
```

## 以后加新模块

按 `modules/ADDING.md`：新文件 + catalog 一行。不要写回旧的整份 `模块.md`，也不要放进实验方法文件夹。

## 推到 GitHub

在本目录：

```bash
git init -b main
git add SKILL.md constraints.md workflow.md prompts.md README.md .gitignore modules
git commit -m "Add Codex model-improve skill and module catalog."
gh repo create codex-model-improve --private --source . --remote origin --push
```

`--public` 若你要公开。不要把 `e:\Codex模型改进\项目代码及数据` 放进这个仓库。
