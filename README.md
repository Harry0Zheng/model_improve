# model_improve

把「分析 Baseline → 选 1～3 个模块 → 最小改动 → Forward Test → 公平消融」写成 Cursor Skill。零件实现以 Markdown 为准。

## 上传 GitHub 的范围（只有这个文件夹）

上传目录：`e:\Codex模型改进\model_improve\`

| 上传 | 不上传 |
|------|--------|
| `SKILL.md` | `e:\Codex模型改进\项目代码及数据\` |
| `constraints.md` `workflow.md` `prompts.md` `README.md` `.gitignore` | 实验权重、数据集 |
| `modules/catalog.md` `modules/ADDING.md` `modules/library.md` | 旧备份 `Codex模型改进Prompt文档.md`（可选，默认不传） |

不要把整个 `e:\Codex模型改进` 当成仓库根目录。

本机没有 GitHub CLI，**不能在未登录的情况下替你一键建远程仓库**。仓库已经在本地初始化；你登录 GitHub 后，在该目录执行：

```powershell
cd "e:\Codex模型改进\model_improve"
gh repo create model_improve --private --source . --remote origin --push
```

没有 `gh` 时：在 GitHub 网页新建空仓库（不要勾 README），然后：

```powershell
cd "e:\Codex模型改进\model_improve"
git remote add origin https://github.com/<你的用户名>/model_improve.git
git push -u origin main
```

装到 Cursor：Windows 已把本目录联接到 `~\.cursor\skills\model_improve`。WSL 里的 Cursor 再 clone 同一仓库到 `~/.cursor/skills/model_improve`。对话里可写：`按 model_improve 做。`

## Windows / WSL

训练在 WSL 跑，不必把 Skill 装进 conda。Skill 只给 Agent 读。Codex CLI 不会自动读 Cursor Skill，把 `prompts.md` 填空并附上 `constraints.md`。

## 以后加新模块

按 `modules/ADDING.md` 追加到 `library.md`，并在 `catalog.md` 加一行。不要拆成 `.py`，也不要写进实验方法文件夹。
