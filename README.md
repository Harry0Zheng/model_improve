# model_improve

模型结构改进 Skill。对话里说「按 model_improve 做」就行。

流程是先看 Baseline 和问题，再从零件库里挑 1～3 个模块，小改、Forward 过了再训练，消融时别动数据划分和训练配置。具体步骤在 `SKILL.md`、`workflow.md`、`constraints.md`。给 Codex 粘贴的短模板在 `prompts.md`。

零件代码在本地 `e:\Codex模型改进\模块.md`，不进这个仓库。先翻 `modules/catalog.md` 对一下适不适合、插哪、有没有坑。新模块追加到 `模块.md` 末尾，并在 catalog 加一行，写法见 `modules/ADDING.md`。实验协议、划分、消融表也不要写进这里。

Windows 上 Cursor 已经联到这个目录。WSL 里的 Cursor / Codex 要再 clone 一份到 `~/.cursor/skills/model_improve`。训练环境不用装这个仓库。Codex CLI 不会自动读 Skill，把 `prompts.md` 填空，带上 `constraints.md` 那段限制。
