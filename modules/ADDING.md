# 向零件库追加新模块

看到新模块时，写进 Markdown 零件库，不要先拆成 `.py`。

## 步骤

1. 在 [library.md](library.md) 文末追加一节，标题格式：`# 27. 模块名（一句话作用）`
2. 节内写「介绍」+「代码实现」代码块（完整可复制的 `nn.Module`）。
3. 在 [catalog.md](catalog.md) 增加一行，`节` 与 `library.md` 编号一致。
4. 未跑过 Forward、缺依赖、接口不是即插头时，`状态` 填 `需核对` 或 `不可用`。

## catalog 字段

| 字段 | 写什么 |
|------|--------|
| id | 短名，如 `ema` |
| 节 | `library.md` 里的编号 |
| 类型 | Channel / Spatial / Multi-scale / Context / Fusion / Lightweight / Token-attn |
| 同shape | 是 = 空间尺寸和通道都不变 |
| 维数 | 2D / 3D / token |
| 依赖 | torch，或 timm / mmcv / einops |
| 优先插入 | Bottleneck / Skip / Encoder 深层 等 |
| 状态 | 可用 / 需核对 / 不可用 |
| 备注 | 和谁功能重复、不适合什么任务 |

## 不要做的事

- 不要把新模块写进实验方法文档。
- 不要为了入库先改成独立 Python 包。
- 不要一次把新旧模块一起塞进网络。
- 不要把未核对的代码标成「可用」。
