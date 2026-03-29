# Demo Guide

当前推荐只看两条主线：

| Demo | 用途 | 命令 |
|---|---|---|
| `demo_sealqa_tree_lifecycle.py` | 完整生命周期：`root -> generate -> evolve -> prune -> merge` | `python -m treeskill` |
| `demo_sealqa_aso.py` | 更接近真实 frontier/beam 的最小 ASO 实验 | `python -m treeskill sealqa-aso` |

## 推荐顺序

### 1. 生命周期 Demo

这是当前仓库的主流 pipeline：
- `Kode` 做前向执行
- `ASO` 做 skill/program 修改
- 使用本地 `search_web/fetch_url` 抽象稳定复现 SealQA 小样本
- 检索策略默认：`search_web_lookup` 只查本地缓存；可通过设置 `SEALQA_ASO_ENABLE_WEB_FALLBACK=1` 打开外部回退，并通过 `SEALQA_WEB_SEARCH_CMD`、`SEALQA_WEB_FETCH_CMD` 注入外部工具命令。

```bash
python -m treeskill
```

输出目录：

```text
demo/outputs/sealqa-tree-lifecycle/
```

### 2. 最小 ASO Demo

如果你要看 frontier / candidate growth：

```bash
python -m treeskill sealqa-aso
```

输出目录：

```text
demo/outputs/sealqa-aso-mini/
```

## 归档 Demo

旧的 prompt-only / `APOEngine` / 早期 tree 实验，已经迁到：

```text
demo/archive/
```

它们保留作历史参考，不再代表当前主流 pipeline。
