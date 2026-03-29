# Kode + MiniMax Thinking 验证记录

## 结论

当前结论很明确：

| 项目 | 结果 |
|---|---|
| 是否需要继续修改本机安装的 `Kode` 源码 | 不需要 |
| 原版 `Kode 2.2.0` + `MiniMax-M2.7` 是否可用 | 可用 |
| `thinkingMode=disabled` 是否可用 | 可用 |
| `thinkingMode=enabled` 是否可用 | 可用 |

也就是说，之前为了绕过 `thinking` 问题而临时改过的本机 `Kode` 安装包，现在已经可以撤掉，不应再作为项目依赖。

## 背景

此前我们为本机安装的 `Kode` 加过一段临时补丁：当请求发往 `https://api.minimaxi.com/anthropic` 且不启用 thinking budget 时，显式传：

```json
{"thinking":{"type":"disabled"}}
```

这个 patch 的目的，是避免某些 Anthropic 兼容端点返回 `thinking block` 但不稳定产出最终文本，导致上层拿到空输出。

## 本次验证动作

本次做了三步验证：

1. 撤掉本机 `Kode` 安装包里的临时 patch  
   目标文件：
   - `/opt/homebrew/lib/node_modules/@shareai-lab/kode/dist/sdk/core.cjs`
   - `/opt/homebrew/lib/node_modules/@shareai-lab/kode/dist/sdk/core.js`

2. 修复本机 `~/.kode.json`  
   在排查过程中发现该文件曾一度成为非法 JSON，导致 `kode models list` 读不到任何 model profile。修复后恢复正常。

3. 重新在**无源码补丁**前提下验证两种模式：
   - `thinkingMode = "disabled"`
   - `thinkingMode = "enabled"`

## Probe 结果

统一使用：

```bash
kode -p "Reply with exactly: <TOKEN>" \
  --model "MiniMax-M2.7" \
  --output-format json \
  --dangerously-skip-permissions
```

### Disabled

| 项目 | 结果 |
|---|---|
| thinkingMode | `disabled` |
| return code | `0` |
| result | `DISABLED_OK` |
| duration | 约 `2.9s` |

### Enabled

| 项目 | 结果 |
|---|---|
| thinkingMode | `enabled` |
| return code | `0` |
| result | `ENABLED_OK` |
| duration | 约 `3.6s` |

## 工程结论

### 1. 不再依赖本机 patch

仓库不应把“手改 Homebrew 安装目录中的 Kode 源码”当成正式方案。  
当前验证表明，这条临时路径已经可以去掉。

### 2. 保留仓库内的 Anthropic `extra_body` 支持

TreeSkill 自己的 LLM 调用层仍然应该保留：
- `judge_extra_body`
- `rewrite_extra_body`
- Anthropic `extra_body` 透传

原因是这属于**仓库内可控的兼容层**，不是本机 hack。

### 3. 当前推荐配置

对于 `MiniMax-M2.7`：

| 层 | 推荐 |
|---|---|
| Kode | 使用原版，不改安装包源码 |
| `~/.kode.json` | 合法 JSON，包含有效 `MiniMax-M2.7` profile |
| thinkingMode | `enabled` 或 `disabled` 都可；默认按任务选择 |
| TreeSkill judge/rewrite | 继续允许通过 `extra_body` 传 Anthropic 兼容参数 |

## 完整实验复跑

在**无本机 Kode 源码补丁**、`thinkingMode = "enabled"` 的前提下，重新执行了当前主线 demo：

```bash
python -m treeskill
```

输出：

```text
demo/outputs/sealqa-tree-lifecycle/summary.json
```

本次复跑结果：

| 阶段 | 准确率 |
|---|---:|
| root | 16.7% |
| generated | 100.0% |
| evolved | 100.0% |
| pruned | 83.3% |
| merged | 83.3% |

这次在修正 `prune_program()` 的统计逻辑后，`merged` 没有再虚高到 100%，而是暴露出更真实的后半段结果：

| 阶段 | 最终 skill | 备注 |
|---|---|---|
| pruned | `answer-format + enumeration_verification` | 剪掉了 `search_web_lookup` 和 `recency_guard` |
| merged | `answer-format + verified_fact_lookup` | `merged_from = [enumeration_verification, search_web_lookup]`，说明 merge 已经显式生成 |

这说明两件事：

1. `thinking enabled` 不会阻塞当前主线 pipeline。
2. 当前推荐的 `Kode + ASO + SealQA lifecycle` 在原版 Kode 上可复现，并且 merge 阶段已经能显式生成 merged skill，但后半段收益仍有继续优化空间。

## 与当前主线的关系

当前主线 pipeline 是：

```text
Kode forward pass + ASO skill evolution
```

对应命令：

```bash
python -m treeskill
```

本次文档的意义是确认：

- `Kode` 作为前向执行器可继续使用
- `MiniMax-M2.7` 可在原版 Kode 下运行
- 不必再要求使用者手工修改本机安装的 `Kode`
