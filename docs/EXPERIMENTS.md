# 实验记录

> 当前主线请优先看 `SealQA lifecycle (cached search)`。旧的 `APOEngine` / prompt-only demo 已迁到 `demo/archive/`，这里只保留历史结果说明。

相关兼容性验证见：[KODE_MINIMAX_THINKING.md](./KODE_MINIMAX_THINKING.md)

## 实验 0: SealQA lifecycle (当前主线)

**日期**: 2026-03-29
**Demo**: `python -m treeskill`
**数据**: `demo/data/sealqa_tree_samples.json` (6 条 SealQA 抽样)
**模型**: MiniMax-M2.7 (actor)
**前向引擎**: Kode CLI
**检索层**: 本地 `search_web.py + fetch_url.py` 缓存抽象。`search_web_lookup` 先走 `search_cache.json`，当 `SEALQA_ASO_ENABLE_WEB_FALLBACK=1` 且配置了 `SEALQA_WEB_SEARCH_CMD`/`SEALQA_WEB_FETCH_CMD` 时才会走外部检索分支。

| 阶段 | 准确率 | 技能变化 |
|------|--------|----------|
| Root | 16.7% | `answer-format` |
| Generated | 100.0% | + `search_web_lookup` + `enumeration_verification` + `recency_guard` |
| Evolved | 100.0% | 强化 skill 使用策略 |
| Pruned | 83.3% | 剪掉低收益 skill 后回退 |
| Merged | 83.3% | 合并/重整后回归到真实收益 |

**结论**: `Kode + ASO` 在一个简单 SealQA 子集上已经能稳定演示完整 skill 生命周期，并且在验证中支持从 `generate -> evolve -> prune -> merge` 全链路。

## 实验 1: 前端 Distill ASO（完整链路）

**日期**: 2026-03-29
**Demo**: `python -m treeskill sealqa-aso`（内部包含前端蒸馏任务入口和主链路实现）
**数据**: `demo/data/frontend_tasks.json`（4 / 2 / 2）
**模型**:
- Student/Executor: Intern-S1-Pro（通过 Kode CLI）
- Teacher/Judge/Rewrite: MiniMax-M2.7
**前向引擎**: Kode CLI

| 阶段 | 分数 |
|------|------|
| Baseline | 0.50 |
| Round 1 | 1.00 |
| Final | 1.00 |

**结论**: 可完整验证蒸馏任务中 skill program 的生成、进化、剪枝与合并。当前样本内精度显著抬升，主要用于验证链路通畅。

---

## 实验 2: 论文分类 (3 类)

**日期**: 2026-03-23
**Demo**: `demo/archive/legacy-apo/demo_tree_minimal_10min.py`
**数据**: `demo/data/intern_camp5.csv` (51 train / 12 test, 3 类: A/E/M)
**模型**: Qwen3.5-4B (actor) + GLM-5 (judge)
**详细报告**: [DEMO_PAPER_CLASSIFICATION.md](./DEMO_PAPER_CLASSIFICATION.md)

| 轮次 | 准确率 | 变化 |
|------|--------|------|
| Baseline | 50.0% | — |
| Round 1 | 75.0% | +25.0% |
| Round 2 | 75.0% | +0.0% |
| **Round 3** | **91.7%** | **+16.7%** |

**结论**: APO 在有明确对错的分类任务上效果显著。

---

## 实验 3: 论文分类 (6 类, auto-split)

**日期**: 2026-03-23
**Demo**: `demo/archive/legacy-apo/demo_tree_split.py`
**数据**: 60 train / 30 test, 6 类跨 3 大领域
**模型**: Qwen3.5-4B (actor) + GLM-5 (judge)

| 轮次 | 准确率 | 事件 |
|------|--------|------|
| Baseline | 10.0% | — |
| Round 1 | 70.0% | **auto-split → physics / cs / math** |
| **Round 2** | **86.7%** | +16.7% |
| Round 3 | 83.3% | 小波动 |

**结论**: auto-split 自动发现了 3 个领域，单个 prompt 无法覆盖时框架会自动分裂成子技能。

---

## 实验 4: SealQA Benchmark

**日期**: 2026-03-24
**Demo**: `demo/archive/legacy-apo/demo_sealqa.py`
**数据**: SealQA seal-0 (111 题, 16 train / 10 val / 85 test)
**模型**: Qwen3.5-9B via SGLang (actor) + GLM-5 (judge)
**部署**: InternLM A100 服务器

| 阶段 | 准确率 |
|------|--------|
| Baseline | 3.5% |
| Round 1 | val=0.0% |
| Round 2 | val=0.0% |
| Round 3 | val=0.0% |
| **Final** | **2.4%** (-1.2%) |

**结论**: SealQA 需要搜索能力和最新知识，纯 prompt 优化无法给模型增加它不具备的知识。知识型问答不适合 TreeSkill。

---

## 实验 5: 前端 Copy 蒸馏

**日期**: 2026-03-24
**Demo**: `demo/archive/legacy-apo/demo_distill_copy.py`
**数据**: 10 copy 任务 (6 train / 2 val / 2 test)
**模型**: Intern-S1-Pro (student) + MiniMax-M2.7 (teacher/judge)

| 阶段 | 分数 |
|------|------|
| Baseline | 0.61 |
| Round 1 | **val=0.69** ★ Accepted |
| Round 2 | val=0.67 Rejected |
| Final | 0.56 (-0.05) |

**结论**: val 上有微弱提升，但 test 上没泛化。主观任务的 judge 评分噪声大，数据量太少。

---

## 实验 6: Kode CLI + APO 端到端

**日期**: 2026-03-27
**Demo**: `demo/archive/legacy-apo/demo_kode_apo.py`
**数据**: 8 编码任务 (4 train / 2 val / 2 test)
**模型**: qwen3.5-plus via OneAPI (actor, 通过 Kode CLI) + GLM-5 (judge)
**前向引擎**: Kode CLI

| 阶段 | 分数 |
|------|------|
| Baseline | 1.00 |
| Final | 0.50 |

**结论**: 链路完全跑通（4.2 分钟），但 qwen3.5-plus 太强（baseline 满分），需要更难的任务才能体现优化效果。验证了 Kode CLI 作为前向引擎的可行性。

---

## 关键发现

1. **APO 在明确对错的任务上效果好**（分类: +41.7%），主观任务效果差
2. **auto-split 能自动发现领域**并生成专业子技能
3. **Judge 评分噪声是最大瓶颈** — 硬验证（文件存在 + 代码输出）远比 LLM judge 可靠
4. **Kode CLI 可作为前向引擎** — Skill 在真实 agent loop 中执行，评估更真实
5. **Qwen3.5 思考模式** 需要 `enable_thinking: false` 或 `chat_template_kwargs`
