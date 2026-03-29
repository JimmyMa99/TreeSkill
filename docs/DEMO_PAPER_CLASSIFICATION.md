# Demo: Paper Classification — APO Prompt Optimization

> 日期: 2026-03-23
> 模型: Qwen3.5-4B (task) + GLM-5 (judge)
> 数据: 3 类 x 17 条训练 + 3 类 x 4 条测试 = 63 条

## 概述

使用 TreeSkill 的 APO (Automatic Prompt Optimization) 引擎，对论文分类任务进行自动 prompt 优化。评分机制对齐 [Microsoft Agent-Lightning](https://github.com/microsoft/agent-lightning/) —— 真实模型跑分类 + judge 评分，而非 judge 主观打分。

## 实验配置

| 参数 | 值 |
|------|-----|
| 任务模型 | `SilliconCloud/Qwen3.5-4B` (enable_thinking=False) |
| Judge 模型 | `bailian/glm-5` |
| 分类类别 | A (量子物理) / E (机器人) / M (计算机视觉) |
| 训练集 | 51 条 (每类 17 条) |
| 测试集 | 12 条 (每类 4 条) |
| 优化轮数 | 3 轮 |
| Beam search | beam_width=2, branch_factor=2, beam_rounds=2 |
| 总耗时 | 37.1 分钟 |

## 结果

| 阶段 | 准确率 | 变化 |
|------|--------|------|
| Baseline | **50.0%** (6/12) | — |
| 第 1 轮 | **75.0%** (9/12) | +25.0% |
| 第 2 轮 | **75.0%** (9/12) | +0.0% |
| 第 3 轮 | **91.7%** (11/12) | +16.7% |
| **总提升** | | **+41.7%** |

## 优化前 Prompt (Baseline)

```
You are a paper classifier. Given a scientific paper, classify it.

A - physics
E - engineering
M - computers

Output only the letter.
```

- 类别描述过于简略（"physics" / "engineering" / "computers"）
- 没有分类提示和边界决策规则
- 没有处理跨领域论文的指导

## 优化后 Prompt (v1.2)

```
You are a paper classifier. Given a scientific paper, classify it into one of the following categories:

A - Physics
   Studies fundamental laws of nature, matter, energy, and physical phenomena.
   Includes: mechanics, quantum physics, thermodynamics, electromagnetism, particle physics, astrophysics, optics, condensed matter.
   Key indicators: theoretical models of natural laws, experimental studies of physical phenomena.

E - Engineering
   Designs, builds, and operates physical systems, devices, and hardware.
   Includes: robotics, mechanical/electrical systems, aerospace, control systems, manufacturing, sensors, actuators, civil structures.
   Key indicators: physical devices, hardware platforms, real-world implementation, manipulation systems.

M - Computers
   Develops software, algorithms, computational theory, and information processing.
   Includes: software engineering, AI/ML algorithms, data science, computer graphics, networking, databases, theoretical CS.
   Key indicators: software systems, algorithmic methods, computational theory, data processing.

CLASSIFICATION HINTS:
• Robotics and autonomous systems → E (Engineering), even when algorithms are involved
• Physical manipulation, hardware platforms, sensors, actuators → E (Engineering)
• Pure algorithms, software architectures, computational theory → M (Computers)
• Fundamental physical principles and phenomena → A (Physics)
• Computational physics, quantum simulations, numerical studies of physical systems → A (Physics)
• Using algorithms (QAOA, VQE, Monte Carlo, DFT) to investigate physical phenomena → A (Physics)
• Novel algorithm development as primary contribution → M (Computers)

BOUNDARY DECISIONS:
- If a paper describes a physical device, robot, or hardware system, classify as E regardless of computational components.
- If computation is the primary contribution without physical hardware, classify as M.
- If the study of natural physical phenomena is primary, classify as A.
- If algorithms are tools to study physical systems, classify by the subject domain (physics → A), not the method.
- Quantum computing: physics investigations → A; algorithm development → M; hardware → E.

Output only the letter (A, E, or M).
```

## 优化分析

APO 引擎通过 3 轮优化自动发现并修复了以下问题:

1. **类别描述不足** — 自动补充了每个类别的详细定义、包含子领域、关键指标
2. **边界模糊** — 添加了 CLASSIFICATION HINTS 处理跨领域论文（如机器人用 AI 算法→仍归 E）
3. **量子计算歧义** — 明确区分了量子物理研究(A)、量子算法开发(M)、量子硬件(E)
4. **方法 vs 领域** — 建立了"按研究对象而非研究方法分类"的原则

## 关键技术点

### 评分机制 (Agent-Lightning 对齐)

传统 APO 评分: Judge 看 prompt 文本 → 主观打分 → 虚高 (0.90+)，但实际精度不变

本实现评分:
1. **真实模型跑任务** — 候选 prompt 实际用 Qwen3.5-4B 分类
2. **Judge 对比评分** — GLM-5 比较模型输出 vs 真实答案，打 0-1 分
3. **平均 reward** — 作为 prompt 的最终得分

### Qwen3 思考模式适配

Qwen3 系列模型默认开启思考模式，回答放在 `reasoning_content` 中，`content` 返回空字符串。需要 `enable_thinking: False` 关闭思考模式才能正常使用。

已通过 `LLMConfig.extra_body` 配置支持，无需硬编码。

## 运行方式

```bash
conda activate pr
TREE_LLM_MODEL=SilliconCloud/Qwen3.5-4B \
TREE_LLM_JUDGE_MODEL=bailian/glm-5 \
python demo/archive/legacy-apo/demo_tree_minimal_10min.py
```

输出目录: `demo/outputs/tree-minimal-10min/`
