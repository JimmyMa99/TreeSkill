# Tree Optimization 最佳实践 — 论文分类 Demo

本文档以一个 10 分钟可跑完的最小化 demo 为例，展示如何使用 EvoSkill 的树优化功能。

对应代码: [`demo/demo_tree_minimal_10min.py`](../demo/demo_tree_minimal_10min.py)

## 场景

将科学论文分为 3 类:

| 标签 | 领域 |
|------|------|
| A | Quantum Physics |
| E | Robotics |
| M | Computer Vision |

数据来自 `demo/data/intern_camp5.csv`，每类取 8 条训练 + 4 条测试。

## 核心设计

### 1. 故意写烂的 baseline

```
Classify the paper into one category. Reply with a single letter.

Categories: A, E, M

Reply ONLY the letter.
```

不给任何类别描述，模型只能猜。baseline 准确率约 33%，留出足够优化空间。

> **要点**: 如果 baseline 已经很好（>80%），APO 很难产生显著提升。设计 demo 时应让初始 prompt 有明显缺陷。

### 2. Trace 路由 (`node_path`)

收集 trace 时设置 `node_path`，确保每个节点只用属于自己的数据优化:

```python
t = Trace(
    inputs=[Message(role="user", content=f"Paper:\n{question}")],
    prediction=Message(role="assistant", content=pred),
    node_path="paper-classifier",  # 标记归属节点
)
```

路由规则:
- `node_path == dotpath` — 精确匹配
- `node_path` 以 `dotpath.` 开头 — 子节点 trace，父节点也能看到
- `node_path is None` — 旧数据，所有节点都用（向后兼容）

### 3. 树结构与 auto-split

初始只有一个根节点。当某轮反馈中存在矛盾（不同类型的论文需要不同分类策略），APO 会自动拆分:

```
paper-classifier (v1.1)
├── Agriculture (v1.0)
├── Engineering (v1.0)
└── Machine Learning (v1.0)
```

拆分由 `evolve_tree(tree, traces, auto_split=True)` 控制。

### 4. 进度条

`evolve_tree` 内置 rich 进度条，显示当前节点、完成数、已用时间和 ETA:

```
  paper-classifier ✓ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4/4 0:00:38 ETA 0:00:00
```

auto-split 产生新节点时进度条自动扩展 total。

## 运行

```bash
conda activate pr
python demo/demo_tree_minimal_10min.py
```

约 2-3 分钟完成。

## 配置说明

```python
MAIN_MODEL = "Qwen/Qwen3-8B"          # 执行分类的弱模型
JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # APO judge（分析失败 + 评分）
NUM_ROUNDS = 2                          # 优化轮数
NUM_CANDIDATES = 2                      # 每轮候选数
```

| 参数 | 效果 | 建议 |
|------|------|------|
| `NUM_ROUNDS` | 轮数越多越精细，但耗时线性增长 | demo 用 2，生产用 3-5 |
| `NUM_CANDIDATES` | 候选越多越可能找到好 prompt，但 API 成本增加 | demo 用 2，生产用 3-4 |
| `gradient_accumulation_steps` | 每次优化看多少条反馈 | 不超过训练集大小 |

## 实测结果

| 阶段 | 准确率 | 说明 |
|------|--------|------|
| baseline | 33.3% | 无类别描述，纯猜 |
| 第 1 轮 | 50.0% | APO 补上了类别 hints，触发 auto-split |
| 第 2 轮 | 50.0% | judge 打分持平，保持原 prompt |

优化后的 prompt 自动学到了类别描述:

```
Categories:
- A: Agriculture (实际应为 Quantum Physics)
- E: Engineering
- M: Machine Learning

Use these hints:
- A: Focus on farming, crops, and livestock.
- E: Involves technology, robotics, and structural design.
- M: Deals with algorithms, data, and predictive modeling.
```

> **注意**: judge 模型对类别的理解取决于 baseline prompt 的信息量。如果 baseline 完全不提类别含义，judge 只能从失败案例中推断，可能推断错误（如把 A=Quantum Physics 理解为 Agriculture）。生产环境中 baseline 至少应包含类别名称。

## 改进方向

1. **baseline 加上类别名称** — 即使故意写烂，也至少写 `A = Quantum Physics`，让 judge 有正确锚点
2. **增加训练数据** — 每类 8 条偏少，20+ 条效果更稳定
3. **更多轮次** — 2 轮可能不够收敛，3-5 轮更稳妥
4. **子节点独立评估** — 拆分后应按子节点分别评估，而非只看根节点 prompt
