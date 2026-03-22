# 跨模型 Skill 迁移

## 问题

在大模型（Claude、GPT-4o）上优化好的 Skill，直接移植到小参数量模型（Qwen-7B、InternLM-7B 等）时，效果往往断崖式下降。

根本原因：

1. **能力差异** — 小模型的指令遵循、推理链、格式控制能力弱
2. **Prompt 敏感度不同** — 大模型容忍模糊指令，小模型需要更显式、结构化的表述
3. **隐性依赖** — Skill 隐性依赖大模型的常识推理能力，小模型不具备

## 方案：双模型 TGD

核心思路：**梯度计算用大模型，Skill 执行用小模型**。

```
                    ┌─────────────────┐
                    │  Optimizer Model │  ← 大模型（Claude / GPT-4o）
                    │  负责：           │
                    │  · 分析失败原因    │
                    │  · 计算文本梯度    │
                    │  · 改写 Prompt    │
                    └────────┬────────┘
                             │ 新 Prompt
                             ▼
                    ┌─────────────────┐
                    │  Executor Model  │  ← 小模型（Qwen-7B / InternLM-7B）
                    │  负责：           │
                    │  · 执行 Skill     │
                    │  · 产生输出       │
                    └────────┬────────┘
                             │ 输出 + 反馈
                             ▼
                    ┌─────────────────┐
                    │  评估 / 反馈      │
                    │  （人工或 Judge）  │
                    └────────┬────────┘
                             │ 失败案例
                             ▼
                      回到 Optimizer Model
```

### 为什么可行

小模型自己分析不了自己为什么失败——让大模型来做这件事。大模型理解小模型的局限性，能生成适配小模型能力的 Prompt：更显式的指令、更多 few-shot 示例、更简单的推理步骤。

### 迁移流程

1. **种子 Prompt** — 用大模型上已验证的 Skill 作为起点（warm start）
2. **执行** — 用小模型跑测试集，收集失败案例
3. **梯度计算** — 大模型分析"小模型在这个 Prompt 下为什么失败"
4. **Prompt 改写** — 大模型生成适配小模型的新 Prompt
5. **循环** — 重复 2-4 直到收敛

## 框架改动

现有架构已具备双模型基础：

- `GlobalConfig` 已有 `model`（执行）和 `judge_model`（评判）两个配置
- `LLMClient.generate()` 支持 per-call 的 model 覆盖
- `APOEngine` 的梯度计算和执行本身已解耦

### 配置变更

`config.yaml` 新增 `optimizer_model` 字段：

```yaml
llm:
  model: "Qwen/Qwen2.5-7B-Instruct"        # executor — 小模型
  optimizer_model: "claude-sonnet-4-20250514"   # optimizer — 大模型
  judge_model: "claude-sonnet-4-20250514"       # judge（可复用 optimizer）
```

环境变量：

```bash
TREE_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
TREE_LLM_OPTIMIZER_MODEL=claude-sonnet-4-20250514
```

### 代码改动

| 文件 | 改动 | 规模 |
|------|------|------|
| `config.py` | 新增 `optimizer_model` 字段 | ~3 行 |
| `optimizer.py` | `_compute_gradient` / `_apply_update` 使用 `optimizer_model` | ~5 行 |
| `core/optimizer.py` | `TrainFreeOptimizer` 接受可选的 `optimizer_adapter` | ~15 行 |

总计约 **20-30 行核心代码改动**，无需修改框架架构。

### `TrainFreeOptimizer` 改动示意

```python
class TrainFreeOptimizer:
    def __init__(
        self,
        adapter: ModelAdapter,                        # executor（小模型）
        optimizer_adapter: ModelAdapter | None = None, # optimizer（大模型，可选）
        config: OptimizerConfig | None = None,
    ):
        self.adapter = adapter
        self.optimizer_adapter = optimizer_adapter or adapter  # 不传则退化为单模型
        self.config = config or OptimizerConfig()

    def optimize(self, prompt, experiences, validator=None):
        # ...
        # 梯度计算 → 用大模型
        gradient = self.optimizer_adapter.compute_gradient(...)
        # 梯度应用（改写 Prompt）→ 用大模型
        new_prompt = self.optimizer_adapter.apply_gradient(...)
        # 验证 → 用小模型执行
        score = validator(new_prompt)  # validator 内部调用 self.adapter
        # ...
```

## 使用场景

### 场景 1：Claude → Qwen-7B 迁移

```yaml
llm:
  api_key: "sk-xxx"
  base_url: "https://api.siliconflow.cn/v1"
  model: "Qwen/Qwen2.5-7B-Instruct"
  optimizer_model: "claude-sonnet-4-20250514"
```

大模型（Claude）负责分析和改写，小模型（Qwen-7B）负责执行，TGD 自动找到适合小模型的 Prompt 表述。

### 场景 2：同族模型降级

```yaml
llm:
  model: "Qwen/Qwen2.5-7B-Instruct"       # 部署用
  optimizer_model: "Qwen/Qwen2.5-72B-Instruct"  # 优化用
```

用 72B 帮 7B 优化 Prompt，同族模型理解彼此的能力边界更准确。

### 场景 3：成本优化

在大模型上验证 Skill 效果后，用双模型 TGD 迁移到便宜的小模型部署，降低推理成本。
