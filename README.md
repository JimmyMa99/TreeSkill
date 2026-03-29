# EvoSkill

> **Note**: 本项目由于与其他仓库重名，已迁移至 [TreeSkill](https://github.com/JimmyMa99/TreeSkill)，后续所有更新均在 TreeSkill 进行。

[English](./docs_en/README.md) | 中文

**Kode 前向 + AS(skill)O 自进化框架**

当前主流 pipeline 已经收敛为：

```text
Kode 前向执行 → 失败样本 / 反馈 → ASO 生成/修改 skill program → 评估 → 剪枝 / 合并
```

也就是说，优化对象不再只是单个 prompt，而是完整的：

```text
program = root prompt + skills + selection policy
```

```
问题 / 数据集 → Kode 执行 → 文本梯度 / proposal → skill growth → prune / merge → 更好的 Agent
```

## 核心理念

TreeSkill 将 LLM prompt 优化类比为深度学习的训练循环，但**完全不需要训练模型**：

| 深度学习 | TreeSkill |
|----------|----------|
| 模型权重 | System Prompt |
| 训练数据 | 交互反馈（人工 or LLM Judge） |
| 损失函数 | 失败案例分析 |
| 梯度 | 文本梯度（自然语言的失败归因） |
| 参数更新 | Prompt 重写（Beam Search 多候选） |
| Epoch | 优化轮次（支持断点续跑） |

## 特性

- **免训练 (Train-Free)** — 纯 API 调用，无需 GPU、无需微调、无需标注数据
- **Human-in-the-Loop** — 人工反馈驱动优化：`/bad` 标记失败、`/rewrite` 提供理想回答、`/target` 设置优化方向，让领域专家直接参与 prompt 进化
- **Beam Search APO** — 对齐 [Agent-Lightning](https://github.com/microsoft/agent-lightning/) 的优化算法：多模板梯度分析 × 多候选生成 × Beam Search 选择，持续保留 top-k prompt 跨轮优化
- **模型无关** — 支持 OpenAI、Anthropic、任何 OpenAI 兼容 API（硅基流动、Ollama 等）
- **Agent Skills 标准** — Skill 文件遵循 [agentskills.io](https://agentskills.io) 开放标准
- **层级 Skill 树** — 自动拆分、剪枝、嫁接，递归 bottom-up 优化
- **多协议工具系统** — Skill 可声明和调用多种格式的外部工具：
  - **Python 脚本** — `script.py` 中的函数自动注册为工具
  - **HTTP API** — 声明式调用任意 REST 端点
  - **MCP 服务器** — 兼容 [Model Context Protocol](https://modelcontextprotocol.io/) 的工具调用
- **断点续跑** — 优化中断后从上次进度恢复，不浪费已完成的 API 调用
- **插件注册机制** — 通过 `@scorer`、`@gradient`、`@rewriter`、`@skill_format` 装饰器自定义优化流程的每个环节（[详细文档](./docs/REGISTRY_GUIDE.md)）
- **多端点支持** — Actor / Judge / Rewrite 可配置不同的 API 端点、模型和协议（OpenAI / Anthropic）
- **Kode CLI 集成** — 使用 [Kode](https://github.com/shareAI-lab/Kode-Agent) 作为 Agent 前向引擎，Skill 在真实 agent loop 中执行和验证（[配置指南](#kode-集成)）
- **AgentHarness** — 内置轻量 agent loop，支持 bash / 文件读写 / skill 加载，无需外部依赖即可评估 Skill

## 安装

```bash
git clone https://github.com/JimmyMa99/TreeSkill.git
cd TreeSkill
pip install -e .
```

## 当前主流 Pipeline

推荐直接跑 SealQA 生命周期 demo。它使用：
- `Kode` 作为前向执行器
- `ASO` 作为 program / skill 修改器
- 一个免费、稳定的本地 `search_web/fetch_url` 抽象来模拟检索

```bash
# 默认就是当前主流 pipeline
python -m treeskill

# 等价写法
python -m treeskill sealqa-lifecycle
```

输出会落到：

```text
demo/outputs/sealqa-tree-lifecycle/
├── root/
├── generated/
├── evolved/
├── pruned/
├── merged/
└── summary.json
```

如果你要跑更接近真实 ASO frontier 的最小实验：

```bash
python -m treeskill sealqa-aso
```

旧的交互式 APO/chat 入口仍保留为兼容模式：

```bash
python -m treeskill legacy-chat -- --skill demo/writing-skills
# 或
python -m treeskill.main --skill demo/writing-skills
```

## 快速开始

### 1. 配置 API

```bash
cp demo/example/config.yaml my-config.yaml
# 编辑 my-config.yaml，填入 API Key
```

或用环境变量：

```env
TREE_LLM_API_KEY=your-api-key
TREE_LLM_BASE_URL=https://api.siliconflow.cn/v1
TREE_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
```

### 2. 启动当前主流 demo

```bash
python -m treeskill
```

### 3. 启动旧版兼容 chat / APO

```bash
# 用已有 skill 目录
python -m treeskill.main --config my-config.yaml --skill skills/fast-test-skill

# 用默认 skill（自动创建）
python -m treeskill.main --skill default

# 用 skill 树目录
python -m treeskill.main --skill my-skills/
```

### 4. Human-in-the-Loop 优化（兼容模式）

TreeSkill 的核心交互模式是**人机协作优化**：领域专家通过自然语言反馈，引导 APO 引擎改进 prompt。

```
You: 帮我写一段关于春天的短文

🤖 Assistant: [生成结果]

You: /bad 太像AI写的，缺乏生活气息       ← 标记失败 + 原因
You: /rewrite 春天来了，小区的玉兰花...    ← 提供理想回答（可选）
You: /target 更像人，有生活气息            ← 设置优化方向（可选）
You: /optimize                            ← 触发 APO 优化

✓ Skill optimized → writing-assistant (v1.0 → v1.1) (checkpoint saved)
```

每次 `/bad` 和 `/rewrite` 仍然只更新当前这条 Trace；同一次会话里所有 Trace 共享同一个 `session_id`，但每条交互都有自己独立的 `trace.id`。`/optimize` 时 APO 引擎从这些 Trace 中提取失败模式，计算文本梯度，重写 prompt。**领域专家不需要懂 prompt engineering，只需要判断回答好不好。**

也支持数据集驱动的模式：

```bash
# 全自动：LLM Judge 评分 → APO 优化，无需人工
python -m treeskill.main --optimize --dataset train.jsonl --skill my-skill --no-resume

# 人机协作标注：auto-judge 打分，人可随时 override（偏好信号 → 指导 judge）
python -m treeskill.main --annotate --dataset train.jsonl --skill my-skill

# 纯手动标注
python -m treeskill.main --annotate --dataset train.jsonl --skill my-skill --manual
```

标注模式中，人工反馈是自然语言偏好信号，既作为 APO 梯度的输入，也可导出为 DPO 微调数据。

CLI 中输入 `/` 时会弹出 slash 命令候选列表，输入命令前缀时会自动收窄候选，便于快速发现和选择可用命令。

## Skill 文件格式

遵循 [Agent Skills 标准](https://agentskills.io/specification)，每个 Skill 是一个目录：

```
my-skill/
├── SKILL.md          # YAML frontmatter + Markdown body（= system prompt）
├── config.yaml       # 可选：few-shot、temperature、工具声明、日程等
└── script.py         # 可选：Python 工具函数
```

**SKILL.md 示例：**

```markdown
---
name: my-writing-assistant
description: 专业中文写作助手，帮助撰写和润色各类文本。
metadata:
  version: "1.0"
  target: "更像真人说话，有温度"
---

你是一位专业的中文写作助手。
语言要自然流畅，避免 AI 腔。
```

> 详见 [docs/design/skill-format.md](./docs/design/skill-format.md)

## 工具系统

Skill 可以声明外部工具，agent 在对话中按需调用。支持三种协议：

**config.yaml 声明示例：**

```yaml
tools:
  # HTTP API 工具
  - name: weather
    type: http
    endpoint: https://api.weather.com/current
    method: GET
    description: 获取当前天气

  # MCP 工具
  - name: database
    type: mcp
    mcp_server: localhost:5000
    tool_name: query
    description: 查询数据库
```

**script.py 自动注册：**

```python
# script.py 中的公开函数自动成为工具
def search_docs(query: str) -> str:
    """搜索文档库"""
    ...
```

工具继承：子 Skill 自动继承父 Skill 的工具声明，同名覆盖。

> 详见 [docs/TOOLS_GUIDE.md](./docs/TOOLS_GUIDE.md)

## Skill 树

Skill 通过目录嵌套形成层级：

```
writing-skills/
├── SKILL.md              # 根：通用写作
├── social/
│   ├── SKILL.md          # 社交写作
│   └── moments/
│       └── SKILL.md      # 朋友圈专精
└── business/
    ├── SKILL.md          # 商务写作
    └── email/
        └── SKILL.md
```

优化时 bottom-up：先叶子后父节点。反馈矛盾时自动建议拆分。支持 graft（嫁接）跨树复用 skill。

> 详见 [docs/design/tree-optimization.md](./docs/design/tree-optimization.md)

## 命令一览

| 命令 | 作用 |
|------|------|
| `/bad <原因>` | 标记上条回复不好 |
| `/rewrite <文本>` | 提供理想回答（同时积累 DPO 偏好数据） |
| `/export-dpo <output.jsonl>` | 导出 DPO 偏好数据（用于微调） |
| `/target <方向>` | 设置优化方向 |
| `/optimize` | 触发 APO 优化（支持断点续跑） |
| `/image <路径>` | 附加图片（多模态） |
| `/audio <路径>` | 附加音频（语音输入） |
| `/save` | 保存当前 skill |
| `/tree` | 显示技能树 |
| `/select <路径>` | 切换子技能（如 `social.moments`） |
| `/split` | 分析是否需要拆分 |
| `/ckpt` | 列出 checkpoint |
| `/restore <名称>` | 从 checkpoint 恢复 |
| `/tools` | 查看可用工具 |
| `/quit` | 退出 |

## AS(skill)O 主线

当前推荐的优化对象是 skill program，而不是单个 prompt：

| 层级 | 当前主线 |
|------|----------|
| 前向执行 | `Kode` |
| 优化对象 | `root prompt + skills + selection policy` |
| 修改动作 | `add_skill / revise_skill / drop_skill / merge_skills / adjust_selection_policy` |
| 生命周期 | `root -> generate -> evolve -> prune -> merge` |
| 推荐 demo | `python -m treeskill` |

其中，SealQA lifecycle demo 会显式展示：
- 从弱 root 起步
- 自动长出检索 / 校验 / 时效性 skill
- 自进化强化 skill 用法
- 剪枝低收益 skill
- 合并 / 重整 skill 集合

## APO 优化原理（兼容层）

APO（Automatic Prompt Optimization）引擎对齐 [Agent-Lightning](https://github.com/microsoft/agent-lightning/) 的设计，核心是 **Beam Search + 文本梯度下降**：

```
                    ┌─ 梯度模板 1 ─┐     ┌─ 编辑模板 1（激进重写）─┐
失败 Traces ──→     ├─ 梯度模板 2 ─┤ ──→ ├─ 编辑模板 2（保守修复）─┤ ──→ 评分 ──→ Top-K Beam
                    └─ 梯度模板 3 ─┘     └─ branch_factor 个候选  ─┘
```

**单轮流程：**

1. **采样 Traces** — 从反馈中选取失败案例
2. **计算文本梯度** — 随机选梯度模板（3 种），让 judge 模型分析"prompt 哪里导致了失败"
3. **生成候选** — 随机选编辑模板（激进重写 / 保守修复），每个 parent prompt 生成 `branch_factor` 个候选
4. **评分选择** — 对所有候选 + 原 beam 评分，保留 top `beam_width` 个

**两种模式：**

| 模式 | 配置 | 行为 |
|------|------|------|
| 单轨 (默认) | `beam_width=1` | 一次梯度 → N 候选 → 选最佳，兼容旧版 |
| Beam Search | `beam_width>1` | 保留 top-k prompt 跨轮优化，更稳定 |

支持断点续跑——中断后 `.evo_resume.json` 记录已完成的节点，下次自动跳过。

> 详见 [docs/design/apo-optimization.md](./docs/design/apo-optimization.md)

## Checkpoint

每次优化自动保存 checkpoint：

```
ckpt/
└── writing-assistant_v1.2_20260306_140000/
    ├── skill/          # 完整 skill 树
    │   └── SKILL.md
    └── mem/
        ├── traces.jsonl
        └── meta.json
```

```bash
# CLI 恢复
/ckpt                                                # 列出
/restore writing-assistant_v1.2_20260306_140000       # 恢复

# 命令行恢复
python -m treeskill.main --ckpt ckpt/writing-assistant_v1.2_20260306_140000
```

## 项目结构

```
treeskill/
├── pipeline_main.py       # 当前主入口：Kode + ASO pipeline
├── core/                   # 核心抽象层
│   ├── abc.py             # 抽象基类
│   ├── optimizer.py       # TrainFreeOptimizer (TGD)
│   ├── tree_optimizer.py  # TreeAwareOptimizer (拆分/剪枝)
│   ├── strategies.py      # 优化策略（保守/激进/自适应）
│   └── validators.py      # 验证器
├── adapters/              # 模型适配器
│   ├── openai.py          # OpenAI / 兼容 API
│   └── anthropic.py       # Anthropic Claude 4.5/4.6
├── schema.py              # 数据模型 (Skill, Message, Trace, ToolRef)
├── skill.py               # SKILL.md 解析器/写入器
├── skill_tree.py          # 层级 Skill 树管理 (graft/split/merge/prune)
├── aso_program.py         # Skill program 数据结构
├── aso_optimizer.py       # AS(skill)O 主循环
├── optimizer.py           # APOEngine（兼容层，旧 pipeline）
├── tools.py               # 工具系统 (HTTP, MCP, script.py)
├── resume.py              # 断点续跑状态管理
├── checkpoint.py          # Checkpoint 快照
├── cli.py                 # 交互式 CLI (兼容层)
├── registry.py            # 插件系统 (@adapter, @optimizer)
└── main.py                # 旧 chat / APO 入口
```

## 配置参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TREE_LLM_API_KEY` | — | API 密钥 |
| `TREE_LLM_BASE_URL` | `https://api.openai.com/v1` | API 地址 |
| `TREE_LLM_MODEL` | `gpt-4o` | 聊天模型 |
| `TREE_LLM_JUDGE_MODEL` | `gpt-4o` | Judge 模型（计算梯度 + 评分） |
| `TREE_LLM_TEMPERATURE` | `0.7` | 生成温度 |
| `TREE_STORAGE_TRACE_PATH` | `./data/traces.jsonl` | Trace 路径 |
| `TREE_STORAGE_SKILL_PATH` | `./skills` | Skill 目录 |
| `TREE_APO_GRADIENT_ACCUMULATION_STEPS` | `5` | 每次梯度计算的反馈样本数 |
| `TREE_APO_BEAM_WIDTH` | `1` | Beam 宽度（1=单轨，>1=beam search） |
| `TREE_APO_BRANCH_FACTOR` | `2` | 每个 parent 生成的候选数 |
| `TREE_APO_BEAM_ROUNDS` | `3` | Beam search 轮数 |

完整配置模版：[`demo/example/config.yaml`](./demo/example/config.yaml)

## 文档

| 文档 | 说明 |
|------|------|
| [APO 优化原理](./docs/design/apo-optimization.md) | 两种优化模式 + TGD 循环 + 断点续跑 |
| [树感知优化](./docs/design/tree-optimization.md) | 自动拆分、剪枝、部分优化 |
| [Skill 文件格式](./docs/design/skill-format.md) | Agent Skills 标准 + SKILL.md 格式 |
| [快速开始](./docs/QUICKSTART.md) | 5 分钟上手 |
| [使用指南](./docs/USAGE_GUIDE.md) | Skill 加载、配置管理 |
| [架构设计](./docs/ARCHITECTURE.md) | 核心架构和设计理念 |
| [核心抽象](./docs/CORE_ABSTRACTION.md) | Prompt、Gradient、Experience 接口 |
| [工具系统](./docs/TOOLS_GUIDE.md) | Python、HTTP、MCP 工具注册 |
| [Kode + MiniMax Thinking 验证](./docs/KODE_MINIMAX_THINKING.md) | 原版 Kode 与 MiniMax thinking 模式兼容结论 |
| [OpenAI 适配器](./docs/OPENAI_ADAPTER.md) | GPT-4o、o1 等 |
| [Anthropic 适配器](./docs/ANTHROPIC_ADAPTER.md) | Claude 4.5/4.6 系列 |
| [跨模型 Skill 迁移](./docs/design/cross-model-transfer.md) | 双模型 TGD：大模型优化，小模型执行 |
| [优化器详解](./docs/OPTIMIZER_COMPLETE.md) | TrainFreeOptimizer 技术文档 |
| [实验记录](./docs/EXPERIMENTS.md) | 当前主线实验与历史实验 |

## Kode 集成

TreeSkill 当前推荐使用 [Kode](https://github.com/shareAI-lab/Kode-Agent) 作为 Agent 前向引擎。Skill/program 在 Kode 的真实 agent loop 中执行（工具调用、文件操作、代码运行），ASO 根据执行结果修改 skill program。

### 安装 Kode

```bash
npm install -g @shareai-lab/kode
```

### 配置模型

在 `~/.kode.json` 中配置模型（**不要提交此文件到 git**）：

```json
{
  "modelProfiles": [
    {
      "name": "my-model",
      "modelName": "model-name",
      "provider": "anthropic",
      "baseURL": "https://your-api-endpoint",
      "apiKey": "your-api-key",
      "isActive": true,
      "createdAt": 1711500000000
    }
  ],
  "modelPointers": {
    "main": "model-name",
    "task": "model-name",
    "compact": "model-name",
    "quick": "model-name"
  },
  "defaultModelName": "model-name"
}
```

> 注意：`baseURL` 大写 URL，`apiKey` 直接写值（不支持环境变量引用）。

### 验证

```bash
kode -p "Say hello" --output-format json --dangerously-skip-permissions
```

### 运行当前主流 Demo

```bash
# 默认：SealQA lifecycle
python -m treeskill

# 更接近真实 frontier 的 ASO 最小实验
python -m treeskill sealqa-aso
```

### 架构

```
TreeSkill ASO 循环
    ↓ 写入/更新 AGENTS.md / skills/
Kode CLI (前向引擎)
    ↓ 加载 program / Skill → agent loop → 工具调用
    ↓ 返回 JSON 结果
TreeSkill 评估 (verify_fn / judge_fn / cached search)
    ↓ 计算梯度 → 生成/修改/剪枝/合并 Skill
    ↓ 回到顶部
```

## 实验结果

| 实验 | 模型 | 结果 | 说明 |
|------|------|------|------|
| 论文分类 3 类 | Qwen3.5-4B + GLM-5 | 50% → 91.7% (+41.7%) | [详细报告](./docs/DEMO_PAPER_CLASSIFICATION.md) |
| 论文分类 6 类 (split) | Qwen3.5-4B + GLM-5 | 10% → 86.7% (+76.7%) | auto-split 为 physics/cs/math |
| SealQA lifecycle (cached search) | MiniMax-M2.7 | 16.7% → 100% | `root -> generate -> evolve -> prune -> merge` 全链路演示 |
| SealQA ASO mini | MiniMax-M2.7 | 0% → 0% | 证明 skill 可增长，但无检索时不会自动补知识 |

## 致谢

- APO 优化引擎受 [Microsoft Agent-Lightning](https://github.com/microsoft/agent-lightning/) 启发
- Agent Harness 架构受 [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 启发
- Kode CLI 集成基于 [Kode Agent](https://github.com/shareAI-lab/Kode-Agent)
- 论文分类数据集来自 [书生实战营 InternLM Tutorial](https://github.com/InternLM/Tutorial)
