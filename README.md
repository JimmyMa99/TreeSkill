# EvoSkill

**训练无关的 Agent Prompt 自进化框架**

把 System Prompt 当作"权重"，把交互反馈当作"训练信号"，通过文本梯度下降（TGD）让 prompt 自动进化——不需要训练模型，不需要标注数据，只靠 API 调用。

```
用户反馈 → 诊断失败 → 计算文本梯度 → 重写 Prompt → 更好的 Agent
```

## 特性

- **训练无关** — 纯 API 调用，无需 GPU、无需微调
- **模型无关** — 支持 OpenAI、Anthropic、任何 OpenAI 兼容 API
- **Agent Skills 标准** — Skill 文件遵循 [agentskills.io](https://agentskills.io) 开放标准
- **层级 Skill 树** — 自动拆分、剪枝，递归优化
- **断点续跑** — 优化中断后可从上次进度恢复
- **内置工具** — 文件操作、Shell 执行、代码搜索

## 安装

```bash
cd /path/to/evo_agent
pip install -e .
```

## 快速开始

### 1. 配置 API

```bash
cp demo/example/config.yaml my-config.yaml
# 编辑 my-config.yaml，填入 API Key
```

或用环境变量：

```env
EVO_LLM_API_KEY=your-api-key
EVO_LLM_BASE_URL=https://api.siliconflow.cn/v1
EVO_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
```

### 2. 启动

```bash
# 用已有 skill 目录
python -m evoskill.main --config my-config.yaml --skill skills/fast-test-skill

# 用默认 skill（自动创建）
python -m evoskill.main --skill default

# 用 skill 树目录
python -m evoskill.main --skill my-skills/
```

### 3. 在对话中优化

```
You: 帮我写一段关于春天的短文

🤖 Assistant: [生成结果]

You: /bad 太像AI写的，缺乏生活气息       ← 标记不好 + 原因
You: /rewrite 春天来了，小区的玉兰花...    ← 提供理想回答
You: /target 更像人，有生活气息            ← 设置优化方向
You: /optimize                            ← 一键优化

✓ Skill optimized → writing-assistant (v1.0 → v1.1) (checkpoint saved)
```

## Skill 文件格式

遵循 [Agent Skills 标准](https://agentskills.io/specification)，每个 Skill 是一个目录：

```
my-skill/
├── SKILL.md          # YAML frontmatter + Markdown body
└── config.yaml       # 可选：few-shot、temperature 等
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

优化时 bottom-up：先叶子后父节点。反馈矛盾时自动建议拆分。

> 详见 [docs/design/tree-optimization.md](./docs/design/tree-optimization.md)

## 命令一览

| 命令 | 作用 |
|------|------|
| `/bad <原因>` | 标记上条回复不好 |
| `/rewrite <文本>` | 提供理想回答 |
| `/target <方向>` | 设置优化方向 |
| `/optimize` | 触发 APO 优化（支持断点续跑） |
| `/image <路径>` | 附加图片（多模态） |
| `/save` | 保存当前 skill |
| `/tree` | 显示技能树 |
| `/select <路径>` | 切换子技能（如 `social.moments`） |
| `/split` | 分析是否需要拆分 |
| `/ckpt` | 列出 checkpoint |
| `/restore <名称>` | 从 checkpoint 恢复 |
| `/tools` | 查看可用工具 |
| `/quit` | 退出 |

## APO 优化原理

```
失败案例 → 文本梯度（分析原因） → 重写 Prompt → 验证 → 版本 +1
```

- **交互式**：人工 `/bad`、`/rewrite` 反馈 → `/optimize`
- **全自动**：测试集 + LLM Judge → 循环优化直到达标

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
python -m evoskill.main --ckpt ckpt/writing-assistant_v1.2_20260306_140000
```

## 项目结构

```
evoskill/
├── core/                   # 核心抽象层
│   ├── abc.py             # 抽象基类
│   ├── optimizer.py       # TrainFreeOptimizer (TGD)
│   ├── tree_optimizer.py  # TreeAwareOptimizer (拆分/剪枝)
│   ├── strategies.py      # 优化策略（保守/激进/自适应）
│   └── validators.py      # 验证器
├── adapters/              # 模型适配器
│   ├── openai.py          # OpenAI / 兼容 API
│   └── anthropic.py       # Anthropic Claude 4.5/4.6
├── schema.py              # 数据模型 (Skill, Message, Trace)
├── skill.py               # SKILL.md 解析器/写入器
├── skill_tree.py          # 层级 Skill 树管理
├── optimizer.py           # APOEngine (交互式优化)
├── resume.py              # 断点续跑状态管理
├── checkpoint.py          # Checkpoint 快照
├── cli.py                 # 交互式 CLI
├── tools.py               # 工具注册 (@tool, HTTP, MCP)
├── builtin_tools.py       # 内置工具 (文件/Shell)
├── registry.py            # 插件系统 (@adapter, @optimizer)
└── main.py                # 入口
```

## 配置参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EVO_LLM_API_KEY` | — | API 密钥 |
| `EVO_LLM_BASE_URL` | `https://api.openai.com/v1` | API 地址 |
| `EVO_LLM_MODEL` | `gpt-4o` | 聊天模型 |
| `EVO_LLM_JUDGE_MODEL` | `gpt-4o` | Judge 模型 |
| `EVO_LLM_TEMPERATURE` | `0.7` | 生成温度 |
| `EVO_STORAGE_TRACE_PATH` | `./data/traces.jsonl` | Trace 路径 |
| `EVO_STORAGE_SKILL_PATH` | `./skills` | Skill 目录 |
| `EVO_APO_GRADIENT_ACCUMULATION_STEPS` | `5` | 每次优化的反馈样本数 |

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
| [OpenAI 适配器](./docs/OPENAI_ADAPTER.md) | GPT-4o、o1 等 |
| [Anthropic 适配器](./docs/ANTHROPIC_ADAPTER.md) | Claude 4.5/4.6 系列 |
| [跨模型 Skill 迁移](./docs/design/cross-model-transfer.md) | 双模型 TGD：大模型优化，小模型执行 |
| [优化器详解](./docs/OPTIMIZER_COMPLETE.md) | TrainFreeOptimizer 技术文档 |
| [树优化 Demo](./docs/TREE_OPTIMIZATION_DEMO.md) | 10 分钟最小化树优化最佳实践（论文分类） |

## 致谢

- APO 优化引擎受 [Microsoft Agent-Lightning](https://github.com/microsoft/agent-lightning/) 启发，感谢他们的开源贡献。
- 论文分类数据集来自 [书生实战营 InternLM Tutorial](https://github.com/InternLM/Tutorial)。
