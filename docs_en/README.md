# TreeSkill

> **Note**: This project was formerly named **EvoSkill**. It has been renamed to avoid conflicts with another repository. All future updates will be made here at [TreeSkill](https://github.com/JimmyMa99/TreeSkill).

English | [中文](../README.md)

**Train-Free Skill Optimization Framework for LLMs**

Treat the System Prompt as "weights" and interaction feedback as "training signals" -- through Textual Gradient Descent (TGD), prompts evolve automatically. No training, no labeling, just API calls.

```
User Feedback -> Diagnose Failure -> Compute Text Gradient -> Beam Search Rewrite -> Better Agent
```

## Core Philosophy

TreeSkill draws an analogy between LLM prompt optimization and deep learning's training loop, but **requires no model training at all**:

| Deep Learning | TreeSkill |
|----------|----------|
| Model Weights | System Prompt |
| Training Data | Interaction Feedback (Human or LLM Judge) |
| Loss Function | Failure Case Analysis |
| Gradient | Text Gradient (failure attribution in natural language) |
| Parameter Update | Prompt Rewrite (Beam Search with multiple candidates) |
| Epoch | Optimization Round (with checkpoint resume support) |

## Features

- **Train-Free** -- Pure API calls; no GPU, no fine-tuning, no labeled data required
- **Human-in-the-Loop** -- Human feedback drives optimization: `/bad` to flag failures, `/rewrite` to provide ideal responses, `/target` to set optimization direction, letting domain experts directly participate in prompt evolution
- **Beam Search APO** -- Aligned with [Agent-Lightning](https://github.com/microsoft/agent-lightning/)'s optimization algorithm: multi-template gradient analysis x multi-candidate generation x Beam Search selection, continuously retaining top-k prompts across optimization rounds
- **Model Agnostic** -- Supports OpenAI, Anthropic, and any OpenAI-compatible API (SiliconFlow, Ollama, etc.)
- **Agent Skills Standard** -- Skill files follow the [agentskills.io](https://agentskills.io) open standard
- **Hierarchical Skill Tree** -- Automatic splitting, pruning, and grafting with recursive bottom-up optimization
- **Multi-Protocol Tool System** -- Skills can declare and invoke external tools in multiple formats:
  - **Python Scripts** -- Functions in `script.py` are automatically registered as tools
  - **HTTP API** -- Declarative invocation of any REST endpoint
  - **MCP Server** -- Tool invocation compatible with [Model Context Protocol](https://modelcontextprotocol.io/)
- **Checkpoint Resume** -- Resume from last progress after interruption, without wasting completed API calls

## Installation

```bash
git clone https://github.com/JimmyMa99/TreeSkill.git
cd TreeSkill
pip install -e .
```

## Quick Start

### 1. Configure API

```bash
cp demo/example/config.yaml my-config.yaml
# Edit my-config.yaml and fill in your API Key
```

Or use environment variables:

```env
TREE_LLM_API_KEY=your-api-key
TREE_LLM_BASE_URL=https://api.siliconflow.cn/v1
TREE_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
```

### 2. Launch

```bash
# Use an existing skill directory
python -m treeskill.main --config my-config.yaml --skill skills/fast-test-skill

# Use default skill (auto-created)
python -m treeskill.main --skill default

# Use a skill tree directory
python -m treeskill.main --skill my-skills/
```

### 3. Human-in-the-Loop Optimization

The core interaction mode of TreeSkill is **human-machine collaborative optimization**: domain experts guide the APO engine to improve prompts through natural language feedback.

```
You: Write a short essay about spring

Assistant: [generated result]

You: /bad Too AI-like, lacks a personal touch        <- Flag failure + reason
You: /rewrite Spring has arrived. The magnolias...    <- Provide ideal response (optional)
You: /target More human, with warmth and personality  <- Set optimization direction (optional)
You: /optimize                                       <- Trigger APO optimization

Skill optimized -> writing-assistant (v1.0 -> v1.1) (checkpoint saved)
```

Each `/bad` and `/rewrite` generates a Trace (an interaction record with feedback). When `/optimize` is triggered, the APO engine extracts failure patterns from these Traces, computes text gradients, and rewrites the prompt. **Domain experts don't need to understand prompt engineering -- they only need to judge whether the response is good or not.**

Dataset-driven mode is also supported:

```bash
# Fully automatic: LLM Judge scoring -> APO optimization, no human intervention
python -m treeskill.main --optimize --dataset train.jsonl --skill my-skill --no-resume

# Human-machine collaborative annotation: auto-judge scores, humans can override anytime (preference signals -> guide the judge)
python -m treeskill.main --annotate --dataset train.jsonl --skill my-skill

# Purely manual annotation
python -m treeskill.main --annotate --dataset train.jsonl --skill my-skill --manual
```

In annotation mode, human feedback serves as natural language preference signals, used both as input for APO gradients and exportable as DPO fine-tuning data.

In the CLI, typing `/` brings up a list of slash command candidates. As you type a command prefix, the list narrows down automatically, making it easy to discover and select available commands.

## Skill File Format

Following the [Agent Skills Standard](https://agentskills.io/specification), each Skill is a directory:

```
my-skill/
├── SKILL.md          # YAML frontmatter + Markdown body (= system prompt)
├── config.yaml       # Optional: few-shot, temperature, tool declarations, schedules, etc.
└── script.py         # Optional: Python tool functions
```

**SKILL.md Example:**

```markdown
---
name: my-writing-assistant
description: A professional writing assistant that helps draft and polish various texts.
metadata:
  version: "1.0"
  target: "More human-like, with warmth"
---

You are a professional writing assistant.
Your language should be natural and fluent. Avoid sounding like AI.
```

> See [docs/design/skill-format.md](./docs/design/skill-format.md) for details

## Tool System

Skills can declare external tools, and the agent invokes them as needed during conversation. Three protocols are supported:

**config.yaml declaration example:**

```yaml
tools:
  # HTTP API tool
  - name: weather
    type: http
    endpoint: https://api.weather.com/current
    method: GET
    description: Get current weather

  # MCP tool
  - name: database
    type: mcp
    mcp_server: localhost:5000
    tool_name: query
    description: Query the database
```

**script.py auto-registration:**

```python
# Public functions in script.py automatically become tools
def search_docs(query: str) -> str:
    """Search the document repository"""
    ...
```

Tool inheritance: child Skills automatically inherit their parent Skill's tool declarations, with same-name overrides.

> See [docs/TOOLS_GUIDE.md](./docs/TOOLS_GUIDE.md) for details

## Skill Tree

Skills form a hierarchy through directory nesting:

```
writing-skills/
├── SKILL.md              # Root: general writing
├── social/
│   ├── SKILL.md          # Social writing
│   └── moments/
│       └── SKILL.md      # Social media specialist
└── business/
    ├── SKILL.md          # Business writing
    └── email/
        └── SKILL.md
```

Optimization is bottom-up: leaf nodes first, then parent nodes. When feedback conflicts are detected, automatic splitting is suggested. Graft support enables cross-tree skill reuse.

> See [docs/design/tree-optimization.md](./docs/design/tree-optimization.md) for details

## Command Reference

| Command | Description |
|------|------|
| `/bad <reason>` | Flag the last response as bad |
| `/rewrite <text>` | Provide an ideal response (also accumulates DPO preference data) |
| `/export-dpo <output.jsonl>` | Export DPO preference data (for fine-tuning) |
| `/target <direction>` | Set the optimization direction |
| `/optimize` | Trigger APO optimization (supports checkpoint resume) |
| `/image <path>` | Attach an image (multimodal) |
| `/audio <path>` | Attach audio (voice input) |
| `/save` | Save the current skill |
| `/tree` | Display the skill tree |
| `/select <path>` | Switch to a sub-skill (e.g., `social.moments`) |
| `/split` | Analyze whether splitting is needed |
| `/ckpt` | List checkpoints |
| `/restore <name>` | Restore from a checkpoint |
| `/tools` | View available tools |
| `/quit` | Exit |

## APO Optimization Principles

The APO (Automatic Prompt Optimization) engine is aligned with [Agent-Lightning](https://github.com/microsoft/agent-lightning/)'s design. The core mechanism is **Beam Search + Textual Gradient Descent**:

```
                    +- Gradient Template 1 -+     +- Edit Template 1 (aggressive rewrite) -+
Failed Traces -->   +- Gradient Template 2 -+ --> +- Edit Template 2 (conservative fix)    -+ --> Score --> Top-K Beam
                    +- Gradient Template 3 -+     +- branch_factor candidates              -+
```

**Single-round flow:**

1. **Sample Traces** -- Select failure cases from feedback
2. **Compute Text Gradients** -- Randomly select a gradient template (3 types); the judge model analyzes "what in the prompt caused the failure"
3. **Generate Candidates** -- Randomly select an edit template (aggressive rewrite / conservative fix); each parent prompt generates `branch_factor` candidates
4. **Score and Select** -- Score all candidates + original beam, retain the top `beam_width`

**Two modes:**

| Mode | Configuration | Behavior |
|------|------|------|
| Single-track (default) | `beam_width=1` | One gradient -> N candidates -> select best; backward compatible |
| Beam Search | `beam_width>1` | Retain top-k prompts across rounds for more stable optimization |

Checkpoint resume is supported -- after interruption, `.evo_resume.json` records completed nodes, which are automatically skipped on the next run.

> See [docs/design/apo-optimization.md](./docs/design/apo-optimization.md) for details

## Checkpoint

Each optimization automatically saves a checkpoint:

```
ckpt/
└── writing-assistant_v1.2_20260306_140000/
    ├── skill/          # Complete skill tree
    │   └── SKILL.md
    └── mem/
        ├── traces.jsonl
        └── meta.json
```

```bash
# CLI restore
/ckpt                                                # List
/restore writing-assistant_v1.2_20260306_140000       # Restore

# Command-line restore
python -m treeskill.main --ckpt ckpt/writing-assistant_v1.2_20260306_140000
```

## Project Structure

```
treeskill/
├── core/                   # Core abstraction layer
│   ├── abc.py             # Abstract base classes
│   ├── optimizer.py       # TrainFreeOptimizer (TGD)
│   ├── tree_optimizer.py  # TreeAwareOptimizer (split/prune)
│   ├── strategies.py      # Optimization strategies (conservative/aggressive/adaptive)
│   └── validators.py      # Validators
├── adapters/              # Model adapters
│   ├── openai.py          # OpenAI / compatible APIs
│   └── anthropic.py       # Anthropic Claude 4.5/4.6
├── schema.py              # Data models (Skill, Message, Trace, ToolRef)
├── skill.py               # SKILL.md parser/writer
├── skill_tree.py          # Hierarchical Skill tree management (graft/split/merge/prune)
├── optimizer.py           # APOEngine (Beam Search + single-track)
├── tools.py               # Tool system (HTTP, MCP, script.py)
├── resume.py              # Checkpoint resume state management
├── checkpoint.py          # Checkpoint snapshots
├── cli.py                 # Interactive CLI (Human-in-the-Loop)
├── registry.py            # Plugin system (@adapter, @optimizer)
└── main.py                # Entry point
```

## Configuration Reference

| Variable | Default | Description |
|------|--------|------|
| `TREE_LLM_API_KEY` | -- | API key |
| `TREE_LLM_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `TREE_LLM_MODEL` | `gpt-4o` | Chat model |
| `TREE_LLM_JUDGE_MODEL` | `gpt-4o` | Judge model (for gradient computation + scoring) |
| `TREE_LLM_TEMPERATURE` | `0.7` | Generation temperature |
| `TREE_STORAGE_TRACE_PATH` | `./data/traces.jsonl` | Trace file path |
| `TREE_STORAGE_SKILL_PATH` | `./skills` | Skill directory |
| `TREE_APO_GRADIENT_ACCUMULATION_STEPS` | `5` | Number of feedback samples per gradient computation |
| `TREE_APO_BEAM_WIDTH` | `1` | Beam width (1=single-track, >1=beam search) |
| `TREE_APO_BRANCH_FACTOR` | `2` | Number of candidates generated per parent |
| `TREE_APO_BEAM_ROUNDS` | `3` | Beam search rounds |

Full configuration template: [`demo/example/config.yaml`](./demo/example/config.yaml)

## Documentation

| Document | Description |
|------|------|
| [APO Optimization Principles](./docs/design/apo-optimization.md) | Two optimization modes + TGD loop + checkpoint resume |
| [Tree-Aware Optimization](./docs/design/tree-optimization.md) | Automatic splitting, pruning, partial optimization |
| [Skill File Format](./docs/design/skill-format.md) | Agent Skills standard + SKILL.md format |
| [Quick Start](./docs/QUICKSTART.md) | Get started in 5 minutes |
| [Usage Guide](./docs/USAGE_GUIDE.md) | Skill loading, configuration management |
| [Architecture](./docs/ARCHITECTURE.md) | Core architecture and design philosophy |
| [Core Abstractions](./docs/CORE_ABSTRACTION.md) | Prompt, Gradient, Experience interfaces |
| [Tool System](./docs/TOOLS_GUIDE.md) | Python, HTTP, MCP tool registration |
| [OpenAI Adapter](./docs/OPENAI_ADAPTER.md) | GPT-4o, o1, etc. |
| [Anthropic Adapter](./docs/ANTHROPIC_ADAPTER.md) | Claude 4.5/4.6 series |
| [Cross-Model Skill Transfer](./docs/design/cross-model-transfer.md) | Dual-model TGD: large model optimizes, small model executes |
| [Optimizer Deep Dive](./docs/OPTIMIZER_COMPLETE.md) | TrainFreeOptimizer technical documentation |
| [Tree Optimization Demo](./docs/TREE_OPTIMIZATION_DEMO.md) | 10-minute minimal tree optimization best practice (paper classification) |

## Acknowledgments

- The APO optimization engine is inspired by [Microsoft Agent-Lightning](https://github.com/microsoft/agent-lightning/). Thanks to them for their open-source contribution.
- The paper classification dataset comes from [InternLM Tutorial](https://github.com/InternLM/Tutorial).
