# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TreeSkill is a train-free prompt optimization framework using Textual Gradient Descent (TGD). It treats system prompts as "weights" and interaction history as "dataset" to iteratively improve prompts via API calls only—no model training required.

## Commands

```bash
# Install
pip install -e .

# Run tests
pytest test_*.py

# Start interactive CLI with skill
python -m treeskill.main --skill <path-to-skill.yaml-or-directory>

# Start with config file
python -m treeskill.main --config <config.yaml> --skill <skill>

# Run demos
python demo/demo_from_scratch.py    # Build skill from scratch
python demo/demo_from_skill.py      # Optimize existing skill

# Example scripts
python example_optimizer.py              # Interactive optimization demo
python example_tools.py                  # Tool registration demo
python example_fully_automatic.py        # Fully automated optimization
python example_tree_optimization.py      # Tree-aware optimization (split/prune)
python example_load_skill_and_config.py  # Load stored skill + config
```

## Architecture

### Core Layers

```
treeskill/
├── core/              # Core abstraction layer (v0.2)
│   ├── abc.py         # Abstract base classes (OptimizablePrompt, ModelAdapter, etc.)
│   ├── optimizer.py   # TrainFreeOptimizer - TGD algorithm
│   ├── tree_optimizer.py # TreeAwareOptimizer - auto split/prune
│   ├── strategies.py  # Optimization strategies (conservative/aggressive/adaptive)
│   └── validators.py  # Prompt validators
│
├── adapters/          # Model adapters
│   ├── openai.py      # OpenAI-compatible APIs
│   └── anthropic.py   # Anthropic Claude
│
├── registry.py        # Plugin registry (@adapter, @optimizer, @hook)
├── tools.py           # Tool registry (Python functions, HTTP, MCP)
│
└── [legacy]           # v0.1 backward compat
    ├── schema.py, skill.py, skill_tree.py
    ├── checkpoint.py, storage.py
    └── optimizer.py (APOEngine)
```

### Data Flow (Interactive Mode)

```
User Input → compile_messages() → LLM.generate() → Trace
                                                    ↓
User Feedback (/bad, /rewrite, /target) ←───────────┘
                    ↓
APOEngine.optimize() / TrainFreeOptimizer.optimize()
    1. Extract failures (negative feedback)
    2. compute_gradient() - Judge analyzes why prompt failed
    3. apply_gradient() - LLM rewrites prompt
    4. bump_version() - v1.0 → v1.1
                    ↓
CheckpointManager.save() + skill_module.save()
```

### Tree-Aware Optimization

Skill trees use directory structure for hierarchy:

```
my-skills/
├── _meta.yaml
├── root.yaml
├── social/
│   ├── _meta.yaml
│   └── moments.yaml
└── business/
    └── email.yaml
```

Tree optimization (bottom-up):
1. Optimize leaf nodes first
2. `analyze_split_need()` - detect contradictory feedback → suggest split
3. `generate_child_prompts()` - create child-specific prompts
4. `analyze_prune_need()` - remove low-performance nodes
5. Save tree (auto-cleanup pruned directories)

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| OptimizablePrompt | `core/abc.py` | Base class for text/multimodal/structured prompts |
| ModelAdapter | `core/abc.py` | Interface for LLM providers (generate, compute_gradient, apply_gradient) |
| TrainFreeOptimizer | `core/optimizer.py` | TGD optimization loop |
| TreeAwareOptimizer | `core/tree_optimizer.py` | Tree optimization with auto-split/prune |
| Skill | `schema.py` | Versioned prompt container (legacy, still used) |
| SkillTree | `skill_tree.py` | Hierarchical skill management |
| CheckpointManager | `checkpoint.py` | Snapshot management for rollback |

### CLI Commands

| Command | Description |
|---------|-------------|
| `/bad <reason>` | Mark last response as negative |
| `/rewrite <text>` | Provide ideal answer |
| `/target <direction>` | Set optimization direction |
| `/optimize` | Trigger TGD optimization + checkpoint |
| `/tree` | Display skill tree |
| `/select <path>` | Switch active skill node |
| `/split` | Analyze and split skill |
| `/ckpt` | List checkpoints |
| `/restore <name>` | Restore from checkpoint |

### Configuration

Priority: Environment variables > `.env` > YAML config > defaults

```bash
TREE_LLM_API_KEY=...
TREE_LLM_BASE_URL=https://api.siliconflow.cn/v1
TREE_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
TREE_LLM_JUDGE_MODEL=Qwen/Qwen2.5-14B-Instruct
```

Config template: `demo/example/config.yaml`
Skill template: `demo/example/skill.yaml`

## Development Notes

- **v0.2 migration**: New `treeskill/` package with core abstractions; `evo_framework/` legacy still works with deprecation warning
- **YAML encoding**: `skill.save()` uses `allow_unicode=True` for readable Chinese
- **Trace deduplication**: Known issue - `/bad` and `/rewrite` may append same trace twice (should be fixed with ID-based dedup)
- **APO max_steps**: `APOConfig.max_steps` exists but multi-step loop not yet implemented
