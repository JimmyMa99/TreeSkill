# TreeSkill Plugin

Self-evolving skill optimization for Claude Code. Your skills **improve automatically** through APO+ (Automatic Prompt Optimization with capability growth).

Unlike static skill libraries, TreeSkill learns from your usage — fixing prompt issues, growing helper scripts, adding examples, and splitting sub-skills when needed.

## Quick Start

### Install

```bash
# Option 1: Local (for development / testing)
claude --plugin-dir ./treeskill-plugin

# Option 2: From marketplace (coming soon)
/plugin install treeskill@treeskill-marketplace
```

### Verify

Start a session and ask:

```
你有哪些 treeskill 技能？
```

Claude should list `optimize`, `optimize-advanced`, `evolve-status`, `review-session`.

### Basic Usage

```
# 1. Work normally — write code, ask questions, etc.

# 2. When you notice repeated issues, optimize:
/optimize

# 3. Check evolution progress:
/evolve-status

# 4. End-of-session review (optional):
/review-session
```

That's it. Next session, Claude automatically uses the improved skill.

## How It Works

```
┌─────────────────────────────────────────────────┐
│ Session Start                                   │
│   Hook injects: meta-skill + evolved skill      │
│                 + scripts manifest + examples    │
├─────────────────────────────────────────────────┤
│ During Session                                  │
│   You work normally. Claude follows the         │
│   evolved skill. Failures are noted.            │
├─────────────────────────────────────────────────┤
│ /optimize (manual) or /optimize-advanced        │
│   1. Gather failure traces                      │
│   2. Compute text gradient (failure analysis)   │
│   3. Classify: prompt_fix | example_needed      │
│            | script_needed | split_needed        │
│   4. Apply fixes per type                       │
│   5. Score candidates, pick best                │
│   6. Save updated skill bundle                  │
├─────────────────────────────────────────────────┤
│ Session End                                     │
│   Hook extracts traces from transcript (async)  │
│   Optional: auto-optimize if enough traces      │
└─────────────────────────────────────────────────┘
```

## Skills Reference

| Skill | Trigger | What it does |
|-------|---------|-------------|
| `/optimize` | Manual or auto-suggest | APO+ optimization — prompt fixes, examples, scripts, splits |
| `/optimize-advanced` | Manual | Python APO engine with beam search (requires `pip install treeskill`) |
| `/evolve-status` | Manual | Show evolution version, scores, and history |
| `/review-session` | End of session | Reflect on session quality, generate learning signal |

## Two Optimization Modes

### Mode A: Claude-as-Optimizer (`/optimize`)

Zero dependencies. Claude analyzes its own failures and rewrites the skill. Good for:
- Quick iterations
- When you don't have the Python engine installed
- Qualitative improvements (tone, style, constraints)

### Mode B: Python APO Engine (`/optimize-advanced`)

Precise, quantitative optimization with beam search. Requires setup:

```bash
# 1. Install treeskill
cd /path/to/evo_agent
pip install -e .

# 2. Set environment variables
export TREE_LLM_API_KEY="your-key"
export TREE_LLM_BASE_URL="https://api.siliconflow.cn/v1"
export TREE_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"
```

Good for:
- Multi-candidate beam search (tests 3+ rewrites)
- Cross-round beam pool persistence
- Reproducible scoring with judge model

## Auto-Optimization

Enable automatic optimization after each session:

```bash
export TREESKILL_AUTO_OPTIMIZE=1
export TREESKILL_AUTO_MIN_TRACES=10  # trigger after 10 correction traces
```

When enabled, the SessionEnd hook checks accumulated traces and runs the Python APO engine in the background if enough feedback has been collected.

## Skill Bundle Structure

After optimization, your `.evolution/` directory grows:

```
.evolution/
├── current_skill.md          # Optimized prompt (core)
├── scripts/                  # Helper scripts (capability growth)
│   └── validate_json.py      # e.g., JSON validator
├── examples/                 # Few-shot examples (knowledge growth)
│   └── example_001.md        # Extracted from positive traces
├── children/                 # Sub-skills (split growth)
│   ├── child_formal.md       # e.g., formal writing variant
│   └── child_casual.md       # e.g., casual writing variant
├── stats.json                # Version, scores, growth log
├── traces.jsonl              # Accumulated session traces
├── beam_pool.json            # Cross-round beam candidates
└── optimization_log.jsonl    # Full optimization history
```

## Plugin Architecture

```
treeskill-plugin/
├── .claude-plugin/
│   └── plugin.json               # Plugin manifest (name, version, etc.)
├── hooks/
│   ├── hooks.json                # Hook declarations (SessionStart + SessionEnd)
│   ├── session-start.sh          # Injects skill context into Claude
│   └── session-end.sh            # Extracts traces, optional auto-optimize
├── skills/
│   ├── using-treeskill/SKILL.md  # Meta-skill (loaded every session)
│   ├── optimize/SKILL.md         # APO+ optimization workflow
│   ├── optimize-advanced/SKILL.md# Python engine optimization
│   ├── evolve-status/SKILL.md    # Status dashboard
│   └── review-session/SKILL.md   # Session reflection
├── lib/
│   ├── extract_traces.py         # Transcript JSONL → traces parser
│   └── apo_optimize.py           # Python APO engine bridge
└── .evolution/                   # Runtime data (gitignored)
```

## Cross-Platform Support

TreeSkill follows the standard Claude Code plugin format. The same skill bundle works across:

| Platform | Install Method |
|----------|---------------|
| Claude Code | `--plugin-dir` or `/plugin install` |
| Codex (OpenAI) | Copy `SKILL.md` to `AGENTS.md` |
| Cursor | Adapt hooks to `.cursor-plugin/` format |

## License

MIT
