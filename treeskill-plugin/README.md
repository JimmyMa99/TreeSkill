# TreeSkill Plugin

Self-evolving skill optimization for Claude Code. Your skills improve automatically through APO (Automatic Prompt Optimization).

## Install

```bash
# Local development
claude --plugin-dir ./treeskill-plugin

# Or from marketplace (coming soon)
/plugin install treeskill
```

## How It Works

1. **SessionStart** — injects the current best skill + meta-instructions
2. **You work normally** — complete tasks as usual
3. **SessionEnd** — conversation traces are extracted and stored
4. **`/optimize`** — triggers APO: analyzes failures → computes gradient → rewrites skill
5. **Next session** — automatically uses the improved skill

## Skills

| Skill | Trigger | What it does |
|-------|---------|-------------|
| `/optimize` | Manual or auto-suggest | Run APO optimization on current skill |
| `/evolve-status` | Manual | Show evolution progress and history |
| `/review-session` | End of session | Reflect and generate learning signal |

## Architecture

```
treeskill-plugin/
├── .claude-plugin/plugin.json    # Plugin manifest
├── hooks/
│   ├── hooks.json                # Hook declarations
│   ├── session-start.sh          # Inject skill context
│   └── session-end.sh            # Extract traces (async)
├── skills/                       # Skill definitions
│   ├── using-treeskill/          # Meta-skill
│   ├── optimize/                 # APO optimization
│   ├── evolve-status/            # Status dashboard
│   └── review-session/           # Session reflection
├── lib/
│   └── extract_traces.py         # Transcript → Trace parser
└── .evolution/                   # Runtime data (gitignored)
    ├── traces.jsonl              # Accumulated traces
    ├── current_skill.md          # Current best skill
    ├── stats.json                # Version + score
    └── optimization_log.jsonl    # History
```

## Phase B (Coming)

Python APO engine for precise scoring and beam search optimization:
- `pip install treeskill` for the engine
- Configurable judge models
- Multi-candidate beam search
- Cross-session beam pool persistence
