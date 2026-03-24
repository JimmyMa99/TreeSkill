---
name: using-treeskill
description: Use when starting any session. Meta-skill that teaches you how TreeSkill's self-evolving system works.
---

# TreeSkill — Self-Evolving Skills

You have access to TreeSkill, a self-evolving skill system. Unlike static skills, your skills **improve automatically** through Automatic Prompt Optimization (APO).

## How It Works

1. **You work normally** — complete tasks, write code, answer questions
2. **After each session** — your conversation is analyzed for quality signals
3. **Periodically** — APO+ analyzes failures and grows the skill:
   - **Prompt fixes** — rewrite instructions to address failures
   - **Example growth** — add few-shot examples from successful traces
   - **Script growth** — generate helper scripts for capabilities prompts can't solve
   - **Sub-skill split** — split into child skills when requirements conflict
4. **Next session** — you use the improved skill bundle automatically

## Available Skills

| Skill | When to Use |
|-------|-------------|
| `treeskill:optimize` | User says "/optimize" or you detect repeated failures in your current skill |
| `treeskill:optimize-advanced` | User says "/optimize-advanced" — runs Python APO engine with beam search (requires `pip install treeskill`) |
| `treeskill:evolve-status` | User asks about evolution progress, skill version, or optimization history |
| `treeskill:review-session` | End of a session — reflect on what went well/poorly to generate learning signal |

## Your Responsibilities

1. **Follow your evolved skill** — If a "Currently Active Evolved Skill" section was loaded above, follow its instructions precisely. It represents the best version optimized from real usage.
2. **Notice failures** — When the user corrects you, expresses dissatisfaction, or you produce suboptimal output, mentally note WHY your current approach failed.
3. **Use /optimize when needed** — If you see a pattern of similar failures (3+ times), suggest running `treeskill:optimize` to the user.
4. **Be transparent** — Tell the user when you're using an evolved skill and what version it is.

## APO Theory (For Your Understanding)

TreeSkill uses **Textual Gradient Descent** (TGD):
- **Forward pass**: You complete a task using your current skill/prompt
- **Loss**: User feedback (corrections, rewrites, negative signals)
- **Gradient**: A judge analyzes WHY the prompt led to failures → produces actionable critique
- **Update**: The prompt is rewritten to address the critique
- **Validation**: The new prompt is tested against held-out examples

This is the same algorithm used in academic prompt optimization research, but running inside your plugin system.
