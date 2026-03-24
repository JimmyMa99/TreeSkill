# TreeSkill Best Practices

## 1. Getting the Most Out of Self-Evolving Skills

### Give Clear Feedback

The quality of evolution depends entirely on feedback quality. TreeSkill learns from:

- **Corrections** — When you tell Claude "不对" / "wrong" / "fix this", it creates a negative trace
- **Positive signals** — When you say "好" / "perfect" / "thanks", it creates a positive trace
- **Silence** — No feedback = neutral, not used for optimization

**Best practice**: Be explicit. Instead of silently editing Claude's output, tell it what went wrong:

```
# Bad (Claude learns nothing)
*quietly fixes the output yourself*

# Good (Claude learns from the correction)
"不要用emoji，输出里不能有任何emoji符号"
```

### When to Run /optimize

Don't optimize too early. Wait until you have:

- **5+ correction traces** — Enough signal to compute a meaningful gradient
- **A pattern of repeated failures** — Same mistake 3+ times = strong signal
- **Mix of tasks** — Optimize after diverse usage, not just one task type

**Best practice**: Run `/evolve-status` first. If traces < 5, keep working. If you see repeated failure patterns, optimize.

### Review Optimization Results

Always review what `/optimize` changed:

```
/optimize
# Claude shows: gradient → classification → candidates → scores → winner
# Check: does the gradient match your actual complaints?
# Check: did it pick the right candidate?
```

If you disagree with the optimization, tell Claude. It will adjust.

## 2. Skill Bundle Management

### Start Simple, Grow Organically

Don't try to write a perfect skill from scratch. Let it evolve:

```
Session 1: Work normally. No skill yet.
Session 2: /optimize → v0.1 (basic constraints from corrections)
Session 3: Work with v0.1. New failures emerge.
Session 4: /optimize → v0.2 (adds examples, maybe a script)
...
Session N: Skill is well-tuned to your workflow
```

### Monitor Growth

Check your `.evolution/` directory periodically:

```bash
# How many traces?
wc -l .evolution/traces.jsonl

# Current skill version?
head -5 .evolution/stats.json

# Any scripts growing?
ls .evolution/scripts/

# Full history
cat .evolution/optimization_log.jsonl | python3 -m json.tool
```

### Reset When Needed

If the skill drifts in a wrong direction:

```bash
# Nuclear option: start fresh
rm -rf .evolution/current_skill.md .evolution/scripts/ .evolution/examples/

# Soft reset: keep traces but reset skill
rm .evolution/current_skill.md .evolution/stats.json
# Then /optimize will rebuild from all accumulated traces
```

## 3. APO+ Growth Types

### Prompt Fixes (Most Common)

These are standard APO — rewriting instructions to fix failures.

**When it works well**: Missing rules, ambiguous instructions, wrong constraints.

**When it doesn't help**: The task requires computation, external data, or fundamentally different behavior.

**Tip**: If the same prompt fix keeps appearing across multiple optimization rounds, it's probably an `example_needed` or `script_needed` in disguise.

### Example Growth

Few-shot examples extracted from your best interactions.

**When it works well**: Claude keeps making the same formatting mistake despite clear instructions. Showing one concrete example fixes it.

**Watch out for**: Too many examples bloating the context. Keep examples/ to 5-10 max. Prune old ones that overlap.

### Script Growth

Helper scripts generated when prompts can't solve the problem.

**When it works well**:
- JSON/YAML validation
- Date/time formatting
- Data transformation
- Template rendering

**Safety rules**:
- Scripts should be pure functions (input → output)
- No network calls unless you approve
- No file system modifications beyond .evolution/
- Always review generated scripts before trusting them

**Tip**: If a script is useful beyond TreeSkill, move it to your project's actual codebase.

### Sub-Skill Splits

When one skill can't serve contradictory requirements.

**When it works well**: "Write formal emails" vs "Write casual Slack messages" — opposite tone requirements in the same skill.

**When NOT to split**: Different topics but same style. A copywriting skill doesn't need splits for "hero section" vs "pricing section" — those aren't contradictory.

**Tip**: Splitting is expensive (more context, more maintenance). Only split when you see 3+ conflicting traces.

## 4. Advanced: Python APO Engine

### When to Use /optimize-advanced

Use the Python engine when:
- You want reproducible, quantitative scoring
- You need beam search (test 3+ candidates simultaneously)
- You want cross-round persistence (candidates survive between runs)
- You're doing serious prompt engineering, not casual improvement

### Configuring the Engine

```bash
# Required
export TREE_LLM_API_KEY="your-key"
export TREE_LLM_BASE_URL="https://api.siliconflow.cn/v1"
export TREE_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"

# Optional: separate judge model (recommended for better scoring)
export TREE_LLM_JUDGE_MODEL="Qwen/Qwen2.5-72B-Instruct"
```

### Tuning Beam Search

```bash
# Conservative (fast, safe)
/optimize-advanced  # defaults: beam_width=3, beam_rounds=2

# Aggressive (slower, more thorough)
# Tell Claude to run with --beam-width 5 --beam-rounds 3
```

### Auto-Optimization

For power users who want hands-free evolution:

```bash
export TREESKILL_AUTO_OPTIMIZE=1
export TREESKILL_AUTO_MIN_TRACES=10
```

This runs the Python engine after every session that accumulates enough traces. Results appear silently in `.evolution/`.

**Warning**: Auto-optimization uses API credits. Each run costs ~3-5 API calls per beam candidate.

## 5. Team Usage

### Sharing Evolved Skills

After optimizing a skill for your workflow, share it with your team:

```bash
# Export the current skill bundle
cp -r .evolution/current_skill.md ./shared-skills/my-workflow.md
cp -r .evolution/scripts/ ./shared-skills/scripts/
cp -r .evolution/examples/ ./shared-skills/examples/

# Teammates import by placing files in their .evolution/
```

### Version Control

The `.evolution/` directory is gitignored by default (traces contain conversation data). But you CAN version-control the optimized artifacts:

```bash
# Add only the output, not the traces
git add -f .evolution/current_skill.md
git add -f .evolution/scripts/
git add -f .evolution/examples/
git add -f .evolution/stats.json
# Do NOT commit traces.jsonl (contains conversation content)
```

### Per-Project vs Global Skills

- **Per-project**: Each repo evolves its own skill. Good when projects have different requirements.
- **Global**: Set `CLAUDE_PLUGIN_DATA` to a shared path. Good when your workflow is consistent across projects.

```bash
# Global skill evolution
export CLAUDE_PLUGIN_DATA="$HOME/.treeskill-evolution"
```

## 6. Troubleshooting

### "No evolution data yet"

Run `/evolve-status`. If empty: you haven't had enough sessions yet, or the SessionEnd hook isn't extracting traces.

Check: `cat .evolution/traces.jsonl` — if empty, verify hooks are loaded:

```bash
claude --plugin-dir ./treeskill-plugin --verbose
```

### Skill gets worse after optimization

This can happen if:
- Too few traces (< 5) — noise dominates signal
- All traces are similar — overfits to one task type
- Aggressive candidate was selected when conservative was safer

Fix: Reset the skill and re-optimize with more diverse traces:

```bash
rm .evolution/current_skill.md
# Then /optimize
```

### Scripts aren't being used

Check that `session-start.sh` is listing scripts in the context injection:

```bash
ls .evolution/scripts/
bash treeskill-plugin/hooks/session-start.sh
```

If scripts exist but Claude ignores them, the skill instructions may not reference them clearly enough. Run `/optimize` to let it fix the references.

### Python engine errors

```bash
# Test imports
python3 -c "from treeskill.optimizer import APOEngine; print('OK')"

# Test with verbose logging
python3 treeskill-plugin/lib/apo_optimize.py \
    --evolution-dir .evolution/ \
    --verbose
```

Common issues:
- Missing `TREE_LLM_API_KEY` → set the env var
- Missing `treeskill` package → `pip install -e .` from the evo_agent root
- Not enough traces → accumulate more sessions first
