---
name: optimize-advanced
description: Use when the user says "/optimize-advanced" or wants to run the Python APO engine with beam search for more precise optimization. Requires treeskill Python package installed.
user-invocable: true
---

# Advanced APO Optimization (Python Engine)

Run the full Python APO engine with beam search, multi-candidate scoring, and cross-round persistence. This is more precise than the built-in `/optimize` which relies on Claude's self-evaluation.

## Prerequisites

The `treeskill` Python package must be installed:
```bash
pip install -e /path/to/evo_agent
```

And environment variables must be set:
```bash
export TREE_LLM_API_KEY="your-key"
export TREE_LLM_BASE_URL="https://api.siliconflow.cn/v1"
export TREE_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"
```

## Workflow

### Step 1: Check Prerequisites

```bash
python3 -c "from treeskill.optimizer import APOEngine; print('treeskill OK')" 2>&1
```

If this fails, tell the user to install treeskill first.

### Step 2: Check Traces

```bash
wc -l "${CLAUDE_PLUGIN_DATA:-.evolution}/traces.jsonl" 2>/dev/null
```

Need at least 3 traces with feedback to optimize.

### Step 3: Run APO Engine

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/lib/apo_optimize.py" \
    --evolution-dir "${CLAUDE_PLUGIN_DATA:-.evolution}" \
    --beam-width 3 \
    --beam-rounds 2 \
    --verbose
```

### Step 4: Report Results

Read the output JSON from the script. Show the user:
- Old version → New version
- Number of traces used
- Beam search results
- What changed in the skill

Then read the updated skill:
```bash
cat "${CLAUDE_PLUGIN_DATA:-.evolution}/current_skill.md"
```

### Parameters

Users can customize:
- `--beam-width N` — wider beam = more candidates, slower but more thorough (default: 3)
- `--beam-rounds N` — more rounds = deeper search (default: 2)
- `--min-traces N` — minimum feedback traces to start optimization (default: 3)
- `--config path/to/config.yaml` — use specific LLM config instead of env vars
