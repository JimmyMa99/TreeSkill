---
name: evolve-status
description: Use when the user asks about skill evolution progress, version history, optimization logs, or current skill state.
user-invocable: true
---

# Evolution Status

Show the current state of skill evolution.

## Workflow

### 1. Read Current Skill

```
cat "${CLAUDE_PLUGIN_DATA:-.evolution}/current_skill.md"
```

### 2. Read Stats

```
cat "${CLAUDE_PLUGIN_DATA:-.evolution}/stats.json"
```

### 3. Read Optimization History

```
cat "${CLAUDE_PLUGIN_DATA:-.evolution}/optimization_log.jsonl"
```

### 4. Read Recent Traces

```
tail -10 "${CLAUDE_PLUGIN_DATA:-.evolution}/traces.jsonl"
```

### 5. Present Summary

Format as a clear table:

```
## Skill Evolution Status

| Metric | Value |
|--------|-------|
| Current Version | vX.Y |
| Best Score | 0.XX |
| Last Optimized | YYYY-MM-DD |
| Total Optimizations | N |
| Accumulated Traces | N |

### Recent Changes
- vX.Y: [what changed]
- vX.Z: [what changed]

### Current Skill Preview
> First 3 lines of current_skill.md...
```

If no evolution data exists yet, tell the user:
"No evolution data yet. Use your skills normally — after accumulating enough session traces, run `/optimize` to start evolving."
