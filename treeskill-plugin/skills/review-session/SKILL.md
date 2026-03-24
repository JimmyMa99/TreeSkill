---
name: review-session
description: Use at the end of a session or when the user asks to reflect on session quality. Generates learning signal for future optimization.
user-invocable: true
---

# Review Session

Reflect on the current session to generate learning signal for APO.

## Workflow

### 1. Scan Conversation

Review the entire conversation and identify:

**Successes** — moments where:
- User accepted output without correction
- User explicitly praised the result
- Complex task completed smoothly

**Failures** — moments where:
- User corrected the output
- User asked for a redo
- Output missed requirements
- Multiple iterations were needed

**Near-misses** — moments where:
- Output was acceptable but could have been better
- User made minor tweaks

### 2. Score the Session

Rate overall session quality 0.0–1.0:
- 1.0 = all tasks completed perfectly first try
- 0.7 = most tasks good, minor corrections
- 0.5 = mixed results, significant corrections needed
- 0.3 = frequent failures, user frustrated
- 0.0 = session was a disaster

### 3. Extract Learning Signal

For each failure/near-miss, write a structured trace:

```json
{
  "input": "what the user asked",
  "expected": "what the user wanted (based on corrections)",
  "actual": "what you produced",
  "failure_reason": "why the gap exists",
  "skill_gap": "what instruction in the skill could have prevented this"
}
```

### 4. Save Traces

Append each trace to `${CLAUDE_PLUGIN_DATA:-.evolution}/traces.jsonl`

### 5. Suggest Next Steps

Based on accumulated traces, advise:
- If 5+ traces accumulated: "Consider running `/optimize` to evolve the skill"
- If mostly successes: "Skill is performing well. No optimization needed yet."
- If specific pattern of failures: "Noticed repeated issues with [X]. `/optimize` could help."

### 6. Present to User

Show a brief session report:
```
## Session Review

Score: X.X/1.0
Successes: N | Failures: N | Near-misses: N

### Key Learnings
- [learning 1]
- [learning 2]

### Recommendation
[next step suggestion]
```
