---
name: optimize
description: Use when the user says "/optimize", asks to improve a skill, or when you detect repeated failures (3+) in the current session.
user-invocable: true
---

# Optimize Skill via APO

Run Automatic Prompt Optimization on the current skill using conversation history as training signal.

## Workflow

### Step 1: Gather Failure Evidence

Read the evolution traces file to find recent failures:

```
cat "${CLAUDE_PLUGIN_DATA:-.evolution}/traces.jsonl" | tail -20
```

Also review the current session — identify moments where:
- The user corrected you
- The user expressed dissatisfaction
- Your output missed the mark
- You had to retry or significantly revise

### Step 2: Load Current Skill

Read the current evolved skill:

```
cat "${CLAUDE_PLUGIN_DATA:-.evolution}/current_skill.md"
```

If it doesn't exist, start from scratch — the user's use case will define the initial skill.

### Step 3: Compute Gradient (Failure Analysis)

For each failure, analyze using THREE different perspectives:

**Perspective 1 — Direct Issues:**
What specific instructions in the skill are missing, ambiguous, or wrong?

**Perspective 2 — Root Cause:**
What is the deepest reason the skill failed? Is it a structural problem, a missing constraint, or a wrong assumption?

**Perspective 3 — Quality Audit:**
Rate the skill on: instruction clarity, scope constraints, edge case handling, tone control. Which dimension caused the failure?

Synthesize into 3-5 actionable bullet points. This is your "text gradient".

### Step 4: Generate Candidates (Apply Gradient)

Generate **3 candidate** rewrites of the skill:

1. **Conservative**: Fix only the single most critical issue. Minimal changes.
2. **Balanced**: Address top 2-3 issues. Moderate restructuring allowed.
3. **Aggressive**: Full rewrite addressing all issues. May restructure, reorder, add new sections.

Each candidate must:
- Preserve domain knowledge and core intent
- Be a complete, standalone skill (not a diff)
- Be written as Markdown

### Step 5: Score Candidates

For each candidate, mentally simulate: "If I had been using this skill from the start of this session, would the failures have been avoided?"

Score each 0.0–1.0:
- 1.0 = all failures would have been prevented
- 0.5 = some failures prevented, some remain
- 0.0 = no improvement

Also score the **current skill** as baseline.

### Step 6: Select Best & Save

Pick the highest-scoring candidate. If it scores higher than the current skill:

1. Write the new skill to `${CLAUDE_PLUGIN_DATA:-.evolution}/current_skill.md`
2. Update stats:
   ```json
   {
     "version": "0.X",
     "best_score": 0.XX,
     "optimized_at": "ISO-timestamp",
     "gradient_summary": "brief description of what was fixed",
     "candidates_tested": 3
   }
   ```
   Write to `${CLAUDE_PLUGIN_DATA:-.evolution}/stats.json`
3. Tell the user what changed and why.

If no candidate beats the current skill, keep the current skill and tell the user.

### Step 7: Log the Optimization

Append to `${CLAUDE_PLUGIN_DATA:-.evolution}/optimization_log.jsonl`:
```json
{
  "timestamp": "ISO-timestamp",
  "from_version": "0.X",
  "to_version": "0.Y",
  "gradient": ["bullet1", "bullet2"],
  "accepted": true,
  "score_before": 0.XX,
  "score_after": 0.YY
}
```

## Important Notes

- **Never discard domain knowledge** — optimization should add constraints, not remove useful instructions
- **Be conservative by default** — when uncertain, prefer the conservative candidate
- **Version monotonically** — never reuse a version number
- **Show your work** — tell the user the gradient, candidates, and scores so they can verify
