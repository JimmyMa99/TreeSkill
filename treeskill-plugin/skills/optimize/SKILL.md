---
name: optimize
description: Use when the user says "/optimize", asks to improve a skill, or when you detect repeated failures (3+) in the current session.
user-invocable: true
---

# Optimize Skill via APO+

Run Automatic Prompt Optimization on the current skill, with **capability growth** — not just prompt fixes, but also new scripts, examples, and sub-skills when needed.

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

### Step 2: Load Current Skill Bundle

Read the current evolved skill:

```
cat "${CLAUDE_PLUGIN_DATA:-.evolution}/current_skill.md"
```

Also check for existing helper scripts and examples:

```
ls "${CLAUDE_PLUGIN_DATA:-.evolution}/scripts/" 2>/dev/null
ls "${CLAUDE_PLUGIN_DATA:-.evolution}/examples/" 2>/dev/null
```

If nothing exists, start from scratch — the user's use case will define the initial skill.

### Step 3: Compute Gradient (Failure Analysis)

For each failure, analyze using THREE different perspectives:

**Perspective 1 — Direct Issues:**
What specific instructions in the skill are missing, ambiguous, or wrong?

**Perspective 2 — Root Cause:**
What is the deepest reason the skill failed? Is it a structural problem, a missing constraint, or a wrong assumption?

**Perspective 3 — Quality Audit:**
Rate the skill on: instruction clarity, scope constraints, edge case handling, tone control. Which dimension caused the failure?

Synthesize into 3-5 actionable bullet points. This is your "text gradient".

### Step 4: Classify Gradients (Capability Gap Detection)

**This is the key step.** For each gradient bullet, classify it:

| Type | Signal | Action |
|------|--------|--------|
| **prompt_fix** | Instruction is missing/ambiguous/wrong | Rewrite prompt text (standard APO) |
| **example_needed** | Claude keeps making the same mistake despite clear instructions | Add a few-shot example from positive traces |
| **script_needed** | The failure requires computation, parsing, or external data that no prompt can solve | Generate a helper script |
| **split_needed** | Contradictory requirements — two tasks need opposite behaviors | Split into sub-skills |

Present the classification as a table:

```
| # | Gradient | Type | Rationale |
|---|----------|------|-----------|
| 1 | "No emoji in output" | prompt_fix | Clear instruction addition |
| 2 | "JSON output keeps having syntax errors" | script_needed | Need a validator script |
| 3 | "Formal tone for email but casual for chat" | split_needed | Contradictory tone requirements |
| 4 | "Keeps forgetting the date format" | example_needed | Show a concrete example |
```

### Step 5a: Prompt Fixes (Standard APO)

For all `prompt_fix` gradients, generate **3 candidate** rewrites:

1. **Conservative**: Fix only the single most critical issue. Minimal changes.
2. **Balanced**: Address top 2-3 issues. Moderate restructuring allowed.
3. **Aggressive**: Full rewrite addressing all issues. May restructure, reorder, add new sections.

Each candidate must:
- Preserve domain knowledge and core intent
- Be a complete, standalone skill (not a diff)
- Be written as Markdown

### Step 5b: Example Growth

For all `example_needed` gradients:

1. Search traces for **positive examples** (where the user was satisfied):
   ```
   grep '"positive"' "${CLAUDE_PLUGIN_DATA:-.evolution}/traces.jsonl"
   ```

2. If positive traces exist, extract the (input, output) pair as a few-shot example.

3. If no positive traces, construct an ideal example based on the user's corrections.

4. Write examples to `${CLAUDE_PLUGIN_DATA:-.evolution}/examples/`:
   ```
   examples/
   ├── example_001.md    # "Input: ... → Expected output: ..."
   ├── example_002.md
   └── ...
   ```

5. Add a reference in the skill: `## Examples\nSee examples/ directory for reference outputs.`

### Step 5c: Script Growth

For all `script_needed` gradients:

1. Identify what capability is missing (e.g., JSON validation, date parsing, data fetching).

2. Write a **minimal, self-contained** Python/Bash script:
   ```
   ${CLAUDE_PLUGIN_DATA:-.evolution}/scripts/validate_json.py
   ```

3. The script must:
   - Take input via stdin or arguments
   - Output to stdout
   - Have no dependencies beyond Python stdlib (or explicitly listed pip packages)
   - Include a docstring explaining what it does and how to use it

4. Add a reference in the skill:
   ```markdown
   ## Available Helper Scripts
   - `scripts/validate_json.py` — validates JSON output before returning to user
   ```

5. Add usage instructions in the skill telling Claude when and how to call the script.

### Step 5d: Sub-Skill Split

For all `split_needed` gradients:

1. Identify the conflicting requirements.

2. Define 2-3 child skills with clear scoping:
   ```
   parent_skill/
   ├── current_skill.md          # Updated parent with routing logic
   ├── child_formal.md           # Formal tone variant
   └── child_casual.md           # Casual tone variant
   ```

3. Add routing logic to the parent skill:
   ```markdown
   ## Routing
   - If the task involves email/report/documentation → use formal variant
   - If the task involves chat/social/casual → use casual variant
   ```

4. Write child skills to `${CLAUDE_PLUGIN_DATA:-.evolution}/children/`.

### Step 6: Score Candidates

For prompt fix candidates, mentally simulate: "If I had been using this skill from the start of this session, would the failures have been avoided?"

Score each 0.0–1.0:
- 1.0 = all failures would have been prevented
- 0.5 = some failures prevented, some remain
- 0.0 = no improvement

Also score the **current skill** as baseline.

For scripts/examples/splits — these are additive, so accept them if the logic is sound.

### Step 7: Save Skill Bundle

Pick the highest-scoring prompt candidate. Combine with any new examples/scripts/children:

1. Write the new skill to `${CLAUDE_PLUGIN_DATA:-.evolution}/current_skill.md`
2. Write any new scripts to `${CLAUDE_PLUGIN_DATA:-.evolution}/scripts/`
3. Write any new examples to `${CLAUDE_PLUGIN_DATA:-.evolution}/examples/`
4. Write any child skills to `${CLAUDE_PLUGIN_DATA:-.evolution}/children/`
5. Update stats:
   ```json
   {
     "version": "0.X",
     "best_score": 0.XX,
     "optimized_at": "ISO-timestamp",
     "gradient_summary": "brief description of what was fixed",
     "candidates_tested": 3,
     "growth": {
       "scripts_added": ["validate_json.py"],
       "examples_added": ["example_001.md"],
       "children_added": [],
       "prompt_changes": "description of prompt changes"
     }
   }
   ```
   Write to `${CLAUDE_PLUGIN_DATA:-.evolution}/stats.json`
6. Tell the user what changed and why.

If no candidate beats the current skill (and no new capabilities were added), keep the current skill and tell the user.

### Step 8: Log the Optimization

Append to `${CLAUDE_PLUGIN_DATA:-.evolution}/optimization_log.jsonl`:
```json
{
  "timestamp": "ISO-timestamp",
  "from_version": "0.X",
  "to_version": "0.Y",
  "gradient": ["bullet1", "bullet2"],
  "gradient_types": {"prompt_fix": 2, "script_needed": 1, "example_needed": 1, "split_needed": 0},
  "accepted": true,
  "score_before": 0.XX,
  "score_after": 0.YY,
  "artifacts_created": ["scripts/validate_json.py", "examples/example_001.md"]
}
```

## Important Notes

- **Never discard domain knowledge** — optimization should add constraints, not remove useful instructions
- **Be conservative by default** — when uncertain, prefer the conservative candidate
- **Version monotonically** — never reuse a version number
- **Show your work** — tell the user the gradient classification, candidates, and scores so they can verify
- **Scripts must be safe** — no network calls unless explicitly approved, no file deletion, no system modification
- **Examples from real traces** — prefer real positive traces over synthetic examples
- **Split is expensive** — only split when there are genuinely contradictory requirements (3+ conflicting traces), not just different topics
