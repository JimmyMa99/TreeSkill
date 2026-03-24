#!/bin/bash
# TreeSkill SessionStart Hook
# Injects the meta-skill + current best skill into Claude's context

PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVOLUTION_DIR="${CLAUDE_PLUGIN_DATA:-.evolution}"

# Build context to inject
CONTEXT=""

# 1. Load the meta-skill (teaches Claude how to use TreeSkill)
META_SKILL="$PLUGIN_ROOT/skills/using-treeskill/SKILL.md"
if [ -f "$META_SKILL" ]; then
    CONTEXT="$(cat "$META_SKILL")"
fi

# 2. Load current best evolved skill if exists
CURRENT_SKILL="$EVOLUTION_DIR/current_skill.md"
if [ -f "$CURRENT_SKILL" ]; then
    CONTEXT="$CONTEXT

---
## Currently Active Evolved Skill
$(cat "$CURRENT_SKILL")"
fi

# 3. Load helper scripts manifest if exists
SCRIPTS_DIR="$EVOLUTION_DIR/scripts"
if [ -d "$SCRIPTS_DIR" ] && [ "$(ls -A "$SCRIPTS_DIR" 2>/dev/null)" ]; then
    CONTEXT="$CONTEXT

---
## Available Helper Scripts"
    for script in "$SCRIPTS_DIR"/*; do
        SCRIPT_NAME=$(basename "$script")
        # Extract first line of docstring as description
        DESC=$(head -5 "$script" | grep -m1 '"""\|#.*—\|#.*-' | sed 's/^[# "]*//' | head -c 100)
        CONTEXT="$CONTEXT
- \`scripts/$SCRIPT_NAME\` — ${DESC:-no description}"
    done
fi

# 4. Load few-shot examples summary if exists
EXAMPLES_DIR="$EVOLUTION_DIR/examples"
if [ -d "$EXAMPLES_DIR" ] && [ "$(ls -A "$EXAMPLES_DIR" 2>/dev/null)" ]; then
    EXAMPLE_COUNT=$(ls "$EXAMPLES_DIR" | wc -l | tr -d ' ')
    CONTEXT="$CONTEXT

> TreeSkill: $EXAMPLE_COUNT few-shot examples available in examples/ directory"
fi

# 5. Load evolution stats if exists
STATS="$EVOLUTION_DIR/stats.json"
if [ -f "$STATS" ]; then
    VERSION=$(cat "$STATS" | grep -o '"version"[^,]*' | head -1 | cut -d'"' -f4)
    SCORE=$(cat "$STATS" | grep -o '"best_score"[^,]*' | head -1 | cut -d':' -f2 | tr -d ' ')
    CONTEXT="$CONTEXT

> TreeSkill Evolution: v${VERSION:-0.0} | Best score: ${SCORE:-N/A}"
fi

# Output as hookSpecificOutput JSON
ESCAPED_CONTEXT=$(echo "$CONTEXT" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")

cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": $ESCAPED_CONTEXT
  }
}
EOF
