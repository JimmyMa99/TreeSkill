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

# 3. Load evolution stats if exists
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
