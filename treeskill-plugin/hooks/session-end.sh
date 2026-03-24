#!/bin/bash
# TreeSkill SessionEnd Hook
# Reads transcript, extracts traces, stores for future APO optimization
# Optionally triggers auto-optimization if enough traces accumulated
# This runs async so timeout is not an issue

INPUT=$(cat)
TRANSCRIPT_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('transcript_path',''))")
SESSION_ID=$(echo "$INPUT" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('session_id',''))")

if [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
    exit 0
fi

PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVOLUTION_DIR="${CLAUDE_PLUGIN_DATA:-$PLUGIN_ROOT/.evolution}"
mkdir -p "$EVOLUTION_DIR"

# Step 1: Extract traces from transcript
python3 "$PLUGIN_ROOT/lib/extract_traces.py" \
    --transcript "$TRANSCRIPT_PATH" \
    --session-id "$SESSION_ID" \
    --output "$EVOLUTION_DIR/traces.jsonl" \
    2>>"$EVOLUTION_DIR/hook.log"

# Step 2: Auto-optimize if enabled and enough traces
AUTO_OPTIMIZE="${TREESKILL_AUTO_OPTIMIZE:-0}"
MIN_TRACES="${TREESKILL_AUTO_MIN_TRACES:-10}"

if [ "$AUTO_OPTIMIZE" = "1" ]; then
    # Count traces with feedback (correction signals)
    FEEDBACK_COUNT=$(grep -c '"correction"' "$EVOLUTION_DIR/traces.jsonl" 2>/dev/null || echo "0")

    if [ "$FEEDBACK_COUNT" -ge "$MIN_TRACES" ]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Auto-optimizing ($FEEDBACK_COUNT feedback traces)" \
            >> "$EVOLUTION_DIR/hook.log"

        python3 "$PLUGIN_ROOT/lib/apo_optimize.py" \
            --evolution-dir "$EVOLUTION_DIR" \
            --beam-width 3 \
            --beam-rounds 2 \
            >>"$EVOLUTION_DIR/hook.log" 2>&1
    fi
fi

exit 0
