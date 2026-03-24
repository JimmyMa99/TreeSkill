#!/bin/bash
# TreeSkill SessionEnd Hook
# Reads transcript, extracts traces, stores for future APO optimization
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

# Extract conversation pairs (user input -> assistant response) from transcript
# Store as trace for future APO optimization
python3 "$PLUGIN_ROOT/lib/extract_traces.py" \
    --transcript "$TRANSCRIPT_PATH" \
    --session-id "$SESSION_ID" \
    --output "$EVOLUTION_DIR/traces.jsonl" \
    2>/dev/null

exit 0
