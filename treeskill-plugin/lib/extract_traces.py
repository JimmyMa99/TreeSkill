#!/usr/bin/env python3
"""Extract conversation traces from Claude Code transcript JSONL.

Reads a Claude Code session transcript and extracts (user_input, assistant_response)
pairs with quality signals (corrections, retries, user satisfaction).

Output: one JSON line per trace, appended to the output file.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_transcript(transcript_path: str) -> list[dict]:
    """Parse a Claude Code JSONL transcript into structured turns."""
    turns = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")
            if entry_type not in ("user", "assistant"):
                continue

            msg = entry.get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Extract text from content (may be string or list of parts)
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "tool_result":
                            # Skip tool results in user messages
                            pass
                    elif isinstance(part, str):
                        text_parts.append(part)
                text = "\n".join(text_parts)

            if not text.strip():
                continue

            turns.append({
                "role": role,
                "text": text.strip(),
                "timestamp": entry.get("timestamp", ""),
                "uuid": entry.get("uuid", ""),
            })

    return turns


def extract_traces(turns: list[dict], session_id: str) -> list[dict]:
    """Extract (input, response) pairs with quality signals from turns."""
    traces = []
    i = 0
    while i < len(turns):
        # Find user -> assistant pairs
        if turns[i]["role"] != "user":
            i += 1
            continue

        user_turn = turns[i]
        # Find the next assistant response
        j = i + 1
        while j < len(turns) and turns[j]["role"] != "assistant":
            j += 1

        if j >= len(turns):
            break

        assistant_turn = turns[j]

        # Detect quality signals by looking at what comes after
        quality_signals = []
        k = j + 1

        # Check if user immediately corrects or complains
        if k < len(turns) and turns[k]["role"] == "user":
            next_user = turns[k]["text"].lower()
            # Negative signals
            negative_keywords = [
                "不对", "错了", "不是", "重新", "改一下", "wrong", "no ", "fix",
                "redo", "不行", "别这样", "不要", "stop", "don't",
            ]
            for kw in negative_keywords:
                if kw in next_user:
                    quality_signals.append("correction")
                    break

            # Positive signals
            positive_keywords = [
                "好", "对", "perfect", "great", "exactly", "nice", "thanks",
                "谢", "可以", "行", "ok", "good",
            ]
            for kw in positive_keywords:
                if kw in next_user:
                    quality_signals.append("positive")
                    break

        trace = {
            "session_id": session_id,
            "timestamp": user_turn.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "input": user_turn["text"][:2000],  # Truncate for storage
            "output": assistant_turn["text"][:2000],
            "quality_signals": quality_signals,
        }
        traces.append(trace)

        i = j + 1

    return traces


def main():
    parser = argparse.ArgumentParser(description="Extract traces from Claude Code transcript")
    parser.add_argument("--transcript", required=True, help="Path to transcript JSONL file")
    parser.add_argument("--session-id", default="unknown", help="Session ID")
    parser.add_argument("--output", required=True, help="Output JSONL file (append mode)")
    args = parser.parse_args()

    if not Path(args.transcript).exists():
        print(f"Transcript not found: {args.transcript}", file=sys.stderr)
        sys.exit(1)

    turns = parse_transcript(args.transcript)
    if not turns:
        sys.exit(0)

    traces = extract_traces(turns, args.session_id)
    if not traces:
        sys.exit(0)

    # Append to output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    print(f"Extracted {len(traces)} traces from {len(turns)} turns", file=sys.stderr)


if __name__ == "__main__":
    main()
