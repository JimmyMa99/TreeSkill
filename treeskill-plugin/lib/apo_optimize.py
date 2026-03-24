#!/usr/bin/env python3
"""Phase B: Python APO engine for TreeSkill plugin.

Reads accumulated traces from .evolution/traces.jsonl, converts them to
TreeSkill Trace objects, runs APOEngine with beam search, and writes the
optimized skill back to .evolution/current_skill.md.

Usage:
    python apo_optimize.py --evolution-dir .evolution/ [--config config.yaml]

Environment variables:
    TREE_LLM_API_KEY       - API key for the judge/rewrite model
    TREE_LLM_BASE_URL      - API base URL
    TREE_LLM_MODEL         - Model name for generation
    TREE_LLM_JUDGE_MODEL   - Model name for judging (defaults to TREE_LLM_MODEL)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent of treeskill-plugin to path so we can import treeskill
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PLUGIN_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from treeskill.config import GlobalConfig
from treeskill.llm import LLMClient
from treeskill.optimizer import APOEngine
from treeskill.schema import Feedback, Message, Skill, Trace

logger = logging.getLogger(__name__)


def load_traces(traces_path: Path) -> list[dict]:
    """Load raw traces from JSONL file."""
    if not traces_path.exists():
        return []
    traces = []
    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return traces


def raw_to_trace(raw: dict) -> Trace:
    """Convert a raw trace dict (from extract_traces.py) to a TreeSkill Trace."""
    user_input = raw.get("input", "")
    output = raw.get("output", "")
    signals = raw.get("quality_signals", [])

    # Build Trace with Message objects
    input_msg = Message(role="user", content=user_input)
    prediction_msg = Message(role="assistant", content=output)

    # Build feedback from quality signals
    feedback = None
    if "correction" in signals:
        feedback = Feedback(
            score=0.3,
            critique=f"User corrected this response. Original input: {user_input[:200]}",
        )
    elif "positive" in signals:
        feedback = Feedback(
            score=0.9,
            critique="User expressed satisfaction.",
        )
    # Neutral traces (no signal) get no feedback — APO will skip them

    return Trace(
        session_id=raw.get("session_id", "unknown"),
        inputs=[input_msg],
        prediction=prediction_msg,
        feedback=feedback,
    )


def load_current_skill(evolution_dir: Path) -> Skill:
    """Load the current skill from evolution dir, or create a default one."""
    skill_path = evolution_dir / "current_skill.md"
    stats_path = evolution_dir / "stats.json"

    system_prompt = ""
    version = "v1.0"

    if skill_path.exists():
        content = skill_path.read_text(encoding="utf-8")
        # Extract system prompt (everything after frontmatter)
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                system_prompt = parts[2].strip()
                # Try to extract version from frontmatter
                for line in parts[1].strip().split("\n"):
                    if line.strip().startswith("version:"):
                        version = line.split(":", 1)[1].strip().strip('"')
            else:
                system_prompt = content
        else:
            system_prompt = content

    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            version = stats.get("current_version", version)
        except (json.JSONDecodeError, KeyError):
            pass

    return Skill(
        name="treeskill-evolved",
        description="Auto-evolved skill via APO",
        version=version,
        system_prompt=system_prompt,
    )


def save_skill(skill: Skill, evolution_dir: Path, gradient_summary: str = ""):
    """Save optimized skill and update stats."""
    # Write current_skill.md with frontmatter
    skill_path = evolution_dir / "current_skill.md"
    now = datetime.now(timezone.utc).isoformat()
    content = f"""---
name: {skill.name}
version: {skill.version}
optimization_date: "{now}"
description: {skill.description}
---

{skill.system_prompt}
"""
    skill_path.write_text(content, encoding="utf-8")

    # Update stats.json
    stats_path = evolution_dir / "stats.json"
    stats = {}
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    stats.update({
        "current_version": skill.version,
        "best_score": None,  # Will be set if beam provides scores
        "last_optimized": now,
        "gradient_summary": gradient_summary,
    })
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Append to optimization_log.jsonl
    log_path = evolution_dir / "optimization_log.jsonl"
    log_entry = {
        "timestamp": now,
        "to_version": skill.version,
        "gradient_summary": gradient_summary,
        "engine": "python-apo",
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    logger.info("Saved skill %s to %s", skill.version, skill_path)


def save_beam_pool(beam: list[tuple], evolution_dir: Path):
    """Persist beam pool for cross-round optimization."""
    pool_path = evolution_dir / "beam_pool.json"
    pool_data = [{"prompt": p, "score": s} for p, s in beam]
    pool_path.write_text(
        json.dumps(pool_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_beam_pool(evolution_dir: Path) -> list[str]:
    """Load beam pool seeds from previous optimization."""
    pool_path = evolution_dir / "beam_pool.json"
    if not pool_path.exists():
        return []
    try:
        pool_data = json.loads(pool_path.read_text(encoding="utf-8"))
        return [item["prompt"] for item in pool_data[:3]]  # Top 3 as seeds
    except (json.JSONDecodeError, KeyError):
        return []


def main():
    parser = argparse.ArgumentParser(description="Run APO optimization on accumulated traces")
    parser.add_argument(
        "--evolution-dir",
        default=".evolution",
        help="Path to .evolution/ directory",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to TreeSkill config.yaml (optional, uses env vars if not set)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=3,
        help="Beam search width (default: 3)",
    )
    parser.add_argument(
        "--beam-rounds",
        type=int,
        default=2,
        help="Beam search rounds (default: 2)",
    )
    parser.add_argument(
        "--min-traces",
        type=int,
        default=3,
        help="Minimum traces with feedback to trigger optimization (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    evolution_dir = Path(args.evolution_dir)
    if not evolution_dir.exists():
        logger.error("Evolution directory not found: %s", evolution_dir)
        sys.exit(1)

    # Load traces
    raw_traces = load_traces(evolution_dir / "traces.jsonl")
    if not raw_traces:
        logger.info("No traces found. Nothing to optimize.")
        sys.exit(0)

    traces = [raw_to_trace(r) for r in raw_traces]
    traces_with_feedback = [t for t in traces if t.feedback is not None]

    logger.info(
        "Loaded %d traces (%d with feedback)",
        len(traces),
        len(traces_with_feedback),
    )

    if len(traces_with_feedback) < args.min_traces:
        logger.info(
            "Not enough feedback traces (%d < %d). Skipping optimization.",
            len(traces_with_feedback),
            args.min_traces,
        )
        sys.exit(0)

    # Load config
    if args.config:
        config = GlobalConfig.from_yaml(args.config)
    else:
        config = GlobalConfig()  # Uses env vars

    # Override APO params
    config.apo.beam_width = args.beam_width
    config.apo.beam_rounds = args.beam_rounds

    # Initialize
    llm = LLMClient(config)
    engine = APOEngine(config, llm)

    # Load current skill
    skill = load_current_skill(evolution_dir)
    logger.info("Current skill: %s (prompt: %d chars)", skill.version, len(skill.system_prompt))

    # Inject beam seeds from previous optimization
    beam_seeds = load_beam_pool(evolution_dir)
    if beam_seeds:
        engine.initial_beam = beam_seeds
        logger.info("Injected %d beam seeds from previous optimization", len(beam_seeds))

    # Run APO
    logger.info("Starting APO optimization (beam_width=%d, rounds=%d)...", args.beam_width, args.beam_rounds)
    new_skill = engine.optimize(skill, traces_with_feedback)

    # Save results
    if new_skill.version != skill.version:
        gradient_summary = f"Optimized from {skill.version} using {len(traces_with_feedback)} traces"
        save_skill(new_skill, evolution_dir, gradient_summary)

        # Save beam pool for next round
        if engine.last_beam:
            save_beam_pool(engine.last_beam, evolution_dir)
            logger.info("Beam pool saved: %d candidates", len(engine.last_beam))

        logger.info("Optimization complete: %s → %s", skill.version, new_skill.version)
        print(json.dumps({
            "status": "improved",
            "old_version": skill.version,
            "new_version": new_skill.version,
            "traces_used": len(traces_with_feedback),
            "beam_size": len(engine.last_beam),
        }))
    else:
        logger.info("No improvement found. Keeping %s", skill.version)
        print(json.dumps({
            "status": "no_improvement",
            "version": skill.version,
            "traces_used": len(traces_with_feedback),
        }))


if __name__ == "__main__":
    main()
