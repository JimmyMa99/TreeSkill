"""Built-in scorer, gradient, and rewriter registrations.

Auto-imported by the registry to provide default components.
"""

import logging

from treeskill.registry import scorer, gradient, rewriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in Scorers
# ---------------------------------------------------------------------------

@scorer("exact-match")
def exact_match(output: str, expected: str, context: dict) -> float:
    """Simple exact string match."""
    return 1.0 if output.strip().upper() == expected.strip().upper() else 0.0


@scorer("judge-grade", set_default=True)
def judge_grade(output: str, expected: str, context: dict) -> float:
    """Judge compares output vs expected and returns 0-1 score.

    Requires ``context["judge_fn"]`` — a callable that takes
    (output, expected) and returns a float score.
    """
    judge_fn = context.get("judge_fn")
    if judge_fn is None:
        # Fallback to exact match
        return 1.0 if output.strip() == expected.strip() else 0.0
    return judge_fn(output, expected)


# ---------------------------------------------------------------------------
# Built-in Gradient Templates
# ---------------------------------------------------------------------------

@gradient("simple")
def _gradient_simple():
    return (
        "You are an expert prompt engineer. Analyze the conversation failures "
        "below and explain concisely WHY the system prompt led to these problems. "
        "Return a bullet list of specific, actionable issues."
    )


@gradient("root-cause")
def _gradient_root_cause():
    return (
        "You are a senior prompt debugger. For each failure below, identify the "
        "ROOT CAUSE in the system prompt — what instruction is missing, ambiguous, "
        "or misleading? Be specific: quote the problematic part of the prompt and "
        "explain how it caused the failure. Return 3-5 bullets."
    )


@gradient("comprehensive")
def _gradient_comprehensive():
    return (
        "You are a prompt quality auditor. Evaluate the system prompt against "
        "these failures across these dimensions:\n"
        "1. Instruction clarity — are the rules unambiguous?\n"
        "2. Tone/style control — does the prompt prevent AI-sounding language?\n"
        "3. Scope constraints — does the prompt enforce length/format limits?\n"
        "4. Edge cases — does the prompt handle the scenarios that failed?\n"
        "Return a structured critique with specific fixes for each dimension."
    )


# ---------------------------------------------------------------------------
# Built-in Rewriter Templates
# ---------------------------------------------------------------------------

@rewriter("full-rewrite")
def _rewriter_full():
    return (
        "You are an expert prompt engineer. Based on the failure analysis below, "
        "rewrite the System Prompt to fix ALL identified issues. You may "
        "restructure, reorder, or add new instructions as needed. "
        "Preserve the core intent and any domain-specific knowledge. "
        "Return ONLY the new prompt — no commentary, no markdown code fences."
    )


@rewriter("conservative")
def _rewriter_conservative():
    return (
        "You are an expert prompt engineer. Based on the failure analysis below, "
        "revise the System Prompt to address the SINGLE MOST CRITICAL issue. "
        "Make minimal changes — keep the prompt close in tone, length, and "
        "structure to the original. Do not address more than one issue. "
        "Return ONLY the new prompt — no commentary, no markdown code fences."
    )


@rewriter("distill")
def _rewriter_distill():
    return (
        "You are a prompt distillation expert. Based on the failure analysis below, "
        "adapt the System Prompt for a smaller, less capable model. "
        "PRUNE sections the model cannot handle well. "
        "EXPAND key rules with explicit examples and explanations. "
        "Keep all tool/script references intact. "
        "Return ONLY the new prompt — no commentary, no markdown code fences."
    )


# ---------------------------------------------------------------------------
# Kode CLI Scorer — run skill via Kode agent CLI
# ---------------------------------------------------------------------------

@scorer("kode-cli")
def kode_cli_scorer(output: str, expected: str, context: dict) -> float:
    """Score by running a task through Kode CLI with a skill loaded.

    Required context keys:
        task: str — the task to run
        skill_dir: Path — directory containing SKILL.md (optional)
        skill_prompt: str — system prompt to write as AGENTS.md (optional)
        verify_fn: callable(kode_result: dict, workdir: Path) -> float (optional)
        judge_fn: callable(output: str, expected: str) -> float (optional)
    """
    import json as _json
    import subprocess
    import tempfile
    from pathlib import Path

    task = context.get("task", "")
    if not task:
        logger.warning("kode-cli scorer: missing task")
        return 0.0

    # Prepare workspace
    workdir = context.get("workdir")
    if not workdir:
        workdir = Path(tempfile.mkdtemp(prefix="kode_score_"))
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Write skill as AGENTS.md
    skill_prompt = context.get("skill_prompt", "")
    skill_dir = context.get("skill_dir")
    if skill_prompt:
        (workdir / "AGENTS.md").write_text(skill_prompt, encoding="utf-8")
    elif skill_dir:
        skill_md = Path(skill_dir) / "SKILL.md"
        if skill_md.exists():
            (workdir / "AGENTS.md").write_text(
                skill_md.read_text(encoding="utf-8"), encoding="utf-8"
            )

    # Run Kode CLI
    try:
        proc = subprocess.run(
            ["kode", "-p", task, "--cwd", str(workdir),
             "--output-format", "json", "--dangerously-skip-permissions"],
            capture_output=True, text=True, timeout=180,
        )
        result = _json.loads(proc.stdout) if proc.stdout.strip() else {}
    except subprocess.TimeoutExpired:
        logger.warning("kode-cli scorer: timeout")
        return 0.0
    except Exception as e:
        logger.warning(f"kode-cli scorer: {e}")
        return 0.0

    if result.get("is_error"):
        return 0.0

    kode_output = result.get("result", "")

    # Priority 1: verify_fn (hard check)
    verify_fn = context.get("verify_fn")
    if verify_fn is not None:
        return verify_fn(result, workdir)

    # Priority 2: judge_fn
    judge_fn = context.get("judge_fn")
    if judge_fn is not None:
        return judge_fn(kode_output, expected)

    # Priority 3: basic success
    return 1.0 if kode_output.strip() else 0.5


# ---------------------------------------------------------------------------
# Harness Scorer — run skill in real agent loop
# ---------------------------------------------------------------------------

@scorer("harness")
def harness_scorer(output: str, expected: str, context: dict) -> float:
    """Score by running a skill in AgentHarness and evaluating the result.

    Required context keys:
        harness: AgentHarness instance
        task: str — the task to run
        judge_fn: callable(agent_output, expected) -> float (optional)
        verify_fn: callable(HarnessResult) -> float (optional, e.g. run tests)

    Evaluation priority:
        1. verify_fn (if provided) — hard verification (tests pass, file exists, etc.)
        2. judge_fn (if provided) — LLM judge compares output
        3. success check — did the agent complete without error?
    """
    from treeskill.harness import AgentHarness, HarnessResult

    harness = context.get("harness")
    task = context.get("task", "")
    skill_prompt = context.get("skill_prompt", "")

    if not harness or not task:
        logger.warning("harness scorer: missing harness or task in context")
        return 0.0

    # Run the agent
    result: HarnessResult = harness.run(task, system_prompt=skill_prompt)

    if not result.success:
        return 0.0

    # Priority 1: hard verification (e.g. run tests, check files)
    verify_fn = context.get("verify_fn")
    if verify_fn is not None:
        return verify_fn(result)

    # Priority 2: judge compares agent output vs expected
    judge_fn = context.get("judge_fn")
    if judge_fn is not None:
        return judge_fn(result.output, expected)

    # Priority 3: basic success — agent completed and produced output
    return 1.0 if result.output.strip() else 0.5


# ---------------------------------------------------------------------------
# Tool-Aware Gradient
# ---------------------------------------------------------------------------

@gradient("tool-aware")
def _gradient_tool_aware():
    return (
        "You are a skill architect. Analyze the failures below and determine "
        "the root cause for each:\n\n"
        "1. **PROMPT problem** — instructions are unclear, ambiguous, or missing\n"
        "2. **TOOL problem** — the agent lacks a capability it needs:\n"
        "   - Web search (needs current/external information)\n"
        "   - Code execution (needs to compute or verify)\n"
        "   - File access (needs to read/write specific files)\n"
        "   - API calls (needs external service integration)\n"
        "3. **KNOWLEDGE problem** — the model simply doesn't know the answer\n\n"
        "For each failure, classify which type it is and explain why.\n"
        "For TOOL problems, specify exactly what tool is needed.\n"
        "Return a structured analysis with clear categories."
    )


# ---------------------------------------------------------------------------
# Skill-Builder Rewriter
# ---------------------------------------------------------------------------

@rewriter("skill-builder")
def _rewriter_skill_builder():
    return (
        "You are a skill architect. Based on the failure analysis below, "
        "improve the skill. You have three options:\n\n"
        "**Option A: PROMPT FIX** — If the issue is unclear instructions, "
        "rewrite the system prompt. Return ONLY the new prompt text.\n\n"
        "**Option B: ADD TOOL** — If the skill needs a new tool capability, "
        "output the improved prompt AND a tool specification block:\n"
        "```tool\n"
        "name: web_search\n"
        "description: Search the web for current information\n"
        "script: |\n"
        "  import requests\n"
        "  def web_search(query: str) -> str:\n"
        "      # implementation\n"
        "      pass\n"
        "```\n\n"
        "**Option C: SPLIT** — If different failure types need different "
        "strategies, recommend splitting into sub-skills:\n"
        "```split\n"
        "- name: factual-qa\n"
        "  description: Answer factual questions using web search\n"
        "  tools: [web_search]\n"
        "- name: reasoning-qa\n"
        "  description: Answer reasoning questions with step-by-step logic\n"
        "  tools: [code_execution]\n"
        "```\n\n"
        "Choose the option that best addresses the root causes. "
        "If Option A, return ONLY the prompt. "
        "If Option B or C, include the code block markers."
    )
