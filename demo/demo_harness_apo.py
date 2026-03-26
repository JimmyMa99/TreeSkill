#!/usr/bin/env python3
"""
Harness APO Demo — 在真实 agent 环境中优化 Skill

TreeSkill APO + AgentHarness 端到端:
  1. 用一个很烂的 coding skill 作为 baseline
  2. AgentHarness 执行编码任务（真实工具调用）
  3. APO 优化 skill prompt
  4. 对比优化前后的任务完成率

用法:
    conda activate pr
    python demo/demo_harness_apo.py
"""

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

from treeskill.config import GlobalConfig
from treeskill.harness import AgentHarness, HarnessResult
from treeskill.llm import LLMClient
from treeskill.optimizer import APOEngine
from treeskill.registry import registry
from treeskill.schema import Feedback, Message, Skill, Trace
from treeskill.skill import save as save_skill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────
ACTOR_MODEL = "intern-s1-pro"
ACTOR_BASE_URL = "https://chat.intern-ai.org.cn"
ACTOR_API_KEY = os.getenv(
    "INTERN_API_KEY",
    "REDACTED_INTERN_API_KEY",
)

JUDGE_BASE_URL = os.getenv(
    "TREE_LLM_BASE_URL",
    "https://oneapi.liuyanxing.site:8443/v1",
)
JUDGE_MODEL = os.getenv("TREE_LLM_JUDGE_MODEL", "bailian/glm-5")
JUDGE_API_KEY = os.getenv(
    "TREE_LLM_API_KEY",
    "REDACTED_ONEAPI_KEY",
)

OUTPUT_DIR = Path("demo/outputs/harness-apo")
NUM_ROUNDS = 2

# ── 故意很烂的初始 Skill ───────────────────────────────
BAD_SKILL = """You are a coding assistant. Write code when asked."""

# ── 任务数据集 ─────────────────────────────────────────
TASKS = [
    {
        "id": "hello-world",
        "task": "Create a file called hello.py that prints 'Hello World', then run it to verify.",
        "verify": lambda r: _check_file_and_output(r, "hello.py", "Hello World"),
        "split": "train",
    },
    {
        "id": "fibonacci",
        "task": "Create fib.py that prints the first 10 Fibonacci numbers (one per line), then run it.",
        "verify": lambda r: _check_file_and_output(r, "fib.py", "55"),
        "split": "train",
    },
    {
        "id": "word-count",
        "task": "Create a file sample.txt with the text 'hello world hello python world'. Then create count.py that reads sample.txt and prints the count of each unique word. Run it.",
        "verify": lambda r: _check_file_and_output(r, "count.py", "hello"),
        "split": "train",
    },
    {
        "id": "fizzbuzz",
        "task": "Create fizzbuzz.py that prints FizzBuzz for numbers 1-30 (Fizz for multiples of 3, Buzz for 5, FizzBuzz for both). Run it.",
        "verify": lambda r: _check_file_and_output(r, "fizzbuzz.py", "FizzBuzz"),
        "split": "train",
    },
    {
        "id": "csv-create",
        "task": "Create a CSV file data.csv with headers 'name,age,city' and 3 rows of sample data. Then create read_csv.py that reads and prints the CSV contents. Run it.",
        "verify": lambda r: _check_file_and_output(r, "data.csv", None) and _check_file_and_output(r, "read_csv.py", None),
        "split": "val",
    },
    {
        "id": "json-api",
        "task": "Create api.py that fetches data from https://jsonplaceholder.typicode.com/todos/1 and prints the title field. Run it.",
        "verify": lambda r: _check_file_and_output(r, "api.py", "delectus"),
        "split": "val",
    },
    {
        "id": "sort-file",
        "task": "Create numbers.txt with the numbers 5,3,8,1,9 (one per line). Then create sort.py that reads numbers.txt, sorts the numbers, and writes the result to sorted.txt. Print the sorted result. Run it.",
        "verify": lambda r: _check_file_and_output(r, "sort.py", "1"),
        "split": "test",
    },
    {
        "id": "calculator",
        "task": "Create calc.py with functions add, subtract, multiply, divide. Then test all four operations with the numbers 10 and 3, printing results. Run it.",
        "verify": lambda r: _check_file_and_output(r, "calc.py", "30"),
        "split": "test",
    },
]


def _check_file_and_output(result: HarnessResult, filename: str, expected_output: str) -> float:
    """Verify that a file was created and the expected output appeared."""
    file_created = any(filename in str(f) for f in result.files_created)
    bash_outputs = [tc["output"] for tc in result.tool_calls if tc["name"] == "bash"]

    if expected_output is None:
        # Just check file was created
        return 1.0 if file_created else 0.0

    output_found = any(expected_output in out for out in bash_outputs)

    if file_created and output_found:
        return 1.0
    elif file_created or output_found:
        return 0.5
    return 0.0


# ── Main ───────────────────────────────────────────────

def main():
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("Harness APO Demo — Skill Optimization with Real Agent Loop")
    logger.info(f"  Actor: {ACTOR_MODEL}")
    logger.info(f"  Judge: {JUDGE_MODEL}")
    logger.info("=" * 60)

    train = [t for t in TASKS if t["split"] == "train"]
    val = [t for t in TASKS if t["split"] == "val"]
    test = [t for t in TASKS if t["split"] == "test"]
    logger.info(f"Tasks: train={len(train)}, val={len(val)}, test={len(test)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Baseline ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Baseline (bad skill)")
    logger.info(f"Skill: {BAD_SKILL.strip()}")
    logger.info("=" * 60)

    baseline_scores = []
    for task in test:
        workdir = OUTPUT_DIR / "workspaces" / f"baseline_{task['id']}"
        if workdir.exists():
            shutil.rmtree(workdir)
        harness = AgentHarness(
            model=ACTOR_MODEL,
            base_url=ACTOR_BASE_URL,
            api_key=ACTOR_API_KEY,
            workdir=workdir,
        )
        result = harness.run(task["task"], system_prompt=BAD_SKILL)
        score = task["verify"](result)
        baseline_scores.append(score)
        logger.info(f"  {task['id']}: {score:.1f} (turns={result.turns}, tools={len(result.tool_calls)})")

    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    logger.info(f"  Baseline: {baseline_avg:.2f}")

    # ── Phase 2: APO Optimization ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: APO with AgentHarness")
    logger.info("=" * 60)

    # Framework config — force override .env values
    env_overrides = {
        "TREE_LLM_API_KEY": ACTOR_API_KEY,
        "TREE_LLM_BASE_URL": ACTOR_BASE_URL,
        "TREE_LLM_MODEL": ACTOR_MODEL,
        "TREE_LLM_PROTOCOL": "anthropic",
        "TREE_LLM_JUDGE_API_KEY": JUDGE_API_KEY,
        "TREE_LLM_JUDGE_BASE_URL": JUDGE_BASE_URL,
        "TREE_LLM_JUDGE_MODEL": JUDGE_MODEL,
        "TREE_LLM_JUDGE_PROTOCOL": "openai",
        "TREE_LLM_TEMPERATURE": "0.7",
    }
    os.environ.update(env_overrides)

    from treeskill.config import LLMConfig, APOConfig
    config = GlobalConfig(
        llm=LLMConfig(),
        apo=APOConfig(
            gradient_accumulation_steps=4,
            beam_width=1,
            branch_factor=2,
            beam_rounds=1,
        ),
    )
    llm = LLMClient(config)
    engine = APOEngine(config, llm)

    # Score function using harness
    def harness_score_fn(prompt: str, traces: List[Trace]) -> float:
        scores = []
        for t in traces:
            task_text = t.inputs[-1].content
            # Find the matching task to get verify_fn
            matching = [tk for tk in train if tk["task"] == task_text]
            if not matching:
                scores.append(0.5)
                continue

            task = matching[0]
            workdir = OUTPUT_DIR / "workspaces" / f"score_{task['id']}_{int(time.time())}"
            h = AgentHarness(
                model=ACTOR_MODEL,
                base_url=ACTOR_BASE_URL,
                api_key=ACTOR_API_KEY,
                workdir=workdir,
            )
            result = h.run(task_text, system_prompt=prompt)
            scores.append(task["verify"](result))

        return sum(scores) / len(scores) if scores else 0.0

    engine._score_fn = harness_score_fn

    skill = Skill(
        name="coding-assistant",
        description="A coding skill that creates and runs Python scripts",
        system_prompt=BAD_SKILL,
        target="Improve task completion rate. The agent should always: "
               "1) create the requested file, 2) write correct code, "
               "3) run the code to verify output.",
        version="v1.0",
    )

    best_skill = skill
    best_val = -1.0

    for rnd in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n--- Round {rnd}/{NUM_ROUNDS} ---")

        # Collect traces on train
        traces = []
        for task in train:
            workdir = OUTPUT_DIR / "workspaces" / f"train_r{rnd}_{task['id']}"
            if workdir.exists():
                shutil.rmtree(workdir)
            h = AgentHarness(
                model=ACTOR_MODEL,
                base_url=ACTOR_BASE_URL,
                api_key=ACTOR_API_KEY,
                workdir=workdir,
            )
            result = h.run(task["task"], system_prompt=best_skill.system_prompt)
            score = task["verify"](result)

            t = Trace(
                inputs=[Message(role="user", content=task["task"])],
                prediction=Message(role="assistant", content=result.output),
            )

            # Build detailed critique from tool calls
            tool_summary = ", ".join(
                f"{tc['name']}({'ok' if tc.get('success') else 'fail'})"
                for tc in result.tool_calls
            )
            critique = (
                f"Score: {score:.1f}. "
                f"Tools used: [{tool_summary}]. "
                f"Files created: {result.files_created}. "
                f"{'Task completed.' if score >= 0.8 else 'Task incomplete or wrong output.'}"
            )

            t.feedback = Feedback(
                score=score,
                critique=critique,
                correction=f"Expected: create file + run code + verify output for: {task['task']}",
            )
            traces.append(t)
            logger.info(f"  {task['id']}: {score:.1f} — {critique[:80]}")

        failures = [t for t in traces if t.feedback.score < 0.8]
        logger.info(f"  Failures: {len(failures)}/{len(traces)}")

        if not failures:
            logger.info("  All passing, skip")
            continue

        t0 = time.time()
        candidate = engine.optimize(best_skill, traces)
        logger.info(f"  Optimize: {time.time()-t0:.1f}s → {candidate.version}")

        # Validate
        val_scores = []
        for task in val:
            workdir = OUTPUT_DIR / "workspaces" / f"val_r{rnd}_{task['id']}"
            if workdir.exists():
                shutil.rmtree(workdir)
            h = AgentHarness(
                model=ACTOR_MODEL,
                base_url=ACTOR_BASE_URL,
                api_key=ACTOR_API_KEY,
                workdir=workdir,
            )
            result = h.run(task["task"], system_prompt=candidate.system_prompt)
            s = task["verify"](result)
            val_scores.append(s)
            logger.info(f"  val {task['id']}: {s:.1f}")

        val_avg = sum(val_scores) / len(val_scores)
        if val_avg > best_val:
            best_val = val_avg
            best_skill = candidate
            save_skill(best_skill, OUTPUT_DIR)
            logger.info(f"  ★ Accepted {best_skill.version} (val={val_avg:.2f})")
        else:
            logger.info(f"  Rejected (val={val_avg:.2f} <= {best_val:.2f})")

    # ── Phase 3: Final test ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Final test with optimized skill")
    logger.info("=" * 60)

    final_scores = []
    for task in test:
        workdir = OUTPUT_DIR / "workspaces" / f"final_{task['id']}"
        if workdir.exists():
            shutil.rmtree(workdir)
        harness = AgentHarness(
            model=ACTOR_MODEL,
            base_url=ACTOR_BASE_URL,
            api_key=ACTOR_API_KEY,
            workdir=workdir,
        )
        result = harness.run(task["task"], system_prompt=best_skill.system_prompt)
        score = task["verify"](result)
        final_scores.append(score)
        logger.info(f"  {task['id']}: {score:.1f} (turns={result.turns})")

    final_avg = sum(final_scores) / len(final_scores)

    # ── Summary ──
    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! ({elapsed/60:.1f} min)")
    logger.info(f"  Baseline: {baseline_avg:.2f}")
    logger.info(f"  Final:    {final_avg:.2f}")
    logger.info(f"  Delta:    {final_avg - baseline_avg:+.2f}")
    logger.info(f"  Version:  {best_skill.version}")
    logger.info(f"\nOptimized skill:\n{best_skill.system_prompt[:500]}")

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "baseline": baseline_avg,
            "final": final_avg,
            "delta": final_avg - baseline_avg,
            "version": best_skill.version,
            "elapsed_min": elapsed / 60,
        }, f, indent=2)


if __name__ == "__main__":
    main()
