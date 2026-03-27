#!/usr/bin/env python3
"""
Kode CLI + APO Demo — 用 Kode 作为前向引擎优化 Skill

通过 Kode CLI 执行任务，verify_fn 硬验证，APO 优化 skill prompt。
测试：prompt 优化 + split + 工具生成。

用法:
    conda activate pr
    python demo/demo_kode_apo.py
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

from treeskill.config import GlobalConfig
from treeskill.llm import LLMClient
from treeskill.optimizer import APOEngine
from treeskill.registry import registry
from treeskill.schema import Feedback, Message, Skill, Trace
from treeskill.skill import save as save_skill

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("demo/outputs/kode-apo")
NUM_ROUNDS = 2

# ── 故意烂的初始 Skill ──
BAD_SKILL = "You are an assistant. Help the user."

# ── 任务集：混合难度，跨领域 ──
TASKS = [
    # --- 简单编码 (baseline 应该能做) ---
    {
        "id": "hello",
        "task": "Create hello.py that prints 'Hello World' and run it.",
        "verify": lambda r, w: 1.0 if "Hello World" in r.get("result", "") else (0.5 if (w / "hello.py").exists() else 0.0),
        "split": "train",
        "domain": "coding",
    },
    {
        "id": "fizzbuzz",
        "task": "Create fizzbuzz.py for numbers 1-20. Print 'Fizz' for multiples of 3, 'Buzz' for 5, 'FizzBuzz' for both, else the number. Run it.",
        "verify": lambda r, w: 1.0 if "FizzBuzz" in r.get("result", "") else 0.0,
        "split": "train",
        "domain": "coding",
    },
    # --- 文件操作 ---
    {
        "id": "csv-process",
        "task": "Create data.csv with 'name,score' header and rows: Alice,85 / Bob,92 / Carol,78. Then create analyze.py that reads the CSV, calculates the average score, and prints 'Average: X'. Run it.",
        "verify": lambda r, w: 1.0 if "85" in r.get("result", "") or "Average" in r.get("result", "") else (0.5 if (w / "data.csv").exists() else 0.0),
        "split": "train",
        "domain": "data",
    },
    {
        "id": "json-transform",
        "task": "Create users.json with [{\"name\":\"Alice\",\"age\":30},{\"name\":\"Bob\",\"age\":25}]. Then create transform.py that reads it, filters users over 26, and writes result to filtered.json. Print the filtered result. Run it.",
        "verify": lambda r, w: 1.0 if "Alice" in r.get("result", "") and (w / "filtered.json").exists() else 0.0,
        "split": "train",
        "domain": "data",
    },
    # --- 验证集 ---
    {
        "id": "sort-numbers",
        "task": "Create a file numbers.txt with: 5,3,8,1,9 (one per line). Create sort.py that reads, sorts, writes to sorted.txt, and prints the sorted result. Run it.",
        "verify": lambda r, w: 1.0 if ("1" in r.get("result", "") and (w / "sorted.txt").exists()) else 0.0,
        "split": "val",
        "domain": "coding",
    },
    {
        "id": "word-freq",
        "task": "Create text.txt with 'the cat sat on the mat the cat'. Create freq.py that counts word frequencies and prints them sorted by count descending. Run it.",
        "verify": lambda r, w: 1.0 if "the" in r.get("result", "").lower() else 0.0,
        "split": "val",
        "domain": "data",
    },
    # --- 测试集 ---
    {
        "id": "calculator",
        "task": "Create calc.py with add/subtract/multiply/divide functions. Test with 10 and 3, print all results. Run it.",
        "verify": lambda r, w: 1.0 if "30" in r.get("result", "") else 0.0,
        "split": "test",
        "domain": "coding",
    },
    {
        "id": "log-parser",
        "task": "Create sample.log with 5 lines like '2024-01-01 ERROR Something failed'. Create parser.py that counts ERROR vs INFO lines and prints summary. Run it.",
        "verify": lambda r, w: 1.0 if "ERROR" in r.get("result", "") else 0.0,
        "split": "test",
        "domain": "data",
    },
]


def run_kode(task_text: str, skill_prompt: str, workdir: Path) -> dict:
    """Run a task through Kode CLI."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "AGENTS.md").write_text(skill_prompt, encoding="utf-8")

    try:
        proc = subprocess.run(
            ["kode", "-p", task_text, "--cwd", str(workdir),
             "--output-format", "json", "--dangerously-skip-permissions"],
            capture_output=True, text=True, timeout=180,
        )
        return json.loads(proc.stdout) if proc.stdout.strip() else {"is_error": True}
    except subprocess.TimeoutExpired:
        return {"is_error": True, "result": "timeout"}
    except Exception as e:
        return {"is_error": True, "result": str(e)}


def evaluate_tasks(tasks: list, skill_prompt: str, label: str) -> float:
    """Run tasks through Kode and return average score."""
    scores = []
    for task in tasks:
        workdir = OUTPUT_DIR / "workspaces" / f"{label}_{task['id']}"
        if workdir.exists():
            shutil.rmtree(workdir)
        result = run_kode(task["task"], skill_prompt, workdir)
        score = task["verify"](result, workdir) if not result.get("is_error") else 0.0
        scores.append(score)
        logger.info(f"  {task['id']}: {score:.1f} (turns={result.get('num_turns', 0)})")
    avg = sum(scores) / len(scores) if scores else 0
    logger.info(f"  [{label}] avg: {avg:.2f}")
    return avg


def main():
    t_start = time.time()

    train = [t for t in TASKS if t["split"] == "train"]
    val = [t for t in TASKS if t["split"] == "val"]
    test = [t for t in TASKS if t["split"] == "test"]
    logger.info(f"Tasks: train={len(train)}, val={len(val)}, test={len(test)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Baseline ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Baseline (bad skill)")
    logger.info("=" * 60)
    baseline = evaluate_tasks(test, BAD_SKILL, "baseline")

    # ── Phase 2: APO ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: APO with Kode CLI")
    logger.info("=" * 60)

    # Use OneAPI for judge/rewrite
    judge_key = os.getenv("TREE_LLM_API_KEY")
    assert judge_key, "Set TREE_LLM_API_KEY"

    os.environ.update({
        "TREE_LLM_JUDGE_API_KEY": judge_key,
        "TREE_LLM_JUDGE_BASE_URL": os.getenv("TREE_LLM_BASE_URL", "https://oneapi.liuyanxing.site:8443/v1"),
        "TREE_LLM_JUDGE_MODEL": os.getenv("TREE_LLM_JUDGE_MODEL", "bailian/glm-5"),
        "TREE_LLM_JUDGE_PROTOCOL": "openai",
        # Actor not used directly by APO, but needed for config
        "TREE_LLM_API_KEY": judge_key,
        "TREE_LLM_BASE_URL": os.getenv("TREE_LLM_BASE_URL", "https://oneapi.liuyanxing.site:8443/v1"),
        "TREE_LLM_MODEL": "bailian/qwen3.5-plus",
        "TREE_LLM_PROTOCOL": "openai",
    })

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

    # Score function: Kode CLI execution + verify
    def kode_score_fn(prompt: str, traces: List[Trace]) -> float:
        scores = []
        for t in traces:
            task_text = t.inputs[-1].content
            matching = [tk for tk in train if tk["task"] == task_text]
            if not matching:
                scores.append(0.5)
                continue
            task = matching[0]
            workdir = OUTPUT_DIR / "workspaces" / f"score_{task['id']}_{int(time.time())}"
            result = run_kode(task_text, prompt, workdir)
            s = task["verify"](result, workdir) if not result.get("is_error") else 0.0
            scores.append(s)
        return sum(scores) / len(scores) if scores else 0.0

    engine._score_fn = kode_score_fn

    skill = Skill(
        name="coding-assistant",
        description="Help users create and run Python scripts",
        system_prompt=BAD_SKILL,
        target="Improve task completion. Always: 1) create requested files, 2) write correct code, 3) run to verify.",
        version="v1.0",
    )

    best_skill = skill
    best_val = -1.0

    for rnd in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n--- Round {rnd}/{NUM_ROUNDS} ---")

        # Collect traces
        traces = []
        for task in train:
            workdir = OUTPUT_DIR / "workspaces" / f"train_r{rnd}_{task['id']}"
            if workdir.exists():
                shutil.rmtree(workdir)
            result = run_kode(task["task"], best_skill.system_prompt, workdir)
            score = task["verify"](result, workdir) if not result.get("is_error") else 0.0

            t = Trace(
                inputs=[Message(role="user", content=task["task"])],
                prediction=Message(role="assistant", content=result.get("result", "")),
            )
            t.feedback = Feedback(
                score=score,
                critique=f"Score: {score:.1f}. Domain: {task['domain']}. Result: {result.get('result', '')[:100]}",
                correction=f"Should create files and run code for: {task['task']}",
            )
            traces.append(t)
            logger.info(f"  {task['id']}: {score:.1f}")

        failures = [t for t in traces if t.feedback.score < 0.8]
        logger.info(f"  Failures: {len(failures)}/{len(traces)}")

        if not failures:
            logger.info("  All passing, skip")
            continue

        t0 = time.time()
        candidate = engine.optimize(best_skill, traces)
        logger.info(f"  Optimize: {time.time()-t0:.1f}s → {candidate.version}")

        # Check for structured actions
        if engine.pending_actions:
            logger.info(f"  Structured actions: {len(engine.pending_actions)}")
            for action in engine.pending_actions:
                logger.info(f"    {action['action']}: tools={len(action.get('tools', []))}, split={len(action.get('split', []))}")

        # Validate
        val_score = evaluate_tasks(val, candidate.system_prompt, f"val_r{rnd}")
        if val_score > best_val:
            best_val = val_score
            best_skill = candidate
            save_skill(best_skill, OUTPUT_DIR)
            logger.info(f"  ★ Accepted {best_skill.version} (val={val_score:.2f})")
        else:
            logger.info(f"  Rejected (val={val_score:.2f} <= {best_val:.2f})")

    # ── Phase 3: Final ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Final test")
    logger.info("=" * 60)
    final = evaluate_tasks(test, best_skill.system_prompt, "final")

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! ({elapsed/60:.1f} min)")
    logger.info(f"  Baseline: {baseline:.2f}")
    logger.info(f"  Final:    {final:.2f}")
    logger.info(f"  Delta:    {final - baseline:+.2f}")
    logger.info(f"  Version:  {best_skill.version}")
    logger.info(f"\nOptimized skill:\n{best_skill.system_prompt[:500]}")

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "baseline": baseline, "final": final,
            "delta": final - baseline, "version": best_skill.version,
            "elapsed_min": elapsed / 60,
        }, f, indent=2)


if __name__ == "__main__":
    main()
