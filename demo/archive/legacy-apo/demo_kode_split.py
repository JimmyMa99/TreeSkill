#!/usr/bin/env python3
"""
Kode Split Demo — 测试 auto-split 和 prune

用 Kode CLI 跑跨领域任务，触发技能树分裂:
- 数据处理领域: CSV/JSON 操作
- 代码规范领域: 特定格式要求
- 数学计算领域: 精确数值输出

烂 skill + 严格验证 → 高失败率 → auto-split

用法:
    conda activate pr
    export TREE_LLM_API_KEY=your-key
    python demo/demo_kode_split.py
"""

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

from treeskill.config import GlobalConfig
from treeskill.llm import LLMClient
from treeskill.optimizer import APOEngine
from treeskill.resume import ResumeState
from treeskill.schema import Feedback, Message, Skill, Trace
from treeskill.skill import save as save_skill
from treeskill.skill_tree import SkillTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("demo/outputs/kode-split")
NUM_ROUNDS = 3

# ── 故意烂的 Skill（不含任何领域知识） ──
BAD_SKILL = "Do what the user asks."

# ── 跨领域任务（严格验证） ──
TASKS = [
    # === 数据处理 ===
    {
        "id": "csv-stats",
        "task": "Create scores.csv with 'student,math,english' and rows: Alice,85,90 / Bob,72,88 / Carol,91,76 / Dave,68,95. Create stats.py that reads it, calculates average for each subject, and prints EXACTLY in format 'math_avg=XX.X english_avg=XX.X'. Run stats.py.",
        "verify": lambda r, w: 1.0 if "math_avg=79.0" in r.get("result", "") else 0.0,
        "domain": "data",
        "split": "train",
    },
    {
        "id": "json-merge",
        "task": "Create a.json with {\"users\":[{\"name\":\"Alice\",\"age\":30}]} and b.json with {\"users\":[{\"name\":\"Bob\",\"age\":25}]}. Create merge.py that merges both into merged.json (combined users array sorted by name). Print the merged result. Run it.",
        "verify": lambda r, w: 1.0 if (w / "merged.json").exists() and "Alice" in r.get("result", "") and "Bob" in r.get("result", "") else 0.0,
        "domain": "data",
        "split": "train",
    },
    {
        "id": "log-analyze",
        "task": "Create app.log with lines: '2024-01-01 ERROR db connection failed' / '2024-01-01 INFO server started' / '2024-01-02 ERROR timeout' / '2024-01-02 INFO request ok' / '2024-01-02 ERROR disk full'. Create analyze.py that counts ERROR and INFO, prints 'ERRORS=3 INFOS=2'. Run it.",
        "verify": lambda r, w: 1.0 if "ERRORS=3" in r.get("result", "") and "INFOS=2" in r.get("result", "") else 0.0,
        "domain": "data",
        "split": "train",
    },
    # === 代码规范 ===
    {
        "id": "docstring-func",
        "task": "Create mathlib.py with functions: add(a,b), multiply(a,b), power(a,b). Each function MUST have a Google-style docstring with Args, Returns, and Example sections. Then create check.py that imports mathlib, calls help(mathlib.add), and prints 'DOCSTRING_OK' if 'Args:' appears in the help output. Run check.py.",
        "verify": lambda r, w: 1.0 if "DOCSTRING_OK" in r.get("result", "") else 0.0,
        "domain": "code-quality",
        "split": "train",
    },
    {
        "id": "error-handling",
        "task": "Create safediv.py with a function safe_divide(a,b) that returns the result or 'ERROR: division by zero'. Test with: safe_divide(10,2) should print 5.0, safe_divide(10,0) should print 'ERROR: division by zero'. Print both results, each on a new line. Run it.",
        "verify": lambda r, w: 1.0 if "5.0" in r.get("result", "") and "ERROR: division by zero" in r.get("result", "") else 0.0,
        "domain": "code-quality",
        "split": "train",
    },
    {
        "id": "unittest-basic",
        "task": "Create calculator.py with add(a,b) and multiply(a,b). Create test_calculator.py using unittest with at least 3 test methods. Run 'python -m pytest test_calculator.py -v' or 'python -m unittest test_calculator -v'. The output must show all tests passed.",
        "verify": lambda r, w: 1.0 if ("OK" in r.get("result", "") or "passed" in r.get("result", "")) and (w / "test_calculator.py").exists() else 0.0,
        "domain": "code-quality",
        "split": "train",
    },
    # === 验证集 ===
    {
        "id": "pivot-table",
        "task": "Create sales.csv with 'region,product,amount': North,A,100 / North,B,200 / South,A,150 / South,B,50. Create pivot.py that creates a pivot table showing total amount per region. Print exactly: 'North=300 South=200'. Run it.",
        "verify": lambda r, w: 1.0 if "North=300" in r.get("result", "") and "South=200" in r.get("result", "") else 0.0,
        "domain": "data",
        "split": "val",
    },
    {
        "id": "type-hints",
        "task": "Create typed_utils.py with: def greet(name: str, times: int = 1) -> str that returns the greeting repeated. Create verify_types.py that uses typing.get_type_hints() on greet, prints 'TYPES_OK' if name is str and return is str. Run verify_types.py.",
        "verify": lambda r, w: 1.0 if "TYPES_OK" in r.get("result", "") else 0.0,
        "domain": "code-quality",
        "split": "val",
    },
    # === 测试集 ===
    {
        "id": "csv-filter",
        "task": "Create inventory.csv with 'item,quantity,price': Widget,50,9.99 / Gadget,5,29.99 / Doohickey,100,4.99 / Thingamajig,2,99.99. Create filter.py that filters items with quantity < 10 and prints them as 'LOW_STOCK: item (qty)'. Run it.",
        "verify": lambda r, w: 1.0 if "LOW_STOCK: Gadget" in r.get("result", "") and "LOW_STOCK: Thingamajig" in r.get("result", "") else 0.0,
        "domain": "data",
        "split": "test",
    },
    {
        "id": "decorator-log",
        "task": "Create decorators.py with a @log_calls decorator that prints 'CALLING: func_name' before and 'DONE: func_name' after each call. Apply it to a function greet(name) that returns f'Hello {name}'. Call greet('World') and print the return value. Run it.",
        "verify": lambda r, w: 1.0 if "CALLING: greet" in r.get("result", "") and "Hello World" in r.get("result", "") else 0.0,
        "domain": "code-quality",
        "split": "test",
    },
]


def run_kode(task_text: str, skill_prompt: str, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "AGENTS.md").write_text(skill_prompt, encoding="utf-8")
    try:
        proc = subprocess.run(
            ["kode", "-p", task_text, "--cwd", str(workdir),
             "--output-format", "json", "--dangerously-skip-permissions"],
            capture_output=True, text=True, timeout=180,
        )
        return json.loads(proc.stdout) if proc.stdout.strip() else {"is_error": True}
    except Exception as e:
        return {"is_error": True, "result": str(e)}


def main():
    t_start = time.time()

    train = [t for t in TASKS if t["split"] == "train"]
    val = [t for t in TASKS if t["split"] == "val"]
    test = [t for t in TASKS if t["split"] == "test"]
    logger.info(f"Tasks: train={len(train)}, val={len(val)}, test={len(test)}")
    logger.info(f"Domains: {set(t['domain'] for t in TASKS)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Baseline ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Baseline")
    logger.info("=" * 60)

    baseline_scores = []
    for task in test:
        workdir = OUTPUT_DIR / "workspaces" / f"baseline_{task['id']}"
        if workdir.exists():
            shutil.rmtree(workdir)
        result = run_kode(task["task"], BAD_SKILL, workdir)
        score = task["verify"](result, workdir) if not result.get("is_error") else 0.0
        baseline_scores.append(score)
        logger.info(f"  {task['id']} [{task['domain']}]: {score:.1f}")
    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    logger.info(f"  Baseline: {baseline_avg:.2f}")

    # ── Phase 2: APO + evolve_tree (auto-split) ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: APO with evolve_tree (auto-split enabled)")
    logger.info("=" * 60)

    judge_key = os.getenv("TREE_LLM_API_KEY")
    assert judge_key, "Set TREE_LLM_API_KEY"

    os.environ.update({
        "TREE_LLM_API_KEY": judge_key,
        "TREE_LLM_BASE_URL": os.getenv("TREE_LLM_BASE_URL", "https://oneapi.liuyanxing.site:8443/v1"),
        "TREE_LLM_MODEL": "bailian/qwen3.5-plus",
        "TREE_LLM_PROTOCOL": "openai",
        "TREE_LLM_JUDGE_API_KEY": judge_key,
        "TREE_LLM_JUDGE_BASE_URL": os.getenv("TREE_LLM_BASE_URL", "https://oneapi.liuyanxing.site:8443/v1"),
        "TREE_LLM_JUDGE_MODEL": os.getenv("TREE_LLM_JUDGE_MODEL", "bailian/glm-5"),
        "TREE_LLM_JUDGE_PROTOCOL": "openai",
    })

    from treeskill.config import LLMConfig, APOConfig
    config = GlobalConfig(
        llm=LLMConfig(),
        apo=APOConfig(
            gradient_accumulation_steps=4,
            beam_width=2,
            branch_factor=2,
            beam_rounds=1,
        ),
    )
    llm_client = LLMClient(config)
    engine = APOEngine(config, llm_client)

    # Score function
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
            scores.append(task["verify"](result, workdir) if not result.get("is_error") else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    engine._score_fn = kode_score_fn

    # Create skill tree
    skill_file = OUTPUT_DIR / "SKILL.md"
    if skill_file.exists():
        logger.info("Resuming from existing skill tree")
        tree = SkillTree.load(OUTPUT_DIR)
    else:
        root_skill = Skill(
            name="coding-assistant",
            description="Help users with coding tasks across data processing and code quality",
            system_prompt=BAD_SKILL,
            target=(
                "Improve task completion. Tasks span two domains: "
                "1) Data processing (CSV, JSON, log files) — needs precise output formats. "
                "2) Code quality (docstrings, type hints, testing, decorators) — needs Python best practices. "
                "Consider splitting into domain-specific sub-skills if one prompt can't handle both well."
            ),
            version="v1.0",
        )
        save_skill(root_skill, OUTPUT_DIR)
        tree = SkillTree.load(OUTPUT_DIR)

    # Multi-round optimization with evolve_tree
    for rnd in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n--- Round {rnd}/{NUM_ROUNDS} ---")

        # Collect traces
        traces = []
        for task in train:
            workdir = OUTPUT_DIR / "workspaces" / f"train_r{rnd}_{task['id']}"
            if workdir.exists():
                shutil.rmtree(workdir)
            result = run_kode(task["task"], tree.root.skill.system_prompt, workdir)
            score = task["verify"](result, workdir) if not result.get("is_error") else 0.0

            t = Trace(
                inputs=[Message(role="user", content=task["task"])],
                prediction=Message(role="assistant", content=result.get("result", "")),
                node_path="coding-assistant",
            )
            t.feedback = Feedback(
                score=score,
                critique=f"Score={score:.1f}, domain={task['domain']}, result={result.get('result', '')[:100]}",
                correction=f"Domain: {task['domain']}. Expected correct output for: {task['task'][:100]}",
            )
            traces.append(t)
            logger.info(f"  {task['id']} [{task['domain']}]: {score:.1f}")

        failures = [t for t in traces if t.feedback.score < 0.8]
        logger.info(f"  Failures: {len(failures)}/{len(traces)}")

        if not failures:
            logger.info("  All passing, skip")
            continue

        # evolve_tree with auto_split
        resume = ResumeState.create(
            OUTPUT_DIR, total_rounds=NUM_ROUNDS,
            metadata={"round": rnd, "failures": len(failures)},
        )
        resume.round_num = rnd

        t0 = time.time()
        engine.evolve_tree(tree, traces, auto_split=True, resume=resume)
        tree.save()
        resume.clear()

        logger.info(f"  Optimize: {time.time()-t0:.1f}s")
        logger.info(f"  Tree:\n{tree.list_tree()}")
        logger.info(f"  Leaves: {tree.root.leaf_count()}")

        # Check for structured actions
        if engine.pending_actions:
            logger.info(f"  Pending actions: {len(engine.pending_actions)}")
            for a in engine.pending_actions:
                logger.info(f"    {a['action']}")
            engine.pending_actions.clear()

    # ── Phase 3: Final test ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Final test")
    logger.info("=" * 60)

    final_scores = []
    for task in test:
        workdir = OUTPUT_DIR / "workspaces" / f"final_{task['id']}"
        if workdir.exists():
            shutil.rmtree(workdir)
        result = run_kode(task["task"], tree.root.skill.system_prompt, workdir)
        score = task["verify"](result, workdir) if not result.get("is_error") else 0.0
        final_scores.append(score)
        logger.info(f"  {task['id']} [{task['domain']}]: {score:.1f}")
    final_avg = sum(final_scores) / len(final_scores)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! ({elapsed/60:.1f} min)")
    logger.info(f"  Baseline: {baseline_avg:.2f}")
    logger.info(f"  Final:    {final_avg:.2f}")
    logger.info(f"  Delta:    {final_avg - baseline_avg:+.2f}")
    logger.info(f"  Leaves:   {tree.root.leaf_count()}")
    logger.info(f"  Tree:\n{tree.list_tree()}")


if __name__ == "__main__":
    main()
