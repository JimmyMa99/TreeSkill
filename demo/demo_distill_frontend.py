#!/usr/bin/env python3
"""
Skill 蒸馏 Demo — 将强模型 Skill 适配到弱模型

流程:
  Phase 1: Teacher (M2.7) + 原版 SKILL.md → 生成 gold standard
  Phase 2: Student (S1) + 原版 SKILL.md → baseline 评估
  Phase 3: APO 优化 SKILL.md 适配 Student → 蒸馏
  Phase 4: Student (S1) + 蒸馏后 SKILL.md → 最终评估

用法:
    conda activate pr
    python demo/demo_distill_frontend.py
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

from treeskill.config import GlobalConfig
from treeskill.llm import LLMClient
from treeskill.optimizer import APOEngine
from treeskill.schema import Feedback, Message, Skill, Trace
from treeskill.skill import save as save_skill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────
STUDENT_MODEL = "intern-s1-pro"
STUDENT_BASE_URL = "https://chat.intern-ai.org.cn"
STUDENT_API_KEY = os.getenv(
    "INTERN_API_KEY",
    "REDACTED_INTERN_KEY",
)

TEACHER_MODEL = "MiniMax-M2.7"
TEACHER_BASE_URL = "https://api.minimaxi.com/anthropic"
TEACHER_API_KEY = os.getenv(
    "MINIMAX_API_KEY",
    "REDACTED_MINIMAX_KEY",
)

SKILL_PATH = Path("demo/data/minimax_frontend_skill.md")
DATA_PATH = Path("demo/data/frontend_tasks.json")
OUTPUT_DIR = Path("demo/outputs/distill-frontend")
NUM_ROUNDS = 3
MAX_WORKERS = 4


# ── Data ───────────────────────────────────────────────

def load_tasks():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    train = [t for t in tasks if t["split"] == "train"]
    val = [t for t in tasks if t["split"] == "val"]
    test = [t for t in tasks if t["split"] == "test"]
    logger.info(f"Tasks: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def load_skill_md(path: Path) -> Skill:
    """Load SKILL.md and return a Skill object."""
    text = path.read_text(encoding="utf-8")
    # Split frontmatter from body
    if text.startswith("---"):
        parts = text.split("---", 2)
        body = parts[2].strip() if len(parts) > 2 else text
    else:
        body = text
    return Skill(
        name="frontend-dev",
        description="Frontend development skill for building production-ready web pages",
        system_prompt=body,
        target="Adapt this skill for a smaller model while preserving key rules and output quality",
        version="v1.0",
    )


# ── API Helpers ────────────────────────────────────────

def call_model(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    user_text: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str:
    """Call an Anthropic-protocol API and return text content."""
    resp = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_text}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text_parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    return "".join(text_parts).strip()


# ── Phase 1: Gold Standards ───────────────────────────

def generate_gold_standards(
    teacher: anthropic.Anthropic,
    system_prompt: str,
    tasks: List[Dict],
) -> Dict[str, str]:
    """Generate gold standard outputs with the teacher model."""
    cache_path = OUTPUT_DIR / "gold_standards.json"
    if cache_path.exists():
        logger.info("Loading cached gold standards")
        with open(cache_path, "r") as f:
            return json.load(f)

    logger.info(f"Generating gold standards for {len(tasks)} tasks...")
    gold = {}

    def _gen(task):
        output = call_model(teacher, TEACHER_MODEL, system_prompt, task["task"])
        return task["id"], output

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_gen, t) for t in tasks]
        for i, f in enumerate(as_completed(futures)):
            tid, output = f.result()
            gold[tid] = output
            logger.info(f"  [{i+1}/{len(tasks)}] {tid}: {len(output)} chars")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)
    return gold


# ── Evaluation ─────────────────────────────────────────

JUDGE_PROMPT = """You are a strict frontend code reviewer. Compare the Student output against the Gold Standard for the given task.

Task: {task}

Key rules to check: {key_rules}

Gold Standard:
```
{gold}
```

Student Output:
```
{student}
```

Evaluate:
1. Does the student code fulfill the task requirements?
2. Does it follow the key rules listed above?
3. How close is it to the gold standard in quality?

Return ONLY a JSON object: {{"score": 0.X, "critique": "brief explanation"}}"""


def judge_output(
    judge: anthropic.Anthropic,
    task: Dict,
    student_output: str,
    gold_output: str,
) -> Dict[str, Any]:
    """Have the judge compare student output vs gold standard."""
    prompt = JUDGE_PROMPT.format(
        task=task["task"],
        key_rules=", ".join(task["key_rules"]),
        gold=gold_output[:3000],
        student=student_output[:3000],
    )
    raw = call_model(judge, TEACHER_MODEL, "You are a code quality judge.", prompt)

    # Parse score
    try:
        # Try to extract JSON from response
        import re
        m = re.search(r'\{[^}]+\}', raw)
        if m:
            result = json.loads(m.group())
            return {
                "score": max(0.0, min(1.0, float(result.get("score", 0.5)))),
                "critique": result.get("critique", raw[:200]),
            }
    except (json.JSONDecodeError, ValueError):
        pass
    return {"score": 0.5, "critique": raw[:200]}


def evaluate_batch(
    student: anthropic.Anthropic,
    judge: anthropic.Anthropic,
    system_prompt: str,
    tasks: List[Dict],
    gold_standards: Dict[str, str],
    label: str = "",
) -> float:
    """Run student on tasks, judge against gold, return avg score."""
    scores = []

    def _eval_one(task):
        output = call_model(student, STUDENT_MODEL, system_prompt, task["task"])
        gold = gold_standards.get(task["id"], "")
        result = judge_output(judge, task, output, gold)
        return task["id"], result["score"], result["critique"], output

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_eval_one, t) for t in tasks]
        for f in as_completed(futures):
            tid, score, critique, _ = f.result()
            scores.append(score)
            logger.info(f"  {tid}: {score:.2f} — {critique[:80]}")

    avg = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"  [{label}] avg score: {avg:.2f} ({len(scores)} tasks)")
    return avg


# ── Phase 3: Collect Traces for APO ───────────────────

def collect_traces(
    student: anthropic.Anthropic,
    judge: anthropic.Anthropic,
    system_prompt: str,
    tasks: List[Dict],
    gold_standards: Dict[str, str],
) -> List[Trace]:
    """Run student, judge outputs, create Trace objects for APO."""
    traces = []

    def _process(task):
        output = call_model(student, STUDENT_MODEL, system_prompt, task["task"])
        gold = gold_standards.get(task["id"], "")
        result = judge_output(judge, task, output, gold)

        t = Trace(
            inputs=[Message(role="user", content=task["task"])],
            prediction=Message(role="assistant", content=output),
        )
        t.feedback = Feedback(
            score=result["score"],
            critique=result["critique"],
            correction=gold,
        )
        return t

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        traces = list(pool.map(_process, tasks))

    return traces


# ── Main ───────────────────────────────────────────────

def main():
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("Skill Distillation Demo — Frontend Dev")
    logger.info(f"  Teacher: {TEACHER_MODEL}")
    logger.info(f"  Student: {STUDENT_MODEL}")
    logger.info("=" * 60)

    train, val, test = load_tasks()

    # Check skill source
    if not SKILL_PATH.exists():
        logger.error(f"SKILL.md not found at {SKILL_PATH}")
        logger.info("Run: curl -sL https://raw.githubusercontent.com/MiniMax-AI/skills/main/skills/frontend-dev/SKILL.md -o demo/data/minimax_frontend_skill.md")
        return

    skill = load_skill_md(SKILL_PATH)
    logger.info(f"Skill loaded: {len(skill.system_prompt)} chars")

    # API clients
    student_client = anthropic.Anthropic(
        api_key=STUDENT_API_KEY, base_url=STUDENT_BASE_URL,
    )
    teacher_client = anthropic.Anthropic(
        api_key=TEACHER_API_KEY, base_url=TEACHER_BASE_URL,
    )

    # Framework config (for APO engine)
    config = GlobalConfig()
    config = config.model_copy(update={
        "llm": config.llm.model_copy(update={
            "model": STUDENT_MODEL,
            "base_url": STUDENT_BASE_URL,
            "protocol": "anthropic",
            "judge_model": TEACHER_MODEL,
            "judge_base_url": TEACHER_BASE_URL,
            "judge_protocol": "anthropic",
        }),
        "apo": config.apo.model_copy(update={
            "num_candidates": 2,
            "gradient_accumulation_steps": 6,
            "beam_width": 2,
            "branch_factor": 2,
            "beam_rounds": 2,
        }),
    })
    llm = LLMClient(config)
    engine = APOEngine(config, llm)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Gold Standards ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Teacher generating gold standards")
    logger.info("=" * 60)
    all_tasks = train + val + test
    gold_standards = generate_gold_standards(
        teacher_client, skill.system_prompt, all_tasks,
    )
    logger.info(f"Gold standards: {len(gold_standards)} tasks cached")

    # ── Phase 2: Baseline ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Student baseline (original SKILL.md)")
    logger.info("=" * 60)
    baseline_score = evaluate_batch(
        student_client, teacher_client,
        skill.system_prompt, test, gold_standards, "baseline",
    )

    # ── Phase 3: APO Distillation ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: APO optimizing SKILL.md for Student")
    logger.info("=" * 60)

    # Score function: run student + judge grade vs gold
    def real_score_fn(prompt: str, traces: List[Trace]) -> float:
        scores = []
        for t in traces:
            output = call_model(student_client, STUDENT_MODEL, prompt, t.inputs[-1].content)
            gold = t.feedback.correction if t.feedback and t.feedback.correction else ""
            if not gold:
                gold = t.prediction.content
            raw = call_model(
                teacher_client, TEACHER_MODEL,
                "You are a code quality judge. Score the match 0-1. Return ONLY a number.",
                f"Gold:\n{gold[:2000]}\n\nStudent:\n{output[:2000]}\n\nScore (0.0-1.0):",
            )
            try:
                import re
                m = re.search(r'(\d+\.?\d*)', raw)
                s = float(m.group(1)) if m else 0.5
                scores.append(max(0.0, min(1.0, s)))
            except (ValueError, AttributeError):
                scores.append(0.5)
        return sum(scores) / len(scores) if scores else 0.0

    engine._score_fn = real_score_fn

    best_skill = skill
    best_val_score = -1.0

    for round_num in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n--- Round {round_num}/{NUM_ROUNDS} ---")

        t0 = time.time()
        traces = collect_traces(
            student_client, teacher_client,
            best_skill.system_prompt, train, gold_standards,
        )
        failures = [t for t in traces if t.feedback and t.feedback.score < 0.7]
        logger.info(f"  Traces: {len(failures)}/{len(traces)} below 0.7 ({time.time()-t0:.1f}s)")

        if not failures:
            logger.info("  All passing, skip")
            continue

        t0 = time.time()
        candidate = engine.optimize(best_skill, traces)
        logger.info(f"  Optimize: {time.time()-t0:.1f}s → {candidate.version}")

        # Validate
        val_score = evaluate_batch(
            student_client, teacher_client,
            candidate.system_prompt, val, gold_standards,
            f"val round {round_num}",
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_skill = candidate
            save_skill(best_skill, OUTPUT_DIR)
            logger.info(f"  ★ Accepted {best_skill.version} (val={val_score:.2f})")
        else:
            logger.info(f"  Rejected (val={val_score:.2f} <= {best_val_score:.2f})")

    # ── Phase 4: Final Test ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: Final evaluation on test split")
    logger.info("=" * 60)
    final_score = evaluate_batch(
        student_client, teacher_client,
        best_skill.system_prompt, test, gold_standards, "final",
    )

    # ── Summary ──
    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Distillation complete! ({elapsed/60:.1f} min)")
    logger.info(f"  Teacher:  {TEACHER_MODEL}")
    logger.info(f"  Student:  {STUDENT_MODEL}")
    logger.info(f"  Baseline: {baseline_score:.2f} (original SKILL.md)")
    logger.info(f"  Final:    {final_score:.2f} (distilled SKILL.md)")
    logger.info(f"  Delta:    {final_score - baseline_score:+.2f}")
    logger.info(f"  Version:  {best_skill.version}")
    logger.info(f"  Output:   {OUTPUT_DIR}")

    # Save summary
    summary = {
        "teacher": TEACHER_MODEL,
        "student": STUDENT_MODEL,
        "baseline": baseline_score,
        "final": final_score,
        "delta": final_score - baseline_score,
        "rounds": NUM_ROUNDS,
        "version": best_skill.version,
        "elapsed_min": elapsed / 60,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
