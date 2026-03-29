#!/usr/bin/env python3
"""
Skill 蒸馏 Demo（可视化版）— 前端代码生成 + 截图保存

流程:
  1. Teacher (M2.7) + 原版 SKILL.md → gold standard 代码 + 截图
  2. Student (S1) + 原版 SKILL.md → baseline 代码 + 截图
  3. APO 蒸馏 SKILL.md → 适配 Student
  4. Student (S1) + 蒸馏后 SKILL.md → final 代码 + 截图
  5. Judge (M2.7) 看代码打分，截图保存供人工对比

用法:
    conda activate pr
    python demo/demo_distill_visual.py
"""

import base64
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

from treeskill.config import GlobalConfig
from treeskill.llm import LLMClient
from treeskill.optimizer import APOEngine
from treeskill.registry import registry, scorer
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
STUDENT_API_KEY = os.getenv("INTERN_API_KEY")
assert STUDENT_API_KEY, "Set INTERN_API_KEY env var"

TEACHER_MODEL = "MiniMax-M2.7"
TEACHER_BASE_URL = "https://api.minimaxi.com/anthropic"
TEACHER_API_KEY = os.getenv("MINIMAX_API_KEY")
assert TEACHER_API_KEY, "Set MINIMAX_API_KEY env var"

SKILL_PATH = Path("demo/data/minimax_frontend_skill.md")
DATA_PATH = Path("demo/data/frontend_tasks.json")
OUTPUT_DIR = Path("demo/outputs/distill-visual")
NUM_ROUNDS = 2
MAX_WORKERS = 3


# ── Data ───────────────────────────────────────────────

def load_tasks(difficulty: str = "easy"):
    """Load tasks filtered by difficulty."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    filtered = [t for t in tasks if t["difficulty"] == difficulty]
    logger.info(f"Loaded {len(filtered)} {difficulty} tasks")
    return filtered


def load_skill_md(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        body = parts[2].strip() if len(parts) > 2 else text
    else:
        body = text
    return Skill(
        name="frontend-dev",
        description="Frontend development skill for building production-ready web pages",
        system_prompt=body,
        target=(
            "Adapt this skill for a smaller reasoning model (Intern-S1-Pro). "
            "PRUNE sections the model struggles with (e.g. Three.js/WebGL). "
            "EXPAND key rules with explicit examples. "
            "Keep all script paths and tool references intact."
        ),
        version="v1.0",
    )


# ── API Helpers ────────────────────────────────────────

def call_model(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    user_text: str,
    max_tokens: int = 12000,
    temperature: float = 0.7,
    max_retries: int = 5,
) -> str:
    import random as _random
    for attempt in range(max_retries):
        try:
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
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = min(2 ** attempt, 30) * (0.5 + _random.random())
            logger.warning(f"call_model error (attempt {attempt+1}): {e} — retrying in {delay:.1f}s")
            time.sleep(delay)


# ── HTML & Screenshot ──────────────────────────────────

STANDALONE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{task_id} - {model}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://unpkg.com/framer-motion@11/dist/framer-motion.js"></script>
    <style>
        body {{ margin: 0; font-family: system-ui, -apple-system, sans-serif; }}
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
    const {{ useState, useEffect, useRef, useCallback, useMemo }} = React;
    const _FM = window.Motion || window["framer-motion"] || {{}};
    const {{ motion, AnimatePresence, useMotionValue, useTransform, useInView,
             useAnimation, useReducedMotion, useScroll, useSpring }} = _FM;
    {code}

    // Try to render — find first defined component
    const _names = ['App', 'Navbar', 'Hero', 'HeroSection', 'Card', 'ProfileCard',
                    'TeamMemberCard', 'Reveal', 'RevealSection', 'ScrollReveal',
                    'FadeIn', 'ContactForm', 'Footer', 'Page', 'Main', 'Component'];
    let _Root = null;
    for (const _n of _names) {{
        try {{ if (typeof eval(_n) === 'function') {{ _Root = eval(_n); break; }} }} catch(e) {{}}
    }}
    if (_Root) {{
        ReactDOM.createRoot(document.getElementById('root')).render(<_Root />);
    }} else {{
        document.getElementById('root').innerHTML = '<p style="padding:20px;color:red">No component found to render</p>';
    }}
    </script>
</body>
</html>"""


def extract_code(response: str) -> str:
    """Extract code from markdown code blocks."""
    # Try tsx/jsx/html blocks
    patterns = [r'```(?:tsx|jsx|react)\n(.*?)```', r'```(?:html)\n(.*?)```', r'```\n(.*?)```']
    for pat in patterns:
        m = re.search(pat, response, re.DOTALL)
        if m:
            return m.group(1).strip()
    # No code block — return as-is
    return response.strip()


def save_html(code: str, task_id: str, model_name: str, output_dir: Path) -> Path:
    """Save code as standalone HTML file."""
    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    path = html_dir / f"{task_id}_{safe_model}.html"

    # Extract just the code from markdown
    clean_code = extract_code(code)

    # Remove import statements (CDN provides these)
    clean_code = re.sub(r'^import\s+.*?;\s*$', '', clean_code, flags=re.MULTILINE)
    # Remove all export keywords
    clean_code = re.sub(r'^export\s+default\s+', '', clean_code, flags=re.MULTILINE)
    clean_code = re.sub(r'^export\s+', '', clean_code, flags=re.MULTILINE)
    # Remove "use client" directive
    clean_code = clean_code.replace('"use client";', '').replace("'use client';", '')

    html = STANDALONE_HTML_TEMPLATE.format(
        task_id=task_id, model=safe_model, code=clean_code,
    )
    path.write_text(html, encoding="utf-8")
    return path


def screenshot_html(html_path: Path, output_dir: Path) -> Path:
    """Take a headless screenshot of an HTML file."""
    from playwright.sync_api import sync_playwright

    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    png_path = screenshots_dir / f"{html_path.stem}.png"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(f"file://{html_path.resolve()}")
            page.wait_for_timeout(5000)  # Wait for CDN load + animations
            page.screenshot(path=str(png_path), full_page=False)
            browser.close()
        logger.info(f"  Screenshot: {png_path.name}")
    except Exception as e:
        logger.warning(f"  Screenshot failed for {html_path.name}: {e}")
        # Create a placeholder
        png_path.write_bytes(b"")

    return png_path


# ── Judge ──────────────────────────────────────────────

JUDGE_RUBRIC = """You are a strict frontend code reviewer. Compare the Student's code against the Gold Standard for the given task.

Task: {task}

Key rules to check: {key_rules}

Gold Standard code:
```
{gold_code}
```

Student code:
```
{student_code}
```

Evaluate:
1. Does the student code fulfill the task requirements? (0-0.3)
2. Does it follow the key rules listed above? (0-0.4)
3. Code quality: proper structure, no syntax errors, good patterns? (0-0.3)

Return ONLY a JSON object: {{"score": 0.X, "critique": "brief explanation"}}"""


def judge_code(
    judge_client: anthropic.Anthropic,
    task: Dict,
    student_code: str,
    gold_code: str,
) -> Dict[str, Any]:
    """Judge grades student code against gold standard."""
    prompt = JUDGE_RUBRIC.format(
        task=task["task"],
        key_rules=", ".join(task["key_rules"]),
        gold_code=gold_code[:4000],
        student_code=student_code[:4000],
    )
    raw = call_model(
        judge_client, TEACHER_MODEL,
        "You are a code quality judge. Return only JSON.", prompt,
        max_tokens=500,
    )
    try:
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


# ── Register visual scorer ─────────────────────────────

@scorer("code-review")
def code_review_scorer(output: str, expected: str, context: dict) -> float:
    """Score by having judge compare student code vs gold code."""
    judge_client = context.get("judge_client")
    task = context.get("task", {})
    if not judge_client:
        return 0.5
    result = judge_code(judge_client, task, output, expected)
    return result["score"]


# ── Main Pipeline ──────────────────────────────────────

def generate_and_save(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    task: Dict,
    output_dir: Path,
    model_label: str,
) -> tuple:
    """Generate code, save HTML. Returns (code, html_path)."""
    code = call_model(client, model, system_prompt, task["task"])
    html_path = save_html(code, task["id"], model_label, output_dir)
    return code, html_path


def main():
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("Skill Distillation Demo (Visual) — Frontend Dev")
    logger.info(f"  Teacher: {TEACHER_MODEL}")
    logger.info(f"  Student: {STUDENT_MODEL}")
    logger.info("=" * 60)

    # Load easy tasks only
    tasks = load_tasks("easy")

    if not SKILL_PATH.exists():
        logger.error(f"SKILL.md not found at {SKILL_PATH}")
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Teacher gold standards ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Teacher generating gold standards + screenshots")
    logger.info("=" * 60)

    gold_cache = OUTPUT_DIR / "gold_standards.json"
    if gold_cache.exists():
        with open(gold_cache) as f:
            gold = json.load(f)
        logger.info(f"Loaded {len(gold)} cached gold standards")
    else:
        gold = {}
        for task in tasks:
            logger.info(f"  Teacher: {task['id']}...")
            code, html_path = generate_and_save(
                teacher_client, TEACHER_MODEL, skill.system_prompt,
                task, OUTPUT_DIR, "teacher",
            )
            gold[task["id"]] = code
            logger.info(f"    Code: {len(code)} chars, HTML: {html_path.name}")

        with open(gold_cache, "w") as f:
            json.dump(gold, f, ensure_ascii=False, indent=2)

    # ── Phase 2: Student baseline ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Student baseline (original SKILL.md)")
    logger.info("=" * 60)

    baseline_scores = []
    for task in tasks:
        logger.info(f"  Student baseline: {task['id']}...")
        code, html_path = generate_and_save(
            student_client, STUDENT_MODEL, skill.system_prompt,
            task, OUTPUT_DIR, "student_baseline",
        )
        result = judge_code(teacher_client, task, code, gold[task["id"]])
        baseline_scores.append(result["score"])
        logger.info(f"    Score: {result['score']:.2f} — {result['critique'][:80]}")

    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    logger.info(f"\n  Baseline avg: {baseline_avg:.2f}")

    # ── Phase 3: APO Distillation ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: APO distilling SKILL.md for Student")
    logger.info("=" * 60)

    # Setup framework config
    os.environ["TREE_LLM_API_KEY"] = STUDENT_API_KEY
    os.environ["TREE_LLM_BASE_URL"] = STUDENT_BASE_URL
    os.environ["TREE_LLM_MODEL"] = STUDENT_MODEL
    os.environ["TREE_LLM_PROTOCOL"] = "anthropic"
    os.environ["TREE_LLM_JUDGE_API_KEY"] = TEACHER_API_KEY
    os.environ["TREE_LLM_JUDGE_BASE_URL"] = TEACHER_BASE_URL
    os.environ["TREE_LLM_JUDGE_MODEL"] = TEACHER_MODEL
    os.environ["TREE_LLM_JUDGE_PROTOCOL"] = "anthropic"

    from treeskill.config import LLMConfig, APOConfig
    llm_config = LLMConfig()
    apo_config = APOConfig(
        num_candidates=2,
        gradient_accumulation_steps=3,
        beam_width=2,
        branch_factor=2,
        beam_rounds=1,
    )
    config = GlobalConfig(llm=llm_config, apo=apo_config)
    llm = LLMClient(config)
    engine = APOEngine(config, llm)

    # Score function: run student + judge code
    def real_score_fn(prompt: str, traces: List[Trace]) -> float:
        scores = []
        for t in traces:
            output = call_model(student_client, STUDENT_MODEL, prompt, t.inputs[-1].content)
            expected = t.feedback.correction if t.feedback and t.feedback.correction else ""
            raw = call_model(
                teacher_client, TEACHER_MODEL,
                "You are a code quality judge. Score 0-1. Return ONLY a number.",
                f"Gold:\n{expected[:3000]}\n\nStudent:\n{output[:3000]}\n\nScore:",
                max_tokens=100,
            )
            try:
                m = re.search(r'(\d+\.?\d*)', raw)
                s = float(m.group(1)) if m else 0.5
                scores.append(max(0.0, min(1.0, s)))
            except (ValueError, AttributeError):
                scores.append(0.5)
        return sum(scores) / len(scores) if scores else 0.0

    engine._score_fn = real_score_fn

    best_skill = skill
    best_score = baseline_avg

    for round_num in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n--- Round {round_num}/{NUM_ROUNDS} ---")

        # Collect traces
        traces = []
        for task in tasks:
            code = call_model(
                student_client, STUDENT_MODEL,
                best_skill.system_prompt, task["task"],
            )
            gold_code = gold.get(task["id"], "")
            result = judge_code(teacher_client, task, code, gold_code)

            t = Trace(
                inputs=[Message(role="user", content=task["task"])],
                prediction=Message(role="assistant", content=code),
            )
            t.feedback = Feedback(
                score=result["score"],
                critique=result["critique"],
                correction=gold_code,
            )
            traces.append(t)

        failures = [t for t in traces if t.feedback and t.feedback.score < 0.7]
        logger.info(f"  Traces: {len(failures)}/{len(traces)} below 0.7")

        if not failures:
            logger.info("  All passing, skip")
            continue

        t0 = time.time()
        candidate = engine.optimize(best_skill, traces)
        logger.info(f"  Optimize: {time.time()-t0:.1f}s → {candidate.version}")

        # Evaluate candidate
        candidate_scores = []
        for task in tasks:
            code = call_model(
                student_client, STUDENT_MODEL,
                candidate.system_prompt, task["task"],
            )
            result = judge_code(teacher_client, task, code, gold[task["id"]])
            candidate_scores.append(result["score"])

        candidate_avg = sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0
        logger.info(f"  Candidate avg: {candidate_avg:.2f} (prev: {best_score:.2f})")

        if candidate_avg > best_score:
            best_score = candidate_avg
            best_skill = candidate
            save_skill(best_skill, OUTPUT_DIR)
            logger.info(f"  ★ Accepted {best_skill.version}")
        else:
            logger.info(f"  Rejected")

    # ── Phase 4: Final evaluation + screenshots ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: Final evaluation with distilled SKILL.md")
    logger.info("=" * 60)

    final_scores = []
    for task in tasks:
        logger.info(f"  Final: {task['id']}...")
        code, html_path = generate_and_save(
            student_client, STUDENT_MODEL, best_skill.system_prompt,
            task, OUTPUT_DIR, "student_final",
        )
        result = judge_code(teacher_client, task, code, gold[task["id"]])
        final_scores.append(result["score"])
        logger.info(f"    Score: {result['score']:.2f} — {result['critique'][:80]}")

    final_avg = sum(final_scores) / len(final_scores) if final_scores else 0

    # ── Summary ──
    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Distillation complete! ({elapsed/60:.1f} min)")
    logger.info(f"  Teacher:  {TEACHER_MODEL}")
    logger.info(f"  Student:  {STUDENT_MODEL}")
    logger.info(f"  Baseline: {baseline_avg:.2f}")
    logger.info(f"  Final:    {final_avg:.2f}")
    logger.info(f"  Delta:    {final_avg - baseline_avg:+.2f}")
    logger.info(f"  Version:  {best_skill.version}")
    logger.info(f"\nScreenshots saved to: {OUTPUT_DIR / 'screenshots'}")
    logger.info(f"HTML files saved to:  {OUTPUT_DIR / 'html'}")

    summary = {
        "teacher": TEACHER_MODEL,
        "student": STUDENT_MODEL,
        "baseline": baseline_avg,
        "final": final_avg,
        "delta": final_avg - baseline_avg,
        "rounds": NUM_ROUNDS,
        "version": best_skill.version,
        "tasks": len(tasks),
        "elapsed_min": elapsed / 60,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
