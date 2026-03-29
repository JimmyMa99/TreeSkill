#!/usr/bin/env python3
"""Frontend distillation demo using ASO + Kode.

Pipeline:
- Load Frontend skill from MiniMax-AI/skills/frontend-dev
- Use a strong judge (MiniMax) to create gold outputs (Phase: teacher)
- Run student actor (intern) through Kode as baseline
- Evolve `ASOProgram` (skills + selection policy) with `ASOOptimizer`
- Auto merge / prune on the validated frontier

Configuration is intentionally env-driven and kept close to current pipeline
conventions so you can tune without editing this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import load_dotenv

load_dotenv()

from treeskill.aso_optimizer import ASOOptimizer
from treeskill.aso_program import ASOProgram, ASOSkill
from treeskill.config import APOConfig, GlobalConfig, LLMConfig
from treeskill.llm import LLMClient
from treeskill.schema import Message
from treeskill.tasks.sealqa import SealQAExample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


OUTPUT_DIR = Path("demo/outputs/distill-frontend-aso")
WORKSPACES_DIR = OUTPUT_DIR / "workspaces"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
ERROR_PATH = OUTPUT_DIR / "error.json"

DATA_PATH = Path(os.getenv("FRONTEND_DATA_PATH", "demo/data/frontend_tasks.json"))
SKILL_PATH = Path(os.getenv("FRONTEND_SKILL_PATH", "demo/data/minimax_frontend_skill.md"))

ACTOR_MODEL = os.getenv("FRONTEND_ACTOR_MODEL", "intern-s1-pro")
ACTOR_PROTOCOL = os.getenv("FRONTEND_ACTOR_PROTOCOL", "")
ACTOR_BASE_URL = os.getenv("FRONTEND_ACTOR_BASE_URL", "https://chat.intern-ai.org.cn")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
ACTOR_API_KEY = (
    os.getenv("FRONTEND_ACTOR_API_KEY")
    or os.getenv("INTERN_API_KEY")
    or os.getenv("TREE_LLM_API_KEY")
    or ""
)

JUDGE_MODEL = os.getenv("FRONTEND_JUDGE_MODEL", "MiniMax-M2.7")
JUDGE_BASE_URL = os.getenv("FRONTEND_JUDGE_BASE_URL", "https://api.minimaxi.com/anthropic")
JUDGE_PROTOCOL = os.getenv("FRONTEND_JUDGE_PROTOCOL", "")
JUDGE_API_KEY = (
    os.getenv("TREE_LLM_JUDGE_API_KEY")
    or os.getenv("FRONTEND_JUDGE_API_KEY")
    or MINIMAX_API_KEY
    or os.getenv("TREE_LLM_API_KEY")
)
EFFECTIVE_ACTOR_API_KEY = ACTOR_API_KEY or (JUDGE_API_KEY if "minimax" in ACTOR_MODEL.lower() else "")

REWRITE_MODEL = os.getenv("FRONTEND_REWRITE_MODEL", JUDGE_MODEL)
REWRITE_BASE_URL = os.getenv("FRONTEND_REWRITE_BASE_URL", JUDGE_BASE_URL)
REWRITE_PROTOCOL = os.getenv("FRONTEND_REWRITE_PROTOCOL", "")
REWRITE_API_KEY = JUDGE_API_KEY


def _infer_protocol(model: str, base_url: str, explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    lowered_url = (base_url or "").lower()
    if "anthropic" in lowered_url:
        return "anthropic"
    return "openai"


ACTOR_PROTOCOL = _infer_protocol(ACTOR_MODEL, ACTOR_BASE_URL, ACTOR_PROTOCOL)
JUDGE_PROTOCOL = _infer_protocol(JUDGE_MODEL, JUDGE_BASE_URL, JUDGE_PROTOCOL)
REWRITE_PROTOCOL = _infer_protocol(REWRITE_MODEL, REWRITE_BASE_URL, REWRITE_PROTOCOL)

_MINIMAX_MODEL_HINT = "minimax" in JUDGE_MODEL.lower() or "minimax" in REWRITE_MODEL.lower()
JUDGE_EXTRA_BODY = (
    {"thinking": {"type": "disabled"}} if _MINIMAX_MODEL_HINT else None
)
REWRITE_EXTRA_BODY = (
    {"thinking": {"type": "disabled"}} if _MINIMAX_MODEL_HINT else None
)

TRAIN_SIZE = max(1, int(os.getenv("FRONTEND_TRAIN_SIZE", "4")))
VAL_SIZE = max(1, int(os.getenv("FRONTEND_VAL_SIZE", "2")))
TEST_SIZE = max(1, int(os.getenv("FRONTEND_TEST_SIZE", "2")))

MAX_ITERATIONS = max(1, int(os.getenv("FRONTEND_ASO_MAX_ITERATIONS", "2")))
FRONTIER_SIZE = max(1, int(os.getenv("FRONTEND_ASO_FRONTIER_SIZE", "3")))
BRANCH_FACTOR = max(1, int(os.getenv("FRONTEND_ASO_BRANCH_FACTOR", "2")))
EVAL_WORKERS = max(1, int(os.getenv("FRONTEND_EVAL_WORKERS", "4")))
ASO_WORKERS = max(1, int(os.getenv("FRONTEND_ASO_WORKERS", "4")))
AUTO_MERGE = os.getenv("FRONTEND_ASO_AUTO_MERGE", "1") != "0"
AUTO_PRUNE = os.getenv("FRONTEND_ASO_AUTO_PRUNE", "1") != "0"
KODE_TIMEOUT_SECONDS = max(30, int(os.getenv("FRONTEND_KODE_TIMEOUT", "180")))

ROOT_PROMPT = (
    "You are a production-grade frontend engineer."
    " Use the provided skills to generate high-quality React/Tailwind/TSX code"
    " and concise implementation instructions when asked."
)


@dataclass
class DistillTask:
    id: str
    question: str
    answer: str
    split: str
    category: str = "frontend"
    difficulty: str = "easy"
    key_rules: list[str] | None = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_example(self, answer: str) -> SealQAExample:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "split": self.split,
                "category": self.category,
                "difficulty": self.difficulty,
                "key_rules": json.dumps(self.key_rules or []),
            }
        )
        return SealQAExample(
            question=self.question,
            answer=answer,
            topic=self.id,
            metadata=metadata,
        )


def _parse_frontmatter(markdown: str) -> str:
    text = markdown.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text


def _strip_thinking_blocks(text: str) -> str:
    if not text:
        return ""
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def _load_tasks() -> List[DistillTask]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    tasks: List[DistillTask] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tasks.append(
            DistillTask(
                id=str(row.get("id", "")),
                question=str(row.get("task", "")).strip(),
                answer="",
                split=str(row.get("split", "train")),
                category=str(row.get("category", "frontend")),
                difficulty=str(row.get("difficulty", "easy")),
                key_rules=[str(item).strip() for item in row.get("key_rules", []) if str(item).strip()],
                metadata={k: str(v) for k, v in row.items() if isinstance(k, str) and k not in {"id", "task", "answer", "split", "category", "difficulty"} and v is not None},
            )
        )
    return tasks


def _split_tasks(tasks: Iterable[DistillTask]) -> Tuple[List[DistillTask], List[DistillTask], List[DistillTask]]:
    train: List[DistillTask] = []
    val: List[DistillTask] = []
    test: List[DistillTask] = []
    for item in tasks:
        split = item.split.lower()
        if split == "val":
            val.append(item)
        elif split == "test":
            test.append(item)
        else:
            train.append(item)

    return (
        train[:TRAIN_SIZE],
        val[:VAL_SIZE],
        test[:TEST_SIZE],
    )


def _parse_key_rules(task: DistillTask) -> list[str]:
    metadata = task.metadata
    if task.key_rules:
        return task.key_rules
    if not metadata:
        return []
    raw = metadata.get("key_rules")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                cleaned = text.strip("[]")
                if cleaned:
                    return [
                        item.strip().strip("'\"")
                        for item in cleaned.split(",")
                        if item.strip().strip("'\"")
                    ]
        if "," in text:
            return [item.strip() for item in text.split(",") if item.strip()]
        return [text]
    return [str(raw)]


def _run_with_role(
    llm: LLMClient,
    task: DistillTask,
    role: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    try:
        response = llm.generate(
            [
                Message(role="system", content=prompt),
                Message(role="user", content=task.question),
            ],
            role=role,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        return _strip_thinking_blocks(raw).strip()
    except Exception as exc:
        logger.warning("LLM call failed for task=%s role=%s: %s", task.id, role, exc)
        return ""


def _judge_score(llm: LLMClient, task: DistillTask, prediction: str, gold: str) -> float:
    system_prompt = (
        "You are a strict frontend reviewer. Compare the student output with the gold-standard target for the task. "
        "Be conservative: only give high score when implementation quality and requirements are both clearly met."
    )
    rules = ", ".join(_parse_key_rules(task))
    user_prompt = (
        "Task:\n"
        f"{task.question}\n\n"
        "Key rules:\n"
        f"{rules}\n\n"
        "Gold standard:\n"
        f"{gold}\n\n"
        "Student output:\n"
        f"{prediction}\n\n"
        "Return only JSON like {\"score\": 0.0-1.0, \"critique\": \"...\"}."
    )

    try:
        response = llm.generate(
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            role="judge",
            temperature=0.0,
            max_tokens=800,
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        logger.warning("Judge failed for task=%s: %s", task.id, exc)
        return 0.0

    parsed: dict[str, Any] | None = None
    content = _strip_thinking_blocks(raw).strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    try:
        parsed = json.loads(content)
    except Exception:
        match = re.search(r"([01](?:\.\d+)?|0?\.\d+)", content)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                return 0.0
        return 0.0

    if not isinstance(parsed, dict):
        return 0.0

    score_raw = parsed.get("score")
    if isinstance(score_raw, (int, float)):
        return max(0.0, min(1.0, float(score_raw)))
    if isinstance(score_raw, str):
        try:
            return max(0.0, min(1.0, float(score_raw.strip())))
        except ValueError:
            return 0.0
    return 0.0


def _run_kode(program: ASOProgram, task: DistillTask, workspace_root: Path) -> str:
    workspace = workspace_root / task.id
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    if not EFFECTIVE_ACTOR_API_KEY:
        logger.warning("No actor API key, skip kode run for task=%s", task.id)
        return ""

    (workspace / "AGENTS.md").write_text(program.render_agents_markdown(), encoding="utf-8")

    env = os.environ.copy()
    env["TREE_LLM_API_KEY"] = EFFECTIVE_ACTOR_API_KEY
    env["TREE_LLM_BASE_URL"] = ACTOR_BASE_URL
    env["TREE_LLM_MODEL"] = ACTOR_MODEL
    env["TREE_LLM_PROTOCOL"] = ACTOR_PROTOCOL

    command = [
        "kode",
        "-p",
        task.question,
        "--model",
        ACTOR_MODEL,
        "--cwd",
        str(workspace),
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=KODE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Kode timed out task=%s timeout=%ds", task.id, KODE_TIMEOUT_SECONDS)
        return ""
    except Exception as exc:
        logger.warning("Kode exec error task=%s: %s", task.id, exc)
        return ""

    if process.returncode != 0:
        logger.warning(
            "Kode returned non-zero code=%d for task=%s stderr=%s",
            process.returncode,
            task.id,
            process.stderr.strip()[:200],
        )
    if not process.stdout.strip():
        logger.warning("Kode empty output for task=%s: %s", task.id, process.stderr.strip()[:200])
        return ""
    try:
        payload = json.loads(process.stdout)
        return str(payload.get("result", "")).strip()
    except json.JSONDecodeError:
        logger.warning("Invalid Kode JSON for task=%s: %s", task.id, process.stdout[:200])
        return process.stdout.strip()
    except Exception as exc:
        logger.warning("Failed parsing Kode output task=%s: %s", task.id, exc)
        return ""


def _evaluate(
    program: ASOProgram,
    tasks: List[DistillTask],
    llm: LLMClient,
    gold_map: Dict[str, str],
    workspace_root: Path,
    *,
    label: str,
) -> Dict[str, Any]:
    if not tasks:
        return {"score": 0.0, "rows": []}

    def _run_one(task: DistillTask) -> Dict[str, object]:
        start = time.time()
        gold = gold_map.get(task.id, "")
        prediction = ""
        score = 0.0
        kode_seconds = 0.0
        judge_seconds = 0.0
        try:
            t0 = time.time()
            prediction = _run_kode(program, task, workspace_root)
            kode_seconds = time.time() - t0

            t1 = time.time()
            score = _judge_score(llm, task, prediction, gold)
            judge_seconds = time.time() - t1
        except Exception as exc:
            logger.warning("Evaluation failed task=%s label=%s: %s", task.id, label, exc)
        total_seconds = time.time() - start
        return {
            "task_id": task.id,
            "split": task.split,
            "question": task.question,
            "prediction": prediction,
            "gold": gold,
            "score": score,
            "kode_seconds": round(kode_seconds, 2),
            "judge_seconds": round(judge_seconds, 2),
            "total_seconds": round(total_seconds, 2),
        }

    rows: list[Dict[str, object]] = []
    scores: list[float] = []

    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            scores.append(float(row["score"]))

    avg = sum(scores) / len(scores) if scores else 0.0
    total_kode = sum(float(row["kode_seconds"]) for row in rows)
    total_judge = sum(float(row["judge_seconds"]) for row in rows)
    total_elapsed = sum(float(row["total_seconds"]) for row in rows)
    logger.info(
        "[%s] samples=%d avg=%.3f avg_kode=%.2fs avg_judge=%.2fs avg_total=%.2fs workers=%d",
        label,
        len(tasks),
        avg,
        total_kode / len(rows) if rows else 0.0,
        total_judge / len(rows) if rows else 0.0,
        total_elapsed / len(rows) if rows else 0.0,
        EVAL_WORKERS,
    )
    return {"score": avg, "rows": rows}


def _build_base_program(skill_body: str) -> ASOProgram:
    return ASOProgram(
        root_prompt=ROOT_PROMPT,
        skills=[
            ASOSkill(
                name="frontend-dev",
                description="General frontend implementation guidance",
                prompt=skill_body,
                tags=["teacher", "frontend"],
            ),
            ASOSkill(
                name="task-checklist",
                description="Validate requirements before outputting code",
                prompt=(
                    "Before giving final code, check the task requirements and key rules. "
                    "If any rule is missing, fix it in the same response. "
                    "Prefer short, runnable React/TSX snippets with minimal placeholder data."
                ),
                tags=["quality", "procedure"],
            ),
        ],
        selection_policy=(
            "Use frontend-dev first for all code generation tasks. "
            "If style requirements are strict, enable task-checklist to catch missing constraints before final answer."
        ),
    )


def _ensure_skill_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Frontend skill not found: {path}")
    return _parse_frontmatter(path.read_text(encoding="utf-8"))


def _build_client() -> LLMClient:
    actor_api_key = EFFECTIVE_ACTOR_API_KEY
    judge_api_key = JUDGE_API_KEY
    rewrite_api_key = REWRITE_API_KEY

    if not actor_api_key:
        raise RuntimeError(
            "Set actor API key: INTERN_API_KEY (preferred), TREE_LLM_API_KEY, "
            "or MINIMAX_API_KEY when using Minimax actor model"
        )

    if not judge_api_key:
        judge_api_key = actor_api_key
        logger.warning("Judge key not configured; fallback to actor key for judge")

    if not rewrite_api_key:
        rewrite_api_key = judge_api_key

    return LLMClient(
        GlobalConfig(
            llm=LLMConfig(
                api_key=actor_api_key,
                base_url=ACTOR_BASE_URL,
                model=ACTOR_MODEL,
                protocol=ACTOR_PROTOCOL,
                judge_api_key=judge_api_key,
                judge_base_url=JUDGE_BASE_URL,
                judge_model=JUDGE_MODEL,
                judge_protocol=JUDGE_PROTOCOL,
                judge_extra_body=JUDGE_EXTRA_BODY,
                rewrite_api_key=rewrite_api_key,
                rewrite_base_url=REWRITE_BASE_URL,
                rewrite_model=REWRITE_MODEL,
                rewrite_protocol=REWRITE_PROTOCOL,
                rewrite_extra_body=REWRITE_EXTRA_BODY,
            ),
            apo=APOConfig(
                beam_width=2,
                branch_factor=BRANCH_FACTOR,
                beam_rounds=MAX_ITERATIONS,
            ),
        )
    )


def _build_golds(llm: LLMClient, skill_prompt: str, train_tasks: List[DistillTask], tasks: List[DistillTask]) -> Dict[str, str]:
    gold_path = OUTPUT_DIR / "gold_standards.json"
    if gold_path.exists():
        try:
            cached = json.loads(gold_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                return {str(k): str(v) for k, v in cached.items()}
        except Exception:
            logger.warning("Failed to parse existing gold cache; regenerating")

    union = train_tasks + tasks
    outputs: Dict[str, str] = {}

    def _make(task: DistillTask) -> Tuple[str, str]:
        prompt = (
            "Use the provided frontend skill and return concise React/TSX implementation as code. "
            "Focus on meeting all key rules, and avoid placeholders where possible.\n\n"
            f"Skills:\n{skill_prompt}\n\n"
        )
        return task.id, _run_with_role(llm, task, "judge", prompt, temperature=0.0, max_tokens=4000)

    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as executor:
        futures = [executor.submit(_make, task) for task in union]
        for future in as_completed(futures):
            try:
                task_id, output = future.result()
            except Exception as exc:
                logger.warning("Failed generating gold: %s", exc)
                continue
            outputs[task_id] = output
            logger.info("Gold %s len=%d", task_id, len(output))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def _build_samples(tasks: List[DistillTask], gold_map: Dict[str, str]) -> List[SealQAExample]:
    return [task.to_example(gold_map.get(task.id, "")) for task in tasks]


def _to_distill_task(item: DistillTask | SealQAExample) -> DistillTask:
    if isinstance(item, DistillTask):
        return item
    metadata = item.metadata or {}
    task = DistillTask(
        id=item.topic,
        question=item.question,
        answer=item.answer,
        split=metadata.get("split", "val"),
        category=metadata.get("category", "frontend"),
        difficulty=metadata.get("difficulty", "easy"),
        key_rules=None,
        metadata=metadata,
    )
    return task


def _build_programs_summary(programs: List[ASOProgram]) -> list[dict[str, Any]]:
    return [
        {
            "program_id": program.program_id,
            "version": program.version,
            "score": program.score,
            "skills": [skill.name for skill in program.skills],
            "selection_policy": program.selection_policy,
            "metadata": program.metadata,
        }
        for program in programs
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Frontend-ASO distillation demo")
    parser.add_argument("--force", action="store_true", help="Overwrite previous output directory")
    args = parser.parse_args()

    if args.force and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)

    tasks = _load_tasks()
    train_tasks, val_tasks, test_tasks = _split_tasks(tasks)

    logger.info(
        "Frontend ASO distill start: total=%d train=%d val=%d test=%d actor=%s judge=%s rewrite=%s",
        len(tasks),
        len(train_tasks),
        len(val_tasks),
        len(test_tasks),
        ACTOR_MODEL,
        JUDGE_MODEL,
        REWRITE_MODEL,
    )

    skill_text = _ensure_skill_text(SKILL_PATH)
    llm = _build_client()

    # Gold generated by judge model (MiniMax).
    gold_map = _build_golds(llm, skill_text, train_tasks, val_tasks + test_tasks)

    # Attach answers in-place.
    for task in tasks:
        task.answer = gold_map.get(task.id, "")

    # Baseline program from source skill.
    baseline = _build_base_program(skill_text)

    def _run(program: ASOProgram, task: DistillTask) -> str:
        return _run_kode(program, task, WORKSPACES_DIR / program.version.replace(".", "_"))

    def _score(task: DistillTask, prediction: str) -> float:
        return _judge_score(llm, task, prediction, gold_map.get(task.id, ""))

    # Baseline metrics
    baseline_eval = _evaluate(
        baseline,
        test_tasks,
        llm,
        gold_map,
        WORKSPACES_DIR / "baseline",
        label="baseline",
    )
    baseline.score = baseline_eval["score"]

    optimizer = ASOOptimizer(
        llm,
        frontier_size=FRONTIER_SIZE,
        branch_factor=BRANCH_FACTOR,
        max_iterations=MAX_ITERATIONS,
        max_workers=ASO_WORKERS,
        auto_merge=AUTO_MERGE,
        auto_prune=AUTO_PRUNE,
        artifact_dir=OUTPUT_DIR,
    )

    def runner(program: ASOProgram, item: DistillTask | SealQAExample) -> str:
        task = _to_distill_task(item)
        return _run_kode(program, task, WORKSPACES_DIR / program.version.replace(".", "_"))

    def scorer(item: SealQAExample, prediction: str) -> float:
        metadata = item.metadata or {}
        task = DistillTask(
            id=item.topic,
            question=item.question,
            answer=item.answer,
            split=metadata.get("split", "train"),
            category=metadata.get("category", "frontend"),
            difficulty=metadata.get("difficulty", "easy"),
            key_rules=None,
            metadata=metadata,
        )
        return _judge_score(llm, task, prediction, item.answer)

    try:
        result = optimizer.run(
            baseline,
            _build_samples(train_tasks, gold_map),
            _build_samples(val_tasks, gold_map),
            runner=runner,
            scorer=scorer,
        )
        best_program = result.best_program
        history = result.history
        postprocess = result.postprocess
    except Exception:
        logger.exception("ASO run failed")
        ERROR_PATH.write_text(
            json.dumps(
                {
                    "status": "error",
                    "message": "ASO run failed",
                    "config": {
                        "actor_model": ACTOR_MODEL,
                        "actor_base_url": ACTOR_BASE_URL,
                        "judge_model": JUDGE_MODEL,
                        "judge_base_url": JUDGE_BASE_URL,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        traceback.print_exc()
        raise

    final_eval = _evaluate(
        best_program,
        test_tasks,
        llm,
        gold_map,
        WORKSPACES_DIR / "final",
        label="final",
    )
    best_program.score = final_eval["score"]

    best_program.save_to_dir(OUTPUT_DIR / "best_program", clean=True)

    summary = {
        "status": "ok",
        "config": {
            "actor_model": ACTOR_MODEL,
            "actor_base_url": ACTOR_BASE_URL,
            "judge_model": JUDGE_MODEL,
            "judge_base_url": JUDGE_BASE_URL,
            "train_size": len(train_tasks),
            "val_size": len(val_tasks),
            "test_size": len(test_tasks),
            "frontier_size": FRONTIER_SIZE,
            "branch_factor": BRANCH_FACTOR,
            "iterations": MAX_ITERATIONS,
            "eval_workers": EVAL_WORKERS,
            "aso_workers": ASO_WORKERS,
            "auto_merge": AUTO_MERGE,
            "auto_prune": AUTO_PRUNE,
        },
        "baseline": {
            "score": baseline_eval["score"],
            "rows": baseline_eval["rows"],
        },
        "best": {
            "score": final_eval["score"],
            "rows": final_eval["rows"],
        },
        "history": [
            {
                "iteration": item.iteration,
                "best_score": item.best_score,
                "frontier_scores": item.frontier_scores,
                "accepted_program_id": item.accepted_program_id,
                "actions": item.actions,
            }
            for item in history
        ],
        "postprocess": postprocess,
        "frontier": _build_programs_summary(result.frontier),
        "best_program": best_program.to_dict(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Baseline score: %.3f", baseline_eval["score"])
    logger.info("Final score: %.3f", final_eval["score"])
    logger.info("Final score delta: %.3f", final_eval["score"] - baseline_eval["score"])
    logger.info("Saved summary: %s", SUMMARY_PATH)

    print("=== frontend-distill-aso done ===")
    print(f"baseline={baseline_eval['score']:.3f}, final={final_eval['score']:.3f}, delta={final_eval['score'] - baseline_eval['score']:.3f}")
    print(f"best_program={best_program.program_id} version={best_program.version}")
    print(f"history_len={len(history)} postprocess={postprocess}")


if __name__ == "__main__":
    main()
