#!/usr/bin/env python3
"""SealQA AS(skill)O demo driven by TreeSkill + Kode.

This is the first minimal program-level evolution loop:
- Kode runs the forward pass
- TreeSkill computes textual gradients over failures
- Candidates are full programs (root prompt + skills + selection policy)
- Validation keeps the best frontier
- Optional auto-merge / auto-prune postprocessing happens inside ASOOptimizer
"""

from __future__ import annotations

import json
import logging
import os
import ast
import random
import re
import subprocess
import textwrap
import time
import traceback
import shutil
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Tuple

from dotenv import load_dotenv

load_dotenv()

from treeskill.aso_optimizer import ASOIterationResult, ASOOptimizer
from treeskill.aso_program import ASOProgram, ASOSkill
from treeskill.config import APOConfig, GlobalConfig, LLMConfig
from treeskill.llm import LLMClient
from treeskill.schema import Message
from treeskill.tasks.sealqa import SealQAExample, SealQATaskAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("demo/outputs/sealqa-aso-mini")
WORKSPACES_DIR = OUTPUT_DIR / "workspaces"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
ERROR_PATH = OUTPUT_DIR / "error.json"
STATE_PATH = OUTPUT_DIR / "run_state.json"
ITERATIONS_DIR = OUTPUT_DIR / "iterations"
FORCE_NEW = os.getenv("SEALQA_ASO_FORCE_NEW", "0") == "1"
RESUME_ENABLED = os.getenv("SEALQA_ASO_RESUME", "1") != "0"

DATA_PATH = os.getenv("SEALQA_PATH", "/root/sealqa/seal-0.parquet")
KODE_MODEL = os.getenv("KODE_ACTOR_MODEL", "MiniMax-M2.7")
KODE_ACTOR_PROTOCOL = os.getenv(
    "KODE_ACTOR_PROTOCOL",
    "anthropic" if (os.getenv("MINIMAX_API_KEY") or os.getenv("KODE_ACTOR_MODEL", "").lower().startswith("minimax")) else os.getenv("TREE_LLM_PROTOCOL", "openai"),
)
KODE_ACTOR_BASE_URL = os.getenv(
    "KODE_ACTOR_BASE_URL",
    "https://api.minimaxi.com/anthropic" if KODE_ACTOR_PROTOCOL == "anthropic" else os.getenv("TREE_LLM_BASE_URL", "https://oneapi.liuyanxing.site:8443/v1"),
)
KODE_ACTOR_API_KEY = (
    os.getenv("KODE_ACTOR_API_KEY")
    or os.getenv("MINIMAX_API_KEY")
    or os.getenv("TREE_LLM_API_KEY")
    or ""
)
EVAL_MAX_WORKERS = max(1, int(os.getenv("SEALQA_EVAL_MAX_WORKERS", "2")))
ASO_MAX_WORKERS = max(1, int(os.getenv("SEALQA_ASO_MAX_WORKERS", "2")))
MAX_ITERATIONS = max(1, int(os.getenv("SEALQA_ASO_MAX_ITERATIONS", "2")))
TRAIN_SIZE = max(1, int(os.getenv("SEALQA_TRAIN_SIZE", "4")))
VAL_SIZE = max(1, int(os.getenv("SEALQA_VAL_SIZE", "2")))
TEST_SIZE = max(1, int(os.getenv("SEALQA_TEST_SIZE", "4")))
AUTO_MERGE = os.getenv("SEALQA_ASO_AUTO_MERGE", "1") != "0"
AUTO_PRUNE = os.getenv("SEALQA_ASO_AUTO_PRUNE", "1") != "0"
ENABLE_WEB_FALLBACK = os.getenv("SEALQA_ASO_ENABLE_WEB_FALLBACK", "0") == "1"
WEB_FALLBACK_SEARCH_CMD = os.getenv("SEALQA_WEB_SEARCH_CMD", "").strip()
WEB_FALLBACK_FETCH_CMD = os.getenv("SEALQA_WEB_FETCH_CMD", "").strip()
WORKSPACE_SEARCH_CACHE_LIMIT = max(1, int(os.getenv("SEALQA_WEB_CACHE_LIMIT", "200")))
_DEFAULT_JUDGE_PROTOCOL = "anthropic" if os.getenv("MINIMAX_API_KEY") else "openai"
JUDGE_PROTOCOL = os.getenv("TREE_LLM_JUDGE_PROTOCOL", _DEFAULT_JUDGE_PROTOCOL)
JUDGE_BASE_URL = os.getenv("TREE_LLM_JUDGE_BASE_URL") or (
    "https://api.minimaxi.com/anthropic"
    if JUDGE_PROTOCOL == "anthropic"
    else os.getenv("TREE_LLM_BASE_URL", "https://oneapi.liuyanxing.site:8443/v1")
)
JUDGE_MODEL = os.getenv(
    "TREE_LLM_JUDGE_MODEL",
    "MiniMax-M2.7" if JUDGE_PROTOCOL == "anthropic" else "bailian/glm-5",
)
JUDGE_API_KEY = (
    os.getenv("TREE_LLM_JUDGE_API_KEY")
    or (os.getenv("MINIMAX_API_KEY") if JUDGE_PROTOCOL == "anthropic" else None)
    or os.getenv("TREE_LLM_API_KEY")
)
JUDGE_EXTRA_BODY = {"thinking": {"type": "disabled"}} if JUDGE_PROTOCOL == "anthropic" else None
REWRITE_PROTOCOL = os.getenv("TREE_LLM_REWRITE_PROTOCOL", JUDGE_PROTOCOL)
REWRITE_BASE_URL = os.getenv("TREE_LLM_REWRITE_BASE_URL", JUDGE_BASE_URL)
REWRITE_MODEL = os.getenv("TREE_LLM_REWRITE_MODEL", JUDGE_MODEL)
REWRITE_API_KEY = os.getenv("TREE_LLM_REWRITE_API_KEY", JUDGE_API_KEY)
REWRITE_EXTRA_BODY = {"thinking": {"type": "disabled"}} if REWRITE_PROTOCOL == "anthropic" else None

BASELINE_ROOT_PROMPT = (
    "You are a search-augmented QA assistant working in a coding-agent environment. "
    "Answer the user's question as accurately as possible. Use tools when needed. "
    "Return only the final answer."
)

_SEARCH_CACHE: list[dict[str, str]] = []


def _safe_to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, set):
        return [str(item).strip() for item in sorted(value) if str(item).strip()]
    if not isinstance(value, str):
        return [str(value).strip()] if str(value).strip() else []
    text = value.strip()
    if not text:
        return []
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    values: list[str] = []
    for part in re.split(r"[;,]", text):
        values.extend(item.strip() for item in part.split() if item.strip())
    return values


def _clean_text(value: Any, max_len: int = 1000) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return " ".join(text.split())[:max_len]


def _extract_cache_records(samples: list[SealQAExample]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for idx, sample in enumerate(samples, start=1):
        metadata = sample.metadata or {}
        urls = _safe_to_list(metadata.get("urls"))
        question_types = _safe_to_list(metadata.get("question_types"))
        golds = _safe_to_list(metadata.get("golds"))
        docs = _safe_to_list(metadata.get("12_docs"))
        docs.extend(_safe_to_list(metadata.get("20_docs")))
        docs.extend(_safe_to_list(metadata.get("30_docs")))
        snippet_parts: list[str] = []
        for key in ("search_results", "snippet", "title", "evidence", "canary"):
            snippet_parts.append(_clean_text(metadata.get(key), max_len=350))
        snippet_parts.extend(golds)
        snippet_parts.append(" / ".join([item for item in docs if item]))
        snippet_parts.append(" / ".join(question_types))
        snippet_parts.append(f"topic: {sample.topic}")
        snippet = "; ".join([part for part in snippet_parts if part])[:1000]
        text_parts = [sample.question, sample.answer, snippet]
        text = "\n".join([part for part in text_parts if part]).strip()
        if not text:
            continue
        record = {
            "id": f"sample_{idx:04d}",
            "topic": sample.topic,
            "question": sample.question,
            "keywords": ", ".join(sorted(set(question_types))),
            "title": sample.question,
            "url": urls[0] if urls else "",
            "snippet": snippet,
            "text": text,
        }
        records.append(record)
        for suffix, url in enumerate(urls[1:], start=2):
            alias_record = dict(record)
            alias_record["id"] = f"sample_{idx:04d}_{suffix}"
            alias_record["url"] = url
            records.append(alias_record)
    return records[: WORKSPACE_SEARCH_CACHE_LIMIT * 2]


def _build_search_script() -> str:
    return textwrap.dedent(
        """
        #!/usr/bin/env python3
        import argparse
        import json
        import os
        import re
        import shlex
        import subprocess
        from pathlib import Path

        CACHE_PATH = Path(__file__).resolve().parent / "search_cache.json"

        DEFAULT_TOP_K = 3

        def tokenize(text: str):
            if not text:
                return set()
            return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 1}


        def _parse_keywords(raw: str):
            raw = (raw or "").strip()
            if not raw:
                return []
            return [item.strip() for item in raw.split(",") if item.strip()]


        def score(item, query: str):
            q_tokens = tokenize(query)
            if not q_tokens:
                return 0
            corpus = " ".join(
                [
                    item.get("question", ""),
                    item.get("title", ""),
                    item.get("snippet", ""),
                    item.get("text", ""),
                    " ".join(_parse_keywords(item.get("keywords", ""))),
                ]
            )
            if not corpus:
                return 0
            item_tokens = tokenize(corpus)
            if not item_tokens:
                return 0
            return len(q_tokens & item_tokens)


        def _parse_external_json(text: str):
            text = text.strip()
            if not text:
                return []
            try:
                payload = json.loads(text)
            except Exception:
                return []
            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list):
                return []
            out = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "")).strip()
                if not url:
                    continue
                out.append(
                    {
                        "id": str(item.get("id", url)),
                        "topic": str(item.get("topic", "")),
                        "title": str(item.get("title", "")),
                        "url": url,
                        "snippet": str(item.get("snippet", "")),
                        "source": "web_fallback",
                    }
                )
            return out


        def _parse_external_lines(text: str):
            out = []
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("http"):
                    continue
                out.append(
                    {
                        "id": line[:64],
                        "topic": "",
                        "title": "",
                        "url": line,
                        "snippet": "",
                        "source": "web_fallback",
                    }
                )
            return out


        def _fallback_search_command(query: str, top_k: int):
            cmd = os.getenv("SEALQA_WEB_SEARCH_CMD", "").strip()
            if not cmd:
                return []
            if "{query}" in cmd:
                cmd = cmd.format(query=shlex.quote(query), top_k=top_k)
            elif "{top_k}" in cmd:
                cmd = cmd.format(top_k=top_k, query=shlex.quote(query))
            else:
                cmd = f"{cmd} {shlex.quote(query)} {int(top_k)}"
            try:
                completed = subprocess.run(
                    cmd,
                    shell=True,
                    text=True,
                    check=False,
                    capture_output=True,
                    timeout=int(os.getenv("SEALQA_WEB_SEARCH_TIMEOUT", "10")),
                )
            except Exception:
                return []
            if completed.returncode != 0:
                return []
            parsed = _parse_external_json(completed.stdout)
            if parsed:
                return parsed[: top_k]
            return _parse_external_lines(completed.stdout)[: top_k]


        def _load_cache():
            if not CACHE_PATH.exists():
                return []
            try:
                return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                return []


        def _search_cached(cache, query: str, top_k: int):
            scored = []
            for item in cache:
                scored.append((score(item, query), item))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            results = []
            for item_score, item in scored[:top_k]:
                if item_score <= 0 and not os.getenv("SEALQA_ASO_ENABLE_WEB_FALLBACK", "0") == "1":
                    break
                results.append(
                    {
                        "id": item.get("id", ""),
                        "topic": item.get("topic", ""),
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "cache",
                    }
                )
            if results:
                return results
            if os.getenv("SEALQA_ASO_ENABLE_WEB_FALLBACK", "0") != "1":
                return []
            return _fallback_search_command(query, top_k)


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--query", required=True)
            parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
            args = parser.parse_args()
            cache = _load_cache()
            results = _search_cached(cache, args.query, args.top_k)
            print(json.dumps(results, ensure_ascii=False, indent=2))


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"


def _build_fetch_script() -> str:
    return textwrap.dedent(
        """
        #!/usr/bin/env python3
        import argparse
        import json
        import os
        import shlex
        import subprocess
        from pathlib import Path

        CACHE_PATH = Path(__file__).resolve().parent / "search_cache.json"

        def _load_cache():
            if not CACHE_PATH.exists():
                return []
            try:
                return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                return []


        def _fallback_fetch(url: str):
            cmd = os.getenv("SEALQA_WEB_FETCH_CMD", "").strip()
            if not cmd:
                return ""
            if "{url}" in cmd:
                cmd = cmd.format(url=shlex.quote(url))
            else:
                cmd = f"{cmd} {shlex.quote(url)}"
            try:
                completed = subprocess.run(
                    cmd,
                    shell=True,
                    text=True,
                    check=False,
                    capture_output=True,
                    timeout=int(os.getenv("SEALQA_WEB_FETCH_TIMEOUT", "10")),
                )
            except Exception:
                return ""
            if completed.returncode != 0:
                return ""
            return completed.stdout.strip()


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--url", required=True)
            args = parser.parse_args()

            refs = _load_cache()
            for item in refs:
                if item.get("url", "") == args.url:
                    print(item.get("text", ""))
                    return
            if os.getenv("SEALQA_ASO_ENABLE_WEB_FALLBACK", "0") != "1":
                return
            fallback = _fallback_fetch(args.url)
            if fallback:
                print(fallback)


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"


def _prepare_kode_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    cache_rows = _SEARCH_CACHE[:WORKSPACE_SEARCH_CACHE_LIMIT * 2]
    (workspace / "search_web.py").write_text(_build_search_script(), encoding="utf-8")
    (workspace / "search_web.py").chmod(0o755)
    (workspace / "fetch_url.py").write_text(_build_fetch_script(), encoding="utf-8")
    (workspace / "fetch_url.py").chmod(0o755)
    (workspace / "search_cache.json").write_text(
        json.dumps(cache_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _judge_answer(llm: LLMClient, sample: SealQAExample, prediction: str) -> float:
    prompt = (
        f"Question: {sample.question}\n"
        f"Ground truth answer: {sample.answer}\n"
        f"Predicted answer: {prediction}\n\n"
        "Is the predicted answer correct? Consider semantic equivalence, not exact wording. "
        "Return ONLY 1 or 0."
    )
    try:
        response = llm.generate(
            [
                Message(role="system", content="You are a strict answer judge. Return only 1 or 0."),
                Message(role="user", content=prompt),
            ],
            role="judge",
            max_tokens=64,
            temperature=0.0,
        )
        raw = response.content.strip()
        match = re.search(r"\b([01])\b", raw)
        return 1.0 if match and match.group(1) == "1" else 0.0
    except Exception as exc:
        logger.warning("judge error: %s", exc)
        return 0.0


def _run_kode(program: ASOProgram, sample: SealQAExample, label: str) -> str:
    workspace = WORKSPACES_DIR / f"{label}_{abs(hash(sample.question)) % 10_000_000}"
    if workspace.exists():
        for item in workspace.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    _prepare_kode_workspace(workspace)
    (workspace / "AGENTS.md").write_text(program.render_agents_markdown(), encoding="utf-8")

    env = os.environ.copy()
    if KODE_ACTOR_API_KEY:
        env["TREE_LLM_API_KEY"] = KODE_ACTOR_API_KEY
    env["TREE_LLM_BASE_URL"] = KODE_ACTOR_BASE_URL
    env["TREE_LLM_PROTOCOL"] = KODE_ACTOR_PROTOCOL
    env["TREE_LLM_MODEL"] = KODE_MODEL

    command = [
        "kode",
        "-p",
        sample.question,
        "--model",
        KODE_MODEL,
        "--cwd",
        str(workspace),
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
    ]
    try:
        proc = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ""

    if not proc.stdout.strip():
        logger.warning("Kode returned empty stdout: %s", proc.stderr.strip())
        return ""

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        logger.warning("Invalid Kode JSON: %s", proc.stdout[:200])
        return ""
    return str(data.get("result", "")).strip()


def _run_sample(
    program: ASOProgram,
    sample: SealQAExample,
    llm: LLMClient,
    *,
    label: str,
    idx: int,
    total: int,
) -> Tuple[int, Dict[str, object]]:
    t0 = time.time()
    prediction = _run_kode(program, sample, f"{label}_{idx}")
    kode_seconds = time.time() - t0

    t1 = time.time()
    score = _judge_answer(llm, sample, prediction)
    judge_seconds = time.time() - t1
    total_seconds = time.time() - t0

    logger.info(
        "  [%s] %d/%d topic=%s score=%.0f kode=%.2fs judge=%.2fs total=%.2fs",
        label,
        idx,
        total,
        sample.topic,
        score,
        kode_seconds,
        judge_seconds,
        total_seconds,
    )
    return idx, {
        "question": sample.question,
        "answer": sample.answer,
        "topic": sample.topic,
        "prediction": prediction,
        "score": score,
        "kode_seconds": round(kode_seconds, 2),
        "judge_seconds": round(judge_seconds, 2),
        "total_seconds": round(total_seconds, 2),
    }


def _evaluate_program(
    program: ASOProgram,
    data: list[SealQAExample],
    llm: LLMClient,
    *,
    label: str,
) -> Dict[str, object]:
    rows: list[Dict[str, object]] = []
    if not data:
        return {"accuracy": 0.0, "rows": rows}

    correct = 0.0
    total_kode_seconds = 0.0
    total_judge_seconds = 0.0
    total_seconds = 0.0

    with ThreadPoolExecutor(max_workers=EVAL_MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                _run_sample,
                program,
                sample,
                llm,
                label=label,
                idx=idx,
                total=len(data),
            )
            for idx, sample in enumerate(data, start=1)
        ]
        indexed_rows: Dict[int, Dict[str, object]] = {}
        for future in as_completed(futures):
            idx, row = future.result()
            indexed_rows[idx] = row
            correct += float(row["score"])
            total_kode_seconds += float(row["kode_seconds"])
            total_judge_seconds += float(row["judge_seconds"])
            total_seconds += float(row["total_seconds"])
        rows = [indexed_rows[idx] for idx in sorted(indexed_rows)]

    logger.info(
        "  [%s] done acc=%.1f%% avg_kode=%.2fs avg_judge=%.2fs avg_total=%.2fs workers=%d",
        label,
        100 * correct / len(data),
        total_kode_seconds / len(data),
        total_judge_seconds / len(data),
        total_seconds / len(data),
        EVAL_MAX_WORKERS,
    )
    return {
        "accuracy": correct / len(data),
        "rows": rows,
    }


def _load_state() -> Dict[str, Any] | None:
    if not STATE_PATH.exists():
        return None
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _aso_program_from_payload(payload: Dict[str, Any]) -> ASOProgram | None:
    if not isinstance(payload, dict):
        return None
    skills: list[ASOSkill] = []
    for item in payload.get("skills", []):
        if not isinstance(item, dict):
            continue
        if not item.get("name"):
            continue
        skills.append(
            ASOSkill(
                name=str(item.get("name", "")),
                description=str(item.get("description", "")),
                prompt=str(item.get("prompt", "")),
                version=str(item.get("version", "v1.0")),
                tags=list(item.get("tags", [])),
            )
        )
    return ASOProgram(
        root_prompt=str(payload.get("root_prompt", BASELINE_ROOT_PROMPT)),
        skills=skills,
        selection_policy=str(payload.get("selection_policy", "")),
        version=str(payload.get("version", "v1.0")),
        score=payload.get("score"),
        parent_id=payload.get("parent_id"),
        program_id=str(payload.get("program_id", "")),
        metadata=dict(payload.get("metadata", {})),
    )


def _load_program_from_snapshot(snapshot_dir: Path) -> ASOProgram | None:
    program_json = snapshot_dir / "program.json"
    if not program_json.exists():
        return None
    try:
        payload = json.loads(program_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return _aso_program_from_payload(payload)


def _load_frontier_from_snapshot(iteration_dir: Path) -> list[ASOProgram]:
    frontier_file = iteration_dir / "frontier.json"
    if not frontier_file.exists():
        return []
    try:
        payload = json.loads(frontier_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    frontier_items = payload.get("frontier", [])
    if not isinstance(frontier_items, list):
        return []
    frontier: list[ASOProgram] = []
    for item in frontier_items:
        if not isinstance(item, dict):
            continue
        loaded = _aso_program_from_payload(item)
        if loaded is not None:
            frontier.append(loaded)
    return frontier


def _load_history_from_summary(summary_path: Path) -> list[dict]:
    if not summary_path.exists():
        return []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    raw_history = payload.get("history")
    if not isinstance(raw_history, list):
        return []
    return [item for item in raw_history if isinstance(item, dict)]


def _as_iteration_results(records: list[dict]) -> list[ASOIterationResult]:
    out: list[ASOIterationResult] = []
    for item in records:
        try:
            iteration = int(item.get("iteration"))
            best_score = float(item.get("best_score", 0.0))
            frontier_scores = item.get("frontier_scores", [])
            if not isinstance(frontier_scores, list):
                frontier_scores = []
            frontier_scores = [float(value) for value in frontier_scores if isinstance(value, (int, float))]
            accepted_program_id = item.get("accepted_program_id")
            if not isinstance(accepted_program_id, str):
                accepted_program_id = ""
            actions = item.get("actions", [])
            if not isinstance(actions, list):
                actions = []
            out.append(
                ASOIterationResult(
                    iteration=iteration,
                    best_score=best_score,
                    frontier_scores=frontier_scores,
                    accepted_program_id=accepted_program_id,
                    actions=actions,
                )
            )
        except Exception:
            continue
    return out


def _latest_iteration_dir() -> tuple[int, Path | None]:
    if not ITERATIONS_DIR.exists():
        return 0, None
    latest_n = 0
    latest_path = None
    for child in ITERATIONS_DIR.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("iteration_"):
            continue
        try:
            n = int(child.name.split("_", 1)[1])
        except Exception:
            continue
        if n > latest_n:
            latest_n = n
            latest_path = child
    return latest_n, latest_path


def _write_state(
    stage: str,
    *,
    status: str = "running",
    iteration: int | None = None,
    best_score: float | None = None,
    frontier_size: int | None = None,
    postprocess: list[dict] | None = None,
    error: str | None = None,
) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "status": status,
        "iteration": iteration,
        "best_score": best_score,
        "frontier_size": frontier_size,
        "postprocess": postprocess,
        "error": error,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    assert JUDGE_API_KEY, "Set TREE_LLM_API_KEY or TREE_LLM_JUDGE_API_KEY"
    baseline_program = None
    baseline_eval = None
    result = None
    final_eval = None
    resumed_from_iteration = 0
    start = time.time()

    try:
        previous_state = _load_state() if RESUME_ENABLED else None
        if FORCE_NEW or (OUTPUT_DIR.exists() and not RESUME_ENABLED):
            import shutil
            shutil.rmtree(OUTPUT_DIR)
        elif previous_state and previous_state.get("status") == "done" and str(previous_state.get("stage")) == "done":
            logger.info("Existing run has status done; pass SEALQA_ASO_FORCE_NEW=1 to override")
            _write_state(
                "done",
                status="done",
                iteration=int(previous_state.get("iteration", 0) or 0),
                best_score=previous_state.get("best_score") if isinstance(previous_state.get("best_score"), (int, float)) else None,
                frontier_size=previous_state.get("frontier_size") if isinstance(previous_state.get("frontier_size"), int) else None,
                postprocess=previous_state.get("postprocess") if isinstance(previous_state.get("postprocess"), list) else None,
            )
            return

        resumed_from_iteration = 0
        resume_frontier: list[ASOProgram] = []
        resume_history: list[ASOIterationResult] = []
        resume_start_iteration = 1
        if OUTPUT_DIR.exists():
            latest_iteration, latest_iteration_dir = _latest_iteration_dir()
            resumed_from_iteration = latest_iteration
            if latest_iteration_dir:
                checkpoint = _load_program_from_snapshot(latest_iteration_dir / "best_program")
                if checkpoint is not None:
                    baseline_program = checkpoint
                    logger.info("Resume checkpoint found at iteration_%d", latest_iteration)
                    logger.info("Resume checkpoint score=%s", checkpoint.score)
                    resume_frontier = _load_frontier_from_snapshot(latest_iteration_dir)
                    if not resume_frontier:
                        resume_frontier = [checkpoint]
                    if previous_state is not None and str(previous_state.get("stage")) == "failed":
                        resume_start_iteration = resumed_from_iteration + 1
                        resume_history = _as_iteration_results(_load_history_from_summary(SUMMARY_PATH))
                    _write_state(
                        "resuming",
                        status="running",
                        iteration=latest_iteration,
                        best_score=previous_state.get("best_score") if isinstance(previous_state, dict) else None,
                        frontier_size=previous_state.get("frontier_size") if isinstance(previous_state, dict) else None,
                        postprocess=previous_state.get("postprocess") if isinstance(previous_state, dict) and isinstance(previous_state.get("postprocess"), list) else None,
                    )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
        _write_state("load_dataset", status="running")

        adapter = SealQATaskAdapter(DATA_PATH, train_ratio=0.18, val_ratio=0.12, seed=42)
        train, val, test = adapter.split()

        # Keep the demo small enough to iterate quickly.
        train = train[: min(TRAIN_SIZE, len(train))]
        val = val[: min(VAL_SIZE, len(val))]
        test = test[: min(TEST_SIZE, len(test))]
        global _SEARCH_CACHE
        _SEARCH_CACHE = _extract_cache_records(train + val + test)

        logger.info("SealQA ASO demo")
        logger.info("  dataset=%s", DATA_PATH)
        logger.info("  actor=%s", KODE_MODEL)
        logger.info("  judge=%s (%s)", JUDGE_MODEL, JUDGE_PROTOCOL)
        logger.info("  concurrency: eval=%d aso=%d", EVAL_MAX_WORKERS, ASO_MAX_WORKERS)
        logger.info("  postprocess: auto_merge=%s auto_prune=%s", AUTO_MERGE, AUTO_PRUNE)
        logger.info("  split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))
        logger.info(
            "  search cache: rows=%d fallback=%s cmd_search=%s cmd_fetch=%s cache_limit=%d",
            len(_SEARCH_CACHE),
            ENABLE_WEB_FALLBACK,
            bool(WEB_FALLBACK_SEARCH_CMD),
            bool(WEB_FALLBACK_FETCH_CMD),
            WORKSPACE_SEARCH_CACHE_LIMIT,
        )
        if baseline_program is None:
            baseline_program = ASOProgram(
                root_prompt=BASELINE_ROOT_PROMPT,
                skills=[
                    ASOSkill(
                        name="answer-format",
                        description="Keep final answers concise and avoid extra explanation.",
                        prompt=(
                            "When you have enough evidence, produce a short final answer. "
                            "Do not include chain-of-thought. Prefer direct factual wording."
                        ),
                        tags=["baseline"],
                    ),
                    ASOSkill(
                        name="search_web_lookup",
                        description="Use the local search helper first, then fallback helper only when needed.",
                        prompt=(
                            "For factual questions, always run:\n"
                            "python search_web.py --query \"<full user question>\" --top-k 3\n"
                            "Then run `python fetch_url.py --url \"<best url>\"` and answer from fetched evidence.\n"
                            "The search tool should use cached references first; fallback tools are optional and should only be used when cache returns no useful hit."
                        ),
                        tags=["retrieval", "baseline"],
                    ),
                    ASOSkill(
                        name="verified_fact_lookup",
                        description="If fetched evidence is weak, abstain instead of guessing.",
                        prompt=(
                            "Before finalizing, require evidence from search/fetch output. "
                            "If no sufficient evidence exists, answer with uncertainty instead of fabrication."
                        ),
                        tags=["verification", "baseline"],
                    )
                ],
                selection_policy=(
                    "Use `search_web_lookup` for factual and knowledge-heavy questions first. "
                    "If retrieval fails from local cache, use fallback only when enabled by `SEALQA_ASO_ENABLE_WEB_FALLBACK=1`. "
                    "Answer only after checking evidence with `fetch_url.py`; avoid unsupported extrapolation."
                ),
            )
            _write_state(
                "baseline_started",
                status="running",
                iteration=0,
                frontier_size=0,
            )
        else:
            _write_state(
                "baseline_skipped",
                status="running",
                iteration=resumed_from_iteration,
                frontier_size=1,
            )

        llm = LLMClient(
            GlobalConfig(
                llm=LLMConfig(
                    api_key=os.getenv("TREE_LLM_API_KEY", JUDGE_API_KEY),
                    base_url=os.getenv("TREE_LLM_BASE_URL", JUDGE_BASE_URL),
                    model=os.getenv("TREE_LLM_MODEL", JUDGE_MODEL),
                    protocol=os.getenv("TREE_LLM_PROTOCOL", JUDGE_PROTOCOL),
                    judge_api_key=JUDGE_API_KEY,
                    judge_base_url=JUDGE_BASE_URL,
                    judge_model=JUDGE_MODEL,
                    judge_protocol=JUDGE_PROTOCOL,
                    judge_extra_body=JUDGE_EXTRA_BODY,
                    rewrite_api_key=REWRITE_API_KEY,
                    rewrite_base_url=REWRITE_BASE_URL,
                    rewrite_model=REWRITE_MODEL,
                    rewrite_protocol=REWRITE_PROTOCOL,
                    rewrite_extra_body=REWRITE_EXTRA_BODY,
                ),
                apo=APOConfig(beam_width=2, branch_factor=2, beam_rounds=1),
            )
        )
        if baseline_program.score is None or not RESUME_ENABLED:
            baseline_eval = _evaluate_program(baseline_program, test, llm, label="baseline")
            baseline_program.score = float(baseline_eval["accuracy"])
            _write_state(
                "baseline_complete",
                status="running",
                iteration=0,
                best_score=baseline_eval["accuracy"],
                frontier_size=1,
            )
        else:
            baseline_eval = {"accuracy": float(baseline_program.score)}
            _write_state(
                "baseline_complete",
                status="running",
                iteration=0,
                best_score=baseline_program.score,
                frontier_size=1,
            )

        optimizer = ASOOptimizer(
            llm,
            frontier_size=2,
            branch_factor=2,
            max_iterations=MAX_ITERATIONS,
            max_workers=ASO_MAX_WORKERS,
            auto_merge=AUTO_MERGE,
            auto_prune=AUTO_PRUNE,
            artifact_dir=ITERATIONS_DIR,
        )

        def runner(program: ASOProgram, sample: SealQAExample) -> str:
            return _run_kode(program, sample, label=program.version.replace(".", "_"))

        def scorer(sample: SealQAExample, prediction: str) -> float:
            return _judge_answer(llm, sample, prediction)

        skip_aso = (
            baseline_program is not None
            and baseline_program.score is not None
            and resumed_from_iteration >= MAX_ITERATIONS
            and previous_state is not None
            and str(previous_state.get("stage")) != "failed"
        )

        if skip_aso:
            final_program = baseline_program
            _write_state(
                "aso_complete",
                status="running",
                iteration=resumed_from_iteration,
                best_score=float(final_program.score or 0.0),
                frontier_size=max(1, previous_state.get("frontier_size") if isinstance(previous_state, dict) and isinstance(previous_state.get("frontier_size"), int) else 1),
            )
            result = None
        else:
            result = optimizer.run(
                baseline_program,
                train,
                val,
                runner=runner,
                scorer=scorer,
                start_iteration=resume_start_iteration,
                initial_frontier=resume_frontier if resume_frontier else None,
                initial_best_program=baseline_program,
                initial_history=resume_history,
                initial_baseline_score=baseline_program.score if isinstance(baseline_program.score, (int, float)) else None,
            )
            final_program = result.best_program
            _write_state(
                "aso_complete",
                status="running",
                iteration=len(result.history),
                best_score=result.final_score,
                frontier_size=len(result.frontier),
                postprocess=result.postprocess,
            )

        final_eval = _evaluate_program(final_program, test, llm, label="final")
        snapshot_dir = OUTPUT_DIR / "best_program"
        final_program.save_to_dir(snapshot_dir, clean=True)
        final_iteration = len(result.history) if result else resumed_from_iteration
        final_frontier = len(result.frontier) if result else max(
            1,
            previous_state.get("frontier_size")
            if isinstance(previous_state, dict) and isinstance(previous_state.get("frontier_size"), int)
            else 1,
        )
        final_postprocess = result.postprocess if result else []
        _write_state(
            "final_eval_complete",
            status="running",
            iteration=final_iteration,
            best_score=final_program.score,
            frontier_size=final_frontier,
            postprocess=final_postprocess,
        )

        summary = {
            "status": "ok",
            "config": {
                "dataset_path": DATA_PATH,
                "kode_actor_model": KODE_MODEL,
                "judge_model": JUDGE_MODEL,
                "judge_protocol": JUDGE_PROTOCOL,
                "judge_base_url": JUDGE_BASE_URL,
                "eval_max_workers": EVAL_MAX_WORKERS,
                "aso_max_workers": ASO_MAX_WORKERS,
                "train_size": len(train),
                "val_size": len(val),
                "test_size": len(test),
                "search_cache_limit": WORKSPACE_SEARCH_CACHE_LIMIT,
                "web_fallback_enabled": ENABLE_WEB_FALLBACK,
                "web_search_cmd_configured": bool(WEB_FALLBACK_SEARCH_CMD),
                "web_fetch_cmd_configured": bool(WEB_FALLBACK_FETCH_CMD),
            },
            "baseline_accuracy": baseline_eval["accuracy"],
            "final_accuracy": final_eval["accuracy"],
            "best_program": final_program.to_dict(),
            "frontier": [program.to_dict() for program in (result.frontier if result else [final_program])],
            "history": [
                {
                    "iteration": item.iteration,
                    "best_score": item.best_score,
                    "frontier_scores": item.frontier_scores,
                    "accepted_program_id": item.accepted_program_id,
                    "actions": item.actions,
                }
                for item in result.history
            ] if result else [],
            "postprocess": result.postprocess if result else [],
            "elapsed_minutes": round((time.time() - start) / 60, 2),
        }
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("Baseline acc: %.1f%%", 100 * baseline_eval["accuracy"])
        logger.info("Final acc: %.1f%%", 100 * final_eval["accuracy"])
        logger.info("Saved best program to %s", snapshot_dir)
        logger.info("Summary saved to %s", SUMMARY_PATH)
        _write_state(
            "done",
            status="done",
            iteration=(len(result.history) if result else resumed_from_iteration),
            best_score=final_eval["accuracy"],
            frontier_size=(len(result.frontier) if result else max(1, previous_state.get("frontier_size") if isinstance(previous_state, dict) and isinstance(previous_state.get("frontier_size"), int) else 1)),
            postprocess=result.postprocess if result else [],
        )
    except Exception as exc:
        _write_state(
            "failed",
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            postprocess=result.postprocess if result else [],
        )
        error_payload = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_minutes": round((time.time() - start) / 60, 2),
            "baseline_program": baseline_program.to_dict() if baseline_program else None,
            "baseline_accuracy": baseline_eval["accuracy"] if baseline_eval else None,
            "partial_best_program": result.best_program.to_dict() if result else None,
            "partial_history": [
                {
                    "iteration": item.iteration,
                    "best_score": item.best_score,
                    "frontier_scores": item.frontier_scores,
                    "accepted_program_id": item.accepted_program_id,
                    "actions": item.actions,
                }
                for item in result.history
            ] if result else [],
            "partial_postprocess": result.postprocess if result else [],
            "final_accuracy": final_eval["accuracy"] if final_eval else None,
        }
        ERROR_PATH.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.exception("SealQA ASO demo failed")
        raise


if __name__ == "__main__":
    random.seed(42)
    main()
