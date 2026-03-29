#!/usr/bin/env python3
"""
RepoOps Demo — TreeSkill + Kode for split / prune / compose experiments.

This demo is designed for the AS(skill)O direction:
1. Start from one weak root skill
2. Collect hard-verified failures through Kode
3. Auto-split into specialized child skills
4. Prune leaves that are unused or dominated on validation
5. Compose closely-related leaves back into a coarser skill

The task set is intentionally small, local, and fully hard-verifiable.

Usage:
    conda activate pr
    export TREE_LLM_API_KEY=...
    python demo/demo_repoops_tree.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from treeskill.config import GlobalConfig
from treeskill.llm import LLMClient
from treeskill.optimizer import APOEngine
from treeskill.schema import Feedback, Message, Skill, Trace
from treeskill.skill import save as save_skill
from treeskill.skill_tree import SkillNode, SkillTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("demo/outputs/repoops-tree-v3")
WORKSPACES_DIR = OUTPUT_DIR / "workspaces"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
NUM_ROUNDS = 2
KODE_MODEL = os.getenv("KODE_ACTOR_MODEL", "SilliconCloud/Qwen3.5-4B")

ROOT_NAME = "repoops-assistant"
BAD_SKILL = (
    "You are an assistant working under strict time pressure. "
    "Prefer the smallest visible change, avoid reading too many files, "
    "avoid running tests unless explicitly necessary, and do not refactor existing code."
)
ROOT_TARGET = (
    "Improve task completion in a real coding-agent loop. "
    "The tasks span several different styles: "
    "CSV processing, JSON processing, Python quality tasks, and small repository maintenance tasks. "
    "Always create or edit the requested files, run commands to verify the result, and prefer precise output. "
    "If one prompt cannot reliably handle all task families, split into specialized child skills."
)


@dataclass
class TaskSpec:
    id: str
    split: str
    domain: str
    family: str
    task: str
    verify: Callable[[Path, dict], float]
    setup: Optional[Callable[[Path], None]] = None


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run(command: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _stdout(path: Path, command: List[str]) -> str:
    proc = _run(command, path)
    return (proc.stdout + proc.stderr).strip()


def _contains_all(text: str, expected: List[str]) -> bool:
    return all(item in text for item in expected)


def _setup_csv_average_bug(workdir: Path) -> None:
    _write(
        workdir / "scores.csv",
        "student,math,english\nAlice,85,90\nBob,72,88\nCarol,91,76\nDave,68,95\n",
    )
    _write(
        workdir / "stats.py",
        "import csv\n\n"
        "with open('scores.csv', newline='', encoding='utf-8') as f:\n"
        "    rows = list(csv.DictReader(f))\n\n"
        "math_avg = sum(int(r['math']) for r in rows) / len(rows)\n"
        "english_avg = sum(int(r['math']) for r in rows) / len(rows)\n"
        "print(f'math_avg={math_avg:.1f} english_avg={english_avg:.1f}')\n",
    )
    _write(
        workdir / "test_stats.py",
        "import subprocess, sys\n\n"
        "proc = subprocess.run([sys.executable, 'stats.py'], capture_output=True, text=True, check=True)\n"
        "output = proc.stdout.strip()\n"
        "assert output == 'math_avg=79.0 english_avg=87.2'\n",
    )


def _setup_csv_filter_bug(workdir: Path) -> None:
    _write(
        workdir / "inventory.csv",
        "item,quantity,price\nWidget,50,9.99\nGadget,5,29.99\nThingamajig,2,99.99\nDoohickey,100,4.99\n",
    )
    _write(
        workdir / "filter.py",
        "import csv\n\n"
        "with open('inventory.csv', newline='', encoding='utf-8') as f:\n"
        "    rows = list(csv.DictReader(f))\n\n"
        "for row in rows:\n"
        "    if int(row['quantity']) <= 10:\n"
        "        print(f\"LOW_STOCK: {row['item']} ({row['quantity']})\")\n",
    )
    _write(
        workdir / "test_filter.py",
        "import subprocess, sys\n\n"
        "proc = subprocess.run([sys.executable, 'filter.py'], capture_output=True, text=True, check=True)\n"
        "lines = sorted(line.strip() for line in proc.stdout.splitlines() if line.strip())\n"
        "assert lines == ['LOW_STOCK: Gadget (5)', 'LOW_STOCK: Thingamajig (2)']\n",
    )


def _setup_csv_city_bug(workdir: Path) -> None:
    _write(
        workdir / "orders.csv",
        "city,count\nBeijing,3\nShanghai,5\nBeijing,2\nShenzhen,4\n",
    )
    _write(
        workdir / "summarize.py",
        "import csv\n\n"
        "totals = {}\n"
        "with open('orders.csv', newline='', encoding='utf-8') as f:\n"
        "    for row in csv.DictReader(f):\n"
        "        totals[row['city']] = int(row['count'])\n"
        "print(' '.join(f'{city}={total}' for city, total in totals.items()))\n",
    )
    _write(
        workdir / "test_summarize.py",
        "import subprocess, sys\n\n"
        "proc = subprocess.run([sys.executable, 'summarize.py'], capture_output=True, text=True, check=True)\n"
        "assert proc.stdout.strip() == 'Beijing=5 Shanghai=5 Shenzhen=4'\n",
    )


def _setup_csv_top_bug(workdir: Path) -> None:
    _write(
        workdir / "ranking.csv",
        "name,score\nAlice,88\nBob,93\nCarol,91\n",
    )
    _write(
        workdir / "top.py",
        "import csv\n\n"
        "with open('ranking.csv', newline='', encoding='utf-8') as f:\n"
        "    rows = list(csv.DictReader(f))\n"
        "top_row = rows[0]\n"
        "print(f\"TOP={top_row['name']} SCORE={top_row['score']}\")\n",
    )
    _write(
        workdir / "test_top.py",
        "import subprocess, sys\n\n"
        "proc = subprocess.run([sys.executable, 'top.py'], capture_output=True, text=True, check=True)\n"
        "assert proc.stdout.strip() == 'TOP=Bob SCORE=93'\n",
    )


def _setup_json_merge_bug(workdir: Path) -> None:
    _write(workdir / "a.json", '{"users":[{"name":"Alice","age":30}]}\n')
    _write(workdir / "b.json", '{"users":[{"name":"Bob","age":25}]}\n')
    _write(
        workdir / "merge.py",
        "import json\n\n"
        "with open('a.json', encoding='utf-8') as fa, open('b.json', encoding='utf-8') as fb:\n"
        "    a = json.load(fa)\n"
        "    b = json.load(fb)\n\n"
        "merged = {'users': b['users'] + a['users']}\n"
        "with open('merged.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump(merged, f)\n"
        "print('MERGED=' + ','.join(user['name'] for user in merged['users']))\n",
    )
    _write(
        workdir / "test_merge.py",
        "import json, subprocess, sys\n\n"
        "proc = subprocess.run([sys.executable, 'merge.py'], capture_output=True, text=True, check=True)\n"
        "assert proc.stdout.strip() == 'MERGED=Alice,Bob'\n"
        "data = json.load(open('merged.json', encoding='utf-8'))\n"
        "assert [u['name'] for u in data['users']] == ['Alice', 'Bob']\n",
    )


def _setup_json_active_bug(workdir: Path) -> None:
    _write(
        workdir / "users.json",
        '[{"name":"Alice","active":true},{"name":"Bob","active":false},{"name":"Carol","active":true}]\n',
    )
    _write(
        workdir / "active.py",
        "import json\n\n"
        "users = json.load(open('users.json', encoding='utf-8'))\n"
        "active = [user['name'] for user in users if user['active'] == 'true']\n"
        "print('ACTIVE=' + ','.join(active))\n",
    )
    _write(
        workdir / "test_active.py",
        "import subprocess, sys\n\n"
        "proc = subprocess.run([sys.executable, 'active.py'], capture_output=True, text=True, check=True)\n"
        "assert proc.stdout.strip() == 'ACTIVE=Alice,Carol'\n",
    )


def _setup_json_status_bug(workdir: Path) -> None:
    _write(
        workdir / "tasks.json",
        '[{"id":1,"status":"todo"},{"id":2,"status":"done"},{"id":3,"status":"done"}]\n',
    )
    _write(
        workdir / "status_count.py",
        "import json\n\n"
        "items = json.load(open('tasks.json', encoding='utf-8'))\n"
        "done = len([item for item in items if item['status'] == 'done'])\n"
        "todo = len(items) - done - 1\n"
        "print(f'done={done} todo={todo}')\n",
    )
    _write(
        workdir / "test_status_count.py",
        "import subprocess, sys\n\n"
        "proc = subprocess.run([sys.executable, 'status_count.py'], capture_output=True, text=True, check=True)\n"
        "assert proc.stdout.strip() == 'done=2 todo=1'\n",
    )


def _setup_json_sum_bug(workdir: Path) -> None:
    _write(workdir / "metrics.json", '{"scores":[4,7,9]}\n')
    _write(
        workdir / "sum_scores.py",
        "import json\n\n"
        "data = json.load(open('metrics.json', encoding='utf-8'))\n"
        "print('SUM=' + str(max(data['scores'])))\n",
    )
    _write(
        workdir / "test_sum_scores.py",
        "import subprocess, sys\n\n"
        "proc = subprocess.run([sys.executable, 'sum_scores.py'], capture_output=True, text=True, check=True)\n"
        "assert proc.stdout.strip() == 'SUM=20'\n",
    )


def _setup_decorator_bug(workdir: Path) -> None:
    _write(
        workdir / "decorators.py",
        "def log_calls(func):\n"
        "    def wrapper(*args, **kwargs):\n"
        "        result = func(*args, **kwargs)\n"
        "        print(f'CALLING: {func.__name__}')\n"
        "        print(f'DONE: {func.__name__}')\n"
        "        return result\n"
        "    return wrapper\n\n"
        "@log_calls\n"
        "def greet(name):\n"
        "    return f'Hello {name}'\n",
    )
    _write(
        workdir / "test_decorators.py",
        "import io, contextlib\n"
        "from decorators import greet\n\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    result = greet('World')\n"
        "lines = [line.strip() for line in buf.getvalue().splitlines() if line.strip()]\n"
        "assert lines == ['CALLING: greet', 'DONE: greet']\n"
        "assert result == 'Hello World'\n"
        "assert greet.__name__ == 'greet'\n",
    )


def _setup_type_hints_bug(workdir: Path) -> None:
    _write(
        workdir / "typed_utils.py",
        "def greet(name, times=1):\n"
        "    return ' '.join([f'Hello {name}'] * times)\n",
    )
    _write(
        workdir / "verify_types.py",
        "from typing import get_type_hints\n"
        "from typed_utils import greet\n\n"
        "hints = get_type_hints(greet)\n"
        "ok = hints.get('name') is str and hints.get('times') is int and hints.get('return') is str\n"
        "print('TYPES_OK' if ok else 'TYPES_BAD')\n",
    )


def _setup_docstring_bug(workdir: Path) -> None:
    _write(
        workdir / "mathlib.py",
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def multiply(a, b):\n"
        "    return a * b\n",
    )
    _write(
        workdir / "check.py",
        "import mathlib\n\n"
        "doc = mathlib.add.__doc__ or ''\n"
        "parts = ['Args:', 'Returns:', 'Example']\n"
        "print('DOCSTRING_OK' if all(part in doc for part in parts) else 'DOCSTRING_BAD')\n",
    )


def _setup_cli_bug(workdir: Path) -> None:
    _write(
        workdir / "cli.py",
        "import argparse\n\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--name', default='RepoOps')\n"
        "args = parser.parse_args()\n"
        "print(f'HELLO {args.name}')\n",
    )
    _write(
        workdir / "test_cli.py",
        "import subprocess, sys\n\n"
        "proc = subprocess.run([sys.executable, 'cli.py', '--verbose', '--name', 'RepoOps'], capture_output=True, text=True)\n"
        "assert proc.returncode == 0\n"
        "assert proc.stdout.strip() == 'HELLO RepoOps VERBOSE=1'\n",
    )


def _setup_config_env_bug(workdir: Path) -> None:
    _write(
        workdir / "config.py",
        "import os\n\n"
        "DEFAULT_TIMEOUT = 10\n\n"
        "def load_timeout():\n"
        "    return int(os.getenv('APP_TIMEOUT', '0'))\n",
    )
    _write(
        workdir / "app.py",
        "from config import load_timeout\n\n"
        "print(f'TIMEOUT={load_timeout()}')\n",
    )
    _write(
        workdir / "test_config.py",
        "import os, subprocess, sys\n\n"
        "env = dict(os.environ)\n"
        "env.pop('APP_TIMEOUT', None)\n"
        "proc = subprocess.run([sys.executable, 'app.py'], env=env, capture_output=True, text=True, check=True)\n"
        "assert proc.stdout.strip() == 'TIMEOUT=10'\n",
    )


def _setup_readme_bug(workdir: Path) -> None:
    _write(
        workdir / "README.md",
        "# Mini Repo\n\n## Usage\n\nTBD\n",
    )
    _write(
        workdir / "check_readme.py",
        "text = open('README.md', encoding='utf-8').read()\n"
        "assert '# Mini Repo' in text\n"
        "assert 'python app.py' in text\n"
        "assert 'python cli.py --name RepoOps --verbose' in text\n",
    )


def _setup_help_bug(workdir: Path) -> None:
    _write(
        workdir / "cli.py",
        "import argparse\n\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--name', default='RepoOps')\n"
        "args = parser.parse_args()\n"
        "print(f'HELLO {args.name}')\n",
    )
    _write(
        workdir / "test_help.py",
        "import subprocess, sys\n\n"
        "proc = subprocess.run([sys.executable, 'cli.py', '--help'], capture_output=True, text=True)\n"
        "assert 'Mini Repo CLI' in proc.stdout\n"
        "assert '--verbose' in proc.stdout\n"
        "assert '--name' in proc.stdout\n",
    )


TASKS: List[TaskSpec] = [
    TaskSpec(
        id="csv_subject_avg_bugfix",
        split="train",
        domain="csv_ops",
        family="data_ops",
        setup=_setup_csv_average_bug,
        task=(
            "This repo already contains scores.csv, stats.py, and test_stats.py. "
            "Fix the implementation so test_stats.py passes. "
            "Do not edit the CSV file or the test file. Run the test."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "test_stats.py"], w).returncode == 0 else 0.0,
    ),
    TaskSpec(
        id="csv_low_stock_bugfix",
        split="train",
        domain="csv_ops",
        family="data_ops",
        setup=_setup_csv_filter_bug,
        task=(
            "This repo already contains inventory.csv, filter.py, and test_filter.py. "
            "Fix the implementation so the test passes. "
            "Do not edit the CSV file or the test file. Run the test."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "test_filter.py"], w).returncode == 0 else 0.0,
    ),
    TaskSpec(
        id="json_merge_bugfix",
        split="train",
        domain="json_ops",
        family="data_ops",
        setup=_setup_json_merge_bug,
        task=(
            "This repo already contains a.json, b.json, merge.py, and test_merge.py. "
            "Fix merge.py so the test passes. Do not edit the JSON fixtures or the test file. Run the test."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "test_merge.py"], w).returncode == 0 else 0.0,
    ),
    TaskSpec(
        id="json_active_bugfix",
        split="train",
        domain="json_ops",
        family="data_ops",
        setup=_setup_json_active_bug,
        task=(
            "This repo already contains users.json, active.py, and test_active.py. "
            "Fix active.py so the test passes. Do not edit the JSON fixture or the test file. Run the test."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "test_active.py"], w).returncode == 0 else 0.0,
    ),
    TaskSpec(
        id="quality_decorator_bugfix",
        split="train",
        domain="python_quality",
        family="quality_ops",
        setup=_setup_decorator_bug,
        task=(
            "This repo already contains decorators.py and test_decorators.py. "
            "Fix decorators.py so the test passes. Keep the decorator-based API. Run the test."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "test_decorators.py"], w).returncode == 0 else 0.0,
    ),
    TaskSpec(
        id="quality_type_hints_fix",
        split="train",
        domain="python_quality",
        family="quality_ops",
        setup=_setup_type_hints_bug,
        task=(
            "This repo already contains typed_utils.py and verify_types.py. "
            "Update typed_utils.py so verify_types.py prints TYPES_OK. "
            "Do not edit verify_types.py. Run it."
        ),
        verify=lambda w, _r: 1.0 if "TYPES_OK" in _stdout(
            w, ["python", "verify_types.py"]
        ) else 0.0,
    ),
    TaskSpec(
        id="repo_cli_verbose_bugfix",
        split="train",
        domain="repo_maintenance",
        family="repo_ops",
        setup=_setup_cli_bug,
        task=(
            "This repo already contains cli.py and test_cli.py. "
            "Fix the existing CLI so the test passes. "
            "Do not edit the test file. Run the test."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "test_cli.py"], w).returncode == 0 else 0.0,
    ),
    TaskSpec(
        id="repo_config_env_bugfix",
        split="train",
        domain="repo_maintenance",
        family="repo_ops",
        setup=_setup_config_env_bug,
        task=(
            "This repo already contains config.py, app.py, and test_config.py. "
            "Fix the configuration loading behavior so the test passes. "
            "Do not edit the test file. Run the test."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "test_config.py"], w).returncode == 0 else 0.0,
    ),
    TaskSpec(
        id="csv_city_totals_bugfix",
        split="val",
        domain="csv_ops",
        family="data_ops",
        setup=_setup_csv_city_bug,
        task=(
            "This repo already contains orders.csv, summarize.py, and test_summarize.py. "
            "Fix summarize.py so the test passes. Do not edit the CSV fixture or the test file. Run the test."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "test_summarize.py"], w).returncode == 0 else 0.0,
    ),
    TaskSpec(
        id="json_status_count",
        split="val",
        domain="json_ops",
        family="data_ops",
        setup=_setup_json_status_bug,
        task=(
            "This repo already contains tasks.json, status_count.py, and test_status_count.py. "
            "Fix status_count.py so the test passes. Do not edit the JSON fixture or the test file. Run the test."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "test_status_count.py"], w).returncode == 0 else 0.0,
    ),
    TaskSpec(
        id="quality_docstrings_fix",
        split="val",
        domain="python_quality",
        family="quality_ops",
        setup=_setup_docstring_bug,
        task=(
            "This repo already contains mathlib.py and check.py. "
            "Update mathlib.py so check.py prints DOCSTRING_OK. "
            "Do not edit check.py. Run it."
        ),
        verify=lambda w, _r: 1.0 if "DOCSTRING_OK" in _stdout(
            w, ["python", "check.py"]
        ) else 0.0,
    ),
    TaskSpec(
        id="repo_readme_fix",
        split="val",
        domain="repo_maintenance",
        family="repo_ops",
        setup=_setup_readme_bug,
        task=(
            "This repo already contains README.md and check_readme.py. "
            "Update README.md so check_readme.py passes. Keep the title '# Mini Repo'. "
            "Run the check script."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "check_readme.py"], w).returncode == 0 else 0.0,
    ),
    TaskSpec(
        id="csv_top_student_bugfix",
        split="test",
        domain="csv_ops",
        family="data_ops",
        setup=_setup_csv_top_bug,
        task=(
            "This repo already contains ranking.csv, top.py, and test_top.py. "
            "Fix top.py so the test passes. Do not edit the CSV fixture or the test file. Run the test."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "test_top.py"], w).returncode == 0 else 0.0,
    ),
    TaskSpec(
        id="json_score_sum_bugfix",
        split="test",
        domain="json_ops",
        family="data_ops",
        setup=_setup_json_sum_bug,
        task=(
            "This repo already contains metrics.json, sum_scores.py, and test_sum_scores.py. "
            "Fix sum_scores.py so the test passes. Do not edit the JSON fixture or the test file. Run the test."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "test_sum_scores.py"], w).returncode == 0 else 0.0,
    ),
    TaskSpec(
        id="quality_docstrings_regression",
        split="test",
        domain="python_quality",
        family="quality_ops",
        setup=_setup_docstring_bug,
        task=(
            "This repo already contains mathlib.py and check.py. "
            "Update mathlib.py so check.py prints DOCSTRING_OK. "
            "Do not edit check.py. Run it."
        ),
        verify=lambda w, _r: 1.0 if "DOCSTRING_OK" in _stdout(
            w, ["python", "check.py"]
        ) else 0.0,
    ),
    TaskSpec(
        id="repo_help_bugfix",
        split="test",
        domain="repo_maintenance",
        family="repo_ops",
        setup=_setup_help_bug,
        task=(
            "This repo already contains cli.py and test_help.py. "
            "Fix the existing CLI help behavior so the test passes. Do not edit the test file. Run the test."
        ),
        verify=lambda w, _r: 1.0 if _run(["python", "test_help.py"], w).returncode == 0 else 0.0,
    ),
]


def run_kode(task_text: str, skill_prompt: str, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "AGENTS.md").write_text(skill_prompt, encoding="utf-8")

    try:
        proc = subprocess.run(
            [
                "kode",
                "-p",
                task_text,
                "--model",
                KODE_MODEL,
                "--cwd",
                str(workdir),
                "--output-format",
                "json",
                "--dangerously-skip-permissions",
            ],
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        if not proc.stdout.strip():
            return {"is_error": True, "result": proc.stderr.strip() or "empty stdout"}
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return {"is_error": True, "result": "timeout"}
    except json.JSONDecodeError as exc:
        return {"is_error": True, "result": f"invalid json: {exc}"}
    except Exception as exc:
        return {"is_error": True, "result": str(exc)}


def prepare_workspace(task: TaskSpec, workdir: Path) -> None:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if task.setup:
        task.setup(workdir)


def run_task(task: TaskSpec, skill_prompt: str, label: str) -> dict:
    workdir = WORKSPACES_DIR / f"{label}_{task.id}"
    prepare_workspace(task, workdir)
    result = run_kode(task.task, skill_prompt, workdir)
    score = 0.0 if result.get("is_error") else task.verify(workdir, result)
    return {
        "task_id": task.id,
        "domain": task.domain,
        "family": task.family,
        "split": task.split,
        "workdir": str(workdir),
        "score": score,
        "result": result.get("result", ""),
        "is_error": result.get("is_error", False),
        "turns": result.get("num_turns", 0),
    }


def evaluate_tasks(tasks: List[TaskSpec], skill_prompt: str, label: str) -> Dict[str, object]:
    runs = [run_task(task, skill_prompt, label) for task in tasks]
    avg = sum(item["score"] for item in runs) / len(runs) if runs else 0.0
    by_domain: Dict[str, float] = {}
    for domain in sorted({task.domain for task in tasks}):
        subset = [item["score"] for item in runs if item["domain"] == domain]
        by_domain[domain] = sum(subset) / len(subset) if subset else 0.0
    logger.info("  [%s] avg=%.2f by_domain=%s", label, avg, by_domain)
    return {"avg": avg, "runs": runs, "by_domain": by_domain}


def _iter_leaf_paths(node: SkillNode, prefix: str = "") -> List[str]:
    current = f"{prefix}.{node.name}" if prefix else node.name
    if not node.children:
        return [current]
    result: List[str] = []
    for child in node.children.values():
        result.extend(_iter_leaf_paths(child, current))
    return result


def _externalize_path(path: str) -> str:
    parts = path.split(".")
    return ".".join(parts[1:]) if len(parts) > 1 else ""


def _leaf_prompt_map(tree: SkillTree) -> Dict[str, str]:
    prompts: Dict[str, str] = {"": tree.root.skill.system_prompt}
    for full_path in _iter_leaf_paths(tree.root):
        ext = _externalize_path(full_path)
        if ext == "":
            continue
        prompts[ext] = tree.get(ext).skill.system_prompt
    return prompts


def evaluate_leaf_matrix(tree: SkillTree, tasks: List[TaskSpec]) -> Dict[str, Dict[str, float]]:
    prompts = _leaf_prompt_map(tree)
    matrix: Dict[str, Dict[str, float]] = {}
    for leaf_path, prompt in prompts.items():
        matrix[leaf_path] = {}
        for domain in sorted({task.domain for task in tasks}):
            domain_tasks = [task for task in tasks if task.domain == domain]
            label_leaf = leaf_path.replace(".", "_") if leaf_path else "root"
            result = evaluate_tasks(domain_tasks, prompt, f"val_{label_leaf}_{domain}")
            matrix[leaf_path][domain] = float(result["avg"])
    return matrix


def choose_domain_routes(matrix: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    domains = sorted(next(iter(matrix.values())).keys()) if matrix else []
    routes: Dict[str, str] = {}
    for domain in domains:
        best_path = ""
        best_score = -1.0
        for leaf_path, scores in matrix.items():
            score = scores.get(domain, 0.0)
            if score > best_score:
                best_score = score
                best_path = leaf_path
        routes[domain] = best_path
    return routes


def evaluate_with_routes(
    tasks: List[TaskSpec],
    tree: SkillTree,
    routes: Dict[str, str],
    label: str,
) -> Dict[str, object]:
    prompts = _leaf_prompt_map(tree)
    runs = []
    for task in tasks:
        leaf_path = routes.get(task.domain, "")
        prompt = prompts.get(leaf_path, tree.root.skill.system_prompt)
        run = run_task(task, prompt, f"{label}_{leaf_path.replace('.', '_') or 'root'}")
        run["route"] = leaf_path
        runs.append(run)
    avg = sum(item["score"] for item in runs) / len(runs) if runs else 0.0
    by_domain: Dict[str, float] = {}
    for domain in sorted({task.domain for task in tasks}):
        subset = [item["score"] for item in runs if item["domain"] == domain]
        by_domain[domain] = sum(subset) / len(subset) if subset else 0.0
    logger.info("  [%s] routed_avg=%.2f by_domain=%s", label, avg, by_domain)
    return {"avg": avg, "runs": runs, "by_domain": by_domain}


def build_traces(tasks: List[TaskSpec], skill_prompt: str, label: str) -> List[Trace]:
    traces: List[Trace] = []
    for task in tasks:
        run = run_task(task, skill_prompt, label)
        trace = Trace(
            session_id=label,
            inputs=[Message(role="user", content=task.task)],
            prediction=Message(role="assistant", content=str(run.get("result", ""))),
            node_path=ROOT_NAME,
        )
        trace.feedback = Feedback(
            score=float(run["score"]),
            critique=(
                f"Domain={task.domain}, family={task.family}, score={run['score']:.1f}, "
                f"error={run['is_error']}, result={str(run['result'])[:160]}"
            ),
            correction=(
                "The task should be completed exactly as requested, with correct files created or edited "
                "and the final verification command passing."
            ),
        )
        traces.append(trace)
        logger.info("  [trace] %s domain=%s score=%.1f", task.id, task.domain, run["score"])
    return traces


def summarize_tree(tree: SkillTree) -> Dict[str, object]:
    leaf_paths = [_externalize_path(path) for path in _iter_leaf_paths(tree.root)]
    non_root_leaves = [path for path in leaf_paths if path]
    max_depth = max((path.count(".") + 1 for path in non_root_leaves), default=0)
    return {
        "tree": tree.list_tree(),
        "leaf_paths": non_root_leaves,
        "leaf_count": len(non_root_leaves),
        "max_depth": max_depth,
    }


def select_prune_candidates(
    routes: Dict[str, str],
    matrix: Dict[str, Dict[str, float]],
) -> List[str]:
    selected = {path for path in routes.values() if path}
    candidates = []
    for path in matrix:
        if not path:
            continue
        if path in selected:
            continue
        best = max(matrix[path].values()) if matrix[path] else 0.0
        if best <= 0.5:
            candidates.append(path)
    return sorted(candidates)


def compose_merged_prompt(tree: SkillTree, paths: List[str]) -> str:
    prompts = [tree.get(path).skill.system_prompt.strip() for path in paths]
    return (
        "You are a specialized assistant for structured data operations.\n\n"
        "Handle both CSV and JSON tasks accurately. Always create or edit the requested files, "
        "then run the relevant script to verify the result.\n\n"
        "Shared operating rules:\n"
        "- Preserve exact output formatting when the task specifies it.\n"
        "- Avoid inventing extra files unless they help verification.\n"
        "- Prefer small, direct Python scripts over unnecessary abstractions.\n\n"
        "Inherited specialization notes:\n\n"
        + "\n\n---\n\n".join(prompts[:2])
    )


def maybe_merge_data_routes(tree: SkillTree, routes: Dict[str, str]) -> Optional[Dict[str, object]]:
    csv_path = routes.get("csv_ops", "")
    json_path = routes.get("json_ops", "")
    if not csv_path or not json_path or csv_path == json_path:
        return None

    if "." in csv_path or "." in json_path:
        return None

    merged = tree.merge(
        [csv_path, json_path],
        merged_name="data_ops",
        merged_prompt=compose_merged_prompt(tree, [csv_path, json_path]),
    )
    logger.info("Merged %s and %s into %s", csv_path, json_path, merged.name)
    return {
        "merged_paths": [csv_path, json_path],
        "merged_name": merged.name,
    }


def save_summary(summary: Dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    train = [task for task in TASKS if task.split == "train"]
    val = [task for task in TASKS if task.split == "val"]
    test = [task for task in TASKS if task.split == "test"]

    logger.info("RepoOps demo: train=%d val=%d test=%d", len(train), len(val), len(test))
    logger.info("Domains: %s", sorted({task.domain for task in TASKS}))
    logger.info("Kode actor model: %s", KODE_MODEL)

    judge_key = os.getenv("TREE_LLM_API_KEY")
    assert judge_key, "Set TREE_LLM_API_KEY"

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        {
            "TREE_LLM_API_KEY": judge_key,
            "TREE_LLM_BASE_URL": os.getenv("TREE_LLM_BASE_URL", "https://oneapi.liuyanxing.site:8443/v1"),
            "TREE_LLM_MODEL": os.getenv("TREE_LLM_MODEL", "bailian/qwen3.5-plus"),
            "TREE_LLM_PROTOCOL": os.getenv("TREE_LLM_PROTOCOL", "openai"),
            "TREE_LLM_JUDGE_API_KEY": os.getenv("TREE_LLM_JUDGE_API_KEY", judge_key),
            "TREE_LLM_JUDGE_BASE_URL": os.getenv("TREE_LLM_JUDGE_BASE_URL", os.getenv("TREE_LLM_BASE_URL", "https://oneapi.liuyanxing.site:8443/v1")),
            "TREE_LLM_JUDGE_MODEL": os.getenv("TREE_LLM_JUDGE_MODEL", "bailian/glm-5"),
            "TREE_LLM_JUDGE_PROTOCOL": os.getenv("TREE_LLM_JUDGE_PROTOCOL", "openai"),
            "TREE_LLM_REWRITE_API_KEY": os.getenv("TREE_LLM_REWRITE_API_KEY", judge_key),
            "TREE_LLM_REWRITE_BASE_URL": os.getenv("TREE_LLM_REWRITE_BASE_URL", os.getenv("TREE_LLM_BASE_URL", "https://oneapi.liuyanxing.site:8443/v1")),
            "TREE_LLM_REWRITE_MODEL": os.getenv("TREE_LLM_REWRITE_MODEL", os.getenv("TREE_LLM_JUDGE_MODEL", "bailian/glm-5")),
            "TREE_LLM_REWRITE_PROTOCOL": os.getenv("TREE_LLM_REWRITE_PROTOCOL", "openai"),
        }
    )

    from treeskill.config import APOConfig, LLMConfig

    config = GlobalConfig(
        llm=LLMConfig(),
        apo=APOConfig(
            gradient_accumulation_steps=6,
            beam_width=2,
            branch_factor=2,
            beam_rounds=1,
        ),
    )
    llm = LLMClient(config)
    engine = APOEngine(config, llm)

    root_skill = Skill(
        name=ROOT_NAME,
        description="RepoOps root skill for mixed local coding and maintenance tasks",
        system_prompt=BAD_SKILL,
        target=ROOT_TARGET,
        version="v1.0",
    )

    save_skill(root_skill, OUTPUT_DIR)
    tree = SkillTree.load(OUTPUT_DIR)

    start = time.time()
    summary: Dict[str, object] = {
        "config": {
            "rounds": NUM_ROUNDS,
            "train_tasks": len(train),
            "val_tasks": len(val),
            "test_tasks": len(test),
            "kode_actor_model": KODE_MODEL,
        }
    }

    logger.info("\n%s", "=" * 64)
    logger.info("Phase 1: Baseline")
    logger.info("%s", "=" * 64)
    baseline = evaluate_tasks(test, BAD_SKILL, "baseline")
    summary["baseline"] = baseline

    logger.info("\n%s", "=" * 64)
    logger.info("Phase 2: Split via APO + Kode traces")
    logger.info("%s", "=" * 64)
    for round_idx in range(1, NUM_ROUNDS + 1):
        logger.info("--- Round %d/%d ---", round_idx, NUM_ROUNDS)
        traces = build_traces(train, tree.root.skill.system_prompt, f"train_r{round_idx}")
        failures = sum(1 for trace in traces if trace.feedback and trace.feedback.score < 0.8)
        logger.info("Round %d failures: %d/%d", round_idx, failures, len(traces))
        if failures == 0:
            break

        def kode_score_fn(prompt: str, scored_traces: List[Trace]) -> float:
            scores = []
            for trace in scored_traces:
                task = next(item for item in train if item.task == trace.inputs[-1].content)
                run = run_task(task, prompt, f"score_r{round_idx}_{task.id}_{int(time.time())}")
                scores.append(float(run["score"]))
            return sum(scores) / len(scores) if scores else 0.0

        engine._score_fn = kode_score_fn
        tree = engine.evolve_tree(tree, traces, auto_split=True)
        tree.save()

    split_tree = summarize_tree(tree)
    summary["after_split"] = split_tree

    logger.info("\n%s", "=" * 64)
    logger.info("Phase 3: Route leaves on validation")
    logger.info("%s", "=" * 64)
    val_matrix = evaluate_leaf_matrix(tree, val)
    routes = choose_domain_routes(val_matrix)
    split_eval = evaluate_with_routes(test, tree, routes, "split_test")
    summary["validation_matrix"] = val_matrix
    summary["routes_after_split"] = routes
    summary["split_eval"] = split_eval

    logger.info("\n%s", "=" * 64)
    logger.info("Phase 4: Prune dominated leaves")
    logger.info("%s", "=" * 64)
    prune_candidates = select_prune_candidates(routes, val_matrix)
    for candidate in prune_candidates:
        logger.info("Pruning leaf: %s", candidate)
        tree.prune(candidate)
    if prune_candidates:
        tree.save()
    prune_tree = summarize_tree(tree)
    pruned_matrix = evaluate_leaf_matrix(tree, val)
    pruned_routes = choose_domain_routes(pruned_matrix)
    pruned_eval = evaluate_with_routes(test, tree, pruned_routes, "pruned_test")
    summary["pruned_candidates"] = prune_candidates
    summary["after_prune"] = prune_tree
    summary["routes_after_prune"] = pruned_routes
    summary["pruned_eval"] = pruned_eval

    logger.info("\n%s", "=" * 64)
    logger.info("Phase 5: Compose related data leaves")
    logger.info("%s", "=" * 64)
    merge_info = maybe_merge_data_routes(tree, pruned_routes)
    if merge_info:
        tree.save()
    composed_tree = summarize_tree(tree)
    composed_matrix = evaluate_leaf_matrix(tree, val)
    composed_routes = choose_domain_routes(composed_matrix)
    composed_eval = evaluate_with_routes(test, tree, composed_routes, "composed_test")
    summary["merge_info"] = merge_info
    summary["after_compose"] = composed_tree
    summary["routes_after_compose"] = composed_routes
    summary["composed_eval"] = composed_eval

    summary["elapsed_minutes"] = round((time.time() - start) / 60, 2)
    save_summary(summary)

    logger.info("\n%s", "=" * 64)
    logger.info("RepoOps demo complete")
    logger.info("Baseline test avg: %.2f", float(baseline["avg"]))
    logger.info("Split routed test avg: %.2f", float(split_eval["avg"]))
    logger.info("Pruned routed test avg: %.2f", float(pruned_eval["avg"]))
    logger.info("Composed routed test avg: %.2f", float(composed_eval["avg"]))
    logger.info("Summary saved to %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()
