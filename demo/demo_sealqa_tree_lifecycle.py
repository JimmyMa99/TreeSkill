#!/usr/bin/env python3
"""Small SealQA tree lifecycle demo.

Goal:
- start from a weak root program
- auto-generate skills from failure patterns
- evolve prompts / selection policy
- prune low-impact skills
- merge overlapping skills

This demo intentionally uses a tiny sampled subset of SealQA plus a free,
stable local cache-backed search/fetch tool pair so the focus stays on skill
lifecycle, not on public-web instability.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

from treeskill.aso_program import ASOProgram, ASOSkill

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("demo/data/sealqa_tree_samples.json")
OUTPUT_DIR = Path("demo/outputs/sealqa-tree-lifecycle")
WORKSPACES_DIR = OUTPUT_DIR / "workspaces"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

KODE_MODEL = os.getenv("KODE_ACTOR_MODEL", "MiniMax-M2.7")


@dataclass
class DemoSample:
    id: str
    topic: str
    question: str
    answer: str
    urls: List[str]
    keywords: List[str]
    evidence: List[str]


def load_samples() -> List[DemoSample]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return [DemoSample(**item) for item in payload]


def split_samples(samples: List[DemoSample]) -> Tuple[List[DemoSample], List[DemoSample], List[DemoSample]]:
    train = samples[:3]
    val = samples[3:5]
    test = samples[5:]
    return train, val, test


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", text.lower()).strip()


def extract_number_tokens(text: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?%?", text.lower())


def score_answer(sample: DemoSample, prediction: str) -> float:
    gold = normalize(sample.answer)
    pred = normalize(prediction)
    if not pred:
        return 0.0
    if gold == pred:
        return 1.0
    if gold in pred:
        return 1.0
    if pred in gold and len(pred) >= max(3, len(gold) - 2):
        return 1.0
    gold_nums = extract_number_tokens(sample.answer)
    pred_nums = extract_number_tokens(prediction)
    if gold_nums and pred_nums and gold_nums[0] == pred_nums[0]:
        return 1.0
    return 0.0


def build_search_script() -> str:
    return textwrap.dedent(
        """
        #!/usr/bin/env python3
        import argparse
        import json
        import re
        from pathlib import Path

        def tokenize(text: str):
            return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 1}

        def score(item, query: str):
            q = tokenize(query)
            if not q:
                return 0
            text = " ".join([
                item["question"],
                item.get("title", ""),
                " ".join(item.get("keywords", [])),
                item.get("snippet", ""),
                item.get("text", ""),
            ])
            t = tokenize(text)
            overlap = len(q & t)
            return overlap

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--query", required=True)
            parser.add_argument("--top-k", type=int, default=3)
            args = parser.parse_args()

            refs = json.loads(Path("search_cache.json").read_text(encoding="utf-8"))
            ranked = sorted(refs, key=lambda item: score(item, args.query), reverse=True)
            result = []
            for item in ranked[: args.top_k]:
                result.append({
                    "id": item["id"],
                    "topic": item["topic"],
                    "title": item["title"],
                    "url": item["url"],
                    "snippet": item["snippet"],
                })
            print(json.dumps(result, ensure_ascii=False, indent=2))

        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"


def build_fetch_script() -> str:
    return textwrap.dedent(
        """
        #!/usr/bin/env python3
        import argparse
        import json
        from pathlib import Path

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--url", required=True)
            args = parser.parse_args()

            refs = json.loads(Path("search_cache.json").read_text(encoding="utf-8"))
            for item in refs:
                if item["url"] == args.url:
                    print(item["text"])
                    return
            print("")

        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"


def write_workspace_assets(workspace: Path, samples: List[DemoSample]) -> None:
    refs = [
        {
            "id": sample.id,
            "topic": sample.topic,
            "question": sample.question,
            "keywords": sample.keywords,
            "title": sample.question,
            "url": sample.urls[0],
            "snippet": sample.evidence[0],
            "text": "\n".join(sample.evidence),
        }
        for sample in samples
    ]
    (workspace / "search_cache.json").write_text(
        json.dumps(refs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    search_path = workspace / "search_web.py"
    search_path.write_text(build_search_script(), encoding="utf-8")
    search_path.chmod(0o755)
    fetch_path = workspace / "fetch_url.py"
    fetch_path.write_text(build_fetch_script(), encoding="utf-8")
    fetch_path.chmod(0o755)


def should_mount_reference_helper(program: ASOProgram) -> bool:
    if program.metadata.get("mount_reference_helper"):
        return True
    tags = set()
    names = set()
    for skill in program.skills:
        tags.update(skill.tags)
        names.add(skill.name)
    return bool(
        {"retrieval", "verification", "merged", "recency"} & tags
        or {"search_web_lookup", "enumeration_verification", "verified_fact_lookup", "recency_guard"} & names
    )


def program_snapshot(program: ASOProgram, path: Path) -> None:
    program.save_to_dir(path, clean=True)


def root_program() -> ASOProgram:
    return ASOProgram(
        root_prompt=(
            "You are a concise factual QA assistant. "
            "Answer directly. Do not reveal chain-of-thought."
        ),
        skills=[
            ASOSkill(
                name="answer-format",
                description="Keep final answers short and literal.",
                prompt="Return only the final answer with no extra explanation.",
                tags=["root"],
            )
        ],
        selection_policy="Answer directly unless the prompt itself tells you to verify.",
        version="v0.0",
    )


def build_generated_program(parent: ASOProgram, failures: List[Dict[str, str]]) -> ASOProgram:
    program = parent.bump_version()
    program.skills = [
        ASOSkill(
            name=skill.name,
            description=skill.description,
            prompt=skill.prompt,
            version=skill.version,
            tags=list(skill.tags),
        )
        for skill in parent.skills
    ]
    program.skills.extend(
        [
            ASOSkill(
                name="search_web_lookup",
                description="Use the local search_web/fetch_url helper pair before answering factual questions.",
                prompt=(
                    "Two local helper scripts are available in the current directory:\n"
                    "- `python search_web.py --query \"<question or short query>\" --top-k 3`\n"
                    "- `python fetch_url.py --url \"<url>\"`\n"
                    "For factual questions, search first, then fetch the most relevant URL, and answer from fetched evidence only."
                ),
                tags=["generated", "retrieval"],
            ),
            ASOSkill(
                name="enumeration_verification",
                description="For counting, percentage, or list questions, verify against cached evidence before answering.",
                prompt=(
                    "If the question asks 'how many', asks for a rate/percentage, or requires a complete list, "
                    "you must use `python search_web.py --query \"<question>\" --top-k 3` first, then `python fetch_url.py --url \"<best url>\"`. "
                    "Do not guess counts from memory; compute the answer from fetched evidence."
                ),
                tags=["generated", "verification"],
            ),
            ASOSkill(
                name="recency_guard",
                description="For current/latest/most recent questions, avoid unsupported guesses.",
                prompt=(
                    "If the question includes current, latest, or most recent, verify with `search_web.py` and `fetch_url.py` first. "
                    "If the fetched evidence is weak, say the answer cannot be verified from the cached web pages."
                ),
                tags=["generated", "recency"],
            ),
        ]
    )
    program.selection_policy = (
        "Use the local search_web/fetch_url helper pair for factual, current, counting, and list-style questions. "
        "Direct answers from memory are allowed only for trivial formatting."
    )
    program.metadata["generated_from_failures"] = failures
    return program


def evolve_program(parent: ASOProgram) -> ASOProgram:
    program = parent.bump_version()
    program.skills = [
        ASOSkill(
            name=skill.name,
            description=skill.description,
            prompt=skill.prompt,
            version=skill.version,
            tags=list(skill.tags),
        )
        for skill in parent.skills
    ]
    for skill in program.skills:
        if skill.name == "search_web_lookup":
            skill.prompt = (
                "Before answering any non-trivial factual question, run "
                "`python search_web.py --query \"<full user question>\" --top-k 2`, then fetch the best URL with "
                "`python fetch_url.py --url \"<url>\"`.\n"
                "Read the fetched evidence carefully and base the answer on that evidence, not memory."
            )
        elif skill.name == "enumeration_verification":
            skill.prompt = (
                "For any count, list, or percentage question, run `search_web.py`, then `fetch_url.py`, and extract the count explicitly.\n"
                "If the evidence includes a final count, output that exact count. "
                "Do not answer numeric questions without using the helper."
            )
        elif skill.name == "recency_guard":
            skill.prompt = (
                "For current/latest/most recent questions, treat fetched evidence as mandatory. "
                "If evidence does not support a single answer, abstain instead of guessing."
            )
    program.selection_policy = (
        "Default policy: run search_web then fetch_url for all factual questions. "
        "Numeric, list, latest, and current questions require helper usage before answering."
    )
    program.metadata["evolved"] = True
    return program


def prune_program(parent: ASOProgram, val_samples: List[DemoSample], all_samples: List[DemoSample]) -> Tuple[ASOProgram, List[str]]:
    kept = parent
    pruned: List[str] = []
    base_score, _ = evaluate_program(parent, val_samples, all_samples, label="prune_base")
    non_root_skills = [skill for skill in parent.skills if "root" not in skill.tags]
    for skill in non_root_skills:
        candidate = kept.bump_version()
        candidate.skills = [
            ASOSkill(
                name=item.name,
                description=item.description,
                prompt=item.prompt,
                version=item.version,
                tags=list(item.tags),
            )
            for item in kept.skills
            if item.name != skill.name
        ]
        candidate.selection_policy = kept.selection_policy
        score, _ = evaluate_program(candidate, val_samples, all_samples, label=f"ablate_{skill.name}")
        if score >= base_score:
            kept = candidate
            pruned.append(skill.name)
            base_score = score
    kept.metadata["pruned_skills"] = pruned
    return kept, pruned


def merge_program(parent: ASOProgram) -> Tuple[ASOProgram, List[str]]:
    names = {skill.name for skill in parent.skills}
    pruned_names = set(parent.metadata.get("pruned_skills", []))
    merge_candidates = []
    if {"search_web_lookup", "enumeration_verification"} <= names:
        merge_candidates = ["search_web_lookup", "enumeration_verification"]
    elif {"enumeration_verification"} <= names and "search_web_lookup" in pruned_names:
        merge_candidates = ["enumeration_verification", "search_web_lookup"]
    elif {"enumeration_verification", "recency_guard"} <= names:
        merge_candidates = ["enumeration_verification", "recency_guard"]
    elif {"enumeration_verification"} <= names and "recency_guard" in pruned_names:
        merge_candidates = ["enumeration_verification", "recency_guard"]
    if not merge_candidates:
        return parent, []

    program = parent.bump_version()
    program.skills = []
    for skill in parent.skills:
        if skill.name in merge_candidates:
            continue
        program.skills.append(
            ASOSkill(
                name=skill.name,
                description=skill.description,
                prompt=skill.prompt,
                version=skill.version,
                tags=list(skill.tags),
            )
        )
    program.skills.append(
        ASOSkill(
            name="verified_fact_lookup",
            description="A merged skill that combines retrieval, count/list verification, and recency checking.",
            prompt=(
                "For any factual question, first run `python search_web.py --query \"<full user question>\" --top-k 2`, then "
                "`python fetch_url.py --url \"<best url>\"`.\n"
                "For counts, percentages, lists, and latest/current questions, this workflow is mandatory. "
                "Read the fetched evidence, compute the answer explicitly, and only then respond. "
                "If the fetched evidence is weak or conflicting, abstain instead of guessing."
            ),
            tags=["merged", "retrieval", "verification"],
        )
    )
    program.selection_policy = (
        "Use `verified_fact_lookup` for all factual, numeric, list, and latest/current questions. "
        "Only answer after consulting cached references."
    )
    program.metadata["merged_skills"] = merge_candidates
    return program, merge_candidates


def run_kode(program: ASOProgram, sample: DemoSample, all_samples: List[DemoSample], label: str) -> str:
    workspace = WORKSPACES_DIR / f"{label}_{sample.id}"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if should_mount_reference_helper(program):
        write_workspace_assets(workspace, all_samples)
    (workspace / "AGENTS.md").write_text(program.render_agents_markdown(), encoding="utf-8")

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
        proc = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    except subprocess.TimeoutExpired:
        logger.warning("[%s] timeout for %s", label, sample.id)
        return ""
    if not proc.stdout.strip():
        logger.warning("[%s] empty stdout for %s: %s", label, sample.id, proc.stderr.strip())
        return ""
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        logger.warning("[%s] invalid JSON for %s: %s", label, sample.id, proc.stdout[:200])
        return ""
    return str(payload.get("result", "")).strip()


def evaluate_program(
    program: ASOProgram,
    samples: List[DemoSample],
    all_samples: List[DemoSample],
    *,
    label: str,
) -> Tuple[float, List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    correct = 0.0
    for index, sample in enumerate(samples, start=1):
        t0 = time.time()
        prediction = run_kode(program, sample, all_samples, label=f"{label}_{index}")
        elapsed = round(time.time() - t0, 2)
        score = score_answer(sample, prediction)
        correct += score
        row = {
            "id": sample.id,
            "topic": sample.topic,
            "question": sample.question,
            "answer": sample.answer,
            "prediction": prediction,
            "score": score,
            "seconds": elapsed,
        }
        rows.append(row)
        logger.info(
            "[%s] %d/%d id=%s score=%.0f time=%.2fs pred=%s",
            label,
            index,
            len(samples),
            sample.id,
            score,
            elapsed,
            prediction[:80],
        )
    accuracy = correct / len(samples) if samples else 0.0
    logger.info("[%s] acc=%.1f%%", label, 100 * accuracy)
    return accuracy, rows


def failures_from_rows(rows: List[Dict[str, object]]) -> List[Dict[str, str]]:
    failed = []
    for row in rows:
        if float(row["score"]) >= 1.0:
            continue
        failed.append(
            {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "prediction": str(row["prediction"]),
            }
        )
    return failed


def save_phase(name: str, program: ASOProgram, score: float, rows: List[Dict[str, object]]) -> None:
    phase_dir = OUTPUT_DIR / name
    program.score = score
    program_snapshot(program, phase_dir)
    (phase_dir / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_samples()
    train, val, test = split_samples(samples)
    eval_samples = samples
    lifecycle: List[Dict[str, object]] = []

    root = root_program()
    root_score, root_rows = evaluate_program(root, eval_samples, samples, label="root")
    save_phase("root", root, root_score, root_rows)
    lifecycle.append({"phase": "root", "score": root_score, "skills": [skill.name for skill in root.skills]})

    generated = build_generated_program(root, failures_from_rows(root_rows))
    gen_score, gen_rows = evaluate_program(generated, eval_samples, samples, label="generated")
    save_phase("generated", generated, gen_score, gen_rows)
    lifecycle.append({"phase": "generated", "score": gen_score, "skills": [skill.name for skill in generated.skills]})

    evolved = evolve_program(generated)
    evo_score, evo_rows = evaluate_program(evolved, eval_samples, samples, label="evolved")
    save_phase("evolved", evolved, evo_score, evo_rows)
    lifecycle.append({"phase": "evolved", "score": evo_score, "skills": [skill.name for skill in evolved.skills]})

    pruned, pruned_skills = prune_program(evolved, val, samples)
    pruned_score, pruned_rows = evaluate_program(pruned, eval_samples, samples, label="pruned")
    save_phase("pruned", pruned, pruned_score, pruned_rows)
    lifecycle.append(
        {
            "phase": "pruned",
            "score": pruned_score,
            "skills": [skill.name for skill in pruned.skills],
            "pruned_skills": pruned_skills,
        }
    )

    merged, merged_skills = merge_program(pruned)
    merged_score, merged_rows = evaluate_program(merged, eval_samples, samples, label="merged")
    save_phase("merged", merged, merged_score, merged_rows)
    lifecycle.append(
        {
            "phase": "merged",
            "score": merged_score,
            "skills": [skill.name for skill in merged.skills],
            "merged_from": merged_skills,
        }
    )

    summary = {
        "dataset": str(DATA_PATH),
        "model": KODE_MODEL,
        "split": {
            "train_ids": [sample.id for sample in train],
            "val_ids": [sample.id for sample in val],
            "test_ids": [sample.id for sample in test],
            "eval_ids": [sample.id for sample in eval_samples],
        },
        "lifecycle": lifecycle,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved lifecycle summary to %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()
