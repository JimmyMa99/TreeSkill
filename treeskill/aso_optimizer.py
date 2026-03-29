"""AS(skill)O optimizer.

This module keeps the first implementation intentionally small:
- optimize a frontier of agent programs, not a single prompt
- analyze failures with textual gradients
- propose skill-level actions
- generate program candidates and keep the best frontier on validation
"""

from __future__ import annotations

import json
import logging
import traceback
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from treeskill.aso_program import ASOProgram, ASOSkill
from treeskill.llm import LLMClient
from treeskill.schema import Feedback, Message, Trace
from treeskill.tasks.sealqa import SealQAExample

logger = logging.getLogger(__name__)


def _strip_thinking_blocks(text: str) -> str:
    if not text:
        return ""
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def _extract_json_payload(text: str, *, expect_array: bool) -> Optional[Any]:
    candidates: list[str] = []
    cleaned = _strip_thinking_blocks(text)
    if not cleaned:
        return None
    candidates.append(cleaned)

    fenced = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", cleaned, flags=re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    regex = r"\[[\s\S]*?\]" if expect_array else r"\{[\s\S]*?\}"
    candidates.extend(
        match.group(0).strip()
        for match in re.finditer(regex, cleaned)
    )

    for raw in candidates:
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if expect_array and isinstance(parsed, list):
            return parsed
        if not expect_array and isinstance(parsed, dict):
            return parsed
    return None


@dataclass
class ASOSkillAction:
    action: str
    rationale: str = ""
    skill_name: Optional[str] = None
    description: Optional[str] = None
    skill_prompt: Optional[str] = None
    target_skill: Optional[str] = None
    merge_skills: List[str] = field(default_factory=list)
    selection_policy: Optional[str] = None


@dataclass
class ASOIterationResult:
    iteration: int
    best_score: float
    frontier_scores: List[float]
    accepted_program_id: str
    actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ASOResult:
    best_program: ASOProgram
    frontier: List[ASOProgram]
    baseline_score: float
    final_score: float
    history: List[ASOIterationResult] = field(default_factory=list)
    postprocess: List[Dict[str, Any]] = field(default_factory=list)


class ASOOptimizer:
    """Skill-level optimizer driven by textual gradients and validation frontier."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        frontier_size: int = 3,
        branch_factor: int = 2,
        max_iterations: int = 3,
        max_workers: int = 1,
        auto_merge: bool = False,
        auto_prune: bool = False,
        artifact_dir: Optional[Path] = None,
    ) -> None:
        self._llm = llm
        self.frontier_size = frontier_size
        self.branch_factor = branch_factor
        self.max_iterations = max_iterations
        self.max_workers = max(1, max_workers)
        self.auto_merge = auto_merge
        self.auto_prune = auto_prune
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None

    def run(
        self,
        seed_program: ASOProgram,
        train_data: Sequence[SealQAExample],
        val_data: Sequence[SealQAExample],
        runner: Callable[[ASOProgram, SealQAExample], str],
        scorer: Callable[[SealQAExample, str], float],
        *,
        start_iteration: int = 1,
        initial_frontier: Optional[Sequence[ASOProgram]] = None,
        initial_best_program: Optional[ASOProgram] = None,
        initial_history: Optional[Sequence[ASOIterationResult]] = None,
        initial_baseline_score: Optional[float] = None,
    ) -> ASOResult:
        """Run a frontier-based ASO loop."""
        if not initial_frontier:
            frontier = [seed_program]
            if seed_program.score is None:
                seed_program.score = self._evaluate(seed_program, val_data, runner, scorer)
            baseline_score = seed_program.score
            best_program = initial_best_program or seed_program
        else:
            frontier = list(initial_frontier)
            if initial_best_program is not None:
                best_program = initial_best_program
            else:
                best_program = max(
                    frontier,
                    key=lambda item: 0.0 if item.score is None else item.score,
                )
            baseline_score = (
                initial_baseline_score
                if initial_baseline_score is not None
                else best_program.score
            )
            if baseline_score is None:
                baseline_score = self._evaluate(best_program, val_data, runner, scorer)
        if baseline_score is not None:
            best_program.score = baseline_score

        history: List[ASOIterationResult] = list(initial_history or [])
        postprocess: List[Dict[str, Any]] = []
        start_iteration = max(1, int(start_iteration))

        if start_iteration > self.max_iterations:
            logger.info(
                "ASO start iteration %d is beyond max iterations %d; skipping evolution",
                start_iteration,
                self.max_iterations,
            )

        for iteration in range(start_iteration, self.max_iterations + 1):
            logger.info("ASO iteration %d/%d", iteration, self.max_iterations)
            candidates: List[ASOProgram] = list(frontier)
            iteration_actions: List[Dict[str, Any]] = []

            for parent in frontier:
                traces = self._collect_failure_traces(parent, train_data, runner, scorer)
                if not traces:
                    logger.info("  Program %s has no train failures", parent.program_id)
                    continue

                gradient = self.compute_program_gradient(parent, traces)
                logger.info("  Gradient summary: %s", gradient[:160].replace("\n", " "))

                for _ in range(self.branch_factor):
                    actions = self.propose_actions(parent, gradient, traces)
                    iteration_actions.extend([action.__dict__ for action in actions])
                    if not actions:
                        continue
                    candidate = self.apply_actions(parent, actions)
                    candidates.append(candidate)

            scored_candidates: List[ASOProgram] = []
            for candidate in candidates:
                candidate.score = self._evaluate(candidate, val_data, runner, scorer)
                scored_candidates.append(candidate)

            scored_candidates.sort(key=lambda item: item.score or 0.0, reverse=True)
            frontier = scored_candidates[: self.frontier_size]
            if frontier and (frontier[0].score or 0.0) >= (best_program.score or 0.0):
                best_program = frontier[0]
                best_program.score = float(best_program.score or 0.0)

            history.append(
                ASOIterationResult(
                    iteration=iteration,
                    best_score=float(best_program.score or 0.0),
                    frontier_scores=[float(item.score or 0.0) for item in frontier],
                    accepted_program_id=best_program.program_id,
                    actions=iteration_actions,
                )
            )
            self._write_iteration_artifacts(iteration, frontier, best_program, iteration_actions)

        if self.auto_merge:
            best_program, merge_event = self._auto_merge(best_program, val_data, runner, scorer)
            if merge_event:
                postprocess.append(merge_event)

        if self.auto_prune:
            best_program, prune_event = self._auto_prune(best_program, val_data, runner, scorer)
            if prune_event:
                postprocess.append(prune_event)

        best_program.score = self._evaluate(best_program, val_data, runner, scorer)

        return ASOResult(
            best_program=best_program,
            frontier=frontier,
            baseline_score=baseline_score,
            final_score=float(best_program.score or 0.0),
            history=history,
            postprocess=postprocess,
        )

    def _write_iteration_artifacts(
        self,
        iteration: int,
        frontier: Sequence[ASOProgram],
        best_program: ASOProgram,
        iteration_actions: Sequence[Dict[str, Any]],
    ) -> None:
        if self.artifact_dir is None:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        iteration_dir = self.artifact_dir / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        frontier_payload = []
        for index, program in enumerate(frontier, start=1):
            program_dir = iteration_dir / f"frontier_{index}"
            program.save_to_dir(program_dir, clean=True)
            frontier_payload.append(program.to_dict())

        best_program.save_to_dir(iteration_dir / "best_program", clean=True)
        payload = {
            "iteration": iteration,
            "best_program_id": best_program.program_id,
            "best_score": best_program.score,
            "frontier": frontier_payload,
            "actions": list(iteration_actions),
        }
        (iteration_dir / "frontier.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def compute_program_gradient(self, program: ASOProgram, traces: List[Trace]) -> str:
        failure_block = "\n".join(
            (
                f"- Question: {trace.inputs[-1].content}\n"
                f"  Prediction: {trace.prediction.content}\n"
                f"  Critique: {trace.feedback.critique if trace.feedback else 'n/a'}"
            )
            for trace in traces[:8]
        )
        messages = [
            Message(
                role="system",
                content=(
                    "You are analyzing failures for a skill-evolution loop. "
                    "Do not just critique wording. Identify missing skills, weak skills, "
                    "and bad skill-selection behavior. Return a concise bullet list."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Root prompt:\n{program.root_prompt}\n\n"
                    f"Selection policy:\n{program.selection_policy}\n\n"
                    f"Current skills:\n{self._render_skill_inventory(program)}\n\n"
                    f"Failures:\n{failure_block}"
                ),
            ),
        ]
        response = self._llm.generate(messages, role="judge")
        return response.content if isinstance(response.content, str) else str(response.content)

    def propose_actions(
        self,
        program: ASOProgram,
        gradient: str,
        traces: List[Trace],
    ) -> List[ASOSkillAction]:
        messages = [
            Message(
                role="system",
                content=(
                    "You are a proposer for AS(skill)O. Based on the gradient and failures, "
                    "propose up to 2 structured skill actions.\n\n"
                    "Allowed action values:\n"
                    "- add_skill\n"
                    "- revise_skill\n"
                    "- drop_skill\n"
                    "- merge_skills\n"
                    "- adjust_selection_policy\n\n"
                    "Return ONLY valid JSON: "
                    '[{"action":"add_skill","skill_name":"...","description":"...","skill_prompt":"...","rationale":"..."}]\n'
                    "For revise_skill use target_skill + skill_prompt.\n"
                    "For merge_skills use merge_skills + skill_name + description + skill_prompt.\n"
                    "For adjust_selection_policy use selection_policy.\n"
                    "Prefer actionable skill proposals over generic prompt edits."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Current program skills:\n{self._render_skill_inventory(program)}\n\n"
                    f"Selection policy:\n{program.selection_policy}\n\n"
                    f"Gradient:\n{gradient}\n\n"
                    f"Representative failure:\n{traces[0].feedback.critique if traces and traces[0].feedback else 'n/a'}"
                ),
            ),
        ]
        response = self._llm.generate(messages, role="rewrite")
        raw = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _extract_json_payload(raw, expect_array=True)
        if parsed is None:
            logger.warning("Failed to parse ASO actions JSON: %s", raw[:240])
            return []
        if not isinstance(parsed, list):
            return []
        actions: List[ASOSkillAction] = []
        for item in parsed:
            if not isinstance(item, dict) or "action" not in item:
                continue
            actions.append(
                ASOSkillAction(
                    action=str(item.get("action", "")).strip(),
                    rationale=str(item.get("rationale", "")).strip(),
                    skill_name=item.get("skill_name"),
                    description=item.get("description"),
                    skill_prompt=item.get("skill_prompt"),
                    target_skill=item.get("target_skill"),
                    merge_skills=list(item.get("merge_skills", []) or []),
                    selection_policy=item.get("selection_policy"),
                )
            )
        return actions

    def apply_actions(self, parent: ASOProgram, actions: List[ASOSkillAction]) -> ASOProgram:
        candidate = parent.bump_version()
        candidate.skills = [
            ASOSkill(
                name=skill.name,
                description=skill.description,
                prompt=skill.prompt,
                version=skill.version,
                tags=list(skill.tags),
            )
            for skill in parent.skills
        ]
        candidate.metadata.setdefault("applied_actions", [])

        for action in actions:
            candidate.metadata["applied_actions"].append(action.__dict__)
            if action.action == "add_skill" and action.skill_name and action.skill_prompt:
                if any(skill.name == action.skill_name for skill in candidate.skills):
                    continue
                candidate.skills.append(
                    ASOSkill(
                        name=action.skill_name,
                        description=action.description or "",
                        prompt=action.skill_prompt,
                        tags=["generated"],
                    )
                )
            elif action.action == "revise_skill" and action.target_skill and action.skill_prompt:
                for skill in candidate.skills:
                    if skill.name == action.target_skill:
                        skill.prompt = action.skill_prompt
                        if action.description:
                            skill.description = action.description
                        skill.version = _increment_version(skill.version)
                        break
            elif action.action == "drop_skill" and action.target_skill:
                candidate.skills = [
                    skill for skill in candidate.skills if skill.name != action.target_skill
                ]
            elif action.action == "merge_skills" and action.merge_skills and action.skill_name and action.skill_prompt:
                candidate.skills = [
                    skill for skill in candidate.skills if skill.name not in set(action.merge_skills)
                ]
                candidate.skills.append(
                    ASOSkill(
                        name=action.skill_name,
                        description=action.description or "",
                        prompt=action.skill_prompt,
                        tags=["merged"],
                    )
                )
            elif action.action == "adjust_selection_policy" and action.selection_policy:
                candidate.selection_policy = action.selection_policy

        return candidate

    def _auto_prune(
        self,
        program: ASOProgram,
        val_data: Sequence[SealQAExample],
        runner: Callable[[ASOProgram, SealQAExample], str],
        scorer: Callable[[SealQAExample, str], float],
    ) -> tuple[ASOProgram, Optional[Dict[str, Any]]]:
        current = program
        base_score = float(program.score if program.score is not None else self._evaluate(program, val_data, runner, scorer))
        pruned: List[str] = []
        non_root_skills = [skill for skill in current.skills if "root" not in skill.tags]
        for skill in non_root_skills:
            candidate = current.bump_version()
            candidate.skills = [
                ASOSkill(
                    name=item.name,
                    description=item.description,
                    prompt=item.prompt,
                    version=item.version,
                    tags=list(item.tags),
                )
                for item in current.skills
                if item.name != skill.name
            ]
            candidate.selection_policy = current.selection_policy
            candidate.metadata = dict(current.metadata)
            score = self._evaluate(candidate, val_data, runner, scorer)
            if score >= base_score:
                current = candidate
                current.score = score
                pruned.append(skill.name)
                base_score = score
        if not pruned:
            return current, None
        current.metadata["auto_pruned_skills"] = pruned
        return current, {
            "type": "auto_prune",
            "accepted": True,
            "score": base_score,
            "pruned_skills": pruned,
            "skills": [skill.name for skill in current.skills],
        }

    def _auto_merge(
        self,
        program: ASOProgram,
        val_data: Sequence[SealQAExample],
        runner: Callable[[ASOProgram, SealQAExample], str],
        scorer: Callable[[SealQAExample, str], float],
    ) -> tuple[ASOProgram, Optional[Dict[str, Any]]]:
        merge_action = self._propose_merge_action(program)
        if merge_action is None:
            return program, None
        candidate = self.apply_actions(program, [merge_action])
        base_score = float(program.score if program.score is not None else self._evaluate(program, val_data, runner, scorer))
        merge_score = self._evaluate(candidate, val_data, runner, scorer)
        if merge_score < base_score:
            return program, {
                "type": "auto_merge",
                "accepted": False,
                "base_score": base_score,
                "candidate_score": merge_score,
                "merged_from": merge_action.merge_skills,
                "merged_skill": merge_action.skill_name,
            }
        candidate.score = merge_score
        candidate.metadata["auto_merged_from"] = list(merge_action.merge_skills)
        candidate.metadata["auto_merged_skill"] = merge_action.skill_name
        return candidate, {
            "type": "auto_merge",
            "accepted": True,
            "score": merge_score,
            "merged_from": merge_action.merge_skills,
            "merged_skill": merge_action.skill_name,
            "skills": [skill.name for skill in candidate.skills],
        }

    def _propose_merge_action(self, program: ASOProgram) -> Optional[ASOSkillAction]:
        candidates = self._rank_merge_pairs(program)
        if not candidates:
            return None
        left, right = candidates[0]
        left_skill = next(skill for skill in program.skills if skill.name == left)
        right_skill = next(skill for skill in program.skills if skill.name == right)
        messages = [
            Message(
                role="system",
                content=(
                    "You are merging overlapping agent skills for AS(skill)O. "
                    "Return ONLY valid JSON with keys: skill_name, description, skill_prompt, rationale. "
                    "If the pair should not be merged, return [] instead."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Skill A\nname: {left_skill.name}\n"
                    f"description: {left_skill.description}\n"
                    f"prompt:\n{left_skill.prompt}\n\n"
                    f"Skill B\nname: {right_skill.name}\n"
                    f"description: {right_skill.description}\n"
                    f"prompt:\n{right_skill.prompt}\n\n"
                    "Create one merged skill that preserves the useful behavior of both and removes redundancy."
                ),
            ),
        ]
        response = self._llm.generate(messages, role="rewrite")
        raw = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _extract_json_payload(raw, expect_array=False)
        if raw == "[]":
            return None
        if parsed is None:
            logger.warning("Failed to parse ASO merge JSON: %s", raw[:240])
            return None
        if not isinstance(parsed, dict):
            return None
        skill_name = str(parsed.get("skill_name", "")).strip()
        skill_prompt = str(parsed.get("skill_prompt", "")).strip()
        if not skill_name or not skill_prompt:
            return None
        return ASOSkillAction(
            action="merge_skills",
            rationale=str(parsed.get("rationale", "")).strip(),
            skill_name=skill_name,
            description=str(parsed.get("description", "")).strip(),
            skill_prompt=skill_prompt,
            merge_skills=[left, right],
        )

    @staticmethod
    def _rank_merge_pairs(program: ASOProgram) -> List[tuple[str, str]]:
        skills = [skill for skill in program.skills if "root" not in skill.tags]
        ranked: List[tuple[float, tuple[str, str]]] = []
        for idx, left in enumerate(skills):
            for right in skills[idx + 1:]:
                left_tags = set(left.tags)
                right_tags = set(right.tags)
                tag_overlap = len(left_tags & right_tags)
                left_words = set((left.description + " " + left.prompt).lower().split())
                right_words = set((right.description + " " + right.prompt).lower().split())
                text_overlap = len(left_words & right_words)
                score = tag_overlap * 10 + text_overlap
                if score <= 0:
                    continue
                ranked.append((float(score), (left.name, right.name)))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [pair for _score, pair in ranked]

    def _collect_failure_traces(
        self,
        program: ASOProgram,
        train_data: Sequence[SealQAExample],
        runner: Callable[[ASOProgram, SealQAExample], str],
        scorer: Callable[[SealQAExample, str], float],
    ) -> List[Trace]:
        traces: List[Trace] = []
        if self.max_workers == 1:
            for sample in train_data:
                trace = self._score_sample(program, sample, runner, scorer)
                if trace is not None:
                    traces.append(trace)
            return traces

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._score_sample, program, sample, runner, scorer)
                for sample in train_data
            ]
            for future in as_completed(futures):
                try:
                    trace = future.result()
                    if trace is not None:
                        traces.append(trace)
                except Exception:
                    logger.warning("Failure trace worker error:\n%s", traceback.format_exc())
        return traces

    def _evaluate(
        self,
        program: ASOProgram,
        data: Sequence[SealQAExample],
        runner: Callable[[ASOProgram, SealQAExample], str],
        scorer: Callable[[SealQAExample, str], float],
    ) -> float:
        if not data:
            return 0.0

        if self.max_workers == 1:
            total = 0.0
            for sample in data:
                total += self._score_value(program, sample, runner, scorer)
            return total / len(data)

        total = 0.0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._score_value, program, sample, runner, scorer)
                for sample in data
            ]
            for future in as_completed(futures):
                try:
                    total += future.result()
                except Exception:
                    logger.warning("Scoring worker error:\n%s", traceback.format_exc())
                    total += 0.0
        return total / len(data)

    @staticmethod
    def _score_value(
        program: ASOProgram,
        sample: SealQAExample,
        runner: Callable[[ASOProgram, SealQAExample], str],
        scorer: Callable[[SealQAExample, str], float],
    ) -> float:
        try:
            prediction = runner(program, sample)
            return scorer(sample, prediction)
        except Exception:
            logger.warning("Sample scoring failed:\n%s", traceback.format_exc())
            return 0.0

    @staticmethod
    def _score_sample(
        program: ASOProgram,
        sample: SealQAExample,
        runner: Callable[[ASOProgram, SealQAExample], str],
        scorer: Callable[[SealQAExample, str], float],
    ) -> Optional[Trace]:
        try:
            prediction = runner(program, sample)
            score = scorer(sample, prediction)
        except Exception:
            logger.warning("Failure trace scoring failed:\n%s", traceback.format_exc())
            prediction = ""
            score = 0.0
        if score >= 1.0:
            return None
        return Trace(
            session_id=program.program_id,
            inputs=[Message(role="user", content=sample.question)],
            prediction=Message(role="assistant", content=prediction),
            feedback=Feedback(
                score=score,
                critique=(
                    f"Incorrect or incomplete answer for topic={sample.topic}. "
                    f"Expected: {sample.answer}. Got: {prediction}"
                ),
                correction=sample.answer,
            ),
        )

    @staticmethod
    def _render_skill_inventory(program: ASOProgram) -> str:
        if not program.skills:
            return "(none)"
        return "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in program.skills
        )


def _increment_version(version: str) -> str:
    raw = version[1:] if version.startswith("v") else version
    parts = raw.split(".")
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].isdigit():
            parts[idx] = str(int(parts[idx]) + 1)
            return "v" + ".".join(parts)
    return f"{version}.1"
