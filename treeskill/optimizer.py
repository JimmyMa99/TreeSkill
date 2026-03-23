"""APO Optimizer — Automatic Prompt Optimization via Textual Gradient Descent.

Implements the "brain" of the Evo-Framework.  Given a Skill and a batch
of traces (with feedback), it:

1. **Diagnoses** failures using critique / correction annotations.
2. **Computes a "gradient"** by asking a judge model *why* the prompt failed.
   Uses multiple gradient templates for diversity (inspired by Microsoft
   agent-lightning).
3. **Generates N candidate** rewrites using different edit strategies.
4. **Scores** each candidate against the feedback traces.
5. **Selects** the best candidate (or keeps the original if none improve).
6. **Applies** the change, returning a new Skill with an incremented version.

Tree-aware extensions:
- ``analyze_split_need`` — detect when a skill should be split
- ``generate_child_prompts`` — create specialised child prompts
- ``evolve_tree`` — recursively optimise a whole SkillTree
"""

from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from treeskill.config import GlobalConfig
from treeskill.llm import LLMClient
from treeskill.resume import ResumeState
from treeskill.schema import Feedback, Message, Skill, Trace

if TYPE_CHECKING:
    from treeskill.skill_tree import SkillNode, SkillTree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gradient prompt templates — 3 variants for diversity
# ---------------------------------------------------------------------------

_GRADIENT_TEMPLATES = [
    # Variant 1: Simple, direct
    (
        "You are an expert prompt engineer. Analyze the conversation failures "
        "below and explain concisely WHY the system prompt led to these problems. "
        "Return a bullet list of specific, actionable issues."
    ),
    # Variant 2: Root-cause focused
    (
        "You are a senior prompt debugger. For each failure below, identify the "
        "ROOT CAUSE in the system prompt — what instruction is missing, ambiguous, "
        "or misleading? Be specific: quote the problematic part of the prompt and "
        "explain how it caused the failure. Return 3-5 bullets."
    ),
    # Variant 3: Comprehensive evaluation
    (
        "You are a prompt quality auditor. Evaluate the system prompt against "
        "these failures across these dimensions:\n"
        "1. Instruction clarity — are the rules unambiguous?\n"
        "2. Tone/style control — does the prompt prevent AI-sounding language?\n"
        "3. Scope constraints — does the prompt enforce length/format limits?\n"
        "4. Edge cases — does the prompt handle the scenarios that failed?\n"
        "Return a structured critique with specific fixes for each dimension."
    ),
]

# ---------------------------------------------------------------------------
# Edit prompt templates — 2 variants (full rewrite vs conservative)
# ---------------------------------------------------------------------------

_EDIT_TEMPLATES = [
    # Variant 1: Full rewrite
    (
        "You are an expert prompt engineer. Based on the failure analysis below, "
        "rewrite the System Prompt to fix ALL identified issues. You may "
        "restructure, reorder, or add new instructions as needed. "
        "Preserve the core intent and any domain-specific knowledge. "
        "Return ONLY the new prompt — no commentary, no markdown code fences."
    ),
    # Variant 2: Conservative one-point fix
    (
        "You are an expert prompt engineer. Based on the failure analysis below, "
        "revise the System Prompt to address the SINGLE MOST CRITICAL issue. "
        "Make minimal changes — keep the prompt close in tone, length, and "
        "structure to the original. Do not address more than one issue. "
        "Return ONLY the new prompt — no commentary, no markdown code fences."
    ),
]


class APOEngine:
    """Automatic Prompt Optimization engine with multi-candidate scoring.

    Parameters
    ----------
    config : GlobalConfig
        Framework-wide configuration (judge model, APO hyper-params, etc.).
    llm : LLMClient
        Shared LLM client used to talk to the judge model.
    score_fn : callable, optional
        A user-supplied scoring function with signature
        ``(prompt: str, traces: List[Trace]) -> float``.
        When provided, beam search uses this to score candidate prompts
        instead of asking the judge model to score subjectively.
        Typically this runs the real task model and computes accuracy.
    """

    def __init__(
        self,
        config: GlobalConfig,
        llm: LLMClient,
        score_fn: Optional[Callable[[str, List[Trace]], float]] = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._score_fn = score_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, current_skill: Skill, traces: List[Trace]) -> Skill:
        """Run APO optimization and return an improved Skill.

        When ``beam_width == 1`` (default), runs the original single-track
        APO: one gradient → N candidates → pick best.

        When ``beam_width > 1``, runs **beam search** (aligned with
        Microsoft Agent-Lightning):
        1. Initialize beam with the current prompt
        2. For each round:
           a. For each prompt in beam, compute gradient from a sampled batch
           b. Generate ``branch_factor`` candidates per parent
           c. Score all (old beam + new candidates)
           d. Keep top ``beam_width`` prompts
        3. Return the best prompt seen across all rounds
        """
        diagnosed = [t for t in traces if t.feedback is not None]
        if not diagnosed:
            logger.info("No feedback traces available — skipping optimization.")
            return current_skill

        apo = self._config.apo

        if apo.beam_width <= 1:
            return self._optimize_single(current_skill, diagnosed)
        return self._optimize_beam(current_skill, diagnosed)

    # ------------------------------------------------------------------
    # Single-track APO (beam_width=1, backward compatible)
    # ------------------------------------------------------------------

    def _optimize_single(self, skill: Skill, diagnosed: List[Trace]) -> Skill:
        """Original single-track APO: gradient → N candidates → pick best."""
        batch_size = self._config.apo.gradient_accumulation_steps
        batch = diagnosed[-batch_size:]

        # Gradient
        gradient = self._compute_gradient(skill, batch)
        logger.debug("Gradient analysis:\n%s", gradient)

        # Generate N candidates in parallel
        num_candidates = self._config.apo.num_candidates
        edit_batches = [
            self._build_edit_messages(skill, gradient)
            for _ in range(num_candidates)
        ]
        responses = self._llm.generate_batch(
            edit_batches, role="rewrite",
        )
        candidates = [
            r.content if isinstance(r.content, str) else str(r.content)
            for r in responses if r.content
        ]
        logger.info("Generated %d candidates (requested %d)", len(candidates), num_candidates)
        if not candidates:
            logger.warning("All candidate generations failed — keeping original.")
            return skill

        # Score all (original + candidates)
        all_prompts = [skill.system_prompt] + candidates
        scores = self._score_prompts_batch(all_prompts, batch)

        logger.info("Current prompt score: %.2f", scores[0])
        for i, s in enumerate(scores[1:]):
            logger.info("Candidate %d score: %.2f", i + 1, s)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if best_idx == 0:
            logger.info("No candidate improved over current — keeping original.")
            return skill

        new_version = _increment_version(skill.version)
        logger.info("Accepted candidate %d (score=%.2f) → %s", best_idx, scores[best_idx], new_version)
        return skill.model_copy(update={
            "system_prompt": all_prompts[best_idx],
            "version": new_version,
        })

    # ------------------------------------------------------------------
    # Beam search APO (aligned with Agent-Lightning)
    # ------------------------------------------------------------------

    def _optimize_beam(self, skill: Skill, diagnosed: List[Trace]) -> Skill:
        """Beam search APO: maintain top-k prompts across rounds.

        Algorithm (per Agent-Lightning):
        1. beam = [current_prompt]
        2. for each round:
           a. sample parents from beam (up to beam_width)
           b. for each parent: gradient → branch_factor candidates
           c. pool = beam + all new candidates
           d. score pool, keep top beam_width
        3. return history best
        """
        apo = self._config.apo
        beam_width = apo.beam_width
        branch_factor = apo.branch_factor
        beam_rounds = apo.beam_rounds
        batch_size = apo.gradient_accumulation_steps

        beam: List[str] = [skill.system_prompt]
        history_best_prompt = skill.system_prompt
        history_best_score = -1.0

        for round_num in range(1, beam_rounds + 1):
            logger.info("Beam round %d/%d (beam size=%d)", round_num, beam_rounds, len(beam))

            # Pad beam to beam_width by replicating
            parents = beam[:]
            while len(parents) < beam_width:
                parents.append(random.choice(beam))
            parents = parents[:beam_width]

            # For each parent: gradient + branch_factor candidates (parallel)
            def _evolve_parent(pidx_prompt):
                pidx, parent_prompt = pidx_prompt
                batch = random.sample(diagnosed, min(batch_size, len(diagnosed)))
                parent_skill = skill.model_copy(update={"system_prompt": parent_prompt})
                gradient = self._compute_gradient(parent_skill, batch)

                edit_batches = [
                    self._build_edit_messages(parent_skill, gradient)
                    for _ in range(branch_factor)
                ]
                responses = self._llm.generate_batch(
                    edit_batches, role="rewrite",
                )
                new_prompts = [
                    r.content if isinstance(r.content, str) else str(r.content)
                    for r in responses if r.content
                ]
                logger.info(
                    "  Parent %d/%d → %d candidates",
                    pidx + 1, len(parents), len(new_prompts),
                )
                return new_prompts

            all_new: List[str] = []
            with ThreadPoolExecutor(max_workers=len(parents)) as pool:
                for prompts in pool.map(_evolve_parent, enumerate(parents)):
                    all_new.extend(prompts)

            if not all_new:
                logger.warning("Round %d: no candidates generated, keeping beam.", round_num)
                continue

            # Score: old beam + new candidates
            pool = beam + all_new
            # Use a validation batch for scoring
            val_batch = random.sample(diagnosed, min(batch_size, len(diagnosed)))
            scores = self._score_prompts_batch(pool, val_batch)

            # Keep top beam_width
            ranked = sorted(
                zip(pool, scores), key=lambda x: x[1], reverse=True,
            )
            beam = [p for p, _ in ranked[:beam_width]]
            top_score = ranked[0][1]

            logger.info(
                "  Round %d result: top=%.2f, beam=[%s]",
                round_num, top_score,
                ", ".join(f"{s:.2f}" for _, s in ranked[:beam_width]),
            )

            # Track history best
            if top_score > history_best_score:
                history_best_score = top_score
                history_best_prompt = ranked[0][0]

        # Return best prompt found across all rounds
        if history_best_prompt == skill.system_prompt:
            logger.info("Beam search: no improvement found — keeping original.")
            return skill

        new_version = _increment_version(skill.version)
        logger.info(
            "Beam search: best score=%.2f → %s", history_best_score, new_version,
        )
        return skill.model_copy(update={
            "system_prompt": history_best_prompt,
            "version": new_version,
        })

    # ------------------------------------------------------------------
    # Gradient computation (with template diversity)
    # ------------------------------------------------------------------

    def _compute_gradient(self, skill: Skill, traces: List[Trace]) -> str:
        """Ask the judge model to explain *why* the current prompt failed.

        Aligned with Agent-Lightning: the judge sees the full rollout trace
        including the model's prediction, ground truth (correction), and
        reward (score). Randomly selects one of 3 gradient templates.
        """
        rollout_descriptions: List[str] = []
        for t in traces:
            feedback: Feedback = t.feedback  # type: ignore[assignment]
            user_text = _extract_last_user_text(t.inputs)
            agent_text = (
                t.prediction.content
                if isinstance(t.prediction.content, str)
                else "[multimodal response]"
            )

            status = "FAIL" if feedback.score < 0.5 else "PASS"
            parts = [
                f"- [{status}] reward={feedback.score:.1f}",
                f"  Input: \"{user_text}\"",
                f"  Model output: \"{agent_text}\"",
            ]
            if feedback.correction:
                parts.append(f"  Expected answer: \"{feedback.correction}\"")
            if feedback.critique:
                parts.append(f"  Critique: {feedback.critique}")
            rollout_descriptions.append("\n".join(parts))

        rollouts_block = "\n".join(rollout_descriptions)

        # Randomly select gradient template
        system_prompt = random.choice(_GRADIENT_TEMPLATES)

        target_hint = ""
        if skill.target:
            target_hint = (
                f"\n\nIMPORTANT: The user has set an optimization target: "
                f"\"{skill.target}\". Analyze failures with this direction in mind."
            )

        messages = [
            Message(role="system", content=system_prompt + target_hint),
            Message(
                role="user",
                content=(
                    f"The current System Prompt is:\n\"\"\"\n{skill.system_prompt}\n\"\"\"\n\n"
                    f"Rollout results:\n{rollouts_block}\n\n"
                    "Analyze the failures now."
                ),
            ),
        ]
        response = self._llm.generate(
            messages, role="judge"
        )
        return response.content if isinstance(response.content, str) else str(response.content)

    # ------------------------------------------------------------------
    # Apply update (with edit strategy diversity)
    # ------------------------------------------------------------------

    def _build_edit_messages(self, skill: Skill, gradient: str) -> List[Message]:
        """Build the messages for a single edit/rewrite request.

        Each call randomly picks an edit template for diversity.
        """
        system_prompt = random.choice(_EDIT_TEMPLATES)

        target_hint = ""
        if skill.target:
            target_hint = (
                f" The user's optimization target is: \"{skill.target}\". "
                f"Make sure the rewritten prompt aligns with this direction."
            )

        return [
            Message(role="system", content=system_prompt + target_hint),
            Message(
                role="user",
                content=(
                    f"Current System Prompt:\n\"\"\"\n{skill.system_prompt}\n\"\"\"\n\n"
                    f"Failure Analysis:\n{gradient}\n\n"
                    "Rewrite the system prompt now."
                ),
            ),
        ]

    def _apply_update(self, skill: Skill, gradient: str) -> str:
        """Ask the judge model to rewrite the system prompt (sync, single call)."""
        messages = self._build_edit_messages(skill, gradient)
        response = self._llm.generate(
            messages, role="rewrite"
        )
        return response.content if isinstance(response.content, str) else str(response.content)

    # ------------------------------------------------------------------
    # Candidate scoring (real-model rollout + judge grading)
    # ------------------------------------------------------------------

    def _build_grade_messages(
        self, model_output: str, expected: str,
    ) -> List[Message]:
        """Build messages for judge-based grading (Agent-Lightning style).

        The judge compares the model's actual output against the expected
        answer and returns a 0-1 score.
        """
        return [
            Message(
                role="system",
                content=(
                    "You are a strict grader. Compare the task output against "
                    "the expected answer. Score the match on a 0-1 scale.\n"
                    "- 1.0 = correct or semantically equivalent\n"
                    "- 0.5 = partially correct\n"
                    "- 0.0 = wrong\n\n"
                    "Return ONLY a number between 0.0 and 1.0. Nothing else."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Task output:\n{model_output}\n\n"
                    f"Expected answer:\n{expected}\n\n"
                    "Score (0.0-1.0):"
                ),
            ),
        ]

    def _build_score_messages(self, prompt: str, traces: List[Trace]) -> List[Message]:
        """Build the messages for a single scoring request (legacy fallback)."""
        failure_summaries: List[str] = []
        for t in traces:
            fb: Feedback = t.feedback  # type: ignore[assignment]
            user_text = _extract_last_user_text(t.inputs)
            critique = fb.critique or fb.correction or f"score={fb.score}"
            failure_summaries.append(f"- \"{user_text}\" → {critique}")

        failures_block = "\n".join(failure_summaries)

        return [
            Message(
                role="system",
                content=(
                    "You are a prompt quality evaluator. Score how well the "
                    "given System Prompt would handle the listed failure cases. "
                    "Consider whether the prompt's instructions would prevent "
                    "each failure from recurring.\n\n"
                    "Return ONLY a number between 0.0 and 1.0. Nothing else."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"System Prompt to evaluate:\n\"\"\"\n{prompt}\n\"\"\"\n\n"
                    f"Known failure cases:\n{failures_block}\n\n"
                    "Score (0.0-1.0):"
                ),
            ),
        ]

    @staticmethod
    def _parse_score(raw: str) -> float:
        """Parse a score from judge model response.

        Tries JSON first, then regex extraction of any float.
        """
        import re
        raw = raw.strip()

        # Try JSON parse first
        try:
            if raw.startswith("{"):
                data = json.loads(raw)
                score = float(data.get("score", 0.5))
                return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Try direct float
        try:
            score = float(raw)
            return max(0.0, min(1.0, score))
        except ValueError:
            pass

        # Regex: find first float-like pattern
        m = re.search(r"(\d+\.?\d*)", raw)
        if m:
            score = float(m.group(1))
            if score <= 1.0:
                return max(0.0, score)
            elif score <= 100.0:
                return score / 100.0  # handle percentage

        logger.warning("Failed to parse score from: %s", raw[:100])
        return 0.5

    def _score_prompt(self, prompt: str, traces: List[Trace]) -> float:
        """Score a prompt (sync, single call). Used by non-batch code paths."""
        messages = self._build_score_messages(prompt, traces)
        response = self._llm.generate(
            messages, role="judge"
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        return self._parse_score(raw)

    def _score_prompts_batch(self, prompts: List[str], traces: List[Trace]) -> List[float]:
        """Score multiple prompts and return a list of scores.

        When ``score_fn`` is set, runs real task evaluation in parallel
        (aligned with Agent-Lightning). Otherwise falls back to judge scoring.
        """
        if self._score_fn is not None:
            with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
                scores = list(pool.map(
                    lambda p: self._score_fn(p, traces), prompts,
                ))
            return scores

        # Fallback: judge-based scoring
        batches = [self._build_score_messages(p, traces) for p in prompts]
        responses = self._llm.generate_batch(
            batches, role="judge",
        )
        return [
            self._parse_score(
                r.content if isinstance(r.content, str) else str(r.content)
            )
            for r in responses
        ]

    # ------------------------------------------------------------------
    # Tree-aware optimization
    # ------------------------------------------------------------------

    def analyze_split_need(
        self, skill: Skill, traces: List[Trace]
    ) -> Optional[List[Dict]]:
        """Ask the LLM whether *skill* should be split into sub-skills.

        Returns a list of child specs (``[{name, description, focus}]``)
        if a split is recommended, or ``None`` if not.
        """
        diagnosed = [t for t in traces if t.feedback is not None]
        if len(diagnosed) < 2:
            return None

        feedback_block = "\n".join(
            f"- Task: \"{_extract_last_user_text(t.inputs)}\"  "
            f"Critique: \"{t.feedback.critique or 'low score'}\""
            for t in diagnosed
        )

        messages = [
            Message(
                role="system",
                content=(
                    "You are a prompt architecture expert. Analyze the feedback "
                    "below and decide whether this single System Prompt should "
                    "be SPLIT into specialised child prompts for different "
                    "domains/task-types.\n\n"
                    "If YES, return a JSON array of child specs:\n"
                    '[{"name": "...", "description": "...", "focus": "..."}]\n\n'
                    "If NO, return exactly: null\n\n"
                    "Return ONLY the JSON (or null). No commentary."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Current System Prompt:\n\"\"\"\n{skill.system_prompt}\n\"\"\"\n\n"
                    f"Feedback from different tasks:\n{feedback_block}"
                ),
            ),
        ]
        response = self._llm.generate(
            messages, role="judge"
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        raw = raw.strip()

        if raw.lower() == "null" or not raw.startswith("["):
            return None

        try:
            specs = json.loads(raw)
            if isinstance(specs, list) and len(specs) >= 2:
                return specs
        except json.JSONDecodeError:
            logger.warning("Failed to parse split analysis JSON: %s", raw[:200])
        return None

    def generate_child_prompts(
        self, parent_skill: Skill, children_specs: List[Dict]
    ) -> List[Dict]:
        """Generate specialised System Prompts for each child spec.

        Returns the original specs enriched with ``system_prompt`` keys.
        """
        specs_json = json.dumps(children_specs, ensure_ascii=False)
        messages = [
            Message(
                role="system",
                content=(
                    "You are an expert prompt engineer. Given a parent System "
                    "Prompt and a list of child specialisation specs, write a "
                    "tailored System Prompt for each child.\n\n"
                    "Return a JSON array where each item has: "
                    '"name", "description", "system_prompt".\n\n'
                    "Return ONLY valid JSON. No commentary."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Parent System Prompt:\n\"\"\"\n{parent_skill.system_prompt}\n\"\"\"\n\n"
                    f"Child specs:\n{specs_json}"
                ),
            ),
        ]
        response = self._llm.generate(
            messages, role="rewrite"
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        raw = raw.strip()

        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            logger.warning("Failed to parse child prompts JSON: %s", raw[:200])

        # Fallback: return specs with parent prompt
        for spec in children_specs:
            spec.setdefault("system_prompt", parent_skill.system_prompt)
        return children_specs

    def evolve_tree(
        self,
        tree: "SkillTree",
        traces: List[Trace],
        auto_split: bool = True,
        resume: Optional[ResumeState] = None,
        on_node_done: Optional[Callable] = None,
    ) -> "SkillTree":
        """Recursively optimise every skill in a tree with resume support.

        Strategy: optimise leaves first, then parents (bottom-up).
        If *auto_split* is ``True``, the engine will analyse each node
        for potential splits and apply them automatically.

        Traces are routed to the correct node via ``Trace.node_path``.
        Traces with ``node_path=None`` (legacy) are used by all nodes.

        Returns the mutated tree (same object).
        """
        total = _count_nodes(tree.root)
        skipped = 0
        if resume:
            skipped = sum(
                1 for _ in _iter_dotpaths(tree.root, "")
                if resume.is_node_done(_)
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[node]}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                "Evolving",
                total=total,
                completed=skipped,
                node="starting …",
            )
            self._evolve_node(
                tree.root,
                traces,
                tree=tree,
                auto_split=auto_split,
                resume=resume,
                on_node_done=on_node_done,
                _path_prefix="",
                _progress=progress,
                _task_id=task_id,
            )
        return tree

    def _evolve_node(
        self,
        node: "SkillNode",
        traces: List[Trace],
        tree: "SkillTree",
        auto_split: bool = True,
        resume: Optional[ResumeState] = None,
        on_node_done: Optional[Callable] = None,
        _path_prefix: str = "",
        _progress: Optional[Progress] = None,
        _task_id: Optional[int] = None,
    ) -> None:
        """Recursively optimise a single node and its children.

        Order:
        1. Recurse into existing children (bottom-up)
        2. Optimise this node (with multi-candidate scoring)
        3. Auto-split if leaf node with conflicting feedback
        4. Recurse into newly created children (so they get optimised too)
        5. Save checkpoint
        """
        from treeskill.skill_tree import SkillNode  # noqa: deferred

        dotpath = f"{_path_prefix}.{node.name}" if _path_prefix else node.name

        # Step 1: Recurse into existing children (bottom-up)
        for child in list(node.children.values()):
            self._evolve_node(
                child, traces,
                tree=tree,
                auto_split=auto_split,
                resume=resume,
                on_node_done=on_node_done,
                _path_prefix=dotpath,
                _progress=_progress,
                _task_id=_task_id,
            )

        # Update progress bar description
        if _progress is not None and _task_id is not None:
            _progress.update(_task_id, node=dotpath)

        # Skip if already done in a previous (interrupted) run
        if resume and resume.is_node_done(dotpath):
            logger.info("Skipping already-optimised node: %s", dotpath)
            if _progress is not None and _task_id is not None:
                _progress.update(_task_id, advance=1, node=f"{dotpath} (cached)")
            return

        # Step 2: Route traces — only use traces belonging to this node
        node_traces = _filter_traces_for_node(traces, dotpath)
        diagnosed = [t for t in node_traces if t.feedback is not None]
        if not diagnosed:
            if resume:
                resume.mark_node_done(dotpath)
            if _progress is not None and _task_id is not None:
                _progress.update(_task_id, advance=1, node=f"{dotpath} (no traces)")
            return

        logger.info(
            "Optimising '%s' with %d/%d traces (node-specific/total)",
            dotpath, len(diagnosed), len([t for t in traces if t.feedback]),
        )
        node.skill = self.optimize(node.skill, diagnosed)

        # Step 3: Auto-split (only for LEAF nodes to avoid overwriting children)
        new_children: List[SkillNode] = []
        if auto_split and node.is_leaf and len(diagnosed) >= 2:
            specs = self.analyze_split_need(node.skill, diagnosed)
            if specs:
                enriched = self.generate_child_prompts(node.skill, specs)
                child_names = []
                for spec in enriched:
                    if "name" not in spec:
                        continue
                    child_skill = node.skill.model_copy(
                        update={
                            "name": spec["name"],
                            "description": spec.get("description", ""),
                            "system_prompt": spec.get(
                                "system_prompt", node.skill.system_prompt
                            ),
                            "version": "v1.0",
                        }
                    )
                    child_node = SkillNode(
                        name=spec["name"],
                        skill=child_skill,
                    )
                    node.children[spec["name"]] = child_node
                    new_children.append(child_node)
                    child_names.append(spec["name"])
                if child_names:
                    logger.info(
                        "Auto-split '%s' into %d children: %s",
                        node.name, len(child_names), child_names,
                    )
                    if resume:
                        resume.mark_node_split(dotpath, child_names)
                    # Grow the progress bar to account for new nodes
                    if _progress is not None and _task_id is not None:
                        _progress.update(_task_id, total=(_progress.tasks[_task_id].total or 0) + len(child_names))

        # Step 4: Optimise newly created children
        for child_node in new_children:
            self._evolve_node(
                child_node, traces,
                tree=tree,
                auto_split=False,
                resume=resume,
                on_node_done=on_node_done,
                _path_prefix=dotpath,
                _progress=_progress,
                _task_id=_task_id,
            )

        # Step 5: Checkpoint
        if resume:
            tree.save()
            resume.mark_node_done(dotpath)
            logger.info("Progress saved after node: %s", dotpath)

        if _progress is not None and _task_id is not None:
            _progress.update(_task_id, advance=1, node=f"{dotpath} ✓")

        if on_node_done:
            on_node_done(dotpath, node)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _increment_version(version: str) -> str:
    """Bump the last numeric segment of a version string.

    Examples: ``v1.0`` → ``v1.1``, ``1.0`` → ``1.1``,
    ``v1.0.2`` → ``v1.0.3``, ``v2`` → ``v3``.
    """
    prefix = ""
    v = version
    if v.startswith("v"):
        prefix = "v"
        v = v[1:]

    parts = v.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            parts[i] = str(int(parts[i]) + 1)
            return prefix + ".".join(parts)

    return version + ".1"


def _extract_last_user_text(messages: List[Message]) -> str:
    """Pull the text content from the last user message."""
    for msg in reversed(messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                return msg.content
            texts = [
                p.text for p in msg.content if hasattr(p, "text")
            ]
            return " ".join(texts) if texts else "[image-only input]"
    return "[no user message]"


def _count_nodes(node: "SkillNode") -> int:
    """Count total nodes in the subtree (including *node* itself)."""
    return 1 + sum(_count_nodes(c) for c in node.children.values())


def _iter_dotpaths(node: "SkillNode", prefix: str):
    """Yield all dotpaths in the subtree (DFS, children-first like evolve)."""
    dotpath = f"{prefix}.{node.name}" if prefix else node.name
    for child in node.children.values():
        yield from _iter_dotpaths(child, dotpath)
    yield dotpath


def _filter_traces_for_node(traces: List[Trace], dotpath: str) -> List[Trace]:
    """Return traces that belong to *dotpath*.

    Routing rules:
    - ``trace.node_path == dotpath`` → exact match, always included.
    - ``trace.node_path`` starts with ``dotpath + "."`` → child trace,
      included for the parent so it can see its subtree's feedback.
    - ``trace.node_path is None`` → legacy trace (no routing info),
      included for ALL nodes to preserve backward compatibility.
    """
    result: List[Trace] = []
    prefix = dotpath + "."
    for t in traces:
        if t.node_path is None:
            # Legacy trace — no routing info, use everywhere
            result.append(t)
        elif t.node_path == dotpath or t.node_path.startswith(prefix):
            # Exact match or descendant
            result.append(t)
    return result
