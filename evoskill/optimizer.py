"""APO Optimizer — Automatic Prompt Optimization via Textual Gradient Descent.

Implements the "brain" of the Evo-Framework.  Given a Skill and a batch
of traces (with feedback), it:

1. **Diagnoses** failures using critique / correction annotations.
2. **Computes a "gradient"** by asking a judge model *why* the prompt failed.
3. **Updates** the system prompt by asking the judge to rewrite it.
4. **Applies** the change, returning a new Skill with an incremented version.

Tree-aware extensions:
- ``analyze_split_need`` — detect when a skill should be split
- ``generate_child_prompts`` — create specialised child prompts
- ``evolve_tree`` — recursively optimise a whole SkillTree
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from evoskill.config import GlobalConfig
from evoskill.llm import LLMClient
from evoskill.schema import Feedback, Message, Skill, Trace

if TYPE_CHECKING:
    from evoskill.skill_tree import SkillNode, SkillTree

logger = logging.getLogger(__name__)


class APOEngine:
    """Automatic Prompt Optimization engine.

    Parameters
    ----------
    config : GlobalConfig
        Framework-wide configuration (judge model, APO hyper-params, etc.).
    llm : LLMClient
        Shared LLM client used to talk to the judge model.
    """

    def __init__(self, config: GlobalConfig, llm: LLMClient) -> None:
        self._config = config
        self._llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, current_skill: Skill, traces: List[Trace]) -> Skill:
        """Run a single APO cycle and return an improved Skill.

        Parameters
        ----------
        current_skill :
            The skill whose system prompt will be refined.
        traces :
            Interaction traces **that already have feedback attached**.
            Traces without feedback are silently ignored.
        """
        # Step 1 — Diagnosis: select traces with feedback
        diagnosed = [t for t in traces if t.feedback is not None]
        if not diagnosed:
            logger.info("No feedback traces available — skipping optimization.")
            return current_skill

        # Respect gradient_accumulation_steps — take the N most recent
        batch_size = self._config.apo.gradient_accumulation_steps
        diagnosed = diagnosed[-batch_size:]

        # Step 2 — Gradient: ask the judge WHY the prompt failed
        gradient = self._compute_gradient(current_skill, diagnosed)
        logger.debug("Gradient analysis:\n%s", gradient)

        # Step 3 — Update: ask the judge to REWRITE the prompt
        new_prompt = self._apply_update(current_skill, gradient)
        logger.debug("Updated system prompt:\n%s", new_prompt)

        # Step 4 — Apply: build the new Skill with bumped version
        new_version = _increment_version(current_skill.version)
        return current_skill.model_copy(
            update={
                "system_prompt": new_prompt,
                "version": new_version,
            }
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_gradient(self, skill: Skill, traces: List[Trace]) -> str:
        """Ask the judge model to explain *why* the current prompt failed."""
        failure_descriptions: List[str] = []
        for t in traces:
            feedback: Feedback = t.feedback  # type: ignore[assignment]
            feedback_text = ""
            if feedback.correction:
                feedback_text = f"The ideal response should have been: {feedback.correction}"
            elif feedback.critique:
                feedback_text = f"Critique: {feedback.critique}"
            else:
                feedback_text = f"Score: {feedback.score}"

            # Summarize the user turn (last user message)
            user_text = _extract_last_user_text(t.inputs)
            agent_text = (
                t.prediction.content
                if isinstance(t.prediction.content, str)
                else "[multimodal response]"
            )

            failure_descriptions.append(
                f"- User said: \"{user_text}\"\n"
                f"  Agent said: \"{agent_text}\"\n"
                f"  Feedback: {feedback_text}"
            )

        failures_block = "\n".join(failure_descriptions)

        target_hint = ""
        if skill.target:
            target_hint = (
                f"\n\nIMPORTANT: The user has set an optimization target: "
                f"\"{skill.target}\". Analyze failures with this direction in mind."
            )

        messages = [
            Message(
                role="system",
                content=(
                    "You are an expert prompt engineer. Your task is to "
                    "analyze conversation failures and explain WHY the "
                    "system prompt failed to guide the agent correctly."
                    + target_hint
                ),
            ),
            Message(
                role="user",
                content=(
                    f"The current System Prompt is:\n\"\"\"\n{skill.system_prompt}\n\"\"\"\n\n"
                    f"Here are the failures:\n{failures_block}\n\n"
                    "Explain concisely why the system prompt led to these failures."
                ),
            ),
        ]
        response = self._llm.generate(
            messages, model=self._config.llm.judge_model
        )
        return response.content if isinstance(response.content, str) else str(response.content)

    def _apply_update(self, skill: Skill, gradient: str) -> str:
        """Ask the judge model to rewrite the system prompt."""
        target_hint = ""
        if skill.target:
            target_hint = (
                f" The user's optimization target is: \"{skill.target}\". "
                f"Make sure the rewritten prompt aligns with this direction."
            )

        messages = [
            Message(
                role="system",
                content=(
                    "You are an expert prompt engineer. Based on the analysis "
                    "of failures below, rewrite the System Prompt to fix the "
                    "identified issues WITHOUT breaking existing functionality. "
                    "Return ONLY the new prompt — no commentary, no markdown "
                    "code fences, just the raw prompt text."
                    + target_hint
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Current System Prompt:\n\"\"\"\n{skill.system_prompt}\n\"\"\"\n\n"
                    f"Failure Analysis:\n{gradient}\n\n"
                    "Rewrite the system prompt now."
                ),
            ),
        ]
        response = self._llm.generate(
            messages, model=self._config.llm.judge_model
        )
        return response.content if isinstance(response.content, str) else str(response.content)

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
            messages, model=self._config.llm.judge_model
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
            messages, model=self._config.llm.judge_model
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
    ) -> "SkillTree":
        """Recursively optimise every skill in a tree.

        Strategy: optimise leaves first, then parents (bottom-up).
        If *auto_split* is ``True``, the engine will analyse each node
        for potential splits and apply them automatically.

        Returns the mutated tree (same object).
        """
        from evoskill.skill_tree import SkillTree  # noqa: deferred

        self._evolve_node(tree.root, traces, auto_split=auto_split)
        return tree

    def _evolve_node(
        self,
        node: "SkillNode",
        traces: List[Trace],
        auto_split: bool = True,
    ) -> None:
        """Recursively optimise a single node and its children."""
        from evoskill.skill_tree import SkillNode  # noqa: deferred

        # Recurse into children first (bottom-up)
        for child in list(node.children.values()):
            self._evolve_node(child, traces, auto_split=auto_split)

        # Optimise this node's skill
        diagnosed = [t for t in traces if t.feedback is not None]
        if not diagnosed:
            return

        node.skill = self.optimize(node.skill, diagnosed)

        # Auto-split analysis (only for leaf or shallow nodes)
        if auto_split and len(diagnosed) >= 2:
            specs = self.analyze_split_need(node.skill, diagnosed)
            if specs:
                enriched = self.generate_child_prompts(node.skill, specs)
                for spec in enriched:
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
                logger.info(
                    "Auto-split '%s' into %d children: %s",
                    node.name,
                    len(enriched),
                    [s["name"] for s in enriched],
                )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _increment_version(version: str) -> str:
    """Bump a ``vX.Y`` style version string → ``vX.(Y+1)``."""
    if not version.startswith("v"):
        return version + ".1"
    parts = version[1:].split(".")
    if len(parts) == 2 and parts[1].isdigit():
        return f"v{parts[0]}.{int(parts[1]) + 1}"
    return version + ".1"


def _extract_last_user_text(messages: List[Message]) -> str:
    """Pull the text content from the last user message."""
    for msg in reversed(messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                return msg.content
            # Multimodal: collect text parts
            texts = [
                p.text for p in msg.content if hasattr(p, "text")
            ]
            return " ".join(texts) if texts else "[image-only input]"
    return "[no user message]"
