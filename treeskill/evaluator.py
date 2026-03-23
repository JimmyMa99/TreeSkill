"""Evaluator — run a Skill against a dataset and produce scored Traces.

Pipeline::

    Dataset (List[Sample])
        → Runner (generate predictions with target LLM)
        → Judge  (score predictions vs ground truth with rubric)
        → List[Trace] (ready for APO)

The rubric comes from ``config.reward.default_rubric``.
This follows the RL analogy: dataset = environment, rubric = reward function.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from treeskill.config import GlobalConfig
from treeskill.dataset import DataLoader, Sample
from treeskill.llm import LLMClient
from treeskill.schema import Feedback, Message, Skill, Trace
from treeskill.skill import compile_messages

logger = logging.getLogger(__name__)

_DEFAULT_RUBRIC = (
    "Score the response on accuracy and completeness compared to the "
    "reference answer.\n"
    "1.0 = perfect match in meaning and quality\n"
    "0.5 = partially correct or missing key details\n"
    "0.0 = completely wrong or irrelevant"
)


class Evaluator:
    """Runs a Skill against a dataset and produces scored Traces.

    Parameters
    ----------
    config : GlobalConfig
        Framework configuration (LLM endpoints, reward/rubric settings).
    llm : LLMClient
        Shared LLM client for both prediction and judging.
    """

    def __init__(self, config: GlobalConfig, llm: LLMClient) -> None:
        self._config = config
        self._llm = llm

    def evaluate(
        self,
        skill: Skill,
        dataset: DataLoader,
        *,
        max_samples: Optional[int] = None,
        node_path: Optional[str] = None,
    ) -> List[Trace]:
        """Run evaluation and return Traces with Feedback.

        Parameters
        ----------
        skill : Skill
            The skill whose system_prompt is being evaluated.
        dataset : DataLoader
            Loaded dataset (ChatML JSONL).
        max_samples : int | None
            Cap the number of samples to evaluate.
        node_path : str | None
            If set, all generated Traces will have this node_path.

        Returns
        -------
        List[Trace]
            One Trace per sample, each with Feedback (score + critique).
        """
        samples = list(dataset)
        if max_samples and max_samples < len(samples):
            import random
            samples = random.sample(samples, max_samples)

        if not samples:
            logger.warning("No samples to evaluate.")
            return []

        # Step 1: Generate predictions (batch)
        logger.info("Generating predictions for %d samples …", len(samples))
        predictions = self._run_predictions(skill, samples)

        # Step 2: Judge predictions vs ground truth (batch)
        rubric = (
            self._config.reward.default_rubric
            or _DEFAULT_RUBRIC
        )
        logger.info("Judging %d predictions with rubric …", len(samples))
        feedbacks = self._judge_batch(samples, predictions, rubric)

        # Step 3: Assemble Traces
        traces: List[Trace] = []
        for sample, prediction, feedback in zip(samples, predictions, feedbacks):
            compiled = compile_messages(skill, sample.input_messages)
            traces.append(Trace(
                inputs=compiled,
                prediction=prediction,
                feedback=feedback,
                node_path=node_path,
            ))

        scored = [t for t in traces if t.feedback is not None]
        avg = sum(t.feedback.score for t in scored) / len(scored) if scored else 0
        logger.info(
            "Evaluation complete: %d samples, avg_score=%.2f",
            len(scored), avg,
        )
        return traces

    # ------------------------------------------------------------------
    # Prediction generation
    # ------------------------------------------------------------------

    def _run_predictions(
        self, skill: Skill, samples: List[Sample],
    ) -> List[Message]:
        """Generate predictions for all samples using the target LLM."""
        message_batches = [
            compile_messages(skill, s.input_messages)
            for s in samples
        ]
        return self._llm.generate_batch(
            message_batches,
            role="actor",
        )

    # ------------------------------------------------------------------
    # Judging
    # ------------------------------------------------------------------

    def _judge_batch(
        self,
        samples: List[Sample],
        predictions: List[Message],
        rubric: str,
    ) -> List[Feedback]:
        """Score all predictions against ground truths using the Judge LLM."""
        judge_batches = [
            self._build_judge_messages(sample, prediction, rubric)
            for sample, prediction in zip(samples, predictions)
        ]
        responses = self._llm.generate_batch(
            judge_batches, role="judge",
        )

        feedbacks: List[Feedback] = []
        for resp, sample in zip(responses, samples):
            raw = resp.content if isinstance(resp.content, str) else str(resp.content)
            fb = self._parse_judge_response(raw, sample)
            feedbacks.append(fb)
        return feedbacks

    def _build_judge_messages(
        self,
        sample: Sample,
        prediction: Message,
        rubric: str,
    ) -> List[Message]:
        """Build the judge prompt for a single sample."""
        # Extract text from messages for display
        user_text = _extract_text(sample.input_messages)
        gt_text = _extract_content_text(sample.ground_truth.content)
        pred_text = _extract_content_text(prediction.content)

        return [
            Message(
                role="system",
                content=(
                    "You are an expert evaluator. Score the AI assistant's "
                    "response against the reference answer using the following rubric:\n\n"
                    f"{rubric}\n\n"
                    "Return a JSON object with exactly these fields:\n"
                    '{"score": <float 0.0-1.0>, "critique": "<brief explanation>"}\n'
                    "Return ONLY valid JSON. No commentary."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"## Input\n{user_text}\n\n"
                    f"## Reference Answer\n{gt_text}\n\n"
                    f"## Model Response\n{pred_text}\n\n"
                    "Evaluate now."
                ),
            ),
        ]

    @staticmethod
    def _parse_judge_response(raw: str, sample: Sample) -> Feedback:
        """Parse judge LLM response into a Feedback object."""
        raw = raw.strip()

        # Try JSON parse
        try:
            # Handle markdown code fences
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)
            score = float(data.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            critique = data.get("critique", "")
            correction_text = _extract_content_text(sample.ground_truth.content)
            return Feedback(
                score=score,
                critique=critique or None,
                correction=correction_text if score < 0.5 else None,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: try to extract a float score
        m = re.search(r"(\d+\.?\d*)", raw)
        if m:
            score = float(m.group(1))
            if score > 1.0 and score <= 100.0:
                score = score / 100.0
            score = max(0.0, min(1.0, score))
            return Feedback(score=score, critique=raw[:200])

        logger.warning("Failed to parse judge response: %s", raw[:100])
        return Feedback(score=0.5, critique="Judge response unparseable")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_content_text(content) -> str:
    """Extract plain text from a Message content (str or List[ContentPart])."""
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        if hasattr(part, "text"):
            texts.append(part.text)
        elif hasattr(part, "type"):
            texts.append(f"[{part.type}]")
    return " ".join(texts) if texts else "[non-text content]"


def _extract_text(messages: List[Message]) -> str:
    """Concatenate text from a list of messages for display."""
    parts = []
    for msg in messages:
        text = _extract_content_text(msg.content)
        parts.append(f"{msg.role}: {text}")
    return "\n".join(parts)
