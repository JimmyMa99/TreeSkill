"""Annotate — dataset-driven human-in-the-loop feedback collection.

Runs a Skill against a dataset, shows predictions to the user, and
collects feedback (auto-judge, human, or hybrid).  The resulting
Traces can feed APO optimization or be exported as DPO pairs.

This implements a **feedback control loop** where auto-judge and human
are interchangeable controllers:

    Dataset ──► Model Prediction ──► Controller (auto / human) ──► Trace
                                         │
                                    /auto  /manual  (toggle)

In auto mode, the LM judge scores every prediction automatically.
The human can override any judgment by typing.  In manual mode,
every prediction pauses for human feedback.  Human feedback serves
as a preference anchor that calibrates the auto-judge.

Usage::

    python -m treeskill.main --annotate --dataset train.jsonl --skill my-skill
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

from treeskill.config import GlobalConfig
from treeskill.dataset import DataLoader, Sample
from treeskill.llm import LLMClient
from treeskill.schema import Feedback, Message, Skill, Trace
from treeskill.skill import compile_messages
from treeskill.storage import TraceStorage

logger = logging.getLogger(__name__)

_THEME = Theme({
    "info": "dim cyan",
    "success": "bold green",
    "warning": "bold magenta",
    "error": "bold red",
})

_HELP_TEXT = """\
[bold]Annotation Commands[/bold]

  [cyan]Enter[/cyan]          accept auto-judge result (auto mode) / skip (manual mode)
  [cyan]<text>[/cyan]         provide critique (natural language feedback)
  [cyan]/c <text>[/cyan]      provide correction (ideal response → DPO chosen)
  [cyan]/auto[/cyan]          switch to auto-judge mode
  [cyan]/manual[/cyan]        switch to manual mode
  [cyan]/skip[/cyan]          skip this sample
  [cyan]/quit[/cyan]          stop annotation and save
  [cyan]/help[/cyan]          show this help\
"""


class AnnotateCLI:
    """Interactive annotation loop for dataset-driven feedback collection.

    Parameters
    ----------
    config : GlobalConfig
    llm : LLMClient
    skill : Skill
    dataset : DataLoader
    storage : TraceStorage
    auto : bool
        If True (default), start in auto-judge mode.
    """

    def __init__(
        self,
        config: GlobalConfig,
        llm: LLMClient,
        skill: Skill,
        dataset: DataLoader,
        storage: TraceStorage,
        *,
        auto: bool = True,
    ) -> None:
        self._config = config
        self._llm = llm
        self._skill = skill
        self._dataset = dataset
        self._storage = storage
        self._auto = auto
        self._session_id: Optional[str] = None
        self._console = Console(theme=_THEME)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> List[Trace]:
        """Run the annotation loop. Returns all collected Traces."""
        self._session_id = str(uuid.uuid4())
        samples = list(self._dataset)
        traces: List[Trace] = []
        skipped = 0

        mode_label = "[green]AUTO[/green]" if self._auto else "[yellow]MANUAL[/yellow]"
        self._console.print(Panel(
            f"[bold]TreeSkill Annotator[/bold]\n"
            f"Skill: [cyan]{self._skill.name}[/cyan] ({self._skill.version})\n"
            f"Dataset: [cyan]{len(samples)}[/cyan] samples\n"
            f"Mode: {mode_label}\n\n"
            f"[dim]Type /help for commands.  "
            f"/auto and /manual switch judge mode.[/dim]",
            title="🏷️ TreeSkill Annotate",
            border_style="bright_blue",
        ))

        for idx, sample in enumerate(samples):
            self._console.print(f"\n[bold]--- Sample {idx + 1}/{len(samples)} ---[/bold]")

            # Step 1: Show input
            user_text = _extract_text(sample.input_messages)
            gt_text = _content_to_str(sample.ground_truth.content)
            self._console.print(Panel(
                user_text[:500] + ("..." if len(user_text) > 500 else ""),
                title="Input",
                border_style="blue",
            ))

            # Step 2: Generate prediction
            with self._console.status("[dim]Predicting...[/dim]"):
                compiled = compile_messages(self._skill, sample.input_messages)
                prediction = self._llm.generate(
                    compiled, model=self._config.llm.model,
                )
            pred_text = _content_to_str(prediction.content)

            # Step 3: Show prediction vs ground truth
            table = Table(show_header=True, border_style="cyan")
            table.add_column("Model Prediction", style="white", ratio=1)
            table.add_column("Ground Truth", style="green", ratio=1)
            table.add_row(pred_text[:200], gt_text[:200])
            self._console.print(table)

            # Step 4: Get feedback (auto or human)
            feedback = self._collect_feedback(sample, prediction, idx, len(samples))
            if feedback is None:
                skipped += 1
                continue
            if feedback == "QUIT":
                break

            # Step 5: Store trace
            trace = Trace(
                session_id=self._session_id,
                inputs=compiled,
                prediction=prediction,
                feedback=feedback,
            )
            self._storage.append(trace)
            traces.append(trace)

            score_str = f"{feedback.score:.1f}"
            has_correction = "+" if feedback.correction else ""
            self._console.print(
                f"[success]Recorded[/success] score={score_str} "
                f"{'[cyan](has correction → DPO)[/cyan]' if has_correction else ''}"
            )

        # Summary
        self._show_summary(traces, skipped, len(samples))
        return traces

    # ------------------------------------------------------------------
    # Feedback collection
    # ------------------------------------------------------------------

    def _collect_feedback(
        self, sample: Sample, prediction: Message, idx: int, total: int,
    ) -> Optional[Feedback | str]:
        """Collect feedback for one sample.

        Returns Feedback, None (skip), or "QUIT" string.
        """
        mode_tag = "[green]AUTO[/green]" if self._auto else "[yellow]MANUAL[/yellow]"

        if self._auto:
            # Auto-judge first
            with self._console.status("[dim]Auto-judging...[/dim]"):
                auto_fb = self._auto_judge(sample, prediction)
            self._console.print(
                f"  {mode_tag} score={auto_fb.score:.2f}  "
                f"critique: [dim]{(auto_fb.critique or '')[:100]}[/dim]"
            )
            prompt_hint = "Enter=accept, type=override, /c <correction>, /skip, /quit"
        else:
            auto_fb = None
            prompt_hint = "Type critique, /c <correction>, Enter=skip, /quit"

        self._console.print(f"[dim]{prompt_hint}[/dim]")

        try:
            raw = Prompt.ask(f"  {mode_tag} Feedback")
        except (EOFError, KeyboardInterrupt):
            return "QUIT"

        raw = raw.strip()

        # Command dispatch
        if raw.lower() == "/quit":
            return "QUIT"
        if raw.lower() == "/skip":
            return None
        if raw.lower() == "/auto":
            self._auto = True
            self._console.print("[success]Switched to AUTO mode[/success]")
            return self._collect_feedback(sample, prediction, idx, total)
        if raw.lower() == "/manual":
            self._auto = False
            self._console.print("[success]Switched to MANUAL mode[/success]")
            return self._collect_feedback(sample, prediction, idx, total)
        if raw.lower() == "/help":
            self._console.print(Panel(_HELP_TEXT, border_style="cyan"))
            return self._collect_feedback(sample, prediction, idx, total)

        # /c <correction> — provide ideal response
        if raw.lower().startswith("/c "):
            correction = raw[3:].strip()
            gt_text = _content_to_str(sample.ground_truth.content)
            return Feedback(
                score=0.1,
                critique="Human correction provided",
                correction=correction or gt_text,
            )

        # Empty input
        if not raw:
            if self._auto and auto_fb is not None:
                return auto_fb  # accept auto-judge
            return None  # skip in manual mode

        # Non-empty text = human critique (override auto)
        gt_text = _content_to_str(sample.ground_truth.content)
        pred_text = _content_to_str(prediction.content)
        is_correct = pred_text.strip().upper() == gt_text.strip().upper()

        return Feedback(
            score=0.8 if is_correct else 0.2,
            critique=raw,
            correction=gt_text if not is_correct else None,
        )

    # ------------------------------------------------------------------
    # Auto-judge
    # ------------------------------------------------------------------

    def _auto_judge(self, sample: Sample, prediction: Message) -> Feedback:
        """Score a prediction using the LM judge."""
        rubric = self._config.reward.default_rubric or (
            "Score the response on accuracy compared to the reference answer.\n"
            "1.0 = perfect match\n"
            "0.5 = partially correct\n"
            "0.0 = completely wrong"
        )

        gt_text = _content_to_str(sample.ground_truth.content)
        pred_text = _content_to_str(prediction.content)
        user_text = _extract_text(sample.input_messages)

        judge_model = (
            self._config.reward.model
            or self._config.llm.judge_model
        )

        messages = [
            Message(
                role="system",
                content=(
                    "You are an expert evaluator. Score the AI's response "
                    "against the reference answer using this rubric:\n\n"
                    f"{rubric}\n\n"
                    "Return a JSON object: "
                    '{"score": <float 0.0-1.0>, "critique": "<brief explanation>"}\n'
                    "Return ONLY valid JSON."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"## Input\n{user_text[:500]}\n\n"
                    f"## Reference\n{gt_text}\n\n"
                    f"## Model Response\n{pred_text}\n\n"
                    "Evaluate now."
                ),
            ),
        ]
        response = self._llm.generate(messages, model=judge_model)
        return self._parse_judge_response(
            _content_to_str(response.content), sample,
        )

    @staticmethod
    def _parse_judge_response(raw: str, sample: Sample) -> Feedback:
        """Parse judge LLM response into Feedback."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
            score = max(0.0, min(1.0, float(data.get("score", 0.5))))
            critique = data.get("critique", "")
            gt_text = _content_to_str(sample.ground_truth.content)
            return Feedback(
                score=score,
                critique=critique or None,
                correction=gt_text if score < 0.5 else None,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        m = re.search(r"(\d+\.?\d*)", raw)
        if m:
            score = float(m.group(1))
            if score > 1.0 and score <= 100.0:
                score /= 100.0
            score = max(0.0, min(1.0, score))
            return Feedback(score=score, critique=raw[:200])

        return Feedback(score=0.5, critique="Judge response unparseable")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _show_summary(
        self, traces: List[Trace], skipped: int, total: int,
    ) -> None:
        scored = [t for t in traces if t.feedback is not None]
        avg = sum(t.feedback.score for t in scored) / len(scored) if scored else 0
        dpo_count = sum(
            1 for t in traces
            if t.feedback and t.feedback.correction
        )

        self._console.print(Panel(
            f"Annotated: [bold]{len(traces)}[/bold] / {total}\n"
            f"Skipped: {skipped}\n"
            f"Avg score: [cyan]{avg:.2f}[/cyan]\n"
            f"DPO pairs: [cyan]{dpo_count}[/cyan]\n\n"
            f"[dim]Run --optimize to use these traces for APO.\n"
            f"Run /export-dpo in chat mode to export DPO data.[/dim]",
            title="📊 Annotation Summary",
            border_style="green",
        ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        if hasattr(part, "text"):
            texts.append(part.text)
    return " ".join(texts) if texts else ""


def _extract_text(messages: List[Message]) -> str:
    parts = []
    for msg in messages:
        text = _content_to_str(msg.content)
        parts.append(f"{msg.role}: {text}")
    return "\n".join(parts)
