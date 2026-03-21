"""Trace Storage — append-only JSONL persistence for interaction traces.

Each line in the JSONL file is a self-contained JSON object representing
a single ``Trace`` record.  This makes it safe for concurrent appenders
and trivially streamable.

DPO export: traces with ``feedback.correction`` can be exported as
preference pairs for Direct Preference Optimization fine-tuning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from evoskill.config import StorageConfig
from evoskill.schema import Message, Trace

logger = logging.getLogger(__name__)


class TraceStorage:
    """Append-only JSONL store for ``Trace`` objects.

    Parameters
    ----------
    config : StorageConfig
        Must include ``trace_path`` pointing to the JSONL file location.
    """

    def __init__(self, config: StorageConfig) -> None:
        self._path = Path(config.trace_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, trace: Trace) -> None:
        """Serialize *trace* and append it as a single JSON line."""
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(trace.model_dump_json() + "\n")

    def upsert(self, trace: Trace) -> None:
        """Persist *trace*, replacing an existing record with the same ID."""
        traces = self.load_all()
        replaced = False

        for index, existing in enumerate(traces):
            if existing.id == trace.id:
                traces[index] = trace
                replaced = True
                break

        if not replaced:
            traces.append(trace)

        self._write_all(traces)

    def load_all(self) -> List[Trace]:
        """Read every trace from the JSONL file, deduplicated by trace ID."""
        if not self._path.exists():
            return []
        traces_by_id: Dict[str, Trace] = {}
        trace_order: List[str] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    trace = Trace.model_validate_json(line)
                    if trace.id not in traces_by_id:
                        trace_order.append(trace.id)
                    traces_by_id[trace.id] = trace
        return [traces_by_id[trace_id] for trace_id in trace_order]

    def get_feedback_samples(
        self,
        min_score: float = 0.0,
        max_score: float = 0.5,
    ) -> List[Trace]:
        """Return traces whose feedback score falls in [min_score, max_score].

        This surfaces the "bad" examples that the APO optimizer should
        learn from.  Traces without feedback are silently skipped.
        """
        return [
            t
            for t in self.load_all()
            if t.feedback is not None
            and min_score <= t.feedback.score <= max_score
        ]

    # ------------------------------------------------------------------
    # DPO Export
    # ------------------------------------------------------------------

    def get_dpo_pairs(self) -> List[Dict[str, Any]]:
        """Extract DPO preference pairs from stored traces.

        A valid DPO pair requires:
        - ``feedback.correction`` (the human-provided ideal response = chosen)
        - ``prediction`` (the model's original response = rejected)

        Deduplicates by trace id (the /rewrite flow re-appends the same
        trace, so the JSONL may contain duplicates).

        Returns a list of dicts with keys:
        ``prompt``, ``chosen``, ``rejected``, ``score``, ``critique``.
        """
        all_traces = self.load_all()

        # Deduplicate: keep the last occurrence of each trace id
        # (the one with feedback attached)
        seen: Dict[str, Trace] = {}
        for t in all_traces:
            seen[t.id] = t

        pairs: List[Dict[str, Any]] = []
        for t in seen.values():
            if t.feedback is None or not t.feedback.correction:
                continue

            prompt = _messages_to_chatml(t.inputs)
            rejected = _message_content_to_str(t.prediction.content)
            chosen = t.feedback.correction

            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "score": t.feedback.score,
                "critique": t.feedback.critique,
            })

        return pairs
    
    def _write_all(self, traces: List[Trace]) -> None:
        with self._path.open("w", encoding="utf-8") as fh:
            for trace in traces:
                fh.write(trace.model_dump_json() + "\n")

    def export_dpo(
        self,
        output_path: Union[str, Path],
        *,
        include_system: bool = True,
    ) -> int:
        """Export DPO preference pairs to a JSONL file.

        Parameters
        ----------
        output_path : str | Path
            Where to write the DPO JSONL file.
        include_system : bool
            Whether to include system messages in the prompt field.

        Returns
        -------
        int
            Number of pairs exported.
        """
        pairs = self.get_dpo_pairs()
        if not include_system:
            for pair in pairs:
                pair["prompt"] = [
                    m for m in pair["prompt"]
                    if m["role"] != "system"
                ]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for pair in pairs:
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info("Exported %d DPO pairs to %s", len(pairs), output_path)
        return len(pairs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _message_content_to_str(content) -> str:
    """Extract plain text from Message content (str or List[ContentPart])."""
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        if hasattr(part, "text"):
            texts.append(part.text)
    return " ".join(texts) if texts else ""


def _messages_to_chatml(messages: List[Message]) -> List[Dict[str, str]]:
    """Convert a list of Message objects to ChatML dicts."""
    return [
        {"role": m.role, "content": _message_content_to_str(m.content)}
        for m in messages
    ]
