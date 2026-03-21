"""Dataset — ChatML JSONL loader for APO evaluation.

Loads datasets in OpenAI fine-tuning JSONL format::

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "expected"}]}

Convention: the **last assistant message** is the ground truth (expected output).
Everything before it is the input.  Content can be ``str`` or ``List[ContentPart]``
(text, image_url, input_audio) following the OpenAI Chat Completions API format.

Usage::

    loader = DataLoader("train.jsonl")
    for sample in loader:
        print(sample.input_messages)   # user turns
        print(sample.ground_truth)     # expected assistant response
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Union

from evoskill.schema import Message

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """A single dataset sample parsed from ChatML JSONL.

    Attributes
    ----------
    messages : List[Message]
        The full conversation (all messages including ground truth).
    input_messages : List[Message]
        Everything except the last assistant message — the input to the model.
    ground_truth : Message
        The last assistant message — the expected output.
    """

    messages: List[Message]
    input_messages: List[Message]
    ground_truth: Message


class DataLoader:
    """Loads ChatML-format JSONL datasets.

    Each line must be a JSON object with a ``messages`` key containing
    a list of message objects.  The last message with ``role=assistant``
    is treated as the ground truth.

    Parameters
    ----------
    path : str | Path
        Path to the JSONL file.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._samples: Optional[List[Sample]] = None

    def _ensure_loaded(self) -> None:
        if self._samples is None:
            self.load()

    def load(self) -> "DataLoader":
        """Parse the JSONL file and populate samples."""
        if not self._path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {self._path}")

        samples: List[Sample] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Line %d: invalid JSON, skipping — %s", line_num, e)
                    continue

                msgs_raw = raw.get("messages")
                if not msgs_raw or not isinstance(msgs_raw, list):
                    logger.warning("Line %d: missing or invalid 'messages' key, skipping", line_num)
                    continue

                # Parse messages via Pydantic
                try:
                    messages = [Message.model_validate(m) for m in msgs_raw]
                except Exception as e:
                    logger.warning("Line %d: message parse error, skipping — %s", line_num, e)
                    continue

                # Find the last assistant message as ground truth
                gt_idx = None
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].role == "assistant":
                        gt_idx = i
                        break

                if gt_idx is None:
                    logger.warning(
                        "Line %d: no assistant message found (ground truth), skipping", line_num,
                    )
                    continue

                samples.append(Sample(
                    messages=messages,
                    input_messages=messages[:gt_idx],
                    ground_truth=messages[gt_idx],
                ))

        self._samples = samples
        logger.info("Loaded %d samples from %s", len(samples), self._path)
        return self

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._samples)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[Sample]:
        self._ensure_loaded()
        return iter(self._samples)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Sample:
        self._ensure_loaded()
        return self._samples[idx]  # type: ignore[index]

    def sample(self, n: int, *, seed: Optional[int] = None) -> List[Sample]:
        """Return *n* random samples (without replacement if n <= len)."""
        self._ensure_loaded()
        assert self._samples is not None
        rng = random.Random(seed)
        if n >= len(self._samples):
            return list(self._samples)
        return rng.sample(self._samples, n)
