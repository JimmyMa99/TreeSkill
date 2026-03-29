"""SealQA task adapter for AS(skill)O experiments."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class SealQAExample:
    question: str
    answer: str
    topic: str = "unknown"
    metadata: Dict[str, str] | None = None


class SealQATaskAdapter:
    """Loads SealQA-style datasets and produces stable train/val/test splits."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        train_ratio: float = 0.18,
        val_ratio: float = 0.12,
        seed: int = 42,
        limit_per_topic: Optional[int] = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.limit_per_topic = limit_per_topic

    def load(self) -> List[SealQAExample]:
        suffix = self.dataset_path.suffix.lower()
        if suffix == ".csv":
            return self._load_csv()
        if suffix == ".parquet":
            return self._load_parquet()
        raise ValueError(f"Unsupported SealQA dataset format: {self.dataset_path}")

    def split(self) -> Tuple[List[SealQAExample], List[SealQAExample], List[SealQAExample]]:
        examples = self.load()
        rng = random.Random(self.seed)
        grouped: Dict[str, List[SealQAExample]] = {}
        for item in examples:
            grouped.setdefault(item.topic or "unknown", []).append(item)

        train: List[SealQAExample] = []
        val: List[SealQAExample] = []
        test: List[SealQAExample] = []

        for topic, items in grouped.items():
            rng.shuffle(items)
            if self.limit_per_topic is not None:
                items = items[: self.limit_per_topic]
            total = len(items)
            if total == 0:
                continue
            n_train = max(1, int(total * self.train_ratio))
            n_val = max(1, int(total * self.val_ratio))
            train.extend(items[:n_train])
            val.extend(items[n_train:n_train + n_val])
            test.extend(items[n_train + n_val :])

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)
        return train, val, test

    def evaluate_accuracy(
        self,
        data: List[SealQAExample],
        predictor: Callable[[SealQAExample], str],
        scorer: Callable[[SealQAExample, str], float],
    ) -> Tuple[float, List[Dict[str, object]]]:
        rows: List[Dict[str, object]] = []
        for item in data:
            prediction = predictor(item)
            score = scorer(item, prediction)
            rows.append(
                {
                    "question": item.question,
                    "answer": item.answer,
                    "topic": item.topic,
                    "prediction": prediction,
                    "score": score,
                }
            )
        avg = sum(float(row["score"]) for row in rows) / len(rows) if rows else 0.0
        return avg, rows

    def _load_csv(self) -> List[SealQAExample]:
        rows: List[SealQAExample] = []
        with self.dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                question = row.get("question") or row.get("query") or ""
                answer = row.get("answer") or row.get("ground_truth") or ""
                topic = row.get("topic") or row.get("category") or "unknown"
                rows.append(
                    SealQAExample(
                        question=question,
                        answer=answer,
                        topic=topic,
                        metadata={k: str(v) for k, v in row.items() if v is not None},
                    )
                )
        return rows

    def _load_parquet(self) -> List[SealQAExample]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Reading SealQA parquet files requires pandas. "
                "Install pandas or convert the dataset to CSV."
            ) from exc

        df = pd.read_parquet(self.dataset_path)
        rows: List[SealQAExample] = []
        for item in df.to_dict("records"):
            question = item.get("question") or item.get("query") or ""
            answer = item.get("answer") or item.get("ground_truth") or ""
            topic = item.get("topic") or item.get("category") or "unknown"
            rows.append(
                SealQAExample(
                    question=question,
                    answer=answer,
                    topic=str(topic),
                    metadata={k: str(v) for k, v in item.items() if v is not None},
                )
            )
        return rows
