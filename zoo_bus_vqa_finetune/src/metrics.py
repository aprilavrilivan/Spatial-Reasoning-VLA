from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

from src.utils import HELD_OUT_TYPES


def normalize_answer(text: str) -> str:
    """Normalize short VQA answers for robust comparison."""
    if text is None:
        return ""
    text = str(text).strip().lower()

    prefixes = [
        "answer:",
        "the answer is",
        "it is",
        "it's",
        "there are",
        "there is",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    for char in ["\n", "\t", ".", "!", "?", ";"]:
        text = text.replace(char, " ")
    text = " ".join(text.split())
    text = ",".join(part.strip() for part in text.split(","))

    return text


def get_train_question_type_counts(dataset_split) -> Dict[str, int]:
    return dict(Counter(dataset_split["question_type"]))


def split_seen_unseen_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    seen_rows = [row for row in rows if row["question_type"] not in HELD_OUT_TYPES]
    unseen_rows = [row for row in rows if row["question_type"] in HELD_OUT_TYPES]

    def accuracy(subrows: List[Dict[str, Any]]) -> float:
        if not subrows:
            return 0.0
        return sum(int(row["is_correct"]) for row in subrows) / len(subrows)

    metrics: Dict[str, Any] = {
        "overall_accuracy": accuracy(rows),
        "seen_type_accuracy": accuracy(seen_rows),
        "unseen_type_accuracy": accuracy(unseen_rows),
    }

    per_type_accuracy = {}
    rows_by_type = defaultdict(list)
    for row in rows:
        rows_by_type[row["question_type"]].append(row)
    for question_type, question_rows in rows_by_type.items():
        per_type_accuracy[question_type] = accuracy(question_rows)

    metrics["per_type_accuracy"] = per_type_accuracy
    return metrics
