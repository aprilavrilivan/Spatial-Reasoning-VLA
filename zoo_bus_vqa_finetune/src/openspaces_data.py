from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Sequence

from datasets import Dataset as HFDataset
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from src.utils import ensure_rgb


DEFAULT_OPENSPACES_DATASET = "remyxai/OpenSpaces"


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        for key in ("text", "value", "content"):
            if key not in content:
                continue
            value = content[key]
            if value is None:
                continue
            text = _content_text(value)
            if text:
                return text
        return ""
    if isinstance(content, list):
        parts = [_content_text(item) for item in content]
        return " ".join(part for part in parts if part).strip()
    return str(content).strip()


def _coerce_messages(messages: Any) -> List[Dict[str, Any]]:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return []
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, dict)]


def extract_qa_pairs(messages: Any) -> List[tuple[str, str]]:
    qa_pairs: List[tuple[str, str]] = []
    pending_question: str | None = None

    for message in _coerce_messages(messages):
        role = str(message.get("role", "")).lower()
        text = _content_text(message.get("content", message.get("text", ""))).strip()
        if not text:
            continue

        if role in {"user", "human"}:
            pending_question = text
        elif role in {"assistant", "gpt", "model"} and pending_question is not None:
            qa_pairs.append((pending_question, text))
            pending_question = None

    return qa_pairs


def _first_image(images: Any) -> Any:
    if isinstance(images, (list, tuple)):
        if not images:
            raise ValueError("OpenSpaces row has an empty images sequence.")
        return images[0]
    return images


def decode_openspaces_image(images: Any) -> Image.Image:
    image = _first_image(images)
    if isinstance(image, Image.Image):
        return ensure_rgb(image)
    if isinstance(image, bytes):
        with Image.open(BytesIO(image)) as opened:
            return ensure_rgb(opened.copy())
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            with Image.open(BytesIO(image["bytes"])) as opened:
                return ensure_rgb(opened.copy())
        if image.get("path"):
            with Image.open(image["path"]) as opened:
                return ensure_rgb(opened.copy())
    raise TypeError(f"Unsupported OpenSpaces image payload: {type(image)!r}")


def classify_openspaces_question(question: str, answer: str = "") -> str:
    q = question.lower()
    a = answer.lower().strip()
    if re.match(r"^(is|are|do|does|did|can|could|would|will|has|have)\b", q) or a in {"yes", "no"}:
        return "YesNoSpatial"
    if any(token in q for token in ["how far", "distance", "away", "closer", "closest", "nearer", "nearest"]):
        return "DistanceSpatial"
    if any(token in q for token in ["height", "width", "tall", "wide", "large", "larger", "small", "smaller", "size", "bigger", "biggest"]):
        return "SizeSpatial"
    if any(token in q for token in ["left", "right", "above", "below", "under", "over", "front", "behind", "back", "between"]):
        return "RelativePosition"
    if any(token in q for token in ["which", "what object", "what is", "identify"]):
        return "ObjectSelection"
    return "GeneralSpatial"


def classify_answer_type(answer: str) -> str:
    normalized = answer.strip().lower().rstrip(".!?")
    if normalized in {"yes", "no"}:
        return "yes_no"
    if re.search(r"[-+]?\d*\.?\d+", normalized):
        return "numeric"
    if any(word in normalized for word in ["left", "right", "above", "below", "under", "over", "front", "behind"]):
        return "directional"
    if len(normalized.split()) <= 4:
        return "short_phrase"
    return "long_phrase"


@dataclass(frozen=True)
class OpenSpacesExample:
    row_idx: int
    turn_idx: int
    question: str
    answer: str
    question_type: str
    answer_type: str


class OpenSpacesQADataset(Dataset):
    def __init__(
        self,
        hf_split: HFDataset,
        examples: Sequence[OpenSpacesExample],
        split_name: str,
    ):
        self.hf_split = hf_split
        self.examples = list(examples)
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return [self._metadata_row(example)[idx] for example in self.examples]

        example = self.examples[idx]
        row = self.hf_split[example.row_idx]
        item = self._metadata_row(example)
        item["image"] = decode_openspaces_image(row["images"])
        return item

    def _metadata_row(self, example: OpenSpacesExample) -> Dict[str, Any]:
        source_id = f"{self.split_name}_row_{example.row_idx}"
        return {
            "question": example.question,
            "answer": example.answer,
            "question_type": example.question_type,
            "answer_type": example.answer_type,
            "source_id": source_id,
            "id": f"{source_id}_turn_{example.turn_idx}",
        }


def _build_examples(
    hf_split: HFDataset,
    row_indices: Iterable[int],
    *,
    max_answer_chars: int,
    max_question_chars: int,
) -> List[OpenSpacesExample]:
    examples: List[OpenSpacesExample] = []
    for row_idx in row_indices:
        row = hf_split[int(row_idx)]
        for turn_idx, (question, answer) in enumerate(extract_qa_pairs(row.get("messages"))):
            question = question.strip()
            answer = answer.strip()
            if not question or not answer:
                continue
            if len(question) > max_question_chars or len(answer) > max_answer_chars:
                continue
            examples.append(
                OpenSpacesExample(
                    row_idx=int(row_idx),
                    turn_idx=turn_idx,
                    question=question,
                    answer=answer,
                    question_type=classify_openspaces_question(question, answer),
                    answer_type=classify_answer_type(answer),
                )
            )
    return examples


def _limit_examples(examples: List[OpenSpacesExample], max_examples: int | None) -> List[OpenSpacesExample]:
    if max_examples is None or max_examples <= 0:
        return examples
    return examples[:max_examples]


def load_openspaces_qa_datasets(
    dataset_name: str = DEFAULT_OPENSPACES_DATASET,
    *,
    train_split: str = "train",
    test_split: str = "test",
    eval_fraction: float = 0.1,
    seed: int = 42,
    max_train_examples: int | None = None,
    max_eval_examples: int | None = None,
    max_test_examples: int | None = None,
    max_answer_chars: int = 512,
    max_question_chars: int = 1024,
) -> Dict[str, OpenSpacesQADataset]:
    raw = load_dataset(dataset_name)
    if train_split not in raw or test_split not in raw:
        raise ValueError(f"OpenSpaces dataset must contain {train_split!r} and {test_split!r} splits.")

    train_raw = raw[train_split]
    test_raw = raw[test_split]

    train_row_indices = list(range(len(train_raw)))
    rng = random.Random(seed)
    rng.shuffle(train_row_indices)
    eval_rows = max(1, int(round(len(train_row_indices) * eval_fraction))) if eval_fraction > 0 else 0
    eval_row_indices = sorted(train_row_indices[:eval_rows])
    actual_train_row_indices = sorted(train_row_indices[eval_rows:])

    train_examples = _limit_examples(
        _build_examples(
            train_raw,
            actual_train_row_indices,
            max_answer_chars=max_answer_chars,
            max_question_chars=max_question_chars,
        ),
        max_train_examples,
    )
    eval_examples = _limit_examples(
        _build_examples(
            train_raw,
            eval_row_indices,
            max_answer_chars=max_answer_chars,
            max_question_chars=max_question_chars,
        ),
        max_eval_examples,
    )
    test_examples = _limit_examples(
        _build_examples(
            test_raw,
            range(len(test_raw)),
            max_answer_chars=max_answer_chars,
            max_question_chars=max_question_chars,
        ),
        max_test_examples,
    )

    if not train_examples:
        raise ValueError("No OpenSpaces train QA examples were extracted.")
    if not eval_examples:
        raise ValueError("No OpenSpaces eval QA examples were extracted.")
    if not test_examples:
        raise ValueError("No OpenSpaces test QA examples were extracted.")

    return {
        "train": OpenSpacesQADataset(train_raw, train_examples, "train"),
        "evaluation": OpenSpacesQADataset(train_raw, eval_examples, "evaluation"),
        "test": OpenSpacesQADataset(test_raw, test_examples, "test"),
    }


def summarize_openspaces_split(dataset: OpenSpacesQADataset) -> Dict[str, Any]:
    from collections import Counter

    question_counts = Counter(example.question_type for example in dataset.examples)
    answer_counts = Counter(example.answer_type for example in dataset.examples)
    source_rows = {example.row_idx for example in dataset.examples}
    return {
        "num_qa_pairs": len(dataset),
        "num_source_rows": len(source_rows),
        "question_type_counts": dict(sorted(question_counts.items())),
        "answer_type_counts": dict(sorted(answer_counts.items())),
    }


def summarize_openspaces_datasets(datasets: Dict[str, OpenSpacesQADataset]) -> Dict[str, Any]:
    return {split_name: summarize_openspaces_split(split) for split_name, split in datasets.items()}
