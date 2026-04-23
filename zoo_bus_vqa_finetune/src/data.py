from __future__ import annotations

from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from src.utils import ensure_rgb


def build_user_question(question: str) -> str:
    return (
        "Answer the visual question using a short final answer only. "
        "Do not explain your reasoning.\n"
        f"Question: {question}"
    )


def build_messages(question: str, answer: Optional[str] = None) -> List[Dict[str, Any]]:
    user_text = build_user_question(question)
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(answer).strip()}],
            }
        )
    return messages


class ZooBusTrainDataset(Dataset):
    def __init__(self, hf_split):
        self.ds = hf_split

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        return {
            "image": ensure_rgb(row["image"]),
            "question": row["question"],
            "answer": str(row["answer"]),
            "question_type": row["question_type"],
            "source_id": row.get("source_id", ""),
            "id": row.get("id", idx),
        }


class ZooBusEvalDataset(Dataset):
    def __init__(self, hf_split):
        self.ds = hf_split

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        return {
            "image": ensure_rgb(row["image"]),
            "question": row["question"],
            "answer": str(row["answer"]),
            "question_type": row["question_type"],
            "source_id": row.get("source_id", ""),
            "id": row.get("id", idx),
        }
