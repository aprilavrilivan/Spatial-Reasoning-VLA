from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from src.utils import ensure_rgb

DEFAULT_ZOO_BUS_DATASET = "aprilavrilivan/zoo-bus-vqa"
VQA_COLUMNS = ["image", "question", "answer", "question_type", "source_id", "id"]
PARQUET_VQA_COLUMNS = ["image.bytes", "question", "answer", "question_type", "source_id", "id"]


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


class ZooBusParquetSplit(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return [row[idx] for row in self.rows]
        return self.rows[idx]


def _decode_image_from_row(row: Dict[str, Any]) -> Image.Image:
    image = row.get("image")
    if image is None and "image_bytes" in row:
        with Image.open(BytesIO(row["image_bytes"])) as opened:
            image = opened.copy()
    return ensure_rgb(image)


def _load_parquet_split(snapshot_dir: Path, split_name: str) -> ZooBusParquetSplit:
    from fastparquet import ParquetFile

    data_dir = snapshot_dir / "data"
    files = sorted(data_dir.glob(f"{split_name}*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for split {split_name!r} in {data_dir}")

    rows: List[Dict[str, Any]] = []
    for path in files:
        df = ParquetFile(str(path)).to_pandas(columns=PARQUET_VQA_COLUMNS)
        for record in df.to_dict("records"):
            rows.append(
                {
                    "image_bytes": record["image.bytes"],
                    "question": record["question"],
                    "answer": str(record["answer"]),
                    "question_type": record["question_type"],
                    "source_id": record.get("source_id", ""),
                    "id": int(record["id"]),
                }
            )
    return ZooBusParquetSplit(rows)


def load_vqa_dataset(dataset_name: str):
    if dataset_name != DEFAULT_ZOO_BUS_DATASET:
        return load_dataset(dataset_name)

    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            allow_patterns="data/*.parquet",
        )
    )
    return {
        "train": _load_parquet_split(snapshot_dir, "train"),
        "evaluation": _load_parquet_split(snapshot_dir, "evaluation"),
        "test": _load_parquet_split(snapshot_dir, "test"),
    }


class ZooBusTrainDataset(Dataset):
    def __init__(self, hf_split):
        self.ds = hf_split

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        return {
            "image": _decode_image_from_row(row),
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
            "image": _decode_image_from_row(row),
            "question": row["question"],
            "answer": str(row["answer"]),
            "question_type": row["question_type"],
            "source_id": row.get("source_id", ""),
            "id": row.get("id", idx),
        }
