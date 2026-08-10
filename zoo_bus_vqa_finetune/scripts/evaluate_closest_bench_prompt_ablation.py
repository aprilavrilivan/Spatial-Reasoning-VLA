#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DEFAULT_ZOO_BUS_DATASET, ZooBusEvalDataset, _load_parquet_split
from src.metrics import normalize_answer
from src.models.qwen import DEFAULT_MODEL as QWEN_DEFAULT_MODEL
from src.models.qwen import _load_qwen_base_model, generate_qwen_predictions, load_qwen_processor
from src.models.smol import DEFAULT_MODEL as SMOL_DEFAULT_MODEL
from src.models.smol import generate_smol_predictions, load_smol_processor
from src.utils import DEFAULT_DATASET, write_json


QUESTION_TYPE = "ClosestBenchWithPerson"
ORIGINAL_PROMPT = (
    "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
    "Which bench is closest to the clock that has at least one person at it? Answer with the bench ID. "
    "If no benches have people, respond with '0'. "
)
OPTIMIZED_PROMPT = (
    "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
    "First consider only benches that have at least one person at that bench. "
    "Among those benches, which one is closest to the clock? Answer with the bench ID. "
    "If no bench has a person, respond with '0'. "
)
DIRECT_RELATIVE_PROMPT = (
    "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
    "Which bench that has at least one person at it is closest to the clock? Answer with the bench ID. "
    "If no bench has a person, respond with '0'. "
)
AMONG_WITH_PEOPLE_PROMPT = (
    "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
    "Among the benches that have at least one person at them, which bench is closest to the clock? Answer with the bench ID. "
    "If no bench has a person, respond with '0'. "
)
PROMPT_VARIANTS = {
    "original": ORIGINAL_PROMPT,
    "direct_relative": DIRECT_RELATIVE_PROMPT,
    "among_with_people": AMONG_WITH_PEOPLE_PROMPT,
    "filter_first": OPTIMIZED_PROMPT,
}


class QuestionTypePromptDataset(Dataset):
    def __init__(self, base_dataset: Dataset, question_type: str, question_text: str):
        self.rows: List[Dict[str, Any]] = []
        for idx in range(len(base_dataset)):
            item = dict(base_dataset[idx])
            if item["question_type"] == question_type:
                item["question"] = question_text
                self.rows.append(item)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_family", choices=["smol", "qwen"], required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=["evaluation", "test"])
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--variants", nargs="+", default=["original", "filter_first"], choices=sorted(PROMPT_VARIANTS))
    return parser.parse_args()


def load_model_and_processor(model_family: str, model_name: str | None, adapter_path: str):
    if model_family == "smol":
        resolved_model_name = model_name or SMOL_DEFAULT_MODEL
        processor = load_smol_processor(resolved_model_name)
        base_model = AutoModelForImageTextToText.from_pretrained(
            resolved_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        generator = generate_smol_predictions
    else:
        resolved_model_name = model_name or QWEN_DEFAULT_MODEL
        processor = load_qwen_processor(resolved_model_name)
        base_model = _load_qwen_base_model(resolved_model_name, use_flash_attention=True)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        generator = generate_qwen_predictions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, generator, device


def load_requested_splits(dataset_name: str, splits: List[str]) -> Dict[str, Any]:
    if dataset_name == DEFAULT_ZOO_BUS_DATASET:
        from huggingface_hub import snapshot_download

        snapshot_dir = Path(
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                allow_patterns=[f"data/{split}*.parquet" for split in splits],
            )
        )
        return {split: _load_parquet_split(snapshot_dir, split) for split in splits}

    from datasets import load_dataset

    return {split: load_dataset(dataset_name, split=split) for split in splits}


def accuracy(rows: Iterable[Dict[str, Any]]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(int(row["is_correct"]) for row in rows) / len(rows)


def answer_distribution(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else 999, item[0])))


def run_variant(
    *,
    model,
    processor,
    generator,
    device: torch.device,
    dataset: Dataset,
    output_path: Path,
    batch_size: int,
    max_seq_length: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    rows = generator(
        model=model,
        processor=processor,
        eval_dataset=dataset,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return {
        "num_examples": len(rows),
        "accuracy": accuracy(rows),
        "gold_distribution": answer_distribution(rows, "gold_normalized"),
        "prediction_distribution": answer_distribution(rows, "normalized_prediction"),
        "predictions_csv": str(output_path),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_requested_splits(args.dataset_name, args.splits)
    model, processor, generator, device = load_model_and_processor(
        args.model_family,
        args.model_name,
        args.adapter_path,
    )

    summary: Dict[str, Any] = {
        "model_family": args.model_family,
        "adapter_path": args.adapter_path,
        "question_type": QUESTION_TYPE,
        "prompt_variants": {name: PROMPT_VARIANTS[name] for name in args.variants},
        "splits": {},
    }

    for split in args.splits:
        base_eval = ZooBusEvalDataset(dataset[split])
        variant_datasets = {
            name: QuestionTypePromptDataset(base_eval, QUESTION_TYPE, PROMPT_VARIANTS[name])
            for name in args.variants
        }
        sizes = {name: len(ds) for name, ds in variant_datasets.items()}
        if len(set(sizes.values())) != 1:
            raise RuntimeError(f"Prompt variant datasets have different sizes: {sizes}")

        variant_rows_path = output_dir / f"{split}_prompt_variants.json"
        reference_dataset = next(iter(variant_datasets.values()))
        write_json(
            variant_rows_path,
            [
                {
                    "id": item["id"],
                    "source_id": item["source_id"],
                    "question_type": item["question_type"],
                    "answer": str(item["answer"]),
                    "answer_normalized": normalize_answer(str(item["answer"])),
                    "prompt_variants": {name: PROMPT_VARIANTS[name] for name in args.variants},
                }
                for item in reference_dataset
            ],
        )

        split_results: Dict[str, Any] = {"num_examples": next(iter(sizes.values())), "variants": {}}
        for name, variant_dataset in variant_datasets.items():
            result = run_variant(
                model=model,
                processor=processor,
                generator=generator,
                device=device,
                dataset=variant_dataset,
                output_path=output_dir / f"{split}_{name}.csv",
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                max_new_tokens=args.max_new_tokens,
            )
            split_results["variants"][name] = result

        original_accuracy = split_results["variants"].get("original", {}).get("accuracy")
        if original_accuracy is not None:
            split_results["absolute_accuracy_change_vs_original"] = {
                name: result["accuracy"] - original_accuracy
                for name, result in split_results["variants"].items()
                if name != "original"
            }
        summary["splits"][split] = split_results
        print(
            f"{args.model_family} {split}: "
            + ", ".join(
                f"{name}={result['accuracy']:.4f}"
                for name, result in split_results["variants"].items()
            ),
            flush=True,
        )

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
