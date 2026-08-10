from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import ZooBusEvalDataset, ZooBusTrainDataset, load_vqa_dataset
from src.models.qwen import (
    DEFAULT_MODEL,
    QwenVQATrainCollator,
    generate_qwen_predictions,
    load_qwen_lora_model,
    load_qwen_processor,
)
from src.utils import DEFAULT_DATASET, write_json


class FirstNDataset(Dataset):
    def __init__(self, base: Dataset, limit: int):
        self.base = base
        self.limit = min(limit, len(base))

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, idx: int):
        return self.base[idx]


def make_training_arguments(**kwargs) -> TrainingArguments:
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in supported})


def cuda_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    return {
        "gpu_memory_allocated_mb": float(torch.cuda.memory_allocated(device) / (1024**2)),
        "gpu_memory_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024**2)),
        "gpu_memory_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
        "gpu_memory_peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024**2)),
    }


def load_common(args):
    set_seed(args.seed)
    dataset = load_vqa_dataset(args.dataset_name)
    processor = load_qwen_processor(args.model_name)
    model = load_qwen_lora_model(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    return dataset, processor, model


def run_train_benchmark(args) -> Dict[str, Any]:
    dataset, processor, model = load_common(args)
    train_dataset = ZooBusTrainDataset(dataset["train"])
    collator = QwenVQATrainCollator(processor, max_seq_length=args.max_seq_length)

    output_dir = Path(args.output_dir) / (
        f"train_bs{args.train_batch_size}_ga{args.gradient_accumulation_steps}_"
        f"gc{int(args.gradient_checkpointing)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    training_args = make_training_arguments(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=args.max_steps,
        num_train_epochs=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        logging_steps=max(1, min(args.logging_steps, args.max_steps)),
        logging_first_step=True,
        eval_strategy="no",
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        label_names=["labels"],
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    steps = max(int(trainer.state.global_step), 1)
    return {
        "mode": "train",
        "status": "ok",
        "model_name": args.model_name,
        "max_steps": args.max_steps,
        "actual_steps": steps,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "effective_examples_per_second": (
            args.train_batch_size * args.gradient_accumulation_steps * steps / elapsed
            if elapsed > 0
            else 0.0
        ),
        "trainer_metrics": result.metrics,
        **cuda_stats(),
    }


def run_eval_benchmark(args) -> Dict[str, Any]:
    dataset, processor, model = load_common(args)
    eval_dataset = FirstNDataset(ZooBusEvalDataset(dataset["evaluation"]), args.eval_examples)
    device = model.device

    results: List[Dict[str, Any]] = []
    for batch_size in args.eval_batch_sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        started = time.time()
        try:
            rows = generate_qwen_predictions(
                model=model,
                processor=processor,
                eval_dataset=eval_dataset,
                batch_size=batch_size,
                max_seq_length=args.max_seq_length,
                max_new_tokens=args.max_new_tokens_eval,
                device=device,
            )
            elapsed = time.time() - started
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "ok",
                    "examples": len(rows),
                    "elapsed_seconds": elapsed,
                    "examples_per_second": len(rows) / elapsed if elapsed > 0 else 0.0,
                    **cuda_stats(),
                }
            )
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        except RuntimeError as exc:
            elapsed = time.time() - started
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "runtime_error",
                    "error": str(exc),
                    "elapsed_seconds": elapsed,
                    **cuda_stats(),
                }
            )
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "mode": "eval",
        "status": "ok",
        "model_name": args.model_name,
        "eval_examples": len(eval_dataset),
        "max_new_tokens_eval": args.max_new_tokens_eval,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "benchmarks" / "qwen"))
    parser.add_argument("--result_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens_eval", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_examples", type=int, default=256)
    parser.add_argument("--eval_batch_sizes", type=int, nargs="+", default=[8, 16, 24, 32])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = run_train_benchmark(args) if args.mode == "train" else run_eval_benchmark(args)
    except RuntimeError as exc:
        payload = {
            "mode": args.mode,
            "status": "runtime_error",
            "error": str(exc),
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            **cuda_stats(),
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.result_path:
        write_json(args.result_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
