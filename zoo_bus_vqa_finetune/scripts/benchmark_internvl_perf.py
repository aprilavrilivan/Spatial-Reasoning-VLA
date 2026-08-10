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
from src.models.internvl import (
    DEFAULT_MODEL,
    InternVLVQATrainCollator,
    generate_internvl_predictions,
    load_internvl_lora_model,
    load_internvl_processor,
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
        "gpu_memory_allocated_gb": float(torch.cuda.memory_allocated(device) / (1024**3)),
        "gpu_memory_reserved_gb": float(torch.cuda.memory_reserved(device) / (1024**3)),
        "gpu_memory_peak_allocated_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "gpu_memory_peak_reserved_gb": float(torch.cuda.max_memory_reserved(device) / (1024**3)),
    }


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_common(args):
    processor = load_internvl_processor(args.model_name)
    model = load_internvl_lora_model(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    return processor, model


def run_probe(args, dataset) -> Dict[str, Any]:
    set_seed(args.seed)
    processor, model = load_common(args)
    train_dataset = ZooBusTrainDataset(dataset["train"])
    collator = InternVLVQATrainCollator(processor, max_seq_length=args.max_seq_length)
    features = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
    batch = collator(features)
    payload = {
        "mode": "probe",
        "status": "ok",
        "input_shape": list(batch["input_ids"].shape),
        "trainable_label_tokens": int((batch["labels"] != -100).sum().item()),
        "keys": sorted(batch.keys()),
        **cuda_stats(),
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    cleanup(model)
    return payload


def run_train_variant(
    args,
    dataset,
    train_batch_size: int,
    grad_accum: int,
    gradient_checkpointing: bool,
) -> Dict[str, Any]:
    set_seed(args.seed)
    cleanup()
    processor, model = load_common(args)
    train_dataset = ZooBusTrainDataset(dataset["train"])
    collator = InternVLVQATrainCollator(processor, max_seq_length=args.max_seq_length)
    output_dir = Path(args.output_dir) / (
        f"bs{train_batch_size}_ga{grad_accum}_gc{int(gradient_checkpointing)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    training_args = make_training_arguments(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=args.max_steps,
        num_train_epochs=1,
        learning_rate=args.lr,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="no",
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=gradient_checkpointing,
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
    try:
        result = trainer.train()
        elapsed = time.time() - started
        steps = max(int(trainer.state.global_step), 1)
        payload = {
            "mode": "train",
            "status": "ok",
            "max_steps": args.max_steps,
            "actual_steps": steps,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": train_batch_size * grad_accum,
            "gradient_checkpointing": gradient_checkpointing,
            "elapsed_seconds": elapsed,
            "seconds_per_step": elapsed / steps,
            "effective_examples_per_second": (
                train_batch_size * grad_accum * steps / elapsed if elapsed > 0 else 0.0
            ),
            "trainer_metrics": result.metrics,
            **cuda_stats(),
        }
    except RuntimeError as exc:
        payload = {
            "mode": "train",
            "status": "runtime_error",
            "error": str(exc),
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": train_batch_size * grad_accum,
            "gradient_checkpointing": gradient_checkpointing,
            **cuda_stats(),
        }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    cleanup(trainer, model)
    return payload


def run_eval_benchmark(args, dataset) -> Dict[str, Any]:
    set_seed(args.seed)
    cleanup()
    processor, model = load_common(args)
    eval_dataset = FirstNDataset(ZooBusEvalDataset(dataset["evaluation"]), args.eval_examples)
    device = model.device
    results: List[Dict[str, Any]] = []

    for batch_size in args.eval_batch_sizes:
        cleanup()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            rows = generate_internvl_predictions(
                model=model,
                processor=processor,
                eval_dataset=eval_dataset,
                batch_size=batch_size,
                max_seq_length=args.max_seq_length,
                max_new_tokens=args.max_new_tokens_eval,
                device=device,
            )
            elapsed = time.time() - started
            payload = {
                "mode": "eval",
                "status": "ok",
                "batch_size": batch_size,
                "examples": len(rows),
                "elapsed_seconds": elapsed,
                "examples_per_second": len(rows) / elapsed if elapsed > 0 else 0.0,
                **cuda_stats(),
            }
        except RuntimeError as exc:
            payload = {
                "mode": "eval",
                "status": "runtime_error",
                "batch_size": batch_size,
                "error": str(exc),
                **cuda_stats(),
            }
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        results.append(payload)

    cleanup(model)
    return {"mode": "eval_summary", "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "probe", "train", "eval"], default="all")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "benchmarks" / "internvl"))
    parser.add_argument(
        "--result_path",
        default=str(PROJECT_ROOT / "outputs" / "benchmarks" / "internvl" / "results.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens_eval", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--eval_examples", type=int, default=192)
    parser.add_argument("--eval_batch_sizes", type=int, nargs="+", default=[64, 80, 96])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_vqa_dataset(args.dataset_name)
    results: List[Dict[str, Any]] = []
    if args.mode in {"all", "probe"}:
        results.append(run_probe(args, dataset))
    if args.mode in {"all", "train"}:
        for batch_size, grad_accum, gradient_checkpointing in [
            (8, 8, True),
            (4, 16, True),
        ]:
            results.append(
                run_train_variant(
                    args,
                    dataset,
                    train_batch_size=batch_size,
                    grad_accum=grad_accum,
                    gradient_checkpointing=gradient_checkpointing,
                )
            )
    if args.mode in {"all", "eval"}:
        results.append(run_eval_benchmark(args, dataset))

    write_json(result_path, {"results": results})
    print(f"RESULT_PATH {result_path}", flush=True)


if __name__ == "__main__":
    main()
