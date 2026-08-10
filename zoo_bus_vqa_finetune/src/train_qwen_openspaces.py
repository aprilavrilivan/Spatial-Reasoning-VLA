from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

try:
    import wandb
except ImportError:
    wandb = None

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.metrics import normalize_answer
from src.models.qwen import (
    DEFAULT_MODEL as QWEN_DEFAULT_MODEL,
    QwenVQATrainCollator,
    generate_qwen_predictions,
    load_qwen_lora_model,
    load_qwen_processor,
)
from src.openspaces_data import (
    DEFAULT_OPENSPACES_DATASET,
    OpenSpacesQADataset,
    classify_answer_type,
    load_openspaces_qa_datasets,
    summarize_openspaces_datasets,
)
from src.utils import get_gpu_metrics, write_json


def build_openspaces_question(question: str) -> str:
    # OpenSpaces already stores full instruction-style spatial questions, and
    # its gold answers are often sentences or measurements. Reusing the zoo-bus
    # "short final answer only" prompt conflicts with this target format.
    return str(question).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL on remyxai/OpenSpaces.")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_OPENSPACES_DATASET)
    parser.add_argument("--model_name", type=str, default=QWEN_DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--eval_fraction", type=float, default=0.1)
    parser.add_argument("--max_train_examples", type=int, default=0)
    parser.add_argument("--max_eval_examples", type=int, default=0)
    parser.add_argument("--max_test_examples", type=int, default=0)
    parser.add_argument("--max_question_chars", type=int, default=1024)
    parser.add_argument("--max_answer_chars", type=int, default=512)
    parser.add_argument("--image_min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--image_max_pixels", type=int, default=768 * 28 * 28)

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens_eval", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=2949)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=300)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str, default="openspaces-qwen-finetune")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--skip_pre_finetune_baseline", action="store_true")

    return parser.parse_args()


def make_training_arguments(**kwargs) -> TrainingArguments:
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"Skipping unsupported TrainingArguments: {', '.join(dropped)}")
    return TrainingArguments(**filtered)


def accuracy(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(int(row["is_correct"]) for row in rows) / len(rows)


def grouped_accuracy(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return {name: accuracy(group_rows) for name, group_rows in sorted(grouped.items())}


def grouped_counts(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, ""))] += 1
    return dict(sorted(counts.items()))


def macro_average(values: Dict[str, float]) -> float:
    if not values:
        return 0.0
    return sum(values.values()) / len(values)


_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")
_DIRECTION_WORDS = {
    "left": "left",
    "right": "right",
    "above": "above",
    "over": "above",
    "higher": "above",
    "below": "below",
    "under": "below",
    "lower": "below",
    "front": "front",
    "behind": "behind",
    "back": "behind",
    "closer": "closer",
    "nearest": "closer",
    "nearer": "closer",
    "farther": "farther",
    "further": "farther",
    "distant": "farther",
}


def _polarity_label(text: str) -> str | None:
    normalized = normalize_answer(text)
    if not normalized:
        return None
    if normalized.startswith(("yes", "correct", "indeed", "true", "that is true")):
        return "yes"
    if normalized.startswith(("no", "incorrect", "false", "that is false")):
        return "no"
    return None


def _measurement_in_meters(text: str) -> float | None:
    match = _NUMBER_RE.search(str(text).lower())
    if match is None:
        return None
    value = float(match.group(0))
    suffix = str(text).lower()[match.end() : match.end() + 24]
    if any(unit in suffix for unit in ("centimeter", "centimetre", " cm")):
        return value / 100.0
    if any(unit in suffix for unit in ("millimeter", "millimetre", " mm")):
        return value / 1000.0
    if any(unit in suffix for unit in ("feet", "foot", " ft")):
        return value * 0.3048
    if any(unit in suffix for unit in ("inch", " in")):
        return value * 0.0254
    return value


def _direction_set(text: str) -> set[str]:
    normalized = normalize_answer(text)
    labels: set[str] = set()
    for word, label in _DIRECTION_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            labels.add(label)
    return labels


def openspaces_relaxed_correct(row: Dict[str, Any]) -> int:
    gold = str(row.get("gold_answer", ""))
    prediction = str(row.get("raw_prediction", ""))
    if normalize_answer(gold) == normalize_answer(prediction):
        return 1

    gold_polarity = _polarity_label(gold)
    prediction_polarity = _polarity_label(prediction)
    if gold_polarity is not None and prediction_polarity is not None:
        return int(gold_polarity == prediction_polarity)

    gold_measurement = _measurement_in_meters(gold)
    prediction_measurement = _measurement_in_meters(prediction)
    if gold_measurement is not None and prediction_measurement is not None:
        tolerance = max(0.05, abs(gold_measurement) * 0.15)
        return int(abs(gold_measurement - prediction_measurement) <= tolerance)

    gold_directions = _direction_set(gold)
    prediction_directions = _direction_set(prediction)
    if gold_directions and prediction_directions:
        return int(gold_directions == prediction_directions)

    return 0


def compute_openspaces_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    for row in rows:
        row.setdefault("answer_type", classify_answer_type(str(row.get("gold_answer", ""))))
        row.setdefault("is_relaxed_correct", openspaces_relaxed_correct(row))

    per_type_accuracy = grouped_accuracy(rows, "question_type")
    per_answer_type_accuracy = grouped_accuracy(rows, "answer_type")
    strict_correct = sum(int(row["is_correct"]) for row in rows)
    relaxed_correct = sum(int(row["is_relaxed_correct"]) for row in rows)

    relaxed_rows = [
        {**row, "is_correct": row["is_relaxed_correct"]}
        for row in rows
    ]
    per_type_relaxed_accuracy = grouped_accuracy(relaxed_rows, "question_type")
    per_answer_type_relaxed_accuracy = grouped_accuracy(relaxed_rows, "answer_type")
    return {
        "overall_accuracy": accuracy(rows),
        "relaxed_accuracy": accuracy(relaxed_rows),
        "macro_question_type_accuracy": macro_average(per_type_accuracy),
        "macro_question_type_relaxed_accuracy": macro_average(per_type_relaxed_accuracy),
        "macro_answer_type_accuracy": macro_average(per_answer_type_accuracy),
        "macro_answer_type_relaxed_accuracy": macro_average(per_answer_type_relaxed_accuracy),
        "num_samples": len(rows),
        "num_correct": strict_correct,
        "num_relaxed_correct": relaxed_correct,
        "per_question_type_accuracy": per_type_accuracy,
        "per_question_type_relaxed_accuracy": per_type_relaxed_accuracy,
        "per_answer_type_accuracy": per_answer_type_accuracy,
        "per_answer_type_relaxed_accuracy": per_answer_type_relaxed_accuracy,
        "question_type_counts": grouped_counts(rows, "question_type"),
        "answer_type_counts": grouped_counts(rows, "answer_type"),
    }


class OpenSpacesRuntimeLoggingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_start_time: float | None = None
        self._nonfinite_gradient_skips = 0

    def _infer_model_device(self) -> torch.device | None:
        try:
            return self.model.device
        except AttributeError:
            first_parameter = next(self.model.parameters(), None)
            return first_parameter.device if first_parameter is not None else None

    def _build_compact_train_logs(self) -> Dict[str, float]:
        logs: Dict[str, float] = {}
        if self._train_start_time is not None and self.state.global_step > 0 and self.state.max_steps > 0:
            elapsed = max(time.time() - self._train_start_time, 0.0)
            seconds_per_step = elapsed / max(int(self.state.global_step), 1)
            remaining = max(int(self.state.max_steps) - int(self.state.global_step), 0) * seconds_per_step
            logs["train/elapsed_hours"] = elapsed / 3600.0
            logs["train/eta_hours"] = remaining / 3600.0

        gpu_metrics = get_gpu_metrics(self._infer_model_device())
        if "gpu_utilization_percent" in gpu_metrics:
            logs["train/gpu_utilization_percent"] = float(gpu_metrics["gpu_utilization_percent"])
        if "gpu_memory_used_mb" in gpu_metrics:
            logs["train/gpu_memory_used_gb"] = float(gpu_metrics["gpu_memory_used_mb"] / 1024.0)
        return logs

    def log(self, logs, start_time=None):
        merged_logs = dict(logs)
        if any(key in merged_logs for key in ("loss", "grad_norm", "learning_rate")):
            merged_logs.update(self._build_compact_train_logs())
            if self._nonfinite_gradient_skips:
                merged_logs["train/nonfinite_gradient_skips"] = float(self._nonfinite_gradient_skips)
        return super().log(merged_logs, start_time=start_time)

    def _has_nonfinite_trainable_gradients(self) -> bool:
        for parameter in self.model.parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                return True
        return False

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if self._has_nonfinite_trainable_gradients():
            self._nonfinite_gradient_skips += 1
            print(
                "Detected non-finite OpenSpaces gradients; clearing accumulated gradients "
                f"and skipping this update contribution (count={self._nonfinite_gradient_skips}).",
                flush=True,
            )
            model.zero_grad(set_to_none=True)
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
            return loss.detach() * 0.0
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics: Dict[str, float] = {}
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )
        return metrics

    def train(self, *args, **kwargs):
        self._train_start_time = time.time()
        return super().train(*args, **kwargs)


class OpenSpacesGenerationEvalCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        eval_dataset: OpenSpacesQADataset,
        test_dataset: OpenSpacesQADataset,
        output_dir: str,
        eval_batch_size: int,
        max_seq_length: int,
        max_new_tokens: int,
        question_formatter=build_openspaces_question,
    ):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.output_dir = Path(output_dir)
        self.eval_batch_size = eval_batch_size
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.question_formatter = question_formatter
        self.best_metric = -1.0
        self.best_step: int | None = None
        self.before_finetune_eval_payload: Dict[str, Any] | None = None
        self.before_finetune_test_payload: Dict[str, Any] | None = None
        self.best_dir = self.output_dir / "best_checkpoint_eval_predictions"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_generation_eval(
        self,
        trainer: Trainer,
        split_name: str,
        dataset: OpenSpacesQADataset,
        global_step: int,
        *,
        file_stem: str | None = None,
        phase: str = "after_finetune",
    ) -> Dict[str, Any]:
        rows = generate_qwen_predictions(
            model=trainer.model,
            processor=self.processor,
            eval_dataset=dataset,
            batch_size=self.eval_batch_size,
            max_seq_length=self.max_seq_length,
            max_new_tokens=self.max_new_tokens,
            device=trainer.model.device,
            question_formatter=self.question_formatter,
        )
        for row in rows:
            row["answer_type"] = classify_answer_type(str(row.get("gold_answer", "")))
            row["gold_normalized"] = normalize_answer(row["gold_answer"])
            row["normalized_prediction"] = normalize_answer(row["raw_prediction"])
            row["is_correct"] = int(row["gold_normalized"] == row["normalized_prediction"])

        metrics = compute_openspaces_metrics(rows)
        pred_dir = self.output_dir / f"{split_name}_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        resolved_file_stem = file_stem or f"step_{global_step}"
        pred_path = pred_dir / f"{resolved_file_stem}.csv"
        metrics_path = pred_dir / f"{resolved_file_stem}.json"
        pd.DataFrame(rows).to_csv(pred_path, index=False)

        payload = {
            "global_step": global_step,
            "split": split_name,
            "phase": phase,
            **metrics,
            "predictions_csv": str(pred_path),
            "predictions_json": str(metrics_path),
        }
        write_json(metrics_path, payload)

        grouped_prefix = "eval" if split_name == "eval" else "test"
        log_payload = {
            f"{grouped_prefix}/overall_accuracy": metrics["overall_accuracy"],
            f"{grouped_prefix}/relaxed_accuracy": metrics["relaxed_accuracy"],
            f"{grouped_prefix}/macro_question_type_accuracy": metrics["macro_question_type_accuracy"],
            f"{grouped_prefix}/macro_question_type_relaxed_accuracy": metrics[
                "macro_question_type_relaxed_accuracy"
            ],
            f"{grouped_prefix}/macro_answer_type_accuracy": metrics["macro_answer_type_accuracy"],
            f"{grouped_prefix}/macro_answer_type_relaxed_accuracy": metrics[
                "macro_answer_type_relaxed_accuracy"
            ],
            f"{grouped_prefix}/num_samples": float(metrics["num_samples"]),
        }
        if split_name == "eval" and phase != "before_finetune":
            log_payload["eval_overall_accuracy"] = metrics["overall_accuracy"]
            log_payload["eval_relaxed_accuracy"] = metrics["relaxed_accuracy"]
            log_payload["eval_macro_question_type_accuracy"] = metrics["macro_question_type_accuracy"]
            log_payload["eval_macro_question_type_relaxed_accuracy"] = metrics[
                "macro_question_type_relaxed_accuracy"
            ]
        for question_type, value in metrics["per_question_type_accuracy"].items():
            log_payload[f"{grouped_prefix}/per_question_type/{question_type}"] = value
        for question_type, value in metrics["per_question_type_relaxed_accuracy"].items():
            log_payload[f"{grouped_prefix}/per_question_type_relaxed/{question_type}"] = value
        for answer_type, value in metrics["per_answer_type_accuracy"].items():
            log_payload[f"{grouped_prefix}/per_answer_type/{answer_type}"] = value
        for answer_type, value in metrics["per_answer_type_relaxed_accuracy"].items():
            log_payload[f"{grouped_prefix}/per_answer_type_relaxed/{answer_type}"] = value
        trainer.log(log_payload)
        return payload

    def run_pre_finetune_baseline(self, trainer: Trainer) -> Dict[str, Any]:
        global_step = int(trainer.state.global_step)
        self.before_finetune_eval_payload = self._run_generation_eval(
            trainer,
            "eval",
            self.eval_dataset,
            global_step,
            file_stem=f"before_finetune_step_{global_step}",
            phase="before_finetune",
        )
        self.before_finetune_test_payload = self._run_generation_eval(
            trainer,
            "test",
            self.test_dataset,
            global_step,
            file_stem=f"before_finetune_step_{global_step}",
            phase="before_finetune",
        )
        report = {
            "before_finetune_eval": self.before_finetune_eval_payload,
            "before_finetune_test": self.before_finetune_test_payload,
        }
        write_json(self.output_dir / "before_finetune_report.json", report)
        return report

    def on_evaluate(self, args, state, control, **kwargs):
        trainer: Trainer = kwargs["model"]._hf_peft_trainer_ref
        global_step = int(state.global_step)
        payload = self._run_generation_eval(trainer, "eval", self.eval_dataset, global_step)

        metrics = kwargs.get("metrics")
        if metrics is not None:
            metrics["eval_overall_accuracy"] = payload["overall_accuracy"]
            metrics["eval_macro_question_type_accuracy"] = payload["macro_question_type_accuracy"]

        metric = payload["overall_accuracy"]
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_step = global_step
            self.best_dir.mkdir(parents=True, exist_ok=True)
            payload["best_checkpoint_path"] = str(Path(trainer.args.output_dir) / f"checkpoint-{global_step}")
            write_json(self.best_dir / "best_eval_metrics.json", payload)

    def run_final_test(self, trainer: Trainer) -> Dict[str, Any]:
        payload = self._run_generation_eval(
            trainer,
            "test",
            self.test_dataset,
            int(trainer.state.global_step),
        )
        final_report = {
            "before_finetune": self.before_finetune_test_payload,
            "after_finetune": payload,
            "before_finetune_overall_test_accuracy": (
                self.before_finetune_test_payload["overall_accuracy"]
                if self.before_finetune_test_payload is not None
                else None
            ),
            "after_finetune_overall_test_accuracy": payload["overall_accuracy"],
            "after_finetune_relaxed_test_accuracy": payload["relaxed_accuracy"],
            "after_finetune_macro_question_type_test_accuracy": payload["macro_question_type_accuracy"],
            "after_finetune_macro_question_type_relaxed_test_accuracy": payload[
                "macro_question_type_relaxed_accuracy"
            ],
            "after_finetune_macro_answer_type_test_accuracy": payload["macro_answer_type_accuracy"],
            "after_finetune_macro_answer_type_relaxed_test_accuracy": payload[
                "macro_answer_type_relaxed_accuracy"
            ],
            "after_finetune_per_question_type_test_accuracy": payload["per_question_type_accuracy"],
            "after_finetune_per_question_type_relaxed_test_accuracy": payload[
                "per_question_type_relaxed_accuracy"
            ],
            "after_finetune_per_answer_type_test_accuracy": payload["per_answer_type_accuracy"],
            "after_finetune_per_answer_type_relaxed_test_accuracy": payload[
                "per_answer_type_relaxed_accuracy"
            ],
            "best_eval_overall_accuracy": self.best_metric,
            "best_eval_step": self.best_step,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
        }
        write_json(self.output_dir / "final_test_report.json", final_report)
        return final_report


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.report_to == "wandb":
        if wandb is None:
            raise ImportError("wandb is not installed but --report_to=wandb was requested.")
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    datasets = load_openspaces_qa_datasets(
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        test_split=args.test_split,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        max_train_examples=args.max_train_examples or None,
        max_eval_examples=args.max_eval_examples or None,
        max_test_examples=args.max_test_examples or None,
        max_answer_chars=args.max_answer_chars,
        max_question_chars=args.max_question_chars,
    )
    dataset_summary = summarize_openspaces_datasets(datasets)
    write_json(outdir / "openspaces_dataset_summary.json", dataset_summary)

    train_dataset = datasets["train"]
    eval_dataset = datasets["evaluation"]
    test_dataset = datasets["test"]

    processor = load_qwen_processor(
        args.model_name,
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels,
    )
    model = load_qwen_lora_model(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.print_trainable_parameters()
    collator = QwenVQATrainCollator(
        processor,
        max_seq_length=args.max_seq_length,
        question_formatter=build_openspaces_question,
    )

    steps_per_epoch = math.ceil(
        len(train_dataset)
        / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    estimated_total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    summary = {
        "model": args.model_name,
        "dataset": args.dataset_name,
        "method": "OpenSpaces bf16 LoRA-SFT",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "warmup_steps": args.warmup_steps,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens_eval": args.max_new_tokens_eval,
        "image_min_pixels": args.image_min_pixels,
        "image_max_pixels": args.image_max_pixels,
        "prompt_style": "OpenSpaces raw question text",
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "selection_metric": "OpenSpaces eval overall accuracy",
        "estimated_steps_per_epoch": steps_per_epoch,
        "estimated_total_steps": estimated_total_steps,
        "dataset_summary": dataset_summary,
    }
    write_json(outdir / "run_summary.json", summary)

    training_args_kwargs = {
        "output_dir": str(outdir / "checkpoints"),
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "warmup_steps": args.warmup_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": True,
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "remove_unused_columns": False,
        "dataloader_num_workers": 4,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": args.report_to,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_overall_accuracy",
        "greater_is_better": True,
        "label_names": ["labels"],
    }
    if args.max_steps > 0:
        training_args_kwargs["max_steps"] = args.max_steps

    training_args = make_training_arguments(**training_args_kwargs)
    trainer = OpenSpacesRuntimeLoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    trainer.model._hf_peft_trainer_ref = trainer

    eval_callback = OpenSpacesGenerationEvalCallback(
        processor,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        output_dir=str(outdir),
        eval_batch_size=args.per_device_eval_batch_size,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens_eval,
        question_formatter=build_openspaces_question,
    )
    trainer.add_callback(eval_callback)

    if not args.skip_pre_finetune_baseline:
        eval_callback.run_pre_finetune_baseline(trainer)

    trainer.train()
    trainer.evaluate()
    final_test_report = eval_callback.run_final_test(trainer)
    print(json.dumps(final_test_report, indent=2, ensure_ascii=False))

    if args.report_to == "wandb" and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
