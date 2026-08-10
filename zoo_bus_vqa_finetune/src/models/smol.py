from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainerCallback,
)

from src.data import build_user_question
from src.metrics import normalize_answer, split_seen_unseen_metrics
from src.utils import use_left_padding, write_json


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
METHOD = "bf16 LoRA-SFT"

SMOL_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_smol_messages(
    question: str,
    image: Image.Image,
    answer: str | None = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_user_question(question)},
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


def load_smol_processor(model_name: str):
    return use_left_padding(AutoProcessor.from_pretrained(model_name))


def build_smol_lora_config(lora_r: int, lora_alpha: int, lora_dropout: float) -> LoraConfig:
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=SMOL_LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_smol_lora_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    lora_config = build_smol_lora_config(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    return get_peft_model(model, lora_config)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _cast_floating_tensors_to_bf16(batch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            batch[key] = value.to(dtype=torch.bfloat16)
    return batch


def _assistant_mask_to_tensor(mask: Any, labels: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask_tensor = mask.to(device=labels.device, dtype=torch.bool)
    else:
        mask_tensor = torch.tensor(mask, device=labels.device, dtype=torch.bool)
    if mask_tensor.shape != labels.shape or not bool(mask_tensor.any()):
        return None
    return mask_tensor


def _mask_prompt_tokens_by_length(
    processor: Any,
    features: List[Dict[str, Any]],
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_seq_length: int,
) -> torch.Tensor:
    prompt_conversations = [
        build_smol_messages(feature["question"], feature["image"], answer=None)
        for feature in features
    ]
    prompt_batch = processor.apply_chat_template(
        prompt_conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "padding": True,
            "truncation": True,
            "max_length": max_seq_length,
        },
    )
    prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
    padding_side = getattr(processor.tokenizer, "padding_side", "right")
    for row_idx, prompt_len in enumerate(prompt_lengths):
        if padding_side == "left" and attention_mask is not None:
            full_len = int(attention_mask[row_idx].sum().item())
            prompt_start = max(labels.shape[1] - full_len, 0)
            labels[row_idx, prompt_start : prompt_start + int(prompt_len)] = -100
        else:
            labels[row_idx, : int(prompt_len)] = -100
    return labels


@dataclass
class SmolVQATrainCollator:
    processor: Any
    max_seq_length: int = 512

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        conversations = [
            build_smol_messages(feature["question"], feature["image"], feature["answer"])
            for feature in features
        ]
        template_kwargs = {
            "add_generation_prompt": False,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "processor_kwargs": {
                "padding": True,
                "truncation": True,
                "max_length": self.max_seq_length,
            },
        }
        batch = self.processor.apply_chat_template(conversations, **template_kwargs)
        batch = dict(batch)

        labels = batch["input_ids"].clone()
        assistant_mask = _assistant_mask_to_tensor(batch.pop("assistant_masks", None), labels)
        if assistant_mask is not None:
            labels[~assistant_mask] = -100
        else:
            labels = _mask_prompt_tokens_by_length(
                self.processor,
                features,
                labels,
                batch.get("attention_mask"),
                self.max_seq_length,
            )

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[batch["input_ids"] == pad_token_id] = -100
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100

        batch["labels"] = labels
        return _cast_floating_tensors_to_bf16(batch)


def generate_smol_predictions(
    model,
    processor,
    eval_dataset: Dataset,
    batch_size: int,
    max_seq_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []

    for start in range(0, len(eval_dataset), batch_size):
        batch_items = [
            eval_dataset[i]
            for i in range(start, min(start + batch_size, len(eval_dataset)))
        ]
        conversations = [
            build_smol_messages(item["question"], item["image"], answer=None)
            for item in batch_items
        ]

        model_inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "truncation": True,
                "max_length": max_seq_length,
            },
        )
        model_inputs = _move_batch_to_device(dict(model_inputs), device)

        with torch.no_grad():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        generated_only = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(model_inputs["input_ids"], generated)
        ]
        pred_texts = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for item, prediction in zip(batch_items, pred_texts):
            gold = str(item["answer"])
            pred_norm = normalize_answer(prediction)
            gold_norm = normalize_answer(gold)
            rows.append(
                {
                    "id": item["id"],
                    "source_id": item["source_id"],
                    "question_type": item["question_type"],
                    "question": item["question"],
                    "gold_answer": gold,
                    "gold_normalized": gold_norm,
                    "raw_prediction": prediction,
                    "normalized_prediction": pred_norm,
                    "is_correct": int(pred_norm == gold_norm),
                }
            )

    return rows


class SmolGenerationEvalCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        eval_dataset: Dataset,
        test_dataset: Dataset,
        output_dir: str,
        eval_batch_size: int,
        max_seq_length: int,
        max_new_tokens: int,
        train_type_counts: Dict[str, int],
    ):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.output_dir = Path(output_dir)
        self.eval_batch_size = eval_batch_size
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.train_type_counts = train_type_counts
        self.best_metric = -1.0
        self.best_step = None
        self.before_finetune_eval_payload: Dict[str, Any] | None = None
        self.before_finetune_test_payload: Dict[str, Any] | None = None
        self.best_dir = self.output_dir / "best_checkpoint_eval_predictions"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_eval(
        self,
        trainer: Trainer,
        split_name: str,
        dataset: Dataset,
        global_step: int,
        *,
        file_stem: str | None = None,
        log_prefix: str | None = None,
        phase: str = "after_finetune",
    ) -> Dict[str, Any]:
        rows = generate_smol_predictions(
            model=trainer.model,
            processor=self.processor,
            eval_dataset=dataset,
            batch_size=self.eval_batch_size,
            max_seq_length=self.max_seq_length,
            max_new_tokens=self.max_new_tokens,
            device=trainer.model.device,
        )
        metrics = split_seen_unseen_metrics(rows)

        pred_dir = self.output_dir / f"{split_name}_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        resolved_file_stem = file_stem or f"step_{global_step}"
        resolved_log_prefix = log_prefix or split_name
        pred_path = pred_dir / f"{resolved_file_stem}.csv"
        pd.DataFrame(rows).to_csv(pred_path, index=False)

        metrics_payload = {
            "global_step": global_step,
            "split": split_name,
            "phase": phase,
            "overall_accuracy": metrics["overall_accuracy"],
            "seen_type_accuracy": metrics["seen_type_accuracy"],
            "unseen_type_accuracy": metrics["unseen_type_accuracy"],
            "per_type_accuracy": metrics["per_type_accuracy"],
            "train_count_per_question_type": self.train_type_counts,
            "predictions_csv": str(pred_path),
            "predictions_json": str(pred_dir / f"{resolved_file_stem}.json"),
        }
        write_json(pred_dir / f"{resolved_file_stem}.json", metrics_payload)

        grouped_prefix = "eval" if split_name == "eval" else "test"
        log_payload = {
            f"{grouped_prefix}/overall_accuracy": metrics["overall_accuracy"],
            f"{grouped_prefix}/seen_type_accuracy": metrics["seen_type_accuracy"],
            f"{grouped_prefix}/unseen_type_accuracy": metrics["unseen_type_accuracy"],
        }
        # Keep only the flat after-finetune metric names needed by Trainer's
        # best-checkpoint selection; baseline metrics share the grouped keys.
        if phase != "before_finetune":
            log_payload.update(
                {
                    f"{resolved_log_prefix}_overall_accuracy": metrics["overall_accuracy"],
                    f"{resolved_log_prefix}_seen_type_accuracy": metrics["seen_type_accuracy"],
                    f"{resolved_log_prefix}_unseen_type_accuracy": metrics["unseen_type_accuracy"],
                }
            )
        for question_type, value in metrics["per_type_accuracy"].items():
            log_payload[f"{grouped_prefix}/per_type/{question_type}"] = value
        trainer.log(log_payload)

        return metrics_payload

    def run_pre_finetune_baseline(self, trainer: Trainer) -> Dict[str, Any]:
        global_step = int(trainer.state.global_step)
        self.before_finetune_eval_payload = self._run_eval(
            trainer,
            "eval",
            self.eval_dataset,
            global_step,
            file_stem=f"before_finetune_step_{global_step}",
            log_prefix="before_finetune_eval",
            phase="before_finetune",
        )
        self.before_finetune_test_payload = self._run_eval(
            trainer,
            "test",
            self.test_dataset,
            global_step,
            file_stem=f"before_finetune_step_{global_step}",
            log_prefix="before_finetune_test",
            phase="before_finetune",
        )
        baseline_report = {
            "before_finetune_eval": self.before_finetune_eval_payload,
            "before_finetune_test": self.before_finetune_test_payload,
            "train_count_per_question_type": self.train_type_counts,
        }
        write_json(self.output_dir / "before_finetune_report.json", baseline_report)
        return baseline_report

    def on_evaluate(self, args, state, control, **kwargs):
        trainer: Trainer = kwargs["model"]._hf_peft_trainer_ref
        global_step = int(state.global_step)
        payload = self._run_eval(trainer, "eval", self.eval_dataset, global_step)

        metrics = kwargs.get("metrics")
        if metrics is not None:
            metrics["eval_overall_accuracy"] = payload["overall_accuracy"]
            metrics["eval_seen_type_accuracy"] = payload["seen_type_accuracy"]
            metrics["eval_unseen_type_accuracy"] = payload["unseen_type_accuracy"]

        metric = payload["seen_type_accuracy"]
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_step = global_step
            self.best_dir.mkdir(parents=True, exist_ok=True)
            payload["best_checkpoint_path"] = str(Path(trainer.args.output_dir) / f"checkpoint-{global_step}")
            write_json(self.best_dir / "best_eval_metrics.json", payload)

    def run_final_test(self, trainer: Trainer) -> Dict[str, Any]:
        payload = self._run_eval(
            trainer,
            "test",
            self.test_dataset,
            int(trainer.state.global_step),
        )
        best_model_checkpoint = trainer.state.best_model_checkpoint
        final_report = {
            "before_finetune": self.before_finetune_test_payload,
            "after_finetune": payload,
            "before_finetune_seen_type_test_accuracy": (
                self.before_finetune_test_payload["seen_type_accuracy"]
                if self.before_finetune_test_payload is not None
                else None
            ),
            "before_finetune_unseen_type_test_accuracy": (
                self.before_finetune_test_payload["unseen_type_accuracy"]
                if self.before_finetune_test_payload is not None
                else None
            ),
            "before_finetune_overall_test_accuracy": (
                self.before_finetune_test_payload["overall_accuracy"]
                if self.before_finetune_test_payload is not None
                else None
            ),
            "before_finetune_per_question_type_test_accuracy": (
                self.before_finetune_test_payload["per_type_accuracy"]
                if self.before_finetune_test_payload is not None
                else None
            ),
            "after_finetune_seen_type_test_accuracy": payload["seen_type_accuracy"],
            "after_finetune_unseen_type_test_accuracy": payload["unseen_type_accuracy"],
            "after_finetune_overall_test_accuracy": payload["overall_accuracy"],
            "after_finetune_per_question_type_test_accuracy": payload["per_type_accuracy"],
            "seen_type_test_accuracy": payload["seen_type_accuracy"],
            "unseen_type_test_accuracy": payload["unseen_type_accuracy"],
            "overall_test_accuracy": payload["overall_accuracy"],
            "per_question_type_test_accuracy": payload["per_type_accuracy"],
            "train_count_per_question_type": self.train_type_counts,
            "best_eval_seen_type_accuracy": self.best_metric,
            "best_eval_step": self.best_step,
            "best_model_checkpoint": best_model_checkpoint,
        }
        write_json(self.output_dir / "final_test_report.json", final_report)
        return final_report


GenerationEvalCallback = SmolGenerationEvalCallback
