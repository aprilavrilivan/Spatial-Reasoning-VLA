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

from src.data import build_messages
from src.metrics import normalize_answer, split_seen_unseen_metrics
from src.utils import write_json


DEFAULT_MODEL = "google/gemma-3-4b-it"

GEMMA_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]


def load_gemma_processor(model_name: str):
    return AutoProcessor.from_pretrained(model_name)


def build_gemma_lora_config(lora_r: int, lora_alpha: int, lora_dropout: float) -> LoraConfig:
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=GEMMA_LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_gemma_lora_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    lora_config = build_gemma_lora_config(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    return get_peft_model(model, lora_config)


@dataclass
class GemmaVQATrainCollator:
    processor: Any
    max_seq_length: int = 512

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        images: List[List[Image.Image]] = []

        for feature in features:
            messages = build_messages(feature["question"], feature["answer"])
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            images.append([feature["image"]])

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        batch["labels"] = labels
        return batch


def generate_predictions(
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

        texts = []
        images = []
        for item in batch_items:
            messages = build_messages(item["question"], answer=None)
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)
            images.append([item["image"]])

        model_inputs = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        prompt_len = model_inputs["input_ids"].shape[1]
        generated_only = generated[:, prompt_len:]
        pred_texts = processor.batch_decode(generated_only, skip_special_tokens=True)

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


class GenerationEvalCallback(TrainerCallback):
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
        self.best_dir = self.output_dir / "best_checkpoint_eval_predictions"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_eval(
        self,
        trainer: Trainer,
        split_name: str,
        dataset: Dataset,
        global_step: int,
    ) -> Dict[str, Any]:
        rows = generate_predictions(
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
        pred_path = pred_dir / f"step_{global_step}.csv"
        pd.DataFrame(rows).to_csv(pred_path, index=False)

        metrics_payload = {
            "global_step": global_step,
            "split": split_name,
            "overall_accuracy": metrics["overall_accuracy"],
            "seen_type_accuracy": metrics["seen_type_accuracy"],
            "unseen_type_accuracy": metrics["unseen_type_accuracy"],
            "per_type_accuracy": metrics["per_type_accuracy"],
            "train_count_per_question_type": self.train_type_counts,
            "predictions_csv": str(pred_path),
        }
        write_json(pred_dir / f"step_{global_step}.json", metrics_payload)

        trainer.log(
            {
                f"{split_name}_overall_accuracy": metrics["overall_accuracy"],
                f"{split_name}_seen_type_accuracy": metrics["seen_type_accuracy"],
                f"{split_name}_unseen_type_accuracy": metrics["unseen_type_accuracy"],
            }
        )
        for question_type, value in metrics["per_type_accuracy"].items():
            trainer.log({f"{split_name}_per_type/{question_type}": value})

        return metrics_payload

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
            write_json(self.best_dir / "best_eval_metrics.json", payload)

    def run_final_test(self, trainer: Trainer) -> Dict[str, Any]:
        payload = self._run_eval(
            trainer,
            "test",
            self.test_dataset,
            int(trainer.state.global_step),
        )
        final_report = {
            "seen_type_test_accuracy": payload["seen_type_accuracy"],
            "unseen_type_test_accuracy": payload["unseen_type_accuracy"],
            "overall_test_accuracy": payload["overall_accuracy"],
            "per_question_type_test_accuracy": payload["per_type_accuracy"],
            "train_count_per_question_type": self.train_type_counts,
            "best_eval_seen_type_accuracy": self.best_metric,
            "best_eval_step": self.best_step,
        }
        write_json(self.output_dir / "final_test_report.json", final_report)
        return final_report
