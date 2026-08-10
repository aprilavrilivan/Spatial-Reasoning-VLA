from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch import nn
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


DEFAULT_MODEL = "OpenGVLab/InternVL3_5-4B-HF"
METHOD = "bf16 LoRA-SFT"

INTERNVL_PREFERRED_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]


def build_internvl_messages(
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


def load_internvl_processor(model_name: str):
    try:
        processor = AutoProcessor.from_pretrained(model_name)
    except (OSError, ValueError, ImportError) as exc:
        print(
            "AutoProcessor failed without trust_remote_code for InternVL; retrying "
            f"with trust_remote_code=True. Reason: {exc}",
            flush=True,
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return use_left_padding(processor)


def _flash_attention_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None and torch.cuda.is_available()


def _load_internvl_base_model(
    model_name: str,
    *,
    use_flash_attention: bool = True,
    trust_remote_code: bool = False,
):
    common_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": trust_remote_code,
    }

    attention_attempts: List[str | None] = []
    if use_flash_attention and _flash_attention_available():
        attention_attempts.append("flash_attention_2")
    attention_attempts.extend(["sdpa", None])

    last_error: Exception | None = None
    for attn_implementation in attention_attempts:
        try:
            kwargs = dict(common_kwargs)
            if attn_implementation is not None:
                kwargs["attn_implementation"] = attn_implementation
            return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            last_error = exc
            attn_name = attn_implementation or "default attention"
            print(
                f"Could not load InternVL with {attn_name}; trying fallback if available. "
                f"Reason: {exc}",
                flush=True,
            )

    if not trust_remote_code:
        print(
            "InternVL model loading failed without trust_remote_code; retrying with "
            "trust_remote_code=True.",
            flush=True,
        )
        return _load_internvl_base_model(
            model_name,
            use_flash_attention=use_flash_attention,
            trust_remote_code=True,
        )

    raise RuntimeError(f"Could not load InternVL model {model_name}.") from last_error


def _linear_suffixes_present(model) -> set[str]:
    suffixes: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module_name:
            suffixes.add(module_name.rsplit(".", 1)[-1])
    return suffixes


def resolve_internvl_lora_target_modules(model) -> List[str]:
    suffixes = _linear_suffixes_present(model)
    target_modules = [
        module_name
        for module_name in INTERNVL_PREFERRED_LORA_TARGETS
        if module_name in suffixes
    ]
    missing = [
        module_name
        for module_name in INTERNVL_PREFERRED_LORA_TARGETS
        if module_name not in suffixes
    ]
    if missing:
        print(
            "InternVL LoRA target warning: missing expected linear module suffixes "
            f"{missing}; using existing targets {target_modules}.",
            flush=True,
        )
    if not target_modules:
        raise ValueError(
            "Could not find valid InternVL LoRA target modules. "
            f"Available linear module suffixes include: {sorted(suffixes)[:40]}"
        )
    return target_modules


def build_internvl_lora_config(
    model,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> LoraConfig:
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=resolve_internvl_lora_target_modules(model),
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_internvl_lora_model(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    use_flash_attention: bool = True,
):
    model = _load_internvl_base_model(
        model_name,
        use_flash_attention=use_flash_attention,
    )
    lora_config = build_internvl_lora_config(
        model=model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


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


def _drop_unused_internvl_fields(batch: Dict[str, Any]) -> Dict[str, Any]:
    batch.pop("token_type_ids", None)
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


def _apply_chat_template_tokenized(
    processor: Any,
    conversations: List[List[Dict[str, Any]]],
    *,
    add_generation_prompt: bool,
    max_seq_length: int,
) -> Dict[str, Any]:
    if not hasattr(processor, "apply_chat_template"):
        raise AttributeError("InternVL processor does not expose apply_chat_template.")

    try:
        batch = processor.apply_chat_template(
            conversations,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "truncation": True,
                "max_length": max_seq_length,
            },
        )
        return _drop_unused_internvl_fields(dict(batch))
    except (TypeError, ValueError):
        pass

    try:
        batch = processor.apply_chat_template(
            conversations,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return _drop_unused_internvl_fields(dict(batch))
    except (TypeError, ValueError):
        texts = processor.apply_chat_template(
            conversations,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        images = [
            message["content"][0]["image"]
            for conversation in conversations
            for message in conversation
            if message["role"] == "user"
        ]
        try:
            batch = processor(
                text=texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            batch = processor(
                text=texts,
                images=[[image] for image in images],
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
        return _drop_unused_internvl_fields(dict(batch))


def _mask_prompt_tokens_by_length(
    processor: Any,
    features: List[Dict[str, Any]],
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_seq_length: int,
) -> torch.Tensor:
    prompt_conversations = [
        build_internvl_messages(feature["question"], feature["image"], answer=None)
        for feature in features
    ]
    prompt_batch = _apply_chat_template_tokenized(
        processor,
        prompt_conversations,
        add_generation_prompt=True,
        max_seq_length=max_seq_length,
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
class InternVLVQATrainCollator:
    processor: Any
    max_seq_length: int = 4096

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        conversations = [
            build_internvl_messages(feature["question"], feature["image"], feature["answer"])
            for feature in features
        ]
        batch = _apply_chat_template_tokenized(
            self.processor,
            conversations,
            add_generation_prompt=False,
            max_seq_length=self.max_seq_length,
        )

        labels = batch["input_ids"].clone()
        assistant_mask = _assistant_mask_to_tensor(batch.pop("assistant_masks", None), labels)
        if assistant_mask is not None:
            labels[~assistant_mask] = -100
        else:
            # InternVL uses many visual tokens per image, so prompt masking must
            # be computed at a length that keeps all image tokens intact.
            try:
                labels = _mask_prompt_tokens_by_length(
                    self.processor,
                    features,
                    labels,
                    batch.get("attention_mask"),
                    self.max_seq_length,
                )
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                print(
                    "Could not compute InternVL prompt mask; falling back to pad-only "
                    f"label masking. Reason: {exc}",
                    flush=True,
                )

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[batch["input_ids"] == pad_token_id] = -100
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100

        if not bool((labels != -100).any()):
            raise ValueError(
                "InternVL collator produced no trainable answer tokens. "
                "The prompt/image tokens may have filled max_seq_length; increase "
                "--max_seq_length before running a long training job."
            )

        batch["labels"] = labels
        return _cast_floating_tensors_to_bf16(batch)


def generate_internvl_predictions(
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
            build_internvl_messages(item["question"], item["image"], answer=None)
            for item in batch_items
        ]

        model_inputs = _apply_chat_template_tokenized(
            processor,
            conversations,
            add_generation_prompt=True,
            max_seq_length=max_seq_length,
        )
        model_inputs = _move_batch_to_device(model_inputs, device)

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


class InternVLGenerationEvalCallback(TrainerCallback):
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
        rows = generate_internvl_predictions(
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


GenerationEvalCallback = InternVLGenerationEvalCallback
