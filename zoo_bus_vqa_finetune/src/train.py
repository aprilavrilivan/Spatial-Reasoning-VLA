from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import Trainer, TrainingArguments, set_seed

try:
    import wandb
except ImportError:
    wandb = None

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.data import ZooBusEvalDataset, ZooBusTrainDataset
from src.metrics import get_train_question_type_counts
from src.models.gemma import (
    DEFAULT_MODEL,
    GenerationEvalCallback,
    GemmaVQATrainCollator,
    load_gemma_lora_model,
    load_gemma_processor,
)
from src.utils import DEFAULT_DATASET, HELD_OUT_TYPES, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_new_tokens_eval", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=300)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str, default="zoo-bus-vqa-gemma-finetune")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.report_to == "wandb":
        if wandb is None:
            raise ImportError("wandb is not installed but --report_to=wandb was requested.")
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    dataset = load_dataset(args.dataset_name)
    required_splits = {"train", "evaluation", "test"}
    missing = required_splits - set(dataset.keys())
    if missing:
        raise ValueError(f"Dataset must contain splits {required_splits}. Missing: {missing}")

    train_split = dataset["train"]
    eval_split = dataset["evaluation"]
    test_split = dataset["test"]

    train_type_counts = get_train_question_type_counts(train_split)
    held_out_leakage = {question_type: train_type_counts.get(question_type, 0) for question_type in HELD_OUT_TYPES}
    if any(count != 0 for count in held_out_leakage.values()):
        raise ValueError(f"Held-out question types leaked into train split: {held_out_leakage}")

    processor = load_gemma_processor(args.model_name)
    model = load_gemma_lora_model(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.print_trainable_parameters()

    train_dataset = ZooBusTrainDataset(train_split)
    eval_dataset = ZooBusEvalDataset(eval_split)
    test_dataset = ZooBusEvalDataset(test_split)
    collator = GemmaVQATrainCollator(processor=processor, max_seq_length=args.max_seq_length)

    steps_per_epoch = math.ceil(
        len(train_dataset)
        / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * args.epochs

    summary = {
        "model": args.model_name,
        "method": "bf16 LoRA-SFT",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lr": args.lr,
        "epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "selection_metric": "seen-type eval accuracy",
        "estimated_steps_per_epoch": steps_per_epoch,
        "estimated_total_steps": total_steps,
        "train_count_per_question_type": train_type_counts,
    }
    write_json(outdir / "run_summary.json", summary)

    training_args = TrainingArguments(
        output_dir=str(outdir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to=args.report_to,
        load_best_model_at_end=True,
        metric_for_best_model="eval_seen_type_accuracy",
        greater_is_better=True,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    trainer.model._hf_peft_trainer_ref = trainer

    eval_callback = GenerationEvalCallback(
        processor=processor,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        output_dir=str(outdir),
        eval_batch_size=args.per_device_eval_batch_size,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens_eval,
        train_type_counts=train_type_counts,
    )
    trainer.add_callback(eval_callback)

    trainer.train()
    trainer.evaluate()

    final_test_report = eval_callback.run_final_test(trainer)
    print(json.dumps(final_test_report, indent=2, ensure_ascii=False))

    if args.report_to == "wandb" and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
