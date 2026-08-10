#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from datasets import load_dataset
from peft import PeftModel
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    import wandb
except ImportError:
    wandb = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gemma import DEFAULT_MODEL as GEMMA_DEFAULT_MODEL
from src.models.internvl import (
    DEFAULT_MODEL as INTERNVL_DEFAULT_MODEL,
    _drop_unused_internvl_fields,
)
from src.models.qwen import DEFAULT_MODEL as QWEN_DEFAULT_MODEL, _drop_unused_qwen_fields
from src.models.smol import DEFAULT_MODEL as SMOL_DEFAULT_MODEL
from src.utils import ensure_rgb, use_left_padding, write_json


ERQA_DATASET_NAME = "FlagEval/ERQA"
ROBOSPATIAL_DATASET_NAME = "chanhee-luke/RoboSpatial-Home"

MODEL_DEFAULTS = {
    "gemma": GEMMA_DEFAULT_MODEL,
    "qwen": QWEN_DEFAULT_MODEL,
    "smol": SMOL_DEFAULT_MODEL,
    "internvl": INTERNVL_DEFAULT_MODEL,
}

DEFAULT_MAX_SEQ_LENGTH = {
    "gemma": 2048,
    "qwen": 2048,
    "smol": 2048,
    "internvl": 4096,
}

ERQA_SEPARATE_IMAGE_LIMIT = 6
ERQA_CONTACT_SHEET_CELL_SIZE = 360


@dataclass
class RoboticSpatialSample:
    dataset: str
    split: str
    sample_id: str
    images: List[Image.Image]
    question: str
    gold_answer: str
    answer_format: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    mask: Image.Image | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate base and Zoo-Bus-VQA fine-tuned VLMs on ERQA and "
            "RoboSpatial-Home without additional training."
        )
    )
    parser.add_argument("--dataset", choices=["erqa", "robospatial", "all"], default="all")
    parser.add_argument("--model_family", choices=["gemma", "qwen", "smol", "internvl"], required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--state", choices=["before", "after", "both"], default="both")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "external_eval" / "robotic_spatial_results"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--erqa_dataset_name", default=ERQA_DATASET_NAME)
    parser.add_argument("--robospatial_dataset_name", default=ROBOSPATIAL_DATASET_NAME)
    parser.add_argument(
        "--robospatial_splits",
        nargs="+",
        default=["context", "compatibility", "configuration"],
        choices=["context", "compatibility", "configuration"],
    )
    parser.add_argument(
        "--context_mask_threshold",
        type=int,
        default=128,
        help="Threshold used after per-example mask-polarity inference.",
    )
    parser.add_argument(
        "--max_context_points",
        type=int,
        default=10,
        help="Evaluate at most this many parsed coordinate pairs per context answer.",
    )
    parser.add_argument(
        "--include_prompt_text",
        action="store_true",
        help="Store full prompts in prediction CSVs. Off by default to keep artifacts small.",
    )
    parser.add_argument("--report_to", choices=["none", "wandb"], default="none")
    parser.add_argument("--wandb_project", default="zoo-bus-vqa-robotic-spatial-eval")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_log_artifact", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def build_eval_question(dataset: str, answer_format: str, question: str) -> str:
    question = str(question).strip()
    if dataset == "erqa":
        return (
            "Answer the embodied visual question using only the option letter "
            "(A, B, C, or D). Do not explain your reasoning.\n"
            f"Question: {question}"
        )
    if answer_format == "yes_no":
        return (
            "Answer the spatial question using yes or no only. "
            "Do not explain your reasoning.\n"
            f"Question: {question}"
        )
    if answer_format == "points":
        return (
            "Answer the spatial question using only a Python-style list of normalized "
            "(x, y) coordinate tuples. Do not explain your reasoning.\n"
            f"Question: {question}"
        )
    return (
        "Answer the visual question using a short final answer only. "
        "Do not explain your reasoning.\n"
        f"Question: {question}"
    )


def ensure_image_list(value: Any) -> List[Image.Image]:
    if isinstance(value, Image.Image):
        return [ensure_rgb(value)]
    if isinstance(value, (list, tuple)):
        images = [ensure_rgb(image) for image in value if isinstance(image, Image.Image)]
        if images:
            return images
    raise ValueError(f"Could not decode image list from value of type {type(value)!r}")


def make_contact_sheet(images: Sequence[Image.Image]) -> Image.Image:
    """Preserve many ERQA views without overflowing VLM context windows."""
    if not images:
        raise ValueError("Cannot build a contact sheet from an empty image list.")
    cell = ERQA_CONTACT_SHEET_CELL_SIZE
    pad = 12
    label_h = 24
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    width = cols * cell + (cols + 1) * pad
    height = rows * (cell + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        x0 = pad + col * (cell + pad)
        y0 = pad + row * (cell + label_h + pad)
        draw.text((x0, y0), f"View {index + 1}", fill=(0, 0, 0))
        thumb = ImageOps.contain(ensure_rgb(image), (cell, cell), method=Image.Resampling.LANCZOS)
        x = x0 + (cell - thumb.width) // 2
        y = y0 + label_h + (cell - thumb.height) // 2
        sheet.paste(thumb, (x, y))
    return sheet


def prepare_erqa_samples(dataset_name: str, limit: int | None) -> List[RoboticSpatialSample]:
    rows = load_dataset(dataset_name, split="test")
    samples: List[RoboticSpatialSample] = []
    for idx, row in enumerate(rows):
        if limit is not None and len(samples) >= limit:
            break
        original_images = ensure_image_list(row["images"])
        if len(original_images) > ERQA_SEPARATE_IMAGE_LIMIT:
            images = [make_contact_sheet(original_images)]
            input_mode = "contact_sheet"
        else:
            images = original_images
            input_mode = "separate_images"
        question_type = str(row.get("question_type", "unknown"))
        samples.append(
            RoboticSpatialSample(
                dataset="erqa",
                split="test",
                sample_id=str(row.get("question_id", f"erqa_{idx}")),
                images=images,
                question=build_eval_question("erqa", "letter", row["question"]),
                gold_answer=str(row["answer"]).strip().upper()[:1],
                answer_format="letter",
                metadata={
                    "question_type": question_type,
                    "image_count": len(original_images),
                    "input_image_count": len(images),
                    "input_mode": input_mode,
                    "visual_indices": json.dumps(list(row.get("visual_indices", []))),
                    "row_index": idx,
                },
            )
        )
    return samples


def prepare_robospatial_samples(
    dataset_name: str,
    splits: Sequence[str],
    limit: int | None,
) -> List[RoboticSpatialSample]:
    samples: List[RoboticSpatialSample] = []
    for split in splits:
        rows = load_dataset(dataset_name, split=split)
        for idx, row in enumerate(rows):
            if limit is not None and len(samples) >= limit:
                return samples
            category = str(row.get("category", split))
            answer_format = "points" if category == "context" else "yes_no"
            mask = ensure_rgb(row["mask"]) if category == "context" and row.get("mask") is not None else None
            samples.append(
                RoboticSpatialSample(
                    dataset="robospatial",
                    split=split,
                    sample_id=f"{split}_{idx}",
                    images=[ensure_rgb(row["img"])],
                    question=build_eval_question("robospatial", answer_format, row["question"]),
                    gold_answer=str(row["answer"]).strip(),
                    answer_format=answer_format,
                    mask=mask,
                    metadata={
                        "category": category,
                        "row_index": idx,
                        "has_depth_image": int(row.get("depth_image") is not None),
                        "has_mask": int(mask is not None),
                    },
                )
            )
    return samples


def load_processor(model_name: str):
    try:
        processor = AutoProcessor.from_pretrained(model_name)
    except (OSError, ValueError, ImportError):
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return use_left_padding(processor)


def flash_attention_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("flash_attn") is not None and torch.cuda.is_available()


def load_base_model(model_family: str, model_name: str, use_flash_attention: bool = True):
    common_kwargs = {"torch_dtype": torch.bfloat16}
    if model_family == "qwen":
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2" if use_flash_attention and flash_attention_available() else "sdpa",
            **common_kwargs,
        )

    if model_family == "internvl":
        attention_attempts = (
            ["flash_attention_2"] if use_flash_attention and flash_attention_available() else []
        ) + ["sdpa", None]
        last_error: Exception | None = None
        for attn in attention_attempts:
            try:
                kwargs = dict(common_kwargs)
                if attn is not None:
                    kwargs["attn_implementation"] = attn
                return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
            except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
                last_error = exc
                print(f"InternVL load fallback after {attn or 'default'} failed: {exc}", flush=True)
        return AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            **common_kwargs,
        )

    return AutoModelForImageTextToText.from_pretrained(
        model_name,
        attn_implementation="sdpa",
        **common_kwargs,
    )


def build_conversation(model_family: str, sample: RoboticSpatialSample) -> List[Dict[str, Any]]:
    if model_family == "gemma":
        content = [{"type": "image"} for _ in sample.images]
    else:
        content = [{"type": "image", "image": image} for image in sample.images]
    content.append({"type": "text", "text": sample.question})
    return [{"role": "user", "content": content}]


def prepare_inputs(
    *,
    model_family: str,
    processor,
    batch_items: Sequence[RoboticSpatialSample],
    max_seq_length: int,
) -> Dict[str, Any]:
    conversations = [build_conversation(model_family, item) for item in batch_items]
    if model_family == "gemma":
        texts = [
            processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        images = [item.images for item in batch_items]
        return dict(
            processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        )

    if model_family == "internvl":
        try:
            model_inputs = processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            return _drop_unused_internvl_fields(dict(model_inputs))
        except (TypeError, ValueError):
            texts = [
                processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for conversation in conversations
            ]
            images = [item.images for item in batch_items]
            model_inputs = processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            return _drop_unused_internvl_fields(dict(model_inputs))

    model_inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = dict(model_inputs)
    if model_family == "qwen":
        model_inputs = _drop_unused_qwen_fields(model_inputs)
    elif model_family == "internvl":
        model_inputs = _drop_unused_internvl_fields(model_inputs)
    return model_inputs


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
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


def generate_raw_predictions(
    *,
    model,
    processor,
    model_family: str,
    samples: Sequence[RoboticSpatialSample],
    batch_size: int,
    max_seq_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> List[str]:
    model.eval()
    predictions: List[str] = []
    for start in range(0, len(samples), batch_size):
        batch_items = samples[start : start + batch_size]
        model_inputs = prepare_inputs(
            model_family=model_family,
            processor=processor,
            batch_items=batch_items,
            max_seq_length=max_seq_length,
        )
        model_inputs = move_batch_to_device(model_inputs, device)
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
        decoded = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        predictions.extend(decoded)
    return predictions


YES_NO_MAP = {
    "yes": "yes",
    "true": "yes",
    "correct": "yes",
    "no": "no",
    "false": "no",
    "incorrect": "no",
}


def parse_yes_no(raw_prediction: str) -> tuple[str | None, str]:
    text = raw_prediction.strip().lower()
    text = re.sub(r"^[`'\"\\s({\\[]+", "", text)
    first_token = re.split(r"[^a-z]+", text, maxsplit=1)[0]
    if first_token in YES_NO_MAP:
        return YES_NO_MAP[first_token], "first_token"

    matches = [
        YES_NO_MAP[match.group(0)]
        for match in re.finditer(r"\b(yes|no|true|false|correct|incorrect)\b", text)
    ]
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0], "single_keyword"
    return None, "unparsed"


def parse_letter(raw_prediction: str) -> tuple[str | None, str]:
    upper = raw_prediction.strip().upper()
    patterns = [
        r"^\s*\(?([A-D])\)?(?:[\.\):\s]|$)",
        r"\bOPTION\s+([A-D])\b",
        r"\bANSWER\s+(?:IS\s+)?([A-D])\b",
        r"\bTHE\s+ANSWER\s+IS\s+([A-D])\b",
        r"\bCHOICE\s+([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1), "letter"
    return None, "unparsed"


def normalize_yes_no_gold(answer: str) -> str:
    parsed, _ = parse_yes_no(answer)
    if parsed is None:
        raise ValueError(f"RoboSpatial yes/no gold answer could not be parsed: {answer!r}")
    return parsed


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_points(text: str, *, max_points: int = 10) -> tuple[List[tuple[float, float]], str]:
    points: List[tuple[float, float]] = []
    try:
        literal = ast.literal_eval(text.strip())
        if isinstance(literal, tuple) and len(literal) == 2:
            literal = [literal]
        if isinstance(literal, list):
            for item in literal:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    x, y = float(item[0]), float(item[1])
                    if 1.0 < max(abs(x), abs(y)) <= 100.0:
                        x, y = x / 100.0, y / 100.0
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                        points.append((x, y))
                        if len(points) >= max_points:
                            return points, "literal"
    except (ValueError, SyntaxError, TypeError):
        pass

    pair_pattern = re.compile(
        r"[\(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\)\]]"
    )
    for match in pair_pattern.finditer(text):
        x, y = float(match.group(1)), float(match.group(2))
        if 1.0 < max(abs(x), abs(y)) <= 100.0:
            x, y = x / 100.0, y / 100.0
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            points.append((x, y))
            if len(points) >= max_points:
                return points, "regex_pairs"
    if points:
        return points, "regex_pairs"

    numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if len(numbers) >= 2:
        for idx in range(0, min(len(numbers) - 1, max_points * 2), 2):
            x, y = numbers[idx], numbers[idx + 1]
            if 1.0 < max(abs(x), abs(y)) <= 100.0:
                x, y = x / 100.0, y / 100.0
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                points.append((x, y))
    return (points, "loose_numbers") if points else ([], "unparsed")


def mask_intensity(mask: Image.Image, point: tuple[float, float]) -> int:
    gray = mask.convert("L")
    width, height = gray.size
    x = min(max(int(round(point[0] * (width - 1))), 0), width - 1)
    y = min(max(int(round(point[1] * (height - 1))), 0), height - 1)
    return int(gray.getpixel((x, y)))


def infer_mask_positive_is_bright(
    mask: Image.Image,
    gold_points: Sequence[tuple[float, float]],
    threshold: int,
) -> bool:
    if not gold_points:
        return True
    intensities = [mask_intensity(mask, point) for point in gold_points]
    return (sum(value >= threshold for value in intensities) / len(intensities)) >= 0.5


def point_hits_mask(
    mask: Image.Image,
    point: tuple[float, float],
    *,
    positive_is_bright: bool,
    threshold: int,
) -> bool:
    value = mask_intensity(mask, point)
    return value >= threshold if positive_is_bright else value < threshold


def score_context_points(
    *,
    raw_prediction: str,
    gold_answer: str,
    mask: Image.Image | None,
    threshold: int,
    max_points: int,
) -> Dict[str, Any]:
    predicted_points, parse_method = parse_points(raw_prediction, max_points=max_points)
    gold_points, gold_parse_method = parse_points(gold_answer, max_points=100)
    if mask is None:
        return {
            "parsed_prediction": json.dumps(predicted_points),
            "parse_method": parse_method,
            "num_predicted_points": len(predicted_points),
            "first_point_correct": 0,
            "any_point_correct": 0,
            "valid_point_fraction": 0.0,
            "gold_parse_method": gold_parse_method,
            "mask_positive_is_bright": "",
        }
    positive_is_bright = infer_mask_positive_is_bright(mask, gold_points, threshold)
    valid_flags = [
        point_hits_mask(
            mask,
            point,
            positive_is_bright=positive_is_bright,
            threshold=threshold,
        )
        for point in predicted_points
    ]
    first_point_correct = int(bool(valid_flags) and valid_flags[0])
    any_point_correct = int(any(valid_flags))
    valid_fraction = sum(valid_flags) / len(valid_flags) if valid_flags else 0.0
    return {
        "parsed_prediction": json.dumps(predicted_points),
        "parse_method": parse_method,
        "num_predicted_points": len(predicted_points),
        "first_point_correct": first_point_correct,
        "any_point_correct": any_point_correct,
        "valid_point_fraction": valid_fraction,
        "gold_parse_method": gold_parse_method,
        "mask_positive_is_bright": int(positive_is_bright),
    }


def rows_from_predictions(
    *,
    state: str,
    model_family: str,
    model_name: str,
    samples: Sequence[RoboticSpatialSample],
    raw_predictions: Sequence[str],
    include_prompt_text: bool,
    context_mask_threshold: int,
    max_context_points: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample, raw_prediction in zip(samples, raw_predictions):
        parsed_prediction: str | None = None
        parse_method = "unparsed"
        is_correct = 0
        extra: Dict[str, Any] = {}

        if sample.answer_format == "letter":
            parsed_prediction, parse_method = parse_letter(raw_prediction)
            is_correct = int(parsed_prediction == sample.gold_answer)
        elif sample.answer_format == "yes_no":
            parsed_prediction, parse_method = parse_yes_no(raw_prediction)
            gold = normalize_yes_no_gold(sample.gold_answer)
            is_correct = int(parsed_prediction == gold)
        elif sample.answer_format == "points":
            context = score_context_points(
                raw_prediction=raw_prediction,
                gold_answer=sample.gold_answer,
                mask=sample.mask,
                threshold=context_mask_threshold,
                max_points=max_context_points,
            )
            parsed_prediction = context.pop("parsed_prediction")
            parse_method = str(context.pop("parse_method"))
            # The main context score is intentionally first-point validity:
            # it is stricter than "any point" and avoids rewarding long random lists.
            is_correct = int(context["first_point_correct"])
            extra.update(context)
        else:
            raise ValueError(f"Unsupported answer format: {sample.answer_format}")

        row: Dict[str, Any] = {
            "dataset": sample.dataset,
            "split": sample.split,
            "state": state,
            "model_family": model_family,
            "model_name": model_name,
            "id": sample.sample_id,
            "answer_format": sample.answer_format,
            "image_count": len(sample.images),
            "gold_answer": sample.gold_answer,
            "raw_prediction": raw_prediction,
            "parsed_prediction": parsed_prediction or "",
            "parse_method": parse_method,
            "is_correct": is_correct,
        }
        if include_prompt_text:
            row["question"] = sample.question
        row.update(sample.metadata)
        row.update(extra)
        rows.append(row)
    return rows


def accuracy(rows: Sequence[Dict[str, Any]], key: str = "is_correct") -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def grouped_rows(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return dict(sorted(groups.items()))


def grouped_accuracy(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    return {name: accuracy(items) for name, items in grouped_rows(rows, key).items()}


def grouped_counts(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))


def image_count_bucket(image_count: int) -> str:
    if image_count <= 1:
        return "1"
    if image_count <= 3:
        return "2-3"
    return "4+"


def compute_erqa_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows_with_buckets = []
    for row in rows:
        copied = dict(row)
        copied["image_count_bucket"] = image_count_bucket(int(copied.get("image_count", 1)))
        rows_with_buckets.append(copied)
    question_type_accuracy = grouped_accuracy(rows_with_buckets, "question_type")
    parse_failures = sum(1 for row in rows_with_buckets if not row["parsed_prediction"])
    return {
        "num_samples": len(rows_with_buckets),
        "num_correct": sum(int(row["is_correct"]) for row in rows_with_buckets),
        "overall_accuracy": accuracy(rows_with_buckets),
        "parse_failure_rate": parse_failures / len(rows_with_buckets) if rows_with_buckets else 0.0,
        "question_type_accuracy": question_type_accuracy,
        "question_type_macro_accuracy": mean(question_type_accuracy.values()),
        "question_type_counts": grouped_counts(rows_with_buckets, "question_type"),
        "answer_distribution": grouped_counts(rows_with_buckets, "gold_answer"),
        "prediction_distribution": grouped_counts(rows_with_buckets, "parsed_prediction"),
        "image_count_accuracy": grouped_accuracy(rows_with_buckets, "image_count_bucket"),
        "image_count_counts": grouped_counts(rows_with_buckets, "image_count_bucket"),
    }


def compute_robospatial_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    category_accuracy = grouped_accuracy(rows, "category")
    parse_failures = sum(1 for row in rows if not row["parsed_prediction"] or row["parse_method"] == "unparsed")
    context_rows = [row for row in rows if row.get("answer_format") == "points"]
    binary_rows = [row for row in rows if row.get("answer_format") == "yes_no"]
    return {
        "num_samples": len(rows),
        "num_correct": sum(int(row["is_correct"]) for row in rows),
        "overall_accuracy": accuracy(rows),
        "parse_failure_rate": parse_failures / len(rows) if rows else 0.0,
        "category_accuracy": category_accuracy,
        "category_macro_accuracy": mean(category_accuracy.values()),
        "category_counts": grouped_counts(rows, "category"),
        "binary_accuracy": accuracy(binary_rows) if binary_rows else None,
        "binary_counts": grouped_counts(binary_rows, "gold_answer"),
        "context_first_point_accuracy": accuracy(context_rows, "first_point_correct") if context_rows else None,
        "context_any_point_accuracy": accuracy(context_rows, "any_point_correct") if context_rows else None,
        "context_mean_valid_point_fraction": accuracy(context_rows, "valid_point_fraction") if context_rows else None,
        "context_parse_failure_rate": (
            sum(1 for row in context_rows if row["parse_method"] == "unparsed") / len(context_rows)
            if context_rows
            else None
        ),
        "prediction_distribution": grouped_counts(rows, "parsed_prediction"),
    }


def compute_metrics(dataset_name: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if dataset_name == "erqa":
        return compute_erqa_metrics(rows)
    if dataset_name == "robospatial":
        return compute_robospatial_metrics(rows)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def subtract_nested(after: Any, before: Any) -> Any:
    if isinstance(after, (int, float)) and isinstance(before, (int, float)):
        if math.isnan(float(after)) or math.isnan(float(before)):
            return None
        return after - before
    if isinstance(after, dict) and isinstance(before, dict):
        return {
            key: subtract_nested(after[key], before[key])
            for key in sorted(set(after) & set(before))
        }
    return None


def flatten_metrics(payload: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in payload.items():
        clean_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, bool):
            flat[clean_key] = float(value)
        elif isinstance(value, (int, float)) and value is not None:
            flat[clean_key] = float(value)
        elif isinstance(value, dict):
            flat.update(flatten_metrics(value, clean_key))
    return flat


def maybe_log_wandb(payload: Dict[str, float], step: int | None = None) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(payload, step=step)


def maybe_log_wandb_artifact(output_dir: Path, model_family: str, run_name: str) -> None:
    if wandb is None or wandb.run is None:
        return
    artifact = wandb.Artifact(
        name=f"robotic-spatial-{model_family}-{run_name}",
        type="robotic-spatial-eval-results",
    )
    for path in output_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".csv"}:
            artifact.add_file(str(path), name=str(path.relative_to(output_dir)))
    wandb.log_artifact(artifact)


def write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_samples_for_dataset(args: argparse.Namespace, dataset_name: str) -> List[RoboticSpatialSample]:
    if dataset_name == "erqa":
        return prepare_erqa_samples(args.erqa_dataset_name, args.limit)
    if dataset_name == "robospatial":
        return prepare_robospatial_samples(
            args.robospatial_dataset_name,
            args.robospatial_splits,
            args.limit,
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate_state(
    *,
    state: str,
    model,
    processor,
    model_family: str,
    model_name: str,
    dataset_name: str,
    samples: Sequence[RoboticSpatialSample],
    output_dir: Path,
    batch_size: int,
    max_seq_length: int,
    max_new_tokens: int,
    device: torch.device,
    include_prompt_text: bool,
    context_mask_threshold: int,
    max_context_points: int,
) -> Dict[str, Any]:
    raw_predictions = generate_raw_predictions(
        model=model,
        processor=processor,
        model_family=model_family,
        samples=samples,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    rows = rows_from_predictions(
        state=state,
        model_family=model_family,
        model_name=model_name,
        samples=samples,
        raw_predictions=raw_predictions,
        include_prompt_text=include_prompt_text,
        context_mask_threshold=context_mask_threshold,
        max_context_points=max_context_points,
    )
    metrics = compute_metrics(dataset_name, rows)
    pred_path = output_dir / f"{dataset_name}_{state}_predictions.csv"
    report_path = output_dir / f"{dataset_name}_{state}_report.json"
    write_rows_csv(pred_path, rows)
    payload = {
        "dataset": dataset_name,
        "state": state,
        "model_family": model_family,
        "model_name": model_name,
        "num_samples": len(rows),
        "metrics": metrics,
        "predictions_csv": str(pred_path),
    }
    write_json(report_path, payload)
    maybe_log_wandb(flatten_metrics(metrics, f"{dataset_name}/{state}"))
    return payload


def main() -> None:
    args = parse_args()
    model_name = args.model_name or MODEL_DEFAULTS[args.model_family]
    max_seq_length = args.max_seq_length or DEFAULT_MAX_SEQ_LENGTH[args.model_family]
    device = torch.device(args.device)

    run_name = f"{args.model_family}_{now_timestamp()}"
    output_dir = Path(args.output_dir) / args.model_family / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.report_to == "wandb":
        if wandb is None:
            raise ImportError("wandb is not installed but --report_to=wandb was requested.")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or run_name,
            config=vars(args) | {
                "resolved_model_name": model_name,
                "resolved_max_seq_length": max_seq_length,
            },
        )

    print(f"Loading processor/model: {model_name}", flush=True)
    processor = load_processor(model_name)
    model = load_base_model(args.model_family, model_name)
    model.to(device)
    model.eval()

    datasets_to_run = ["erqa", "robospatial"] if args.dataset == "all" else [args.dataset]
    samples_by_dataset: Dict[str, List[RoboticSpatialSample]] = {}
    reports_by_dataset: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for dataset_name in datasets_to_run:
        print(f"Loading {dataset_name} samples...", flush=True)
        samples = load_samples_for_dataset(args, dataset_name)
        if not samples:
            raise ValueError(f"No samples loaded for {dataset_name}")
        samples_by_dataset[dataset_name] = samples
        print(f"Loaded {len(samples)} {dataset_name} samples.", flush=True)

    summary: Dict[str, Any] = {
        "model_family": args.model_family,
        "model_name": model_name,
        "adapter_path": args.adapter_path,
        "state": args.state,
        "batch_size": args.batch_size,
        "max_seq_length": max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "output_dir": str(output_dir),
        "datasets": {},
    }

    if args.state in {"before", "both"}:
        for dataset_name, samples in samples_by_dataset.items():
            print(f"Evaluating {dataset_name} before fine-tuning...", flush=True)
            reports_by_dataset[dataset_name]["before"] = evaluate_state(
                state="before",
                model=model,
                processor=processor,
                model_family=args.model_family,
                model_name=model_name,
                dataset_name=dataset_name,
                samples=samples,
                output_dir=output_dir,
                batch_size=args.batch_size,
                max_seq_length=max_seq_length,
                max_new_tokens=args.max_new_tokens,
                device=device,
                include_prompt_text=args.include_prompt_text,
                context_mask_threshold=args.context_mask_threshold,
                max_context_points=args.max_context_points,
            )

    if args.state in {"after", "both"}:
        if not args.adapter_path:
            raise ValueError("--adapter_path is required for --state after or --state both")
        print(f"Loading LoRA adapter from {args.adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model.to(device)
        model.eval()
        for dataset_name, samples in samples_by_dataset.items():
            print(f"Evaluating {dataset_name} after fine-tuning...", flush=True)
            reports_by_dataset[dataset_name]["after"] = evaluate_state(
                state="after",
                model=model,
                processor=processor,
                model_family=args.model_family,
                model_name=model_name,
                dataset_name=dataset_name,
                samples=samples,
                output_dir=output_dir,
                batch_size=args.batch_size,
                max_seq_length=max_seq_length,
                max_new_tokens=args.max_new_tokens,
                device=device,
                include_prompt_text=args.include_prompt_text,
                context_mask_threshold=args.context_mask_threshold,
                max_context_points=args.max_context_points,
            )

    for dataset_name, reports in reports_by_dataset.items():
        comparison: Dict[str, Any] = {
            "dataset": dataset_name,
            "model_family": args.model_family,
            "model_name": model_name,
            "num_samples": len(samples_by_dataset[dataset_name]),
            "reports": reports,
        }
        if "before" in reports and "after" in reports:
            comparison["delta_metrics"] = subtract_nested(
                reports["after"]["metrics"],
                reports["before"]["metrics"],
            )
            maybe_log_wandb(flatten_metrics(comparison["delta_metrics"], f"{dataset_name}/delta"))
        write_json(output_dir / f"{dataset_name}_comparison_report.json", comparison)
        summary["datasets"][dataset_name] = comparison

    write_json(output_dir / "robotic_spatial_eval_summary.json", summary)
    if args.report_to == "wandb" and args.wandb_log_artifact:
        maybe_log_wandb_artifact(output_dir, args.model_family, run_name)
    if wandb is not None and wandb.run is not None:
        wandb.finish()
    print(f"Wrote robotic spatial evaluation results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
