#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from datasets import load_dataset
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    import wandb
except ImportError:
    wandb = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_messages
from src.models.gemma import DEFAULT_MODEL as GEMMA_DEFAULT_MODEL
from src.models.internvl import (
    DEFAULT_MODEL as INTERNVL_DEFAULT_MODEL,
    _apply_chat_template_tokenized as apply_internvl_chat_template_tokenized,
    _drop_unused_internvl_fields,
    build_internvl_messages,
)
from src.models.qwen import (
    DEFAULT_MODEL as QWEN_DEFAULT_MODEL,
    _drop_unused_qwen_fields,
    build_qwen_messages,
)
from src.models.smol import DEFAULT_MODEL as SMOL_DEFAULT_MODEL, build_smol_messages
from src.utils import ensure_rgb, use_left_padding, write_json


VSR_DATASET_NAME = "cambridgeltl/vsr_zeroshot"
WHATSUP_REPO_URL = "https://github.com/amitakamath/whatsup_vlms"

WHATSUP_CONTROLLED_DOWNLOADS = {
    "controlled_images_dataset.json": "1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm",
    "controlled_clevr_dataset.json": "1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z",
    "controlled_images.tar.gz": "19KGYVQjrV3syb00GgcavB2nZTW5NXX0H",
    "controlled_clevr.tar.gz": "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG",
}

MODEL_DEFAULTS = {
    "gemma": GEMMA_DEFAULT_MODEL,
    "qwen": QWEN_DEFAULT_MODEL,
    "smol": SMOL_DEFAULT_MODEL,
    "internvl": INTERNVL_DEFAULT_MODEL,
}

DEFAULT_MAX_SEQ_LENGTH = {
    "gemma": 512,
    "qwen": 2048,
    "smol": 2048,
    "internvl": 4096,
}


@dataclass
class ExternalSample:
    dataset: str
    split: str
    sample_id: str
    image: Image.Image
    image_path: str
    question: str
    gold_answer: str
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate base and fine-tuned VLMs on external spatial benchmarks "
            "(VSR and WhatsUp) without training."
        )
    )
    parser.add_argument("--dataset", choices=["vsr", "whatsup", "all"], default="all")
    parser.add_argument("--model_family", choices=["gemma", "qwen", "smol", "internvl"], required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter path for after-finetune evaluation.")
    parser.add_argument("--state", choices=["before", "after", "both"], default="both")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "external_eval" / "results"))
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "external_eval" / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vsr_split", default="test")
    parser.add_argument("--vsr_image_timeout", type=float, default=20.0)
    parser.add_argument("--skip_missing_images", action="store_true")

    parser.add_argument("--whatsup_root", default=None)
    parser.add_argument("--download_whatsup", action="store_true")
    parser.add_argument(
        "--whatsup_subsets",
        nargs="+",
        default=["controlled_images", "controlled_clevr"],
        choices=["controlled_images", "controlled_clevr"],
    )
    parser.add_argument(
        "--include_prompt_text",
        action="store_true",
        help="Keep full prompt text in prediction CSVs. Off by default to keep files smaller.",
    )
    parser.add_argument("--report_to", choices=["none", "wandb"], default="wandb")
    parser.add_argument("--wandb_project", default="zoo-bus-vqa-external-spatial-eval")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument(
        "--wandb_log_artifact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload JSON/CSV result files as a W&B artifact when --report_to=wandb.",
    )
    return parser.parse_args()


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def stable_int_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def safe_filename_from_url(url: str, fallback: str) -> str:
    name = url.rsplit("/", 1)[-1].split("?", 1)[0].strip()
    return name or fallback


def download_file(url: str, path: Path, timeout: float, retries: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "zoo-bus-vqa-external-eval"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                path.write_bytes(response.read())
            return path
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Could not download {url} to {path}: {last_error}")


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ensure_rgb(image.copy())


def bool_to_yes_no(label: Any) -> str:
    return "yes" if int(label) == 1 else "no"


def build_vsr_question(caption: str) -> str:
    return (
        "Does the following sentence correctly describe the image?\n"
        f'Sentence: "{caption}"\n'
        "Answer yes or no only."
    )


def prepare_vsr_samples(
    *,
    split: str,
    data_root: Path,
    limit: int | None,
    timeout: float,
    skip_missing_images: bool,
) -> List[ExternalSample]:
    dataset = load_dataset(VSR_DATASET_NAME, split=split)
    images_dir = data_root / "vsr" / "images" / split
    samples: List[ExternalSample] = []

    for idx, row in enumerate(dataset):
        if limit is not None and len(samples) >= limit:
            break

        image_link = row.get("image_link") or row.get("url") or ""
        image_name = str(row.get("image") or f"{idx}.jpg")
        if image_link:
            image_path = images_dir / safe_filename_from_url(image_link, image_name)
            try:
                download_file(image_link, image_path, timeout=timeout)
            except RuntimeError:
                if skip_missing_images:
                    continue
                raise
        else:
            raise ValueError("VSR row does not contain image_link; cannot load image automatically.")

        caption = str(row["caption"])
        relation = str(row.get("relation", "unknown"))
        sample_id = str(row.get("id", f"{split}_{idx}"))
        samples.append(
            ExternalSample(
                dataset="vsr",
                split=split,
                sample_id=sample_id,
                image=open_image(image_path),
                image_path=str(image_path),
                question=build_vsr_question(caption),
                gold_answer=bool_to_yes_no(row["label"]),
                metadata={
                    "caption": caption,
                    "label": int(row["label"]),
                    "relation": relation,
                    "subject": str(row.get("subj", "")),
                    "object": str(row.get("obj", "")),
                    "image_link": image_link,
                    "row_index": idx,
                },
            )
        )

    return samples


def require_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "WhatsUp automatic download requires gdown. Install it with `pip install gdown`, "
            "or download the WhatsUp controlled dataset manually and pass --whatsup_root."
        ) from exc


def gdown_download(file_id: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    require_gdown()
    import gdown

    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    result = gdown.download(url, str(output_path), quiet=False)
    if result is None or not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to download WhatsUp artifact from Google Drive id {file_id}")


def maybe_download_whatsup(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename, file_id in WHATSUP_CONTROLLED_DOWNLOADS.items():
        gdown_download(file_id, root / filename)

    for archive_name in ["controlled_images.tar.gz", "controlled_clevr.tar.gz"]:
        archive_path = root / archive_name
        extract_marker = root / f".{archive_name}.extracted"
        if extract_marker.exists():
            continue
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(root)
        extract_marker.write_text("ok\n", encoding="utf-8")


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ["data", "examples", "samples", "annotations"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Could not find a list of WhatsUp records in {path}")


def resolve_whatsup_image_path(root: Path, subset: str, image_path_value: str) -> Path:
    raw = Path(str(image_path_value))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend(
        [
            root / raw,
            root / subset / raw,
            root / subset / raw.name,
            root / raw.name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve WhatsUp image path {image_path_value!r}; tried "
        + ", ".join(str(path) for path in candidates)
    )


def find_caption_options(record: Dict[str, Any]) -> List[str]:
    for key in ["caption_options", "captions", "text_options", "options"]:
        value = record.get(key)
        if isinstance(value, list) and value:
            return [str(item) for item in value]
    raise KeyError(f"WhatsUp record has no caption options. Keys: {sorted(record)}")


def find_record_image_path(record: Dict[str, Any]) -> str:
    for key in ["image_path", "image", "filepath", "file_path"]:
        value = record.get(key)
        if value:
            return str(value)
    raise KeyError(f"WhatsUp record has no image path. Keys: {sorted(record)}")


PREPOSITION_PATTERNS = [
    ("in_front_of", re.compile(r"\b(in front of|in-front-of|front of|front)\b", re.I)),
    ("left_of", re.compile(r"\b(left of|to the left of|left)\b", re.I)),
    ("right_of", re.compile(r"\b(right of|to the right of|right)\b", re.I)),
    ("under", re.compile(r"\b(under|below|beneath)\b", re.I)),
    ("behind", re.compile(r"\b(behind|back of)\b", re.I)),
    ("on", re.compile(r"\b(on top of|on)\b", re.I)),
]


def extract_preposition(text: str) -> str:
    for name, pattern in PREPOSITION_PATTERNS:
        if pattern.search(text):
            return name
    return "unknown"


def infer_whatsup_group_key(record: Dict[str, Any], image_path: Path, subset: str, idx: int) -> str:
    for key in ["group_id", "set_id", "pair_id", "source_id"]:
        if record.get(key) is not None:
            return f"{subset}:{record[key]}"

    stem = image_path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return f"{subset}:{parts[0]}:{parts[-1]}"
    if image_path.parent.name:
        return f"{subset}:{image_path.parent.name}:{parts[0]}"
    return f"{subset}:idx_group:{idx // 4}"


def letter_for_index(index: int) -> str:
    return chr(ord("A") + index)


def build_whatsup_question(options: Sequence[str]) -> str:
    lines = ["Which caption best describes the image?", ""]
    for index, option in enumerate(options):
        lines.append(f"{letter_for_index(index)}. {option}")
    lines.extend(["", "Answer with the letter only."])
    return "\n".join(lines)


def prepare_whatsup_samples(
    *,
    root: Path,
    subsets: Sequence[str],
    download: bool,
    limit: int | None,
    seed: int,
) -> List[ExternalSample]:
    if download:
        maybe_download_whatsup(root)

    subset_files = {
        "controlled_images": "controlled_images_dataset.json",
        "controlled_clevr": "controlled_clevr_dataset.json",
    }
    samples: List[ExternalSample] = []

    for subset in subsets:
        json_path = root / subset_files[subset]
        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing WhatsUp metadata file {json_path}. Use --download_whatsup "
                f"or download data from {WHATSUP_REPO_URL} into --whatsup_root."
            )
        records = load_json_records(json_path)
        for idx, record in enumerate(records):
            if limit is not None and len(samples) >= limit:
                return samples

            original_options = find_caption_options(record)
            if len(original_options) < 2:
                continue
            image_path = resolve_whatsup_image_path(root, subset, find_record_image_path(record))
            gold_caption = original_options[0]
            options_with_flags = [(caption, caption == gold_caption) for caption in original_options]
            rng = random.Random(seed + stable_int_hash(f"{subset}:{idx}:{image_path}"))
            rng.shuffle(options_with_flags)
            shuffled_options = [caption for caption, _ in options_with_flags]
            gold_index = [flag for _, flag in options_with_flags].index(True)
            gold_letter = letter_for_index(gold_index)
            sample_id = str(record.get("id", f"{subset}_{idx}"))

            samples.append(
                ExternalSample(
                    dataset="whatsup",
                    split="controlled",
                    sample_id=sample_id,
                    image=open_image(image_path),
                    image_path=str(image_path),
                    question=build_whatsup_question(shuffled_options),
                    gold_answer=gold_letter,
                    metadata={
                        "subset": subset,
                        "row_index": idx,
                        "gold_caption": gold_caption,
                        "gold_preposition": extract_preposition(gold_caption),
                        "group_key": infer_whatsup_group_key(record, image_path, subset, idx),
                        "options_json": json.dumps(shuffled_options, ensure_ascii=False),
                    },
                )
            )

    return samples


def load_processor(model_family: str, model_name: str):
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
        for attn in (
            ["flash_attention_2"] if use_flash_attention and flash_attention_available() else []
        ) + ["sdpa", None]:
            try:
                kwargs = dict(common_kwargs)
                if attn is not None:
                    kwargs["attn_implementation"] = attn
                return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
            except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
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


def build_conversations(model_family: str, batch_items: Sequence[ExternalSample]):
    if model_family == "gemma":
        return [build_messages(item.question, answer=None) for item in batch_items]
    if model_family == "smol":
        return [build_smol_messages(item.question, item.image, answer=None) for item in batch_items]
    if model_family == "qwen":
        return [build_qwen_messages(item.question, item.image, answer=None) for item in batch_items]
    if model_family == "internvl":
        return [build_internvl_messages(item.question, item.image, answer=None) for item in batch_items]
    raise ValueError(f"Unsupported model family: {model_family}")


def prepare_inputs(
    *,
    model_family: str,
    processor,
    batch_items: Sequence[ExternalSample],
    max_seq_length: int,
) -> Dict[str, Any]:
    if model_family == "gemma":
        texts = [
            processor.apply_chat_template(
                build_messages(item.question, answer=None),
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in batch_items
        ]
        images = [[item.image] for item in batch_items]
        return dict(
            processor(
                text=texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
        )

    conversations = build_conversations(model_family, batch_items)
    if model_family == "internvl":
        return apply_internvl_chat_template_tokenized(
            processor,
            conversations,
            add_generation_prompt=True,
            max_seq_length=max_seq_length,
        )

    model_inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "padding": True,
            "truncation": True,
            "max_length": max_seq_length,
        },
    )
    model_inputs = dict(model_inputs)
    if model_family == "qwen":
        model_inputs = _drop_unused_qwen_fields(model_inputs)
    elif model_family == "internvl":
        model_inputs = _drop_unused_internvl_fields(model_inputs)
    return model_inputs


def generate_raw_predictions(
    *,
    model,
    processor,
    model_family: str,
    samples: Sequence[ExternalSample],
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


def normalize_caption(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_letter(raw_prediction: str, options: Sequence[str]) -> tuple[str | None, str]:
    text = raw_prediction.strip()
    upper = text.upper()

    patterns = [
        r"^\s*\(?([A-D])\)?(?:[\.\):\s]|$)",
        r"\bOPTION\s+([A-D])\b",
        r"\bANSWER\s+(?:IS\s+)?([A-D])\b",
        r"\bTHE\s+ANSWER\s+IS\s+([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1), "letter"

    normalized_prediction = normalize_caption(text)
    for index, option in enumerate(options):
        normalized_option = normalize_caption(option)
        if normalized_prediction == normalized_option or normalized_option in normalized_prediction:
            return letter_for_index(index), "caption_match"
    return None, "unparsed"


def options_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    options_json = metadata.get("options_json")
    if not options_json:
        return []
    return [str(item) for item in json.loads(options_json)]


def rows_from_predictions(
    *,
    dataset_name: str,
    state: str,
    model_family: str,
    model_name: str,
    samples: Sequence[ExternalSample],
    raw_predictions: Sequence[str],
    include_prompt_text: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample, raw_prediction in zip(samples, raw_predictions):
        if dataset_name == "vsr":
            parsed, parse_method = parse_yes_no(raw_prediction)
            predicted_preposition = ""
        elif dataset_name == "whatsup":
            options = options_from_metadata(sample.metadata)
            parsed, parse_method = parse_letter(raw_prediction, options)
            predicted_preposition = (
                extract_preposition(options[ord(parsed) - ord("A")])
                if parsed in {"A", "B", "C", "D"} and options
                else "unparsed"
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        row: Dict[str, Any] = {
            "dataset": sample.dataset,
            "split": sample.split,
            "state": state,
            "model_family": model_family,
            "model_name": model_name,
            "id": sample.sample_id,
            "image_path": sample.image_path,
            "gold_answer": sample.gold_answer,
            "raw_prediction": raw_prediction,
            "parsed_prediction": parsed or "",
            "parse_method": parse_method,
            "is_correct": int(parsed == sample.gold_answer),
        }
        if include_prompt_text:
            row["question"] = sample.question
        for key, value in sample.metadata.items():
            row[key] = value
        if dataset_name == "whatsup":
            row["predicted_preposition"] = predicted_preposition
        rows.append(row)
    return rows


def accuracy(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(int(row["is_correct"]) for row in rows) / len(rows)


def grouped_accuracy(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return {name: accuracy(items) for name, items in sorted(groups.items())}


def grouped_counts(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def compute_vsr_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    relation_accuracy = grouped_accuracy(rows, "relation")
    label_accuracy = grouped_accuracy(rows, "gold_answer")
    prediction_distribution = grouped_counts(rows, "parsed_prediction")
    parse_failures = sum(1 for row in rows if not row["parsed_prediction"])
    return {
        "num_samples": len(rows),
        "num_correct": sum(int(row["is_correct"]) for row in rows),
        "overall_accuracy": accuracy(rows),
        "parse_failure_rate": parse_failures / len(rows) if rows else 0.0,
        "label_accuracy": label_accuracy,
        "relation_accuracy": relation_accuracy,
        "relation_macro_accuracy": mean(relation_accuracy.values()),
        "relation_counts": grouped_counts(rows, "relation"),
        "gold_distribution": grouped_counts(rows, "gold_answer"),
        "prediction_distribution": prediction_distribution,
    }


def pair_keys_for_preposition(preposition: str) -> str | None:
    if preposition in {"left_of", "right_of"}:
        return "left_right"
    if preposition in {"on", "under"}:
        return "on_under"
    if preposition in {"in_front_of", "behind"}:
        return "front_behind"
    return None


def compute_whatsup_pair_set_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    group_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_to_rows[str(row.get("group_key", ""))].append(row)

    pair_results: List[int] = []
    set_results: List[int] = []
    for group_rows in group_to_rows.values():
        by_pair: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            pair_key = pair_keys_for_preposition(str(row.get("gold_preposition", "unknown")))
            if pair_key:
                by_pair[pair_key].append(row)
        for pair_rows in by_pair.values():
            if len(pair_rows) >= 2:
                pair_results.append(int(all(int(row["is_correct"]) for row in pair_rows)))
        if len(group_rows) >= 4:
            set_results.append(int(all(int(row["is_correct"]) for row in group_rows)))

    return {
        "pair_accuracy": sum(pair_results) / len(pair_results) if pair_results else None,
        "pair_count": len(pair_results),
        "set_accuracy": sum(set_results) / len(set_results) if set_results else None,
        "set_count": len(set_results),
    }


def compute_confusion_matrix(
    rows: Sequence[Dict[str, Any]],
    *,
    gold_key: str,
    pred_key: str,
) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        matrix[str(row.get(gold_key, "unknown"))][str(row.get(pred_key, "unknown"))] += 1
    return {gold: dict(sorted(pred_counts.items())) for gold, pred_counts in sorted(matrix.items())}


def compute_whatsup_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    subset_accuracy = grouped_accuracy(rows, "subset")
    preposition_accuracy = grouped_accuracy(rows, "gold_preposition")
    parse_failures = sum(1 for row in rows if not row["parsed_prediction"])
    metrics = {
        "num_samples": len(rows),
        "num_correct": sum(int(row["is_correct"]) for row in rows),
        "individual_accuracy": accuracy(rows),
        "overall_accuracy": accuracy(rows),
        "parse_failure_rate": parse_failures / len(rows) if rows else 0.0,
        "subset_accuracy": subset_accuracy,
        "preposition_accuracy": preposition_accuracy,
        "preposition_macro_accuracy": mean(preposition_accuracy.values()),
        "subset_counts": grouped_counts(rows, "subset"),
        "preposition_counts": grouped_counts(rows, "gold_preposition"),
        "prediction_distribution": grouped_counts(rows, "parsed_prediction"),
        "confusion_matrix": compute_confusion_matrix(
            rows,
            gold_key="gold_preposition",
            pred_key="predicted_preposition",
        ),
    }
    metrics.update(compute_whatsup_pair_set_metrics(rows))
    return metrics


def compute_metrics(dataset_name: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if dataset_name == "vsr":
        return compute_vsr_metrics(rows)
    if dataset_name == "whatsup":
        return compute_whatsup_metrics(rows)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def subtract_nested(after: Any, before: Any) -> Any:
    if isinstance(after, (int, float)) and isinstance(before, (int, float)):
        return after - before
    if isinstance(after, dict) and isinstance(before, dict):
        keys = sorted(set(after) & set(before))
        return {key: subtract_nested(after[key], before[key]) for key in keys}
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
        name=f"external-spatial-{model_family}-{run_name}",
        type="external-eval-results",
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


def load_samples_for_dataset(args: argparse.Namespace, dataset_name: str) -> List[ExternalSample]:
    data_root = Path(args.data_root)
    if dataset_name == "vsr":
        return prepare_vsr_samples(
            split=args.vsr_split,
            data_root=data_root,
            limit=args.limit,
            timeout=args.vsr_image_timeout,
            skip_missing_images=args.skip_missing_images,
        )
    if dataset_name == "whatsup":
        root = Path(args.whatsup_root) if args.whatsup_root else data_root / "whatsup"
        return prepare_whatsup_samples(
            root=root,
            subsets=args.whatsup_subsets,
            download=args.download_whatsup,
            limit=args.limit,
            seed=args.seed,
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
    samples: Sequence[ExternalSample],
    output_dir: Path,
    batch_size: int,
    max_seq_length: int,
    max_new_tokens: int,
    device: torch.device,
    include_prompt_text: bool,
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
        dataset_name=dataset_name,
        state=state,
        model_family=model_family,
        model_name=model_name,
        samples=samples,
        raw_predictions=raw_predictions,
        include_prompt_text=include_prompt_text,
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
    metric_prefix = f"{dataset_name}/{state}"
    maybe_log_wandb(flatten_metrics(metrics, metric_prefix))
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
            config=vars(args) | {"resolved_model_name": model_name, "resolved_max_seq_length": max_seq_length},
        )

    print(f"Loading processor/model: {model_name}", flush=True)
    processor = load_processor(args.model_family, model_name)
    model = load_base_model(args.model_family, model_name)
    model.to(device)
    model.eval()

    datasets_to_run = ["vsr", "whatsup"] if args.dataset == "all" else [args.dataset]
    summary = {
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

    samples_by_dataset: Dict[str, List[ExternalSample]] = {}
    reports_by_dataset: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for dataset_name in datasets_to_run:
        print(f"Loading {dataset_name} samples...", flush=True)
        samples = load_samples_for_dataset(args, dataset_name)
        if not samples:
            raise ValueError(f"No samples loaded for {dataset_name}")
        samples_by_dataset[dataset_name] = samples

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

    write_json(output_dir / "external_spatial_eval_summary.json", summary)
    if args.report_to == "wandb" and args.wandb_log_artifact:
        maybe_log_wandb_artifact(output_dir, args.model_family, run_name)
    if wandb is not None and wandb.run is not None:
        wandb.finish()
    print(f"Wrote external spatial evaluation results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
