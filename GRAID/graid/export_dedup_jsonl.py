#!/usr/bin/env python
"""
Export a HF dataset (saved via save_to_disk) into:
  - A deduplicated JPEG image folder (e.g. images/*.jpg)
  - A JSONL file for LLaVA finetuning.

Each JSON line:
{
  "image": "images/img_<hash>.jpg",
  "question_type": "<question_type>",
  "conversations": [
    {"value": "<question>"},
    {"value": "<answer>"}
  ]
}
"""

import argparse
import json
import io
import hashlib
from pathlib import Path
from typing import Any, Dict

from datasets import load_from_disk, DatasetDict
from datasets import Image as HFImage
from PIL import Image as PILImage


def pil_to_jpeg_bytes(img: PILImage.Image, quality: int = 90) -> bytes:
    """Convert a PIL image to JPEG bytes (RGB, lossy, smaller)."""
    buf = io.BytesIO()
    img = img.convert("RGB")  # JPEG 不支持 alpha
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def get_image_jpeg_bytes(img_field: Any) -> bytes:
    """
    Normalize various image formats to *JPEG-encoded* bytes:
    - HF Image dict with 'bytes' or 'path'
    - PIL.Image
    """
    # case 1: HF Image(decode=False) → dict
    if isinstance(img_field, dict):
        # If we already have raw bytes, decode to PIL then re-encode as JPEG
        if "bytes" in img_field and img_field["bytes"] is not None:
            raw = img_field["bytes"]
            img = PILImage.open(io.BytesIO(raw))
            return pil_to_jpeg_bytes(img)

        # If we only have a path, open and re-encode as JPEG
        if "path" in img_field and img_field["path"] is not None:
            with open(img_field["path"], "rb") as f:
                raw = f.read()
            img = PILImage.open(io.BytesIO(raw))
            return pil_to_jpeg_bytes(img)

    # case 2: PIL.Image 直接转 JPEG
    if isinstance(img_field, PILImage.Image):
        return pil_to_jpeg_bytes(img_field)

    raise ValueError(
        f"Unsupported image field type: {type(img_field)}. "
        "Make sure your dataset has an Image column or PIL images."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="datasets/zoo_bus_vqa",
        help="Directory of the dataset saved by save_to_disk().",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to export (train / validation / test).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="train.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="Name of the image column.",
    )
    parser.add_argument(
        "--question_column",
        type=str,
        default="question",
        help="Name of the question column.",
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        default="answer",
        help="Name of the answer column.",
    )
    parser.add_argument(
        "--question_type_column",
        type=str,
        default="question_type",
        help="Name of the question_type column (optional).",
    )
    parser.add_argument(
        "--image_out_dir",
        type=str,
        default="images",
        help="Directory to save deduplicated JPEG images.",
    )
    parser.add_argument(
        "--hash_prefix_len",
        type=int,
        default=16,
        help="Length of SHA256 hash prefix used in filenames.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=90,
        help="JPEG quality (1–100, higher = larger files, better quality).",
    )
    args = parser.parse_args()

    ds_obj = load_from_disk(args.dataset_dir)
    dataset = ds_obj[args.split] if isinstance(ds_obj, DatasetDict) else ds_obj

    # Ensure the image column is a HF Image so we get dict/path/bytes nicely.
    if args.image_column in dataset.column_names:
        try:
            dataset = dataset.cast_column(args.image_column, HFImage())
        except Exception:
            # If column is already str paths, this may fail; keep as-is.
            pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_out_dir = Path(args.image_out_dir)
    image_out_dir.mkdir(parents=True, exist_ok=True)

    has_qtype = args.question_type_column in dataset.column_names

    # Map: hash -> relative JPEG path
    hash_to_relpath: Dict[str, str] = {}

    print(f"Exporting {len(dataset)} examples to {out_path} ...")
    num_new_images = 0

    with out_path.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(dataset):
            img_field = ex.get(args.image_column, None)
            if img_field is None:
                raise KeyError(f"Missing image column: {args.image_column}")

            # 1) JPEG bytes & hash
            jpeg_bytes = get_image_jpeg_bytes(img_field)
            h = hashlib.sha256(jpeg_bytes).hexdigest()[: args.hash_prefix_len]

            # 2) If new image, write JPEG to disk
            if h not in hash_to_relpath:
                filename = f"img_{h}.jpg"
                rel_path = image_out_dir / filename
                with open(rel_path, "wb") as img_f:
                    img_f.write(jpeg_bytes)
                hash_to_relpath[h] = str(rel_path)
                num_new_images += 1

            image_path = hash_to_relpath[h]

            # 3) Build JSON record
            q = ex.get(args.question_column, "")
            a = ex.get(args.answer_column, "")

            record = {
                "image": image_path,  # many QAs may share this JPEG
                "conversations": [
                    {"value": str(q)},
                    {"value": str(a)},
                ],
            }

            if has_qtype:
                record["question_type"] = str(ex.get(args.question_type_column, ""))

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"  Total QA examples: {len(dataset)}")
    print(f"  Unique JPEG images written: {num_new_images}")
    print(f"  JSONL saved to: {out_path.resolve()}")
    print(f"  Images saved under: {image_out_dir.resolve()}")


if __name__ == "__main__":
    main()