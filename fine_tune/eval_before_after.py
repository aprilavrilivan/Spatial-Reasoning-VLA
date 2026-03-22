#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import gc
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
from datasets import load_dataset
from PIL import Image, ImageFile

from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_norm_space = re.compile(r"\s+")

DIR_CHOICES = ["keep straight", "go left", "go right", "go up", "go down"]


# ----------------- normalization / parsing -----------------
def normalize_answer(s: str) -> str:
    s = str(s).strip().lower()
    s = _norm_space.sub(" ", s)
    # remove quotes/backticks
    s = re.sub(r"[\"\'`]", "", s)
    # remove periods (common)
    s = re.sub(r"[\.]", "", s)
    # unify comma spacing
    s = s.replace(" ,", ",").replace(", ", ",")
    return s


def parse_yesno(s: str) -> str:
    s = normalize_answer(s)
    if re.search(r"\byes\b", s):
        return "yes"
    if re.search(r"\bno\b", s):
        return "no"
    return ""


def looks_like_yesno_question(q: str) -> bool:
    qn = normalize_answer(q)
    return ("respond with" in qn and "yes" in qn and "no" in qn)


def extract_first_int(s: str) -> Optional[str]:
    m = re.search(r"\b\d+\b", str(s))
    return m.group(0) if m else None


def extract_int_list(s: str) -> Optional[str]:
    nums = re.findall(r"\b\d+\b", str(s))
    if not nums:
        return None
    return ",".join(nums)


def extract_direction(s: str) -> Optional[str]:
    sn = normalize_answer(s)
    for d in DIR_CHOICES:
        if d in sn:
            return d
    # very common variant
    if "straight" in sn:
        return "keep straight"
    return None


def is_pure_int(s: str) -> bool:
    return re.fullmatch(r"\d+", str(s).strip()) is not None


def is_int_list(s: str) -> bool:
    return re.fullmatch(r"\d+(,\s*\d+)*", str(s).strip()) is not None


def is_direction(s: str) -> bool:
    return normalize_answer(s) in DIR_CHOICES


def smart_match(pred: str, gt: str, is_yesno: bool) -> bool:
    """
    Route-1 evaluation:
    - Yes/No: parse_yesno
    - GT is integer: extract_first_int(pred)
    - GT is integer list: extract_int_list(pred)
    - GT is direction: extract_direction(pred)
    - else: normalized exact match
    """
    gt_raw = str(gt).strip()

    if is_yesno:
        p = parse_yesno(pred)
        g = parse_yesno(gt_raw)
        return (p != "" and g != "" and p == g)

    # numeric single
    if is_pure_int(gt_raw):
        p = extract_first_int(pred)
        return (p is not None and p == gt_raw)

    # numeric list
    if is_int_list(gt_raw):
        p = extract_int_list(pred)
        return (p is not None and normalize_answer(p) == normalize_answer(gt_raw))

    # direction
    if is_direction(gt_raw):
        p = extract_direction(pred)
        return (p is not None and normalize_answer(p) == normalize_answer(gt_raw))

    # fallback
    return normalize_answer(pred) == normalize_answer(gt_raw)


# ----------------- path / sampling helpers -----------------
def resolve_image_path(base_dir: Path, p: str) -> str:
    p = str(p)
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand1 = base_dir / p
    if cand1.exists():
        return str(cand1)
    cand2 = base_dir / "images" / p
    if cand2.exists():
        return str(cand2)
    cand3 = base_dir / "image" / p
    if cand3.exists():
        return str(cand3)
    return str(cand1)


def pick_k_images(image_dir: Path, k: int, seed: int) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    files = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not files:
        raise RuntimeError(f"No images found under: {image_dir}")
    rnd = random.Random(seed)
    rnd.shuffle(files)
    return files[: min(k, len(files))]


def get_input_device(model) -> torch.device:
    # For device_map="auto" models, inputs should go to the first cuda device in hf_device_map (usually cuda:0)
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for _, dev in model.hf_device_map.items():
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    return next(model.parameters()).device


# ----------------- model loading -----------------
def load_llava_model(model_id: str, device_map: str, load_4bit: bool = True):
    processor = LlavaNextProcessor.from_pretrained(model_id)
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    quant_config = None
    if load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    ).eval()

    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    return model, processor


def attach_lora(model, lora_dir: str):
    return PeftModel.from_pretrained(model, lora_dir).eval()


# ----------------- dataset selection -----------------
def build_examples(
    ds,
    base_dir: Path,
    chosen_basenames: set,
    question_type: str,
) -> List[Dict[str, Any]]:
    out = []
    for ex in ds:
        if str(ex.get("question_type", "")).strip() != question_type:
            continue
        conv = ex.get("conversations", [])
        if len(conv) < 2:
            continue

        raw_img = ex.get("image", "")
        img_path = resolve_image_path(base_dir, raw_img)

        # match by basename against chosen set
        if Path(img_path).name not in chosen_basenames:
            continue

        if not os.path.exists(img_path):
            continue

        out.append(ex)
    return out


# ----------------- eval -----------------
@torch.no_grad()
def eval_model(
    model,
    processor,
    examples: List[Dict[str, Any]],
    base_dir: Path,
    max_new_tokens: int,
    batch_size: int,
    print_each: bool,
) -> Tuple[float, int, int]:
    tok = processor.tokenizer
    device = get_input_device(model)

    total = 0
    correct = 0

    for start in range(0, len(examples), batch_size):
        batch = examples[start: start + batch_size]

        images = []
        prompts = []
        gts = []
        yesno_flags = []

        for ex in batch:
            img_path = resolve_image_path(base_dir, ex["image"])
            with Image.open(img_path) as im:
                images.append(im.convert("RGB"))

            conv = ex["conversations"]
            q = str(conv[0]["value"])
            gt = str(conv[1]["value"]).strip()

            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": q}, {"type": "image"}],
            }]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            prompts.append(prompt)
            gts.append(gt)
            yesno_flags.append(looks_like_yesno_question(q))

        inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

        for i in range(gen_ids.size(0)):
            out_ids = gen_ids[i, int(prompt_lens[i]):]
            pred = tok.decode(out_ids, skip_special_tokens=True).strip()
            gt = gts[i]

            ok = smart_match(pred=pred, gt=gt, is_yesno=yesno_flags[i])

            total += 1
            correct += int(ok)

            if print_each:
                # show parsed view for debugging
                if yesno_flags[i]:
                    pr_show = parse_yesno(pred)
                    gt_show = parse_yesno(gt)
                    extra = f" | parsed(PR)={pr_show!r} parsed(GT)={gt_show!r}"
                else:
                    if is_pure_int(gt):
                        extra = f" | extracted_int(PR)={extract_first_int(pred)!r}"
                    elif is_int_list(gt):
                        extra = f" | extracted_list(PR)={extract_int_list(pred)!r}"
                    elif is_direction(gt):
                        extra = f" | extracted_dir(PR)={extract_direction(pred)!r}"
                    else:
                        extra = ""
                print(f"[{total:04d}] {'RIGHT' if ok else 'WRONG'} | GT={gt!r} | PR={pred!r}{extra}")

    acc = (correct / total) if total else 0.0
    return acc, correct, total


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, default="train.jsonl")
    ap.add_argument("--base_dir", type=str, default=".", help="用于解析 jsonl 里的 image 路径")
    ap.add_argument("--image_dir", type=str, default="", help="用于随机抽图的目录，默认优先 ./images 否则 ./image")
    ap.add_argument("--questionclass", type=str, required=True, help="数据里的 question_type，例如 OneOrMorePeople")
    ap.add_argument("--k_images", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--base_model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    ap.add_argument("--lora_dir", type=str, required=True, help="LoRA adapter 目录，例如 ./llava-graid-lora")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--print_each", action="store_true")

    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()

    # choose image_dir
    if args.image_dir.strip():
        image_dir = Path(args.image_dir).resolve()
    else:
        cand = base_dir / "images"
        image_dir = cand if cand.exists() else (base_dir / "image")

    # sample images
    chosen = pick_k_images(image_dir, args.k_images, args.seed)
    chosen_basenames = {p.name for p in chosen}
    print(f"[INFO] sampled_images={len(chosen)} from {image_dir}")

    # load dataset
    train_jsonl = args.train_jsonl if os.path.isabs(args.train_jsonl) else str((base_dir / args.train_jsonl).resolve())
    ds = load_dataset("json", data_files={"train": train_jsonl})["train"]
    print(f"[INFO] dataset_rows={len(ds)} | jsonl={train_jsonl}")

    # build eval examples
    examples = build_examples(ds, base_dir, chosen_basenames, args.questionclass)
    print(f"[INFO] questionclass={args.questionclass} | matched_QA={len(examples)}")
    if not examples:
        print("[ERROR] No matched QA. Check: question_type spelling, image_dir, base_dir, jsonl image paths.")
        return

    # -------- base model eval --------
    print("\n================ BASE MODEL (no LoRA) ================")
    base_model, processor = load_llava_model(args.base_model_id, device_map=args.device_map, load_4bit=True)
    acc_base, c_base, t_base = eval_model(
        base_model, processor, examples, base_dir,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        print_each=args.print_each,
    )
    print(f"[BASE] acc={acc_base:.4f} ({c_base}/{t_base})")

    # free base model
    del base_model
    free_cuda()

    # -------- lora model eval --------
    print("\n================ LoRA MODEL (finetuned) ================")
    ft_model, processor2 = load_llava_model(args.base_model_id, device_map=args.device_map, load_4bit=True)
    ft_model = attach_lora(ft_model, args.lora_dir)
    acc_ft, c_ft, t_ft = eval_model(
        ft_model, processor2, examples, base_dir,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        print_each=args.print_each,
    )
    print(f"[LoRA] acc={acc_ft:.4f} ({c_ft}/{t_ft})")

    # summary
    print("\n================ SUMMARY ================")
    print(f"questionclass: {args.questionclass}")
    print(f"sampled images: {len(chosen)} (dir={image_dir})")
    print(f"eval QA: {len(examples)}")
    print(f"BASE acc: {acc_base:.4f} ({c_base}/{t_base})")
    print(f"LoRA acc: {acc_ft:.4f} ({c_ft}/{t_ft})")


if __name__ == "__main__":
    main()