#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import torch
from datasets import load_dataset
from PIL import Image, ImageFile

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig
from peft import PeftModel

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# 1) 你的 12 个问题模板（用于 fallback 匹配）
# =========================
Q1  = "Are there one or more people in the image? Respond with 'Yes' or 'No'."
Q2  = "Are there people at the bench closest to the clock? Respond with 'Yes' or 'No'."
Q3  = "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. Which bench is closest to the clock that has at least one person at it? Answer the bench ID. If no benches have people, respond with '0'."
Q4P = "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. Is there a person at bench number {bench_number}? Respond with 'Yes' or 'No'."
Q5  = "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. Which bench is closest to the clock? Answer with the bench ID."
Q6  = "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. List the benches in order from closest to furthest from the clock, separated by commas. For example, '2, 1, 4, 3'."
Q7P = "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs.  Is the clock close to bench number {bench_number}? Respond with 'Yes' or 'No'."
Q8  = "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. Which stop sign is closest to the clock? Answer with its ID."
Q9  = "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. List the stop signs in order from closest to furthest from the clock, separated by commas. For example, '2, 1, 4, 3'."
Q10P= "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. For each stop sign, consider all animals that are spatially closest to that stop sign. Is the clock close to the animals around stop sign number {stop_sign_number}? Respond with 'Yes' or 'No'."
Q11P= "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. For bench number {bench_number}, choose the best direction for the clock to move to reach that bench while avoiding obstacles between them. Answer with exactly one of: 'keep straight', 'go left', 'go right', 'go up', or 'go down'. 'keep straight' if the clock can move directly toward bench {bench_number} without colliding with any obstacle. 'go left' or 'go right' if the bench is mainly above or below the clock and it is better for the clock to pass the nearest obstacle between them on its left or right side. Answer 'go up' or 'go down' if the bench is mainly to the left or right of the clock and it is better for the clock to pass the nearest obstacle between them above or below it."
Q12P= "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. For stop sign number {stop_sign_number}, choose the best direction for the clock to move to reach that stop sign while avoiding obstacles between them. Answer with exactly one of: 'keep straight', 'go left', 'go right', 'go up', or 'go down'. Answer 'keep straight' if the clock can move directly toward stop sign {stop_sign_number} without colliding with any objects. Answer 'go left' or 'go right' if the stop sign is mainly above or below the clock and it is better for the clock to pass the nearest obstacle between them on its left or right side. Answer 'go up' or 'go down' if the stop sign is mainly to the left or right of the clock and it is better for the clock to pass the nearest obstacle between them above or below it."


def _re_template(s: str) -> str:
    # 把模板里的 {xxx} 变成捕获数字（你的占位都是 number）
    s = re.escape(s)
    s = s.replace(re.escape("{bench_number}"), r"(\d+)")
    s = s.replace(re.escape("{stop_sign_number}"), r"(\d+)")
    # 容忍多余空格
    s = s.replace(r"\ ", r"\s+")
    return r"^\s*" + s + r"\s*$"


PATTERNS = {
    "Q1_OneOrMorePeople": re.compile(_re_template(Q1), re.IGNORECASE),
    "Q2_PersonAtClosestBench": re.compile(_re_template(Q2), re.IGNORECASE),
    "Q3_ClosestBenchWithPerson": re.compile(_re_template(Q3), re.IGNORECASE),
    "Q4_IsPersonAtBench": re.compile(_re_template(Q4P), re.IGNORECASE),
    "Q5_ClosestBench": re.compile(_re_template(Q5), re.IGNORECASE),
    "Q6_ClosestToFurthestBenches": re.compile(_re_template(Q6), re.IGNORECASE),
    "Q7_ArrivedAtBench": re.compile(_re_template(Q7P), re.IGNORECASE),
    "Q8_ClosestStopSigns": re.compile(_re_template(Q8), re.IGNORECASE),
    "Q9_ClosestToFurthestStopSigns": re.compile(_re_template(Q9), re.IGNORECASE),
    "Q10_ArrivedAtAnimalsAroundStopSigns": re.compile(_re_template(Q10P), re.IGNORECASE),
    "Q11_PathBlockedBetweenBusAndBench": re.compile(_re_template(Q11P), re.IGNORECASE),
    "Q12_PathBlockedBetweenBusAndStopSign": re.compile(_re_template(Q12P), re.IGNORECASE),
}


# =========================
# 2) 解析/评分：四类答案
# =========================
_norm_space = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = _norm_space.sub(" ", s)
    return s

def parse_yesno(s: str) -> Optional[str]:
    s = normalize_text(s)
    if re.search(r"\byes\b", s): return "yes"
    if re.search(r"\bno\b", s):  return "no"
    return None

def parse_int(s: str) -> Optional[int]:
    s = normalize_text(s)
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None

def parse_int_list(s: str) -> Optional[List[int]]:
    s = normalize_text(s)
    nums = re.findall(r"-?\d+", s)
    if not nums:
        return None
    return [int(x) for x in nums]

ALLOWED_DIR = ["keep straight", "go left", "go right", "go up", "go down"]

def parse_dir(s: str) -> Optional[str]:
    s = normalize_text(s)
    # 优先精确包含
    for d in ALLOWED_DIR:
        if d in s:
            return d
    # 兜底：常见简写
    if "straight" in s: return "keep straight"
    if re.search(r"\bleft\b", s): return "go left"
    if re.search(r"\bright\b", s): return "go right"
    if re.search(r"\bup\b", s): return "go up"
    if re.search(r"\bdown\b", s): return "go down"
    return None


# 每个 task 的判分方式
TASK_METRIC = {
    "Q1_OneOrMorePeople": "yesno",
    "Q2_PersonAtClosestBench": "yesno",
    "Q3_ClosestBenchWithPerson": "int",
    "Q4_IsPersonAtBench": "yesno",
    "Q5_ClosestBench": "int",
    "Q6_ClosestToFurthestBenches": "int_list",
    "Q7_ArrivedAtBench": "yesno",
    "Q8_ClosestStopSigns": "int",
    "Q9_ClosestToFurthestStopSigns": "int_list",
    "Q10_ArrivedAtAnimalsAroundStopSigns": "yesno",
    "Q11_PathBlockedBetweenBusAndBench": "dir",
    "Q12_PathBlockedBetweenBusAndStopSign": "dir",
}

# 如果你的 train.jsonl 有 question_type，就把它映射到我们的 task key（更稳）
QTYPE_TO_TASK = {
    "OneOrMorePeople": "Q1_OneOrMorePeople",
    "PersonAtClosestBench": "Q2_PersonAtClosestBench",
    "ClosestBenchWithPerson": "Q3_ClosestBenchWithPerson",
    "IsPersonAtBench": "Q4_IsPersonAtBench",
    "ClosestBench": "Q5_ClosestBench",
    "ClosestToFurthestBenches": "Q6_ClosestToFurthestBenches",
    "ArrivedAtBench": "Q7_ArrivedAtBench",
    "ClosestStopSigns": "Q8_ClosestStopSigns",
    "ClosestToFurthestStopSigns": "Q9_ClosestToFurthestStopSigns",
    "ArrivedAtAnimalsAroundStopSigns": "Q10_ArrivedAtAnimalsAroundStopSigns",
    "PathBlockedBetweenBusAndBench": "Q11_PathBlockedBetweenBusAndBench",
    "PathBlockedBetweenBusAndStopSign": "Q12_PathBlockedBetweenBusAndStopSign",
}


# =========================
# 3) 模型加载（支持 LoRA）
# =========================
def load_model(model_id: str, lora_dir: Optional[str], device_map: str):
    processor = LlavaNextProcessor.from_pretrained(model_id)

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
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

    if lora_dir:
        model = PeftModel.from_pretrained(model, lora_dir).eval()

    return model, processor


# =========================
# 4) 数据筛选：识别属于哪个 task
# =========================
def match_task(ex: Dict[str, Any]) -> Optional[str]:
    # 1) 优先用 question_type（最稳）
    qt = ex.get("question_type")
    if qt in QTYPE_TO_TASK:
        return QTYPE_TO_TASK[qt]

    # 2) fallback 用题面模板匹配
    conv = ex.get("conversations", [])
    if len(conv) < 1:
        return None
    q = str(conv[0].get("value", "")).strip()
    for task, pat in PATTERNS.items():
        if pat.match(q):
            return task
    return None


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
    return str(cand1)


# =========================
# 5) 评估：按 task 做结构化对比
# =========================
def score(metric: str, pred_raw: str, gt_raw: str) -> Tuple[bool, str, str]:
    """
    返回 (ok, pred_parsed, gt_parsed) 用于打印更直观
    """
    if metric == "yesno":
        p = parse_yesno(pred_raw)
        g = parse_yesno(gt_raw)
        return (p is not None and g is not None and p == g), str(p), str(g)

    if metric == "int":
        p = parse_int(pred_raw)
        g = parse_int(gt_raw)
        return (p is not None and g is not None and p == g), str(p), str(g)

    if metric == "int_list":
        p = parse_int_list(pred_raw)
        g = parse_int_list(gt_raw)
        return (p is not None and g is not None and p == g), str(p), str(g)

    if metric == "dir":
        p = parse_dir(pred_raw)
        g = parse_dir(gt_raw)
        return (p is not None and g is not None and p == g), str(p), str(g)

    # fallback：严格文本
    return (normalize_text(pred_raw) == normalize_text(gt_raw)), pred_raw, gt_raw


@torch.no_grad()
def run_test(
    model,
    processor,
    samples: List[Dict[str, Any]],
    base_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    log_every: int,
    print_each: bool,
):
    tok = processor.tokenizer
    device = next(model.parameters()).device

    total = 0
    correct = 0
    per_task_total = defaultdict(int)
    per_task_correct = defaultdict(int)

    n_batches = (len(samples) + batch_size - 1) // batch_size
    print(f"[TEST] total={len(samples)} | batch_size={batch_size} | batches={n_batches} | device={device}")

    for bidx in range(n_batches):
        batch = samples[bidx * batch_size : (bidx + 1) * batch_size]

        images, prompts, gts, tasks, img_paths = [], [], [], [], []

        for ex in batch:
            img_path = resolve_image_path(base_dir, ex["image"])
            img_paths.append(img_path)
            with Image.open(img_path) as im:
                images.append(im.convert("RGB"))

            conv = ex["conversations"]
            user_text = str(conv[0]["value"])
            gt = str(conv[1]["value"]).strip()

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image"},
                ],
            }]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            prompts.append(prompt)
            gts.append(gt)
            tasks.append(match_task(ex) or "Unknown")

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

        # decode + score
        for i in range(gen_ids.size(0)):
            out = gen_ids[i, int(prompt_lens[i]):]
            pred_raw = tok.decode(out, skip_special_tokens=True).strip()

            task = tasks[i]
            metric = TASK_METRIC.get(task, "text")
            ok, pred_parsed, gt_parsed = score(metric, pred_raw, gts[i])

            total += 1
            correct += int(ok)
            per_task_total[task] += 1
            per_task_correct[task] += int(ok)

            if print_each:
                rel = os.path.relpath(img_paths[i], start=str(base_dir))
                status = "RIGHT" if ok else "WRONG"
                print(f"[{total:05d}] {task} | {rel}")
                print(f"  GT   : {gts[i]!r} -> {gt_parsed}")
                print(f"  PRED : {pred_raw!r} -> {pred_parsed}")
                print(f"  ==> {status}\n")

        if (bidx == 0) or ((bidx + 1) % log_every == 0) or (bidx + 1 == n_batches):
            acc = correct / total if total else 0.0
            print(f"[TEST] batch {bidx+1:>4d}/{n_batches} | qa {total:>6d}/{len(samples)} | running_acc={acc:.4f}")

    print("\n[RESULT] Overall accuracy:")
    print(f"  acc={correct/total:.4f} ({correct}/{total})")

    print("\n[RESULT] Accuracy by task:")
    for task in sorted(per_task_total.keys()):
        t = per_task_total[task]
        c = per_task_correct[task]
        print(f"  {task:35s}  {c/t:.4f}  ({c}/{t})")


def sample_examples(
    ds,
    base_dir: Path,
    seed: int,
    k_images: int,
    mode: str,
) -> List[Dict[str, Any]]:
    """
    mode:
      - "by_image": 从全体样本里按 image 去重抽 k_images 张图（每图可能多条 QA，仍会全部进入评估）
      - "flat": 直接从样本里抽 k_images 条（不保证图片去重）
    """
    rnd = random.Random(seed)

    # 先过滤到属于 12 tasks 的样本
    candidates = []
    for ex in ds:
        task = match_task(ex)
        if task is None:
            continue
        imgp = resolve_image_path(base_dir, ex["image"])
        if not os.path.exists(imgp):
            continue
        candidates.append(ex)

    if not candidates:
        raise RuntimeError("No candidate examples matched your 12 tasks. Check question_type or templates.")

    if mode == "flat":
        rnd.shuffle(candidates)
        return candidates[:k_images]

    # by_image：同一张图上的多条 QA 都保留
    # 做法：先抽图片集合，再把属于这些图片的样本全放进来
    img_to_examples = defaultdict(list)
    for ex in candidates:
        imgp = resolve_image_path(base_dir, ex["image"])
        img_to_examples[imgp].append(ex)

    imgs = list(img_to_examples.keys())
    rnd.shuffle(imgs)
    imgs = imgs[:min(k_images, len(imgs))]

    out = []
    for imgp in imgs:
        out.extend(img_to_examples[imgp])

    rnd.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default=".")
    ap.add_argument("--train_jsonl", type=str, default="train.jsonl")
    ap.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    ap.add_argument("--lora_dir", type=str, default="", help="可选：LoRA adapter 目录")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_images", type=int, default=200, help="抽多少张图片（by_image 模式）")
    ap.add_argument("--sample_mode", type=str, default="by_image", choices=["by_image", "flat"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--print_each", action="store_true", help="打印每条 QA 的 GT/PRED/RIGHT|WRONG")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    train_jsonl = args.train_jsonl if os.path.isabs(args.train_jsonl) else str((base_dir / args.train_jsonl).resolve())
    lora_dir = args.lora_dir.strip() or None

    print(f"[INFO] base_dir={base_dir}")
    print(f"[INFO] train_jsonl={train_jsonl}")
    print(f"[INFO] model_id={args.model_id}")
    print(f"[INFO] lora_dir={lora_dir}")
    print(f"[INFO] sample_mode={args.sample_mode}, k_images={args.k_images}, seed={args.seed}")

    ds = load_dataset("json", data_files={"train": train_jsonl})["train"]
    samples = sample_examples(ds, base_dir, args.seed, args.k_images, args.sample_mode)

    print(f"[INFO] Samples picked: {len(samples)} QA items (after filtering to 12 tasks)")
    # 统计每个 task 的数量
    cnt = defaultdict(int)
    for ex in samples:
        cnt[match_task(ex) or "Unknown"] += 1
    for k in sorted(cnt.keys()):
        print(f"  {k:35s}: {cnt[k]}")

    model, processor = load_model(args.model_id, lora_dir, args.device_map)

    run_test(
        model=model,
        processor=processor,
        samples=samples,
        base_dir=base_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        log_every=args.log_every,
        print_each=args.print_each,
    )


if __name__ == "__main__":
    main()