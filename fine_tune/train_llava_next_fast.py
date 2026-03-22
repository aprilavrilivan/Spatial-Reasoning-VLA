#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from PIL import Image, ImageFile

from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------- Config (env-overridable) -----------------
MODEL_ID   = os.environ.get("MODEL_ID", "llava-hf/llava-v1.6-mistral-7b-hf")
TRAIN_JSON = os.environ.get("TRAIN_JSONL", "train.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./llava-graid-lora")

IMAGE_KEY  = "image"
QTYPE_KEY  = "question_type"
CONV_KEY   = "conversations"

SEED = int(os.environ.get("SEED", "42"))

# big VRAM -> default longer context to avoid prompt overflow
MAX_LEN = int(os.environ.get("MAX_LEN", "4096"))

# you can keep 336; if you ever hit prompt too long, reduce this (e.g. 224)
FORCE_IMAGE_SIZE = int(os.environ.get("FORCE_IMAGE_SIZE", "336"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))
BASE_LR    = float(os.environ.get("BASE_LR", "5e-5"))

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "16"))
PREFETCH    = int(os.environ.get("PREFETCH", "4"))

VAL_IMAGE_COUNT = int(os.environ.get("VAL_IMAGE_COUNT", "100"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "8"))
EVAL_MAX_NEW_TOKENS = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "32"))
EVAL_MAX_QA = int(os.environ.get("EVAL_MAX_QA", "0"))  # 0=all

# replay to prevent forgetting (stage2/3/4 mix in previous stages)
REPLAY_RATIO_PER_PREV_STAGE = float(os.environ.get("REPLAY_RATIO_PER_PREV_STAGE", "0.20"))
REPLAY_CAP_PER_PREV_STAGE   = int(os.environ.get("REPLAY_CAP_PER_PREV_STAGE", "6000"))
REPLAY_MIN_PER_PREV_STAGE   = int(os.environ.get("REPLAY_MIN_PER_PREV_STAGE", "300"))

# optional: downsample yes/no majority within a stage (helps if data is very skewed)
BALANCE_YESNO = int(os.environ.get("BALANCE_YESNO", "1"))  # 1=on, 0=off
BALANCE_MAX_KEEP_RATIO = float(os.environ.get("BALANCE_MAX_KEEP_RATIO", "1.2"))  # majority kept <= 1.2 * minority

STAGE_SPECS = [
    ("stage1_basic_presence_and_closest",
     {"OneOrMorePeople", "PersonAtClosestBench"},
     1, 1.0),
    ("stage2_indexing_benches_and_stops",
     {"IsPersonAtBench", "ClosestBench", "ClosestStopSigns"},
     1, 0.7),
    ("stage3_ordering_arrival_groups",
     {
         "ClosestToFurthestBenches",
         "ClosestToFurthestStopSigns",
         "ClosestBenchWithPerson",
         "ArrivedAtBench",
         "ArrivedAtAnimalsAroundStopSigns",
     },
     1, 0.5),
    ("stage4_path_blocking_and_detour",
     {"PathBlockedBetweenBusAndBench", "PathBlockedBetweenBusAndStopSign"},
     1, 0.5),
]

_norm_space = re.compile(r"\s+")
def normalize_answer(s: str) -> str:
    s = str(s).strip().lower()
    s = _norm_space.sub(" ", s)
    s = re.sub(r"[\,\.\!\?\:\;\(\)\[\]\{\}\"\'`]", "", s)
    return s

def parse_yesno(s: str) -> str:
    s = normalize_answer(s)
    if re.search(r"\byes\b", s):
        return "yes"
    if re.search(r"\bno\b", s):
        return "no"
    return ""

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

def set_processor_image_size(processor: LlavaNextProcessor, sz: int):
    if not hasattr(processor, "image_processor"):
        return
    ip = processor.image_processor
    try:
        ip.size = {"shortest_edge": sz}
    except Exception:
        pass
    try:
        ip.crop_size = {"height": sz, "width": sz}
    except Exception:
        pass

# ----------------- Fixed Collator (MOST IMPORTANT) -----------------
class LlavaNextBatchCollatorFixed:
    """
    Key points:
    1) prompt_len must be computed from processor encoding (includes image tokens)
    2) never use truncation='max_length' inside processor for multimodal text (causes mismatch)
    3) trim ONLY the answer by "room = MAX_LEN - prompt_len - 1"
    """

    def __init__(self, processor: LlavaNextProcessor, base_dir: Path, max_len: int):
        self.processor = processor
        self.tok = processor.tokenizer
        self.base_dir = base_dir
        self.max_len = max_len
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images: List[Image.Image] = []
        prompts: List[str] = []
        answers: List[str] = []

        eos = self.tok.eos_token or ""

        for ex in features:
            img_path = resolve_image_path(self.base_dir, ex[IMAGE_KEY])
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path} (raw={ex.get(IMAGE_KEY)})")

            with Image.open(img_path) as im:
                images.append(im.convert("RGB"))

            conv = ex[CONV_KEY]
            user_text = str(conv[0]["value"])
            assistant_text = str(conv[1]["value"])

            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": user_text}, {"type": "image"}],
            }]
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt)
            answers.append(assistant_text)

        # Encode prompt-only to get TRUE prompt_len (includes image tokens)
        enc_prompt = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prompt_lens = enc_prompt["attention_mask"].sum(dim=1).tolist()

        full_texts: List[str] = []
        for prompt, ans, plen in zip(prompts, answers, prompt_lens):
            plen = int(plen)
            if plen >= self.max_len - 1:
                raise ValueError(
                    f"Prompt length already >= MAX_LEN (prompt_len={plen} MAX_LEN={self.max_len}). "
                    f"Fix: increase MAX_LEN (e.g. 4096/8192) or reduce FORCE_IMAGE_SIZE (e.g. 224)."
                )

            room = self.max_len - plen - 1
            ans_ids = self.tok(ans, add_special_tokens=False)["input_ids"][:room]
            ans_trunc = self.tok.decode(ans_ids, skip_special_tokens=True)
            full_texts.append(prompt + ans_trunc + eos)

        # Encode full prompt+answer (still no truncation)
        enc = self.processor(
            images=images,
            text=full_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # hard clamp to MAX_LEN (safe because we only cut answer tail)
        for k in ["input_ids", "attention_mask"]:
            if k in enc and enc[k].size(1) > self.max_len:
                enc[k] = enc[k][:, :self.max_len]
        # some llava processors also return position_ids; clamp if exists
        if "position_ids" in enc and enc["position_ids"].size(1) > self.max_len:
            enc["position_ids"] = enc["position_ids"][:, :self.max_len]

        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            plen = min(int(plen), labels.size(1))
            labels[i, :plen] = -100
        labels[attn == 0] = -100

        enc["labels"] = labels
        return enc

# ----------------- Split by unique images -----------------
def split_by_images(full_ds: Dataset, base_dir: Path, val_image_count: int, seed: int) -> Tuple[Set[str], Set[str]]:
    seen = set()
    uniq = []
    for ex in full_ds:
        p = resolve_image_path(base_dir, ex[IMAGE_KEY])
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    rnd = random.Random(seed)
    rnd.shuffle(uniq)

    if val_image_count > len(uniq):
        val_image_count = max(1, int(0.2 * len(uniq)))

    val_imgs = set(uniq[:val_image_count])
    train_imgs = set(uniq[val_image_count:])
    return train_imgs, val_imgs

def filter_by_qtype_and_images(ds: Dataset, allowed_qtypes: Set[str], images_set: Set[str], base_dir: Path) -> Dataset:
    def _f(ex):
        if ex.get(QTYPE_KEY, "") not in allowed_qtypes:
            return False
        p = resolve_image_path(base_dir, ex[IMAGE_KEY])
        return p in images_set
    return ds.filter(_f)

def maybe_balance_yesno(ds: Dataset, seed: int) -> Dataset:
    if not BALANCE_YESNO or len(ds) == 0:
        return ds

    ys, ns, other = [], [], []
    for i in range(len(ds)):
        ex = ds[i]
        conv = ex.get(CONV_KEY, [])
        if len(conv) < 2:
            other.append(i)
            continue
        y = parse_yesno(conv[1].get("value", ""))
        if y == "yes":
            ys.append(i)
        elif y == "no":
            ns.append(i)
        else:
            other.append(i)

    if not ys or not ns:
        return ds

    rnd = random.Random(seed)
    if len(ys) < len(ns):
        minority, majority = ys, ns
    else:
        minority, majority = ns, ys

    keep_major = int(min(len(majority), max(len(minority) * BALANCE_MAX_KEEP_RATIO, len(minority))))
    rnd.shuffle(majority)
    keep = set(minority + majority[:keep_major] + other)
    keep = sorted(list(keep))
    return ds.select(keep)

def build_stage_train_with_replay(
    full_ds: Dataset,
    base_dir: Path,
    train_imgs: Set[str],
    curr_qtypes: Set[str],
    prev_stage_qtypes: List[Set[str]],
    seed: int,
) -> Dataset:
    curr_ds = filter_by_qtype_and_images(full_ds, curr_qtypes, train_imgs, base_dir)
    curr_ds = maybe_balance_yesno(curr_ds, seed=seed + 999)
    curr_n = len(curr_ds)
    if curr_n == 0:
        return curr_ds

    replay_parts: List[Dataset] = []
    rnd = random.Random(seed)

    for prev_q in prev_stage_qtypes:
        prev_ds = filter_by_qtype_and_images(full_ds, prev_q, train_imgs, base_dir)
        prev_n = len(prev_ds)
        if prev_n == 0:
            continue

        want = int(curr_n * REPLAY_RATIO_PER_PREV_STAGE)
        want = max(REPLAY_MIN_PER_PREV_STAGE, want)
        want = min(REPLAY_CAP_PER_PREV_STAGE, want)
        want = min(prev_n, want)
        if want <= 0:
            continue

        idx = list(range(prev_n))
        rnd.shuffle(idx)
        replay_parts.append(prev_ds.select(idx[:want]))

    if not replay_parts:
        return curr_ds.shuffle(seed=seed)

    mixed = concatenate_datasets([curr_ds] + replay_parts).shuffle(seed=seed)
    return mixed

# ----------------- Eval (generate exact match) -----------------
@torch.no_grad()
def evaluate_exact_match(
    model: LlavaNextForConditionalGeneration,
    processor: LlavaNextProcessor,
    eval_ds: Dataset,
    base_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    max_qa: int,
) -> float:
    model.eval()
    tok = processor.tokenizer
    device = next(model.parameters()).device

    if max_qa and len(eval_ds) > max_qa:
        idx = list(range(len(eval_ds)))
        random.Random(SEED).shuffle(idx)
        eval_ds = eval_ds.select(idx[:max_qa])

    total = 0
    correct = 0

    for start in range(0, len(eval_ds), batch_size):
        batch = [eval_ds[i] for i in range(start, min(start + batch_size, len(eval_ds)))]

        images, prompts, gts = [], [], []
        for ex in batch:
            img_path = resolve_image_path(base_dir, ex[IMAGE_KEY])
            with Image.open(img_path) as im:
                images.append(im.convert("RGB"))

            conv = ex[CONV_KEY]
            user_text = str(conv[0]["value"])
            gt = str(conv[1]["value"]).strip()
            gts.append(gt)

            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": user_text}, {"type": "image"}],
            }]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompts.append(prompt)

        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
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
            out = gen_ids[i, int(prompt_lens[i]):]
            pred = tok.decode(out, skip_special_tokens=True).strip()
            if normalize_answer(pred) == normalize_answer(gts[i]):
                correct += 1
            total += 1

    return (correct / total) if total else 0.0

# ----------------- Main -----------------
def main():
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()

    random.seed(SEED)

    print(f"[INFO] base_dir={base_dir}")
    print(f"[INFO] MODEL_ID={MODEL_ID}")
    print(f"[INFO] TRAIN_JSONL={TRAIN_JSON}")
    print(f"[INFO] OUTPUT_DIR={OUTPUT_DIR}")
    print(f"[INFO] MAX_LEN={MAX_LEN} FORCE_IMAGE_SIZE={FORCE_IMAGE_SIZE}")
    print(f"[INFO] BATCH_SIZE={BATCH_SIZE} GRAD_ACCUM={GRAD_ACCUM} BASE_LR={BASE_LR}")
    print(f"[INFO] NUM_WORKERS={NUM_WORKERS} PREFETCH={PREFETCH}")

    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    set_processor_image_size(processor, FORCE_IMAGE_SIZE)

    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(use_cuda and (not use_bf16))
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_json_path = str((base_dir / TRAIN_JSON).resolve()) if not os.path.isabs(TRAIN_JSON) else TRAIN_JSON
    print(f"[INFO] Loading JSONL dataset from {train_json_path} ...")
    full_ds: Dataset = load_dataset("json", data_files={"train": train_json_path})["train"]
    print(full_ds)

    train_imgs, val_imgs = split_by_images(full_ds, base_dir, VAL_IMAGE_COUNT, SEED)
    print(f"[Split] unique images: {len(train_imgs) + len(val_imgs)} | train images: {len(train_imgs)} | val images: {len(val_imgs)}")

    collator = LlavaNextBatchCollatorFixed(processor=processor, base_dir=base_dir, max_len=MAX_LEN)

    dl_kwargs = dict(
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    if NUM_WORKERS > 0:
        dl_kwargs["dataloader_prefetch_factor"] = PREFETCH

    current_model = model
    prev_stage_qtypes: List[Set[str]] = []

    for stage_idx, (stage_name, curr_qtypes, num_epochs, lr_scale) in enumerate(STAGE_SPECS, start=1):
        print("=" * 70)
        print(f"[Stage {stage_idx}] {stage_name}")
        print(f"  current question_types: {sorted(curr_qtypes)}")
        print(f"  replay from prev stages: {len(prev_stage_qtypes)}")
        print("=" * 70)

        stage_train = build_stage_train_with_replay(
            full_ds=full_ds,
            base_dir=base_dir,
            train_imgs=train_imgs,
            curr_qtypes=curr_qtypes,
            prev_stage_qtypes=prev_stage_qtypes,
            seed=SEED + stage_idx * 101,
        )

        stage_val = filter_by_qtype_and_images(full_ds, curr_qtypes, val_imgs, base_dir)

        print(f"  train QA (mixed): {len(stage_train)} | val QA (curr only): {len(stage_val)}")
        if len(stage_train) == 0:
            prev_stage_qtypes.append(set(curr_qtypes))
            continue

        stage_output_dir = str((Path(OUTPUT_DIR) / stage_name).resolve())

        args = TrainingArguments(
            output_dir=stage_output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=BASE_LR * lr_scale,
            num_train_epochs=num_epochs,
            logging_steps=20,
            save_steps=500,
            save_total_limit=2,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            max_grad_norm=1.0,
            bf16=use_bf16,
            fp16=use_fp16,
            remove_unused_columns=False,
            report_to="none",
            overwrite_output_dir=True,
            **dl_kwargs,
        )

        trainer = Trainer(
            model=current_model,
            args=args,
            train_dataset=stage_train,
            data_collator=collator,
        )

        trainer.train()
        current_model = trainer.model

        if len(stage_val) > 0:
            acc = evaluate_exact_match(
                model=current_model,
                processor=processor,
                eval_ds=stage_val,
                base_dir=base_dir,
                batch_size=EVAL_BATCH_SIZE,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                max_qa=EVAL_MAX_QA,
            )
            print(f"[Eval] {stage_name} exact-match accuracy = {acc:.4f}")

        Path(stage_output_dir).mkdir(parents=True, exist_ok=True)
        current_model.save_pretrained(stage_output_dir)
        processor.save_pretrained(stage_output_dir)
        print(f"[Save] stage adapter saved to {stage_output_dir}")

        prev_stage_qtypes.append(set(curr_qtypes))

    out_root = Path(OUTPUT_DIR).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    current_model.save_pretrained(str(out_root))
    processor.save_pretrained(str(out_root))
    print(f"[DONE] Final adapter saved to {out_root}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    main()