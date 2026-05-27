import os
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from answer_parsing import clean_text


DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_ADAPTER_PATH = str(Path(__file__).resolve().parent / "best_checkpoint")
SHORT_ANSWER_WRAPPER = (
    "Answer the visual question using a short final answer only. "
    "Do not explain your reasoning.\n\n"
)


def preferred_torch_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    precision = os.environ.get("SPATIAL_VLA_DTYPE", "fp16").strip().lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


class QwenVLMClient:
    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        use_adapter: bool = True,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ):
        self.model_name = model_name or os.environ.get("SPATIAL_VLA_MODEL_NAME", DEFAULT_MODEL)
        self.use_adapter = use_adapter
        self.adapter_path = adapter_path or os.environ.get("SPATIAL_VLA_ADAPTER_PATH", DEFAULT_ADAPTER_PATH)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.processor = None

    def build_prompt(self, question: str) -> str:
        return SHORT_ANSWER_WRAPPER + question

    def load(self):
        if self.model is not None:
            return self

        adapter_path = Path(self.adapter_path) if self.use_adapter else None
        if self.use_adapter and adapter_path is not None and not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter path not found: {adapter_path}. "
                "Set SPATIAL_VLA_ADAPTER_PATH, pass --adapter-path, or use --no-adapter for the base model."
            )

        dtype = preferred_torch_dtype()

        if self.use_adapter and adapter_path is not None:
            try:
                self.processor = AutoProcessor.from_pretrained(str(adapter_path), use_fast=True)
            except Exception:
                self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)

        device_map = "auto" if torch.cuda.is_available() else None
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )
        if self.use_adapter and adapter_path is not None:
            self.model = PeftModel.from_pretrained(base, str(adapter_path))
        else:
            self.model = base
        self.model.eval()
        return self

    def ask(self, frame_bgr, question: str) -> dict:
        self.load()
        if frame_bgr is None:
            raise ValueError("Cannot run VLM inference on an empty frame.")

        start = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        prompt = self.build_prompt(question)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            images=[image],
            text=[input_text],
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output = self.model.generate(**generation_kwargs)

        input_len = inputs["input_ids"].shape[1]
        gen_ids = output[0][input_len:]
        raw = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        answer = clean_text(raw)
        return {
            "question": question,
            "full_prompt": prompt,
            "raw_answer": raw,
            "answer": answer,
            "latency_sec": time.perf_counter() - start,
        }
