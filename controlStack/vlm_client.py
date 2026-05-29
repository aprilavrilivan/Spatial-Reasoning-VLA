from __future__ import annotations

import os
import time
import json
import base64
import argparse
from contextlib import nullcontext
from pathlib import Path
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request, error

from answer_parsing import clean_text


DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_ADAPTER_PATH = str(Path(__file__).resolve().parent / "best_checkpoint")
DEFAULT_REMOTE_URL = "https://chem-dakota-teams-appropriations.trycloudflare.com/ask"
DEFAULT_REMOTE_TIMEOUT_SEC = 120
SHORT_ANSWER_WRAPPER = (
    "Answer the visual question using a short final answer only. "
    "Do not explain your reasoning.\n\n"
)


def preferred_torch_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    precision = os.environ.get("SPATIAL_VLA_DTYPE", "fp16").strip().lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


def use_remote_by_default() -> bool:
    value = os.environ.get("SPATIAL_VLA_USE_REMOTE", "1").strip().lower()
    return value not in {"0", "false", "no", "local"}


class LocalQwenBackend:
    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        use_adapter: bool = True,
        temperature: float = 0.0,
    ):
        self.model_name = model_name or os.environ.get("SPATIAL_VLA_MODEL_NAME", DEFAULT_MODEL)
        self.use_adapter = use_adapter
        self.adapter_path = adapter_path or os.environ.get("SPATIAL_VLA_ADAPTER_PATH", DEFAULT_ADAPTER_PATH)
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

        import torch

        dtype = preferred_torch_dtype()
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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

    def ask_pil(self, image, question: str, use_adapter: bool | None = None) -> dict:
        self.load()
        import torch
        from PIL import Image

        start = time.perf_counter()
        image = image.convert("RGB")
        requested_use_adapter = self.use_adapter if use_adapter is None else use_adapter
        adapter_enabled = bool(self.use_adapter and requested_use_adapter)
        if requested_use_adapter and not self.use_adapter:
            raise RuntimeError("Adapter inference was requested, but this backend was started without an adapter.")

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
            "max_new_tokens": 32,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        adapter_context = nullcontext()
        if self.use_adapter and not requested_use_adapter:
            if not hasattr(self.model, "disable_adapter"):
                raise RuntimeError("Loaded adapter model does not support disable_adapter(); cannot run base inference safely.")
            adapter_context = self.model.disable_adapter()

        with torch.no_grad(), adapter_context:
            output = self.model.generate(**generation_kwargs)

        input_len = inputs["input_ids"].shape[1]
        gen_ids = output[0][input_len:]
        raw = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        answer = clean_text(raw)
        end = time.perf_counter()
        return {
            "question": question,
            "full_prompt": prompt,
            "raw_answer": raw,
            "answer": answer,
            "latency_sec": end - start,
            "use_adapter": adapter_enabled,
            "model_mode": "adapter" if adapter_enabled else "base",
        }

    def ask(self, frame_bgr, question: str) -> dict:
        if frame_bgr is None:
            raise ValueError("Cannot run VLM inference on an empty frame.")
        import cv2
        from PIL import Image

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        return self.ask_pil(image, question)


class QwenVLMClient:
    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        use_adapter: bool = True,
        temperature: float = 0.0,
        remote_url: str | None = None,
        remote_timeout_sec: float | None = None,
        use_remote: bool | None = None,
    ):
        self.use_remote = use_remote_by_default() if use_remote is None else use_remote
        self.use_adapter = use_adapter
        self.remote_url = remote_url or os.environ.get("SPATIAL_VLA_REMOTE_URL", DEFAULT_REMOTE_URL)
        self.remote_timeout_sec = remote_timeout_sec or float(
            os.environ.get("SPATIAL_VLA_REMOTE_TIMEOUT_SEC", str(DEFAULT_REMOTE_TIMEOUT_SEC))
        )
        self.local_backend = None
        if not self.use_remote:
            self.local_backend = LocalQwenBackend(
                model_name=model_name,
                adapter_path=adapter_path,
                use_adapter=use_adapter,
                temperature=temperature,
            )

    def load(self):
        if self.local_backend is not None:
            self.local_backend.load()
        return self

    def build_prompt(self, question: str) -> str:
        return SHORT_ANSWER_WRAPPER + question

    def ask(self, frame_bgr, question: str) -> dict:
        if not self.use_remote:
            return self.local_backend.ask(frame_bgr, question)
        if frame_bgr is None:
            raise ValueError("Cannot run VLM inference on an empty frame.")

        import cv2

        start = time.perf_counter()
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise ValueError("Failed to JPEG-encode camera frame for remote VLM inference.")

        payload = {
            "question": question,
            "image_jpeg_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "use_adapter": self.use_adapter,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.remote_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.remote_timeout_sec) as response:
                result = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                f"Remote Qwen inference failed at {self.remote_url}. "
                "Make sure the SSH tunnel and remote server are running."
            ) from exc

        end = time.perf_counter()
        result.setdefault("question", question)
        result.setdefault("full_prompt", self.build_prompt(question))
        result.setdefault("raw_answer", result.get("answer", ""))
        result["latency_sec"] = end - start
        return result


class RemoteQwenServer:
    def __init__(self, backend: LocalQwenBackend):
        self.backend = backend

    def ask(self, payload: dict) -> dict:
        image_b64 = payload.get("image_jpeg_b64")
        question = payload.get("question")
        if not image_b64 or not question:
            raise ValueError("Request must include image_jpeg_b64 and question.")
        from PIL import Image

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        use_adapter = payload.get("use_adapter")
        if use_adapter is not None:
            use_adapter = bool(use_adapter)
        return self.backend.ask_pil(image, question, use_adapter=use_adapter)


def make_handler(server_state: RemoteQwenServer):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"ok": True, "model": DEFAULT_MODEL})
                return
            self._send_json(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            start = time.perf_counter()
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if self.path == "/echo":
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "bytes_received": content_length,
                            "server_latency_sec": time.perf_counter() - start,
                        },
                    )
                    return
                if self.path != "/ask":
                    self._send_json(404, {"ok": False, "error": "not found"})
                    return
                result = server_state.ask(payload)
                result["server_latency_sec"] = time.perf_counter() - start
                self._send_json(200, result)
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})

        def log_message(self, format, *args):
            return

    return Handler


def serve_remote_qwen(args):
    backend = LocalQwenBackend(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        use_adapter=not args.no_adapter,
        temperature=args.temperature,
    )
    print("Loading Qwen remote inference backend...")
    backend.load()
    server_state = RemoteQwenServer(backend)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server_state))
    print(f"Remote Qwen server listening on http://{args.host}:{args.port}")
    httpd.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen VLM local/remote inference helpers.")
    parser.add_argument("--serve-remote", action="store_true", help="Run the remote Qwen HTTP inference server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.serve_remote:
        serve_remote_qwen(cli_args)
    else:
        raise SystemExit("Use --serve-remote to start the remote Qwen inference server.")
