from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable

from PIL import Image
import torch

try:
    import pynvml
except ImportError:
    pynvml = None


HELD_OUT_TYPES = {
    "CountPersonAtClosestBench",
    "ClosestBenchWithPerson",
    "AvoidObstacleToReachClosestBench",
    "AvoidObstacleToReachClosestStopSign",
    "DirectionToClosestBench",
    "DirectionToClosestStopSign",
}

DEFAULT_DATASET = "aprilavrilivan/zoo-bus-vqa"


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def use_left_padding(processor_or_tokenizer):
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor_or_tokenizer


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def tensor_l2_norm_sq(tensor: torch.Tensor | None) -> float:
    if tensor is None:
        return 0.0
    detached = tensor.detach()
    if detached.numel() == 0:
        return 0.0
    if detached.is_sparse:
        detached = detached.coalesce().values()
    norm = torch.linalg.vector_norm(detached).item()
    return float(norm * norm)


def parameter_l2_norm_sq(parameters: Iterable[torch.Tensor | None]) -> float:
    return sum(tensor_l2_norm_sq(parameter) for parameter in parameters)


def named_parameter_l2_norm_sq(named_parameters: Iterable[tuple[str, torch.Tensor]]) -> float:
    return sum(tensor_l2_norm_sq(parameter) for _, parameter in named_parameters)


_PYNVML_READY = False
_PYNVML_FAILED = False


def _ensure_pynvml() -> bool:
    global _PYNVML_READY, _PYNVML_FAILED
    if _PYNVML_READY:
        return True
    if _PYNVML_FAILED or pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
    except Exception:
        _PYNVML_FAILED = True
        return False
    _PYNVML_READY = True
    return True


def get_gpu_metrics(device: torch.device | None) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}

    if device is not None and device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
    else:
        device_index = torch.cuda.current_device()

    metrics: Dict[str, float] = {
        "gpu_memory_allocated_mb": float(torch.cuda.memory_allocated(device_index) / (1024 ** 2)),
        "gpu_memory_reserved_mb": float(torch.cuda.memory_reserved(device_index) / (1024 ** 2)),
        "gpu_memory_max_allocated_mb": float(torch.cuda.max_memory_allocated(device_index) / (1024 ** 2)),
    }

    if _ensure_pynvml():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        metrics["gpu_utilization_percent"] = float(util.gpu)
        metrics["gpu_memory_used_mb"] = float(memory.used / (1024 ** 2))
        metrics["gpu_memory_total_mb"] = float(memory.total / (1024 ** 2))
        if memory.total > 0:
            metrics["gpu_memory_used_percent"] = float(memory.used / memory.total * 100.0)

    return metrics


def sqrt_or_zero(value: float) -> float:
    return math.sqrt(value) if value > 0.0 else 0.0
