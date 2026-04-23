from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


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


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
