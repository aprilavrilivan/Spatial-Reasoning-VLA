import csv
import json
from datetime import datetime
from pathlib import Path

import cv2


class ExperimentLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def save_frame(self, frame, stem: str) -> str:
        path = self.image_dir / f"{stem}.jpg"
        cv2.imwrite(str(path), frame)
        return str(path)

    def log(self, **row):
        row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        self.rows.append(row)
        self._write()

    def _write(self):
        json_path = self.output_dir / "results.json"
        csv_path = self.output_dir / "results.csv"
        json_path.write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
        if not self.rows:
            return
        fieldnames = sorted({key for row in self.rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
