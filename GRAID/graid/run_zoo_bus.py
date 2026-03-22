#!/usr/bin/env python
"""
run_zoo_bus.py

Use GRAID to generate a VQA dataset on a custom "zoo + bus" image folder.

Usage:
    python run_zoo_bus.py                  # 默认使用 ./zoo_bus_config.json
    python run_zoo_bus.py path/to/config.json
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T  # 如果没有装 torchvision，需要先 pip 安装

from graid.data.generate_dataset import HuggingFaceDatasetBuilder
from graid.models.Ultralytics import Yolo

logger = logging.getLogger("zoo_bus")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# =======================
# 0. 顶层 collate_fn（关键修复点）
# =======================

def identity_collate(batch):
    """Simple collate_fn that just returns the list of samples as-is."""
    return batch

# =======================
# 1. 自定义 Dataset
# =======================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class ZooBusDataset(Dataset):
    """
    Minimal custom dataset for a flat image folder.

    __getitem__ 返回:
        {
            "image": image_tensor,
            "labels": [],
            "name": "relative/path/to/file.jpg",
        }
    这样既能兼容 GRAID 的 dict 分支，又能把原始文件名传给
    source_id / embedded image format 推断逻辑。
    """

    def __init__(self, root: Path, resize_longest_side: Optional[int] = None):
        self.root = root
        self.paths: List[Path] = sorted(
            [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        )
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")

        self.resize_longest_side = resize_longest_side
        # 关键改动：保持 0–255 的 uint8，不做 /255 归一化，让 Ultralytics 自己处理
        self.to_tensor = T.PILToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def _resize_keep_aspect(self, img: Image.Image) -> Image.Image:
        if self.resize_longest_side is None:
            return img
        w, h = img.size
        longest = max(w, h)
        if longest <= self.resize_longest_side:
            return img
        scale = self.resize_longest_side / float(longest)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), resample=Image.BILINEAR)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self._resize_keep_aspect(img)
        tensor = self.to_tensor(img)  # uint8, (C, H, W), 范围 [0,255]
        return {
            "image": tensor,
            "labels": [],
            "name": str(path.relative_to(self.root)),
        }

# =======================
# 2. 自定义 Builder
# =======================

class ZooBusBuilder(HuggingFaceDatasetBuilder):
    """
    小改版的 HuggingFaceDatasetBuilder：
    - 不用 GRAID 内置的 DatasetLoaderFactory
    - 自己在 _create_data_loader 里构建 DataLoader(ZooBusDataset)
    其它 QA 生成逻辑完全沿用 GRAID。
    """

    def __init__(
        self,
        image_root: Path,
        resize_longest_side: Optional[int],
        *args,
        **kwargs,
    ):
        self._image_root = image_root
        self._resize_longest_side = resize_longest_side
        super().__init__(*args, **kwargs)

    # 覆盖掉父类里对 BDD/NuImages/Waymo 的 transform 选择
    def _get_dataset_transform(self):
        # 我们在自定义 Dataset 里已经做了 to_tensor/resize，这里返回一个空操作即可
        return lambda image, labels: (image, labels)

    # 跳过 DatasetLoaderFactory
    def _init_dataset_loader(self):
        # 我们完全不用它，只是占个位
        self.dataset_loader = None

    # 用我们自己的 Dataset + DataLoader
    def _create_data_loader(self) -> DataLoader:
        dataset = ZooBusDataset(
            root=self._image_root,
            resize_longest_side=self._resize_longest_side,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=identity_collate,  # 顶层函数，避免 lambda 无法 pickle
        )
        return loader

# =======================
# 3. 配置解析 & 模型/问题构建
# =======================

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(raw_path: str, config_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def validate_hub_config(
    upload_to_hub: bool,
    hub_repo_id: Optional[str],
    save_local_copy: bool,
) -> None:
    if upload_to_hub:
        if not hub_repo_id:
            raise ValueError("hub_repo_id is required when upload_to_hub=True")
        if "/" not in hub_repo_id:
            raise ValueError(
                "hub_repo_id must be in format 'username/repo-name' or 'org/repo-name'"
            )
    if not upload_to_hub and not save_local_copy:
        raise ValueError(
            "Nothing to do: set save_local_copy=true or upload_to_hub=true"
        )


def resolve_hub_token(cfg: Dict[str, Any]) -> Optional[str]:
    token_env = cfg.get("hub_token_env")
    candidate_envs = [token_env] if token_env else []
    candidate_envs.extend(["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"])

    seen = set()
    for env_name in candidate_envs:
        if not env_name or env_name in seen:
            continue
        seen.add(env_name)
        token = os.getenv(env_name)
        if token:
            logger.info(f"Using Hugging Face token from ${env_name}")
            return token
    return None


def save_dataset_locally(dataset_dict, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving DatasetDict to disk at {save_path} ...")
    dataset_dict.save_to_disk(str(save_path))
    logger.info("✅ Saved DatasetDict locally")


def upload_dataset_to_hub(
    dataset_dict,
    *,
    hub_repo_id: str,
    hub_private: bool,
    dataset_name: str,
    split: str,
    hub_token: Optional[str],
    hub_commit_message: str,
    hub_max_shard_size: str,
) -> None:
    from huggingface_hub import create_repo

    logger.info(f"Uploading dataset to HuggingFace Hub: {hub_repo_id}")
    create_repo(
        hub_repo_id,
        repo_type="dataset",
        private=hub_private,
        exist_ok=True,
        token=hub_token,
    )
    dataset_dict.push_to_hub(
        repo_id=hub_repo_id,
        private=hub_private,
        token=hub_token,
        commit_message=hub_commit_message or f"Upload {dataset_name} {split} dataset",
        max_shard_size=hub_max_shard_size,
    )
    logger.info("✅ Pushed dataset to HuggingFace Hub")


def build_models(model_cfgs: List[Dict[str, Any]], config_dir: Path) -> List[Any]:
    models: List[Any] = []
    for cfg in model_cfgs:
        backend = cfg["backend"]
        if backend != "ultralytics":
            raise ValueError(f"Only 'ultralytics' backend is handled here, got {backend}")
        # 允许两种写法：model_path 或 model_name
        model_path = cfg.get("model_path")
        model_name = cfg.get("model_name")
        if model_path is None and model_name is None:
            raise ValueError("Ultralytics model needs 'model_path' or 'model_name'")
        model_ref = str(resolve_config_path(model_path, config_dir)) if model_path else model_name
        m = Yolo(model_ref)
        # 覆盖置信度阈值
        if "confidence_threshold" in cfg:
            m.threshold = float(cfg["confidence_threshold"])
        models.append(m)
    return models

# =======================
# 4. 主入口
# =======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="zoo_bus_config.json",
        help="Path to JSON config file (default: zoo_bus_config.json)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config_dir = config_path.parent
    cfg = load_config(config_path)

    dataset_name: str = cfg.get("dataset_name", "zoo_bus")
    split: str = cfg.get("split", "train")

    image_root = resolve_config_path(cfg["image_root"], config_dir)
    save_path = resolve_config_path(cfg["save_path"], config_dir)

    models = build_models(cfg["models"], config_dir)
    allowable_set = cfg.get("allowable_set")
    question_cfgs = cfg.get("questions", [])

    # 如果 config 里没写 transforms，这里会变成 {}，resize_longest_side = None
    resize_longest_side: Optional[int] = None
    transforms_cfg = cfg.get("transforms", {})
    if isinstance(transforms_cfg, dict):
        resize_longest_side = transforms_cfg.get("resize_longest_side")

    conf_threshold = float(cfg.get("confidence_threshold", 0.2))
    batch_size = int(cfg.get("batch_size", 1))
    num_workers = int(cfg.get("num_workers", 4))   # 如遇到别的多进程问题可以先改成 0
    qa_workers = int(cfg.get("qa_workers", 4))
    num_samples = cfg.get("num_samples")  # 可以是 None

    upload_to_hub = bool(cfg.get("upload_to_hub", False))
    hub_repo_id = cfg.get("hub_repo_id") or None
    hub_private = bool(cfg.get("hub_private", True))
    save_local_copy = bool(cfg.get("save_local_copy", True))
    cleanup_local_after_upload = bool(cfg.get("cleanup_local_after_upload", False))
    hub_max_shard_size = str(cfg.get("hub_max_shard_size", "5GB"))
    hub_commit_message = (
        cfg.get("hub_commit_message") or f"Upload {dataset_name} {split} dataset"
    )
    hub_token = resolve_hub_token(cfg)

    validate_hub_config(
        upload_to_hub=upload_to_hub,
        hub_repo_id=hub_repo_id,
        save_local_copy=save_local_copy,
    )

    logger.info(f"Image root: {image_root}")
    logger.info(f"Local dataset path: {save_path}")
    logger.info(f"Using {len(models)} detection model(s)")
    logger.info(f"Local save enabled: {save_local_copy}")
    logger.info(f"Upload to Hub enabled: {upload_to_hub}")
    if upload_to_hub:
        logger.info(f"Hub repo: {hub_repo_id} (private={hub_private})")
    if cleanup_local_after_upload and not save_local_copy:
        logger.warning(
            "cleanup_local_after_upload=true is ignored because save_local_copy=false"
        )

    # ====== 构建自定义 Builder ======
    builder = ZooBusBuilder(
        image_root=image_root,
        resize_longest_side=resize_longest_side,
        dataset_name=dataset_name,
        split=split,
        models=models,
        use_wbf=bool(cfg.get("use_wbf", False)),
        wbf_config=cfg.get("wbf_config"),
        conf_threshold=conf_threshold,
        batch_size=batch_size,
        device=None,  # 让 GRAID 自动挑设备
        allowable_set=allowable_set,
        question_configs=question_cfgs,
        num_workers=num_workers,
        qa_workers=qa_workers,
        num_samples=num_samples,
        save_path=str(save_path),
        use_original_filenames=True,
        filename_prefix=cfg.get("filename_prefix", "zoo_bus"),
    )
    logger.info(f"Questions: {[q.__class__.__name__ for q in builder.questions]}")

    # ====== 生成 HF DatasetDict ======
    dataset_dict = builder.build()

    if save_local_copy:
        save_dataset_locally(dataset_dict, save_path)

    if upload_to_hub:
        upload_dataset_to_hub(
            dataset_dict,
            hub_repo_id=hub_repo_id,
            hub_private=hub_private,
            dataset_name=dataset_name,
            split=split,
            hub_token=hub_token,
            hub_commit_message=hub_commit_message,
            hub_max_shard_size=hub_max_shard_size,
        )
        if save_local_copy and cleanup_local_after_upload and save_path.exists():
            logger.info(f"Removing local dataset copy at {save_path} ...")
            shutil.rmtree(save_path)
            logger.info("✅ Removed local dataset copy after successful upload")

    logger.info("✅ Finished generating zoo_bus VQA dataset")

if __name__ == "__main__":
    main()
