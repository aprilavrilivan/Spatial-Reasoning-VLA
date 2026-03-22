# Spatial-Reasoning-VLA

Synthetic spatial-reasoning data generation, GRAID-based VQA dataset construction, and LLaVA-style fine-tuning utilities for the zoo-bus setting.

## Project Overview

This repository is organized around a simple pipeline:

1. Generate synthetic scenes with benches, stop signs, people, animals, and a clock/bus agent.
2. Use GRAID to turn those images into spatial reasoning Q/A pairs.
3. Optionally export the dataset into JSONL + deduplicated images for VLM fine-tuning.
4. Fine-tune and evaluate a model with the scripts in `fine_tune/`.

### Main folders

```text
Spatial-VLA/
├── scene_gen/      # scene synthesis and image assets
├── GRAID/graid/    # GRAID pipeline + custom zoo-bus question generation
├── fine_tune/      # JSONL / training / evaluation utilities
└── readme.md
```

## End-to-End Workflow

### 1. Scene Generation

The scene generator lives in [scene_gen/generate_scene.py](/Users/xuyifan/Documents/Spatial-VLA/scene_gen/generate_scene.py).

Run it from the repository root:

```bash
cd /Users/xuyifan/Documents/Spatial-VLA
python scene_gen/generate_scene.py
```

What this script does:

- Builds base layouts with benches and stop signs.
- Adds per-variant people and animals.
- Places the clock/bus agent and heading marker.
- Draws visible numeric IDs for benches and stop signs directly on the final image.
- Runs YOLO quality filtering on the exact final JPEG artifact before saving.

Output:

- Final images are written to `scene_gen/output/`.
- Sprite assets are read from `scene_gen/src/`.

Important notes:

- You do **not** need a separate annotation script for bench / stop-sign IDs. The current `generate_scene.py` already handles ID drawing internally.
- The tunable generation parameters are defined at the top of `generate_scene.py`, including:
  - `NUM_BASE_SCENES`
  - `BUS_VARIANTS_PER_SCENE`
  - object count ranges
  - output path and YOLO filtering parameters
- The checked-in defaults are relatively large. For quick smoke tests, reduce these constants before running.

### 2. GRAID Q/A Pair Generation

The custom dataset builder entrypoint is [GRAID/graid/run_zoo_bus.py](/Users/xuyifan/Documents/Spatial-VLA/GRAID/graid/run_zoo_bus.py).

Set up the environment:

```bash
cd /Users/xuyifan/Documents/Spatial-VLA/GRAID/graid
uv venv
source .venv/bin/activate
uv sync
```

If your environment still misses optional model backends, install them as needed:

```bash
uv run install_all
```

Prepare a config file. The repository already contains an example upload-oriented config:

- [GRAID/graid/zoo_bus_upload_config.json](/Users/xuyifan/Documents/Spatial-VLA/GRAID/graid/zoo_bus_upload_config.json)

Before running, make sure these fields are correct for your machine:

- `image_root`: should point to your generated image folder
- `model_path`: should point to your YOLO weights
- `save_path`: local Hugging Face dataset output directory, if you want a local copy
- `upload_to_hub`: whether to upload directly to Hugging Face
- `hub_repo_id`: target dataset repo
- `save_local_copy`: whether to keep a local `save_to_disk()` dataset
- `num_samples`: set to `null` to read all images, or an integer to cap the sample count

Then run:

```bash
python run_zoo_bus.py zoo_bus_upload_config.json
```

What `run_zoo_bus.py` does:

- Reads a flat image folder.
- Runs object detection with the configured YOLO model.
- Applies the custom question classes in [ObjectDetectionQ.py](/Users/xuyifan/Documents/Spatial-VLA/GRAID/graid/graid/src/graid/questions/ObjectDetectionQ.py).
- Builds a Hugging Face `DatasetDict`.
- Optionally saves locally.
- Optionally uploads directly to Hugging Face.

Useful usage patterns:

Local dataset only:

```json
"upload_to_hub": false,
"save_local_copy": true
```

Direct upload without keeping a large local dataset copy:

```json
"upload_to_hub": true,
"save_local_copy": false
```

If uploading to Hugging Face, export your token first:

```bash
export HF_TOKEN=your_huggingface_token
```

## Export to JSONL for Fine-Tuning

If you saved the dataset locally with `save_local_copy=true`, you can export it into:

- a deduplicated image folder
- a LLaVA-style JSONL file

Use [GRAID/graid/export_dedup_jsonl.py](/Users/xuyifan/Documents/Spatial-VLA/GRAID/graid/export_dedup_jsonl.py):

```bash
cd /Users/xuyifan/Documents/Spatial-VLA/GRAID/graid
python export_dedup_jsonl.py \
  --dataset_dir datasets/zoo_bus_vqa \
  --split train \
  --out ../../fine_tune/train.jsonl \
  --image_out_dir ../../fine_tune/images
```

This produces:

- `fine_tune/train.jsonl`
- `fine_tune/images/`

If you uploaded directly to Hugging Face without keeping a local dataset copy, export from a downloaded local copy later instead.

## Fine-Tuning and Evaluation

The fine-tuning utilities are in `fine_tune/`.

Key files:

- [fine_tune/train_llava_next_fast.py](/Users/xuyifan/Documents/Spatial-VLA/fine_tune/train_llava_next_fast.py): training entrypoint
- [fine_tune/eval_before_after.py](/Users/xuyifan/Documents/Spatial-VLA/fine_tune/eval_before_after.py): before/after evaluation
- [fine_tune/test_all_12.py](/Users/xuyifan/Documents/Spatial-VLA/fine_tune/test_all_12.py): evaluation helper
- [fine_tune/fineTuning.ipynb](/Users/xuyifan/Documents/Spatial-VLA/fine_tune/fineTuning.ipynb): notebook workflow

The expected fine-tuning inputs are:

- a JSONL file such as `fine_tune/train.jsonl`
- the referenced image folder

## Practical Notes

- The root repository intentionally ignores local caches and generated artifacts such as `.DS_Store`, virtual environments, installation folders, and `scene_gen/output/`.
- The current scene-generation and question-generation pipeline has already been aligned so that the generated images and the custom GRAID prompts match the intended zoo-bus reasoning tasks.
- If you change the image generation rules or object assets, regenerate the scenes before building a new Q/A dataset.
