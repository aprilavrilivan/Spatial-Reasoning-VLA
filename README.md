# Spatial Reasoning VLM for Robot Navigation

UC Berkeley EECS 149 course project on synthetic spatial-reasoning data, vision-language model fine-tuning, and closed-loop real-robot navigation.

[Dataset](https://huggingface.co/datasets/aprilavrilivan/zoo-bus-vqa) · [Draft report](https://aprilavrilivan.github.io/projects/spatial-vla/spatial-reasoning-vlm-report.pdf)

## Overview

The project combines learned visual reasoning with explicit control logic:

```text
synthetic Zoo-Bus scenes
        ↓
YOLO detector-in-the-loop quality checks
        ↓
GRAID-generated spatial VQA pairs
        ↓
LoRA fine-tuning of four vision-language models
        ↓
Qwen3-VL inference + answer parser + navigation FSM
        ↓
Bluetooth commands + encoder/gyro feedback on a Pololu 3pi+
```

The VLM performs high-level spatial reasoning. A deterministic state machine turns its short text answers into navigation decisions, while the robot controller handles low-level motion feedback.

## Repository structure

```text
Spatial-VLA/
├── scene_gen/                 # synthetic scene generation and YOLO filtering
├── GRAID/graid/               # VQA construction and Zoo-Bus question types
├── zoo_bus_vqa_finetune/      # multi-model LoRA training and evaluation
├── robot_deployment_qwen/     # Qwen inference, FSM, Bluetooth, and robot code
├── results/                   # curated machine-readable experiment results
└── paper_assets/              # final figures and editable figure sources
```

## Zoo-Bus-VQA dataset

The public [Zoo-Bus-VQA dataset](https://huggingface.co/datasets/aprilavrilivan/zoo-bus-vqa) contains 80,295 image-question-answer rows:

| Split | Rows |
| --- | ---: |
| Train | 62,895 |
| Evaluation | 8,700 |
| Test | 8,700 |

It covers 29 spatial question families. Twenty-three appear during training, while six compositional families are reserved for held-out evaluation.

```python
from datasets import load_dataset

dataset = load_dataset("aprilavrilivan/zoo-bus-vqa")
```

## Scene and dataset generation

Generate scenes from the repository root:

```bash
python scene_gen/generate_scene.py
```

Generated images are written to `scene_gen/output/` and excluded from Git because the released dataset is hosted on Hugging Face.

To build VQA pairs, install the GRAID environment and run the project configuration:

```bash
cd GRAID/graid
uv venv
uv sync
python run_zoo_bus.py zoo_bus_upload_config.json
```

Set `upload_to_hub` to `false` for a local-only dataset. Hub authentication is read from `HF_TOKEN`; no credential is stored in this repository.

## Fine-tuning and evaluation

The shared training pipeline supports:

- SmolVLM2-2.2B-Instruct
- Qwen3-VL-4B-Instruct
- Gemma 3 4B IT
- InternVL3.5-4B-HF

From `zoo_bus_vqa_finetune/`, run one model or the sequential launcher:

```bash
./scripts/train_qwen.sh
./scripts/train_all_sequential.sh
```

Final Zoo-Bus-VQA results recorded by the project are:

| Model | Before | After | Seen after | Held-out after |
| --- | ---: | ---: | ---: | ---: |
| SmolVLM2 | 27.7% | 83.0% | 91.0% | 52.3% |
| Gemma 3 | 31.4% | 91.5% | 96.9% | 70.8% |
| InternVL3.5 | 34.6% | 92.6% | 96.8% | 76.6% |
| Qwen3-VL | 46.0% | **94.0%** | **97.3%** | **81.6%** |

Machine-readable reports and compact external-evaluation summaries are preserved under `results/`.

## Real-robot deployment

`robot_deployment_qwen/` contains:

- overhead webcam capture;
- Qwen3-VL inference with an optional LoRA adapter;
- parsing and finite-state navigation logic;
- Bluetooth communication with a Pololu 3pi+ 2040;
- Lingua Franca robot-side encoder and gyro feedback control;
- 17 closed-loop unit-test protocols and a longer multi-stage task.

The deployment client uses a local inference endpoint by default. Point it to a remote server when required:

```bash
export SPATIAL_VLA_REMOTE_URL=https://your-server.example.com/ask
```

Run a dry test before enabling Bluetooth actions:

```bash
cd robot_deployment_qwen/controlStack
python dynamic_unit_test_runner.py --list-tests
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 1
```

The archived physical runs record 2/81 successful trials for the base model and 101/112 for the fine-tuned model. These totals summarize different collections of trials rather than a strictly paired experiment; compact per-test summaries are included for inspection.

## Artifact policy

Source code, configurations, final figures, compact evaluation reports, and robot summary tables are tracked. Regenerable environments, downloaded benchmark data, full prediction dumps, model checkpoints, generated scenes, and raw robot frame sequences stay local and are excluded by `.gitignore`.

Adapter checkpoints are not bundled with the Git repository. Supply one with `SPATIAL_VLA_ADAPTER_PATH` or `--adapter-path` when running local fine-tuned inference.

## Team and attribution

EECS 149 Group 15 project by Ivan Xu, Jungpyo Lee, and Randy Bui.

The bundled GRAID source is derived from [KE7/graid](https://github.com/KE7/graid) and retains its Apache-2.0 license in `GRAID/graid/LICENSE`. The robot firmware template and borrowed Pololu utilities retain their notices under `robot_deployment_qwen/lf-3pi/`.

This repository does not currently grant a project-wide software license; third-party components remain governed by their respective licenses.
