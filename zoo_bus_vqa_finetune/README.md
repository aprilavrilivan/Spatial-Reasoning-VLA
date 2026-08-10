# Zoo-Bus-VQA multi-model fine-tuning

LoRA supervised fine-tuning and before/after evaluation for spatial visual question answering on [aprilavrilivan/zoo-bus-vqa](https://huggingface.co/datasets/aprilavrilivan/zoo-bus-vqa).

## Supported model families

- SmolVLM2-2.2B-Instruct
- Qwen3-VL-4B-Instruct
- Gemma 3 4B IT
- InternVL3.5-4B-HF

All four use a shared training entry point, model-specific adapters, bf16 LoRA-SFT, and exact-match generation evaluation.

## Layout

```text
configs/       model and training hyperparameters
scripts/       training, benchmark, ablation, and external-evaluation launchers
src/           dataset loading, metrics, model adapters, and training code
external_eval/ VSR, WhatsUp, ERQA, and RoboSpatial evaluation code
openspaces_qwen/ comparison-workspace notes
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset loader reads the public Hugging Face release. To train one model:

```bash
./scripts/train_qwen.sh
./scripts/train_smol.sh
./scripts/train_gemma.sh
./scripts/train_internvl.sh
```

Run the shared sequential launcher with:

```bash
./scripts/train_all_sequential.sh
```

## Evaluation artifacts

Large checkpoints, prediction dumps, and downloaded benchmark images are excluded from Git. Compact model, ablation, and external-evaluation summaries are archived in `../results/`.

Pass the appropriate checkpoint path to the external and robotic evaluation launchers after training. The checked-in scripts retain the original experiment run names so the compact reports remain traceable to local artifacts.
