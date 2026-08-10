# External Spatial Evaluation

This folder is reserved for external spatial-reasoning evaluation data and results.

The evaluation script is:

```bash
scripts/evaluate_external_spatial.py
```

It evaluates a model before and/or after zoo-bus-vqa LoRA fine-tuning on:

- VSR (`cambridgeltl/vsr_zeroshot`): yes/no image-sentence spatial verification.
- WhatsUp controlled images: shuffled multiple-choice spatial caption selection.

Example:

```bash
python scripts/evaluate_external_spatial.py \
  --dataset all \
  --model_family qwen \
  --model_name Qwen/Qwen3-VL-4B-Instruct \
  --adapter_path outputs/qwen/qwen_final_evalfix_20260503_073400/adapter/best_checkpoint \
  --state both \
  --download_whatsup \
  --batch_size 64 \
  --max_seq_length 2048
```

Outputs are written under:

```text
external_eval/results/<model_family>/<run_name>/
```

Each run writes prediction CSVs, per-dataset before/after reports, comparison reports, and a top-level `external_spatial_eval_summary.json`.
