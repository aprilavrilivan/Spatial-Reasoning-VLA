# OpenSpaces Qwen Workspace

This workspace fine-tunes `Qwen/Qwen3-VL-4B-Instruct` on
`remyxai/OpenSpaces` for comparison against the existing ZooBus-Qwen run.

The OpenSpaces dataset stores each image with a multi-turn `messages` list.
The training path flattens every user/assistant turn into a single image VQA
example:

```text
image + user question -> assistant answer
```

The split is source-row aware. The original OpenSpaces `train` rows are split
into train/evaluation rows before flattening, so different QA turns from the
same image do not leak across train and evaluation. The original OpenSpaces
`test` split is kept as the final source-domain test split.

## Files

- `src/openspaces_data.py`: OpenSpaces loader, image decoding, message flattening,
  and lightweight question/answer type classification.
- `src/train_qwen_openspaces.py`: Qwen LoRA-SFT trainer for OpenSpaces.
- `configs/qwen_openspaces.yaml`: default experiment configuration.
- `scripts/train_qwen_openspaces.sh`: runnable launcher.
- `outputs/qwen_openspaces/`: default output root.

## Default Run

```bash
cd zoo_bus_vqa_finetune
./scripts/train_qwen_openspaces.sh outputs/qwen_openspaces/openspaces_qwen_default
```

The defaults intentionally keep the Qwen effective batch size at 64:

```text
per_device_train_batch_size = 16
gradient_accumulation_steps = 4
max_steps = 2949
```

This makes the run compute-comparable with the ZooBus-Qwen fine-tuning run.
OpenSpaces contains high-resolution natural images, so the Qwen processor is
loaded with a bounded visual token budget:

```text
image_min_pixels = 256 * 28 * 28
image_max_pixels = 768 * 28 * 28
```

This prevents image-token truncation errors and keeps `max_seq_length=2048`
usable without inflating every batch to very long contexts.

## Reports

The run writes:

- `run_summary.json`
- `openspaces_dataset_summary.json`
- `before_finetune_report.json`
- `eval_predictions/*.csv`
- `eval_predictions/*.json`
- `test_predictions/*.csv`
- `test_predictions/*.json`
- `best_checkpoint_eval_predictions/best_eval_metrics.json`
- `final_test_report.json`

The selection metric is OpenSpaces evaluation overall accuracy. The final
comparison against ZooBus-Qwen should still use the external spatial evaluation
pipeline on VSR and WhatsUp, plus any real deployment images.
