# Curated experiment results

This folder preserves compact, machine-readable outputs needed to audit the reported experiments without checking large checkpoints, full prediction dumps, or thousands of robot frames into Git.

## Contents

- `model_evaluation/`: sanitized run, best-evaluation, and final-test JSON for SmolVLM2, Qwen3-VL, Gemma 3, InternVL3.5, and the OpenSpaces comparison.
- `prompt_ablation/`: summary JSON for the closest-occupied-bench prompt variants.
- `external_evaluation/summary.csv`: VSR, WhatsUp, ERQA, and RoboSpatial before/after results.
- `robot_evaluation/`: compact dynamic-unit-test and complex-navigation summaries.
- `dataset/pre_rebalance_dataset_stats.json`: preserved statistics from a local dataset snapshot before the final Hub rebalance.

The pre-rebalance snapshot contains 80,197 QA rows and is retained only for provenance. The canonical public dataset currently contains 80,295 rows with 62,895/8,700/8,700 train/evaluation/test splits; use the [Hugging Face release](https://huggingface.co/datasets/aprilavrilivan/zoo-bus-vqa) for current statistics.

Some source reports originally contained absolute experiment-machine paths. `scripts/curate_results.py` removes those non-portable fields while preserving metrics.
