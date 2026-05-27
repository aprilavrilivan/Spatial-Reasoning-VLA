# Spatial-VLA Qwen Robot Deployment

This folder contains the real-robot deployment code for the Zoo-Bus-VQA fine-tuned Qwen model. The current setup uses:

- a bird's-eye webcam for scene capture,
- a Qwen3-VL model with the fine-tuned LoRA adapter,
- Bluetooth serial control for a Pololu 3pi robot,
- static and dynamic unit tests for spatial reasoning question types,
- a small navigation FSM for closed-loop tabletop tasks.

## Main folders

- `controlStack/`: active Python deployment code.
- `lf-3pi/`: robot-side Lingua Franca / Pololu control code. The active robot program is `src/BTControlledRobot.lf`; `src/BTModuleSetup.lf` is kept for Bluetooth module setup.

The active entry points are inside `controlStack/`:

```bash
cd robot_deployment_qwen/controlStack
python unit_test_runner.py --test DirectionToClosestBench --trials 10
python dynamic_unit_test_runner.py --list-tests
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 5 --execute-actions
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 5 --no-adapter --run-label base_qwen --execute-actions
python navigation_fsm_runner.py --dry-run --max-steps 10
python gui.py
```

Hardware defaults are documented in `controlStack/README.md`. The working camera and Bluetooth settings are intentionally left unchanged.

The deployment computer is expected to run inference locally on an NVIDIA RTX 3090. The Qwen inference wrapper therefore defaults to CUDA fp16. To override this manually, set `SPATIAL_VLA_DTYPE=bf16` or `SPATIAL_VLA_DTYPE=fp16` before launching a script.

By default, the runners load `Qwen/Qwen3-VL-4B-Instruct` plus the local LoRA adapter at `controlStack/best_checkpoint`. Use `--adapter-path /path/to/best_checkpoint` to switch fine-tuned adapters, or `--no-adapter --run-label base_qwen` to evaluate the base model before fine-tuning.
