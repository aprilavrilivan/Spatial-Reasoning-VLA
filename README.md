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

By default, VLM inference is served by a remote Qwen3-VL endpoint:

```bash
https://referenced-ram-weddings-there.trycloudflare.com/ask
```

The deployment scripts send the captured camera frame and question to that endpoint, then use the returned short answer to control the robot. To point the code at a different remote server, set:

```bash
export SPATIAL_VLA_REMOTE_URL=https://your-server.example.com/ask
```

Use `--no-adapter --run-label base_qwen` to evaluate the base model before fine-tuning. Use `--adapter-path /path/to/best_checkpoint` to switch fine-tuned adapters when running a local server. To force fully local inference on a CUDA machine, set `SPATIAL_VLA_USE_REMOTE=0`; the local path loads `Qwen/Qwen3-VL-4B-Instruct` plus the LoRA adapter at `controlStack/best_checkpoint`.
