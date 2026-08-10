# Qwen real-robot deployment

Real-robot deployment code for the Zoo-Bus-VQA fine-tuned Qwen3-VL model. The system uses:

- a bird's-eye webcam for scene capture;
- Qwen3-VL with an optional LoRA adapter;
- short-answer parsing and a navigation finite-state machine;
- Bluetooth serial control of a Pololu 3pi+ 2040;
- encoder and gyro feedback in the Lingua Franca robot controller;
- static, dynamic, and multi-stage closed-loop tests.

## Layout

- `controlStack/`: Python inference, parsing, FSM, experiment logging, and test runners.
- `lf-3pi/`: Lingua Franca and Pololu robot-side code.
- `../results/robot_evaluation/`: compact summaries from the physical test campaign.

Raw test runs, camera frames, and model checkpoints are retained locally but excluded from Git. Set `SPATIAL_VLA_ADAPTER_PATH` or pass `--adapter-path` to use a local LoRA checkpoint.

## Inference configuration

The deployment client sends frames to `http://127.0.0.1:8899/ask` by default. Start the included server on a CUDA machine:

```bash
cd controlStack
python vlm_client.py --serve-remote --adapter-path /path/to/best_checkpoint
```

For a server on another machine or behind a tunnel, configure the endpoint explicitly:

```bash
export SPATIAL_VLA_REMOTE_URL=https://your-server.example.com/ask
```

To run fully local inference in the control process instead, set `SPATIAL_VLA_USE_REMOTE=0`.

## Running tests

From `controlStack/`:

```bash
python unit_test_runner.py --test DirectionToClosestBench --trials 10
python dynamic_unit_test_runner.py --list-tests
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 5
python navigation_fsm_runner.py --dry-run --max-steps 10
```

Dynamic tests are dry runs unless `--execute-actions` is supplied. Review `DYNAMIC_UNIT_TEST_COMMANDS.md` before enabling physical motion.

The base-model condition is selected with `--no-adapter --run-label base_qwen`. Use `--adapter-path /path/to/best_checkpoint --run-label finetuned_qwen` for a fine-tuned condition.
