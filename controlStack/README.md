# Control Stack

This folder now separates the real-robot deployment code into small scripts:

- `camera_stream.py`: webcam capture wrapper.
- `vlm_client.py`: Qwen3-VL + Zoo-Bus-VQA LoRA inference, using the remote server by default.
- `BluetoothBot.py`: Bluetooth serial transport to the Pololu robot.
- `robot_actions.py`: converts VLM action labels to robot commands.
- `unit_test_runner.py`: repeatable unit tests for individual question types.
- `dynamic_unit_test_runner.py`: closed-loop navigation-skill unit tests.
- `dynamic_test_specs.py`: registry of dynamic test protocols and covered questions.
- `navigation_fsm_runner.py`: closed-loop navigation task runner.
- `gui.py`: lightweight selector UI for unit tests.
- `FSM.py` / `LowLevelFSM.py`: high-level task state and low-level geometry helpers.

The deployment host uses remote Qwen inference by default. The active endpoint is:

```bash
https://referenced-ram-weddings-there.trycloudflare.com/ask
```

Override it with:

```bash
export SPATIAL_VLA_REMOTE_URL=https://your-server.example.com/ask
```

If you intentionally want local inference on a CUDA workstation, set `SPATIAL_VLA_USE_REMOTE=0`. The local path defaults to CUDA fp16; set `SPATIAL_VLA_DTYPE=bf16` or `SPATIAL_VLA_DTYPE=fp16` to override it manually.

The working hardware defaults are intentionally unchanged:

- camera id: `0`
- Bluetooth port: `/dev/rfcomm0`
- baud rate: `9600`
- robot command format: `<mode>,<speed>,<goal>`

Example unit test:

```bash
python unit_test_runner.py --test DirectionToClosestBench --trials 10
```

List dynamic closed-loop unit tests:

```bash
python dynamic_unit_test_runner.py --list-tests
```

Dynamic tests default to dry-run mode and do not send Bluetooth commands unless
`--execute-actions` is passed. Examples:

```bash
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 5
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 5 --execute-actions
python dynamic_unit_test_runner.py --test closest_occupied_bench_navigation --trials 5 --execute-actions
python dynamic_unit_test_runner.py --test obstacle_aware_closest_bench --trials 5 --execute-actions
python dynamic_unit_test_runner.py --test ordered_visit_benches --max-targets 3 --trials 3 --execute-actions
```

Compare the base Qwen model before fine-tuning:

```bash
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 5 --no-adapter --run-label base_qwen --execute-actions
```

Run a specific fine-tuned adapter:

```bash
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 5 --adapter-path /path/to/best_checkpoint --run-label finetuned_qwen --execute-actions
```

Example closed-loop navigation dry run:

```bash
python navigation_fsm_runner.py --dry-run --max-steps 10
```

Set a custom adapter path with either:

```bash
export SPATIAL_VLA_ADAPTER_PATH=/path/to/best_checkpoint
```

or pass `--adapter-path /path/to/best_checkpoint`.

The base model is loaded from `Qwen/Qwen3-VL-4B-Instruct` by default. Override it
with:

```bash
export SPATIAL_VLA_MODEL_NAME=/path/or/hub/id/of/base/model
```
