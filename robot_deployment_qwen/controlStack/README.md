# Control stack

The Python control stack is separated into small components:

- `camera_stream.py`: webcam capture wrapper.
- `vlm_client.py`: local or HTTP-served Qwen3-VL inference.
- `BluetoothBot.py`: Bluetooth serial transport to the Pololu robot.
- `robot_actions.py`: VLM action-label to robot-command conversion.
- `unit_test_runner.py`: repeatable tests for individual question types.
- `dynamic_unit_test_runner.py`: closed-loop navigation-skill tests.
- `dynamic_test_specs.py`: registry of dynamic protocols and covered questions.
- `navigation_fsm_runner.py`: multi-stage closed-loop navigation runner.
- `gui.py`: lightweight unit-test selector.
- `FSM.py` and `LowLevelFSM.py`: task state and geometry helpers.

## Inference

HTTP inference is enabled by default at `http://127.0.0.1:8899/ask`. Override it when the model server runs elsewhere:

```bash
export SPATIAL_VLA_REMOTE_URL=https://your-server.example.com/ask
```

To run the model in the control process on a CUDA workstation:

```bash
export SPATIAL_VLA_USE_REMOTE=0
export SPATIAL_VLA_ADAPTER_PATH=/path/to/best_checkpoint
```

The base model is `Qwen/Qwen3-VL-4B-Instruct`. Use `SPATIAL_VLA_MODEL_NAME` to override it and `SPATIAL_VLA_DTYPE=bf16` or `fp16` to select CUDA precision.

## Hardware defaults

- camera id: `0`
- Bluetooth port: `/dev/rfcomm0`
- baud rate: `9600`
- robot command format: `<mode>,<speed>,<goal>`

Set a different serial device with `SPATIAL_VLA_BLUETOOTH_PORT`.

## Examples

```bash
python unit_test_runner.py --test DirectionToClosestBench --trials 10
python dynamic_unit_test_runner.py --list-tests
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 5
python dynamic_unit_test_runner.py --test closest_occupied_bench_navigation --trials 5
python navigation_fsm_runner.py --dry-run --max-steps 10
```

The commands above do not move the robot. Add `--execute-actions` to a dynamic test only after confirming the scene, Bluetooth link, motion distances, and emergency-stop procedure.

To compare model conditions:

```bash
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 5 --no-adapter --run-label base_qwen --execute-actions
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 5 --adapter-path /path/to/best_checkpoint --run-label finetuned_qwen --execute-actions
```
