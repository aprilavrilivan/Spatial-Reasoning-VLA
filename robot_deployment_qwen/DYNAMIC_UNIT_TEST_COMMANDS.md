# Dynamic Real-Robot Unit Test Commands

This note summarizes the command-line workflow for running the 17 closed-loop
dynamic unit tests in `controlStack/dynamic_unit_test_runner.py`.

The tests are designed for real robot evaluation, not static image scoring. In
each trial, the overhead camera captures the scene, the fine-tuned Qwen model
answers one or more Zoo-Bus-VQA-style spatial questions, and the Pololu 3pi
executes the corresponding action through Bluetooth.

## Basic Setup

Run all commands from the control stack folder:

```bash
cd robot_deployment_qwen/controlStack
```

List all available dynamic tests:

```bash
python dynamic_unit_test_runner.py --list-tests
```

Dry-run a test without sending robot commands:

```bash
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 1
```

Run a real closed-loop test with Bluetooth commands enabled:

```bash
python dynamic_unit_test_runner.py --test face_arrive_bench --bench-number 2 --trials 10 --execute-actions
```

Run the same test with the base Qwen model before fine-tuning:

```bash
python dynamic_unit_test_runner.py \
  --test face_arrive_bench \
  --bench-number 2 \
  --trials 10 \
  --no-adapter \
  --run-label base_qwen \
  --execute-actions
```

Run with a specific fine-tuned LoRA adapter checkpoint:

```bash
python dynamic_unit_test_runner.py \
  --test face_arrive_bench \
  --bench-number 2 \
  --trials 10 \
  --adapter-path /path/to/best_checkpoint \
  --run-label finetuned_qwen \
  --execute-actions
```

By default, the runner:

- uses camera id `0`;
- sends Bluetooth commands only when `--execute-actions` is passed;
- moves forward by `2 cm` per forward step;
- turns by `10 degrees` per turn step;
- during approach, re-checks target facing after every `2` forward steps;
- waits `2` seconds after each Bluetooth command before sending the next one;
- logs images, model answers, parsed answers, actions, and final trial labels;
- asks the operator for an external success/failure label at the end of each trial.

Useful optional flags:

```bash
--camera-id 0
--adapter-path /path/to/best_checkpoint
--no-adapter
--run-label finetuned_qwen
--output-dir real_robot_results/dynamic_unit_tests
--step-cm 2
--turn-deg 10
--max-steps 30
--max-turn-steps 24
--realign-every 2
--command-pause-sec 2.0
--no-display
--no-manual-label
```

`--realign-every 2` means that after every two forward steps, the runner asks the
target-facing question again and corrects the robot until the model answers
`already facing`. Use `--realign-every 0` to disable this correction.

For obstacle-aware tests, `keep straight` sends one forward command. `turn left`
and `turn right` send a turn command followed by one forward command, so the
robot actually moves around the obstacle instead of turning in place and being
pulled back to the original target direction on the next loop.

`--command-pause-sec 2.0` is important when one model answer maps to multiple
low-level commands, such as `turn left` followed by `forward`. The robot-side
controller uses encoder and gyro feedback internally, but it does not send a
completion signal back to Python. The pause gives each primitive time to finish
before the next command is sent. Increase it if the robot is still moving when
the next command begins.

For before/after fine-tuning comparisons, use `--no-adapter --run-label
base_qwen` for the base model and `--adapter-path ... --run-label
finetuned_qwen` for the fine-tuned adapter. The `run_label` value is written into
`results.csv` and `results.json`.

For paper-quality runs, keep manual labeling enabled so every trial has an
external success label from the operator.

## Recommended Trial Count

A good default is:

- `10` randomized trials per dynamic test for the final fine-tuned Qwen model;
- if comparing before/after fine-tuning, use the same physical layouts and the
  same number of trials for both model conditions whenever possible.

Between trials, slightly randomize the robot pose and object layout while keeping
the target IDs valid for that test.

## 17 Dynamic Unit Tests

Replace target IDs with the IDs visible in the current physical scene.

### 1. Face Bench And Arrive

Uses `TurnDirectionToBench` and `ArrivedAtBench`.

```bash
uv run dynamic_unit_test_runner.py \
  --test face_arrive_bench \
  --bench-number 2 \
  --trials 3 \
  --command-pause-sec 1.0 \
  --execute-actions
```

### 2. Face Stop Sign And Arrive

Uses `TurnDirectionToStopSign` and `ArrivedAtAnimalsAroundStopSigns`.

```bash
python dynamic_unit_test_runner.py \
  --test face_arrive_stop \
  --stop-sign-number 1 \
  --trials 10 \
  --execute-actions
```

### 3. Egocentric Bench Controller

Uses `BusHeadingDirection`, `BenchRelativeToHeading`, and `ArrivedAtBench`.

```bash
python dynamic_unit_test_runner.py \
  --test egocentric_bench \
  --bench-number 2 \
  --trials 10 \
  --execute-actions
```

### 4. Egocentric Stop Sign Controller

Uses `BusHeadingDirection`, `StopSignRelativeToHeading`, and
`ArrivedAtAnimalsAroundStopSigns`.

```bash
python dynamic_unit_test_runner.py \
  --test egocentric_stop \
  --stop-sign-number 1 \
  --trials 10 \
  --execute-actions
```

### 5. Navigate To Closest Bench

Uses `ClosestBench`, `DirectionToClosestBench`, `TurnDirectionToBench`, and
`ArrivedAtBench`.

```bash
python dynamic_unit_test_runner.py \
  --test closest_bench_navigation \
  --trials 10 \
  --execute-actions
```

### 6. Navigate To Closest Stop Sign

Uses `ClosestStopSign`, `DirectionToClosestStopSign`, `TurnDirectionToStopSign`,
and `ArrivedAtAnimalsAroundStopSigns`.

```bash
python dynamic_unit_test_runner.py \
  --test closest_stop_navigation \
  --trials 10 \
  --execute-actions
```

### 7. Navigate To Closest Occupied Bench

Uses `CountPeople`, `ClosestBenchWithPerson`, `CountPersonAtClosestBench`,
`TurnDirectionToBench`, and `ArrivedAtBench`.

```bash
python dynamic_unit_test_runner.py \
  --test closest_occupied_bench_navigation \
  --trials 10 \
  --execute-actions
```

### 8. Navigate To A Bench With At Least K People

Uses `ListBenchesWithAtLeastKPeople`, `CountPeopleAtBench`,
`TurnDirectionToBench`, and `ArrivedAtBench`.

```bash
python dynamic_unit_test_runner.py \
  --test bench_with_at_least_k_people \
  --k 2 \
  --trials 10 \
  --execute-actions
```

### 9. Navigate To A Stop Sign With At Least K Animals

Uses `CountAnimals`, `ListStopSignsWithAtLeastKAnimals`,
`CountAnimalsAtStopSign`, `TurnDirectionToStopSign`, and
`ArrivedAtAnimalsAroundStopSigns`.

```bash
python dynamic_unit_test_runner.py \
  --test stop_with_at_least_k_animals \
  --k 3 \
  --trials 10 \
  --execute-actions
```

### 10. Obstacle-Aware Approach To Specified Bench

Uses `TurnDirectionToBench`, `AvoidObstacleToReachBench`, and `ArrivedAtBench`.

```bash
python dynamic_unit_test_runner.py \
  --test obstacle_aware_bench \
  --bench-number 2 \
  --trials 10 \
  --execute-actions
```

### 11. Obstacle-Aware Approach To Specified Stop Sign

Uses `TurnDirectionToStopSign`, `AvoidObstacleToReachStopSign`, and
`ArrivedAtAnimalsAroundStopSigns`.

```bash
python dynamic_unit_test_runner.py \
  --test obstacle_aware_stop \
  --stop-sign-number 1 \
  --trials 10 \
  --execute-actions
```

### 12. Obstacle-Aware Approach To Closest Bench

Uses `DirectionToClosestBench`, `BusHeadingDirection`,
`AvoidObstacleToReachClosestBench`, and `ArrivedAtBench`.

```bash
python dynamic_unit_test_runner.py \
  --test obstacle_aware_closest_bench \
  --trials 10 \
  --execute-actions
```

### 13. Obstacle-Aware Approach To Closest Stop Sign

Uses `DirectionToClosestStopSign`, `BusHeadingDirection`,
`AvoidObstacleToReachClosestStopSign`, and `ArrivedAtAnimalsAroundStopSigns`.

```bash
python dynamic_unit_test_runner.py \
  --test obstacle_aware_closest_stop \
  --trials 10 \
  --execute-actions
```

### 14. Visit Benches From Closest To Furthest

Uses `ClosestToFurthestBenches`, `TurnDirectionToBench`, and `ArrivedAtBench`.

```bash
python dynamic_unit_test_runner.py \
  --test ordered_visit_benches \
  --max-targets 3 \
  --trials 10 \
  --execute-actions
```

### 15. Visit Stop Signs From Closest To Furthest

Uses `ClosestToFurthestStopSigns`, `TurnDirectionToStopSign`, and
`ArrivedAtAnimalsAroundStopSigns`.

```bash
python dynamic_unit_test_runner.py \
  --test ordered_visit_stops \
  --max-targets 3 \
  --trials 10 \
  --execute-actions
```

### 16. Choose Closer Bench Then Visit It

Uses `PairwiseCloserBench`, `TurnDirectionToBench`, and `ArrivedAtBench`.

```bash
python dynamic_unit_test_runner.py \
  --test pairwise_bench_then_visit \
  --bench-i 1 \
  --bench-j 3 \
  --trials 10 \
  --execute-actions
```

### 17. Choose Closer Stop Sign Then Visit It

Uses `PairwiseCloserStopSign`, `TurnDirectionToStopSign`, and
`ArrivedAtAnimalsAroundStopSigns`.

```bash
python dynamic_unit_test_runner.py \
  --test pairwise_stop_then_visit \
  --stop-i 1 \
  --stop-j 2 \
  --trials 10 \
  --execute-actions
```

## Complex Navigation Scene

The full complex navigation task is implemented in `controlStack/zooBus.py`. It
is not a repeated unit test. It runs one longer Zoo-Bus story:

1. Count whether any people remain in the scene.
2. Visit the nearest occupied bench, using ordered bench IDs plus per-bench
   people counts instead of directly relying on `ClosestBenchWithPerson`.
3. Navigate to that bench with obstacle-aware actions.
4. After arrival, manually remove the people at that bench, then continue.
5. Once all people have been picked up, visit stop signs with animals.
6. Stop signs are selected from the remaining unvisited candidates and reached
   with obstacle-aware navigation.

Dry-run the complex task without sending Bluetooth commands:

```bash
python zooBus.py \
  --camera-id 0 \
  --max-pickups 8 \
  --max-stop-visits 6
```

Run the full real-robot complex task with the fine-tuned Qwen adapter:

```bash
python zooBus.py \
  --camera-id 0 \
  --max-pickups 8 \
  --max-stop-visits 6 \
  --command-pause-sec 2.0 \
  --execute-actions
```

During the pickup phase, the default behavior is to pause after the robot reaches
an occupied bench and wait for the operator to remove the people manually. Press
Enter after the people have been removed. If you want the script to continue
automatically after a fixed pause, use:

```bash
python zooBus.py \
  --camera-id 0 \
  --max-pickups 8 \
  --max-stop-visits 6 \
  --pickup-pause-sec 5.0 \
  --auto-continue \
  --execute-actions
```

Run the same complex task with the base Qwen model before fine-tuning:

```bash
python zooBus.py \
  --camera-id 0 \
  --max-pickups 8 \
  --max-stop-visits 6 \
  --no-adapter \
  --run-label base_qwen \
  --execute-actions
```

Run with a specific fine-tuned adapter checkpoint:

```bash
python zooBus.py \
  --camera-id 0 \
  --adapter-path /path/to/best_checkpoint \
  --run-label finetuned_qwen \
  --max-pickups 8 \
  --max-stop-visits 6 \
  --execute-actions
```

Useful complex-task flags:

```bash
--camera-id 0
--output-dir real_robot_results/complex_navigation
--execute-actions
--no-adapter
--adapter-path /path/to/best_checkpoint
--run-label finetuned_qwen
--max-pickups 8
--max-stop-visits 6
--max-steps 20
--max-turn-steps 18
--step-cm 5
--turn-deg 10
--command-pause-sec 2.0
--pickup-pause-sec 0.0
--stop-visit-pause-sec 2.0
--parse-retries 1
--auto-continue
--no-display
--no-manual-label
```

Complex-task logs are written under:

```text
real_robot_results/complex_navigation/
```

The key records are again `results.csv`, `results.json`, and the saved query
images. The final row for `zoo_bus_pickup_and_tour` records whether the whole
pickup-and-tour story completed successfully.

## Logging And Table Construction

Each run creates a timestamped folder under:

```text
real_robot_results/dynamic_unit_tests/
```

The important files are:

- `results.csv`: one row per model query, robot command, and final trial label;
- `results.json`: the same records in JSON format;
- `images/`: camera snapshots used for model queries.

For the paper table, use the final rows in `results.csv`:

- prefer `human_success` as the official trial outcome;
- use `model_loop_success` only as an internal diagnostic;
- report each test as `successes / trials (accuracy)`;
- report `Delta` as the absolute percentage-point change after fine-tuning.

Example table cell:

```text
7/10 (70.0%)
```

Example delta:

```text
+30.0
```
