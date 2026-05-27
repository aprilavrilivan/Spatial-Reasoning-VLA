import argparse
import time
from datetime import datetime
from pathlib import Path

from FSM import SpatialVLMFSM
from answer_parsing import parse_action, parse_int, parse_turn, parse_yes_no
from camera_stream import CameraStream, display
from experiment_logger import ExperimentLogger
from robot_actions import action_to_commands, turn_answer_to_commands
from vlm_client import QwenVLMClient


class BenchNavigationFSMRunner:
    """Closed-loop task: reach the closest occupied bench."""

    def __init__(
        self,
        camera_id: int = 0,
        adapter_path: str | None = None,
        use_adapter: bool = True,
        run_label: str | None = None,
        output_dir: str | Path = "real_robot_results/navigation",
        execute_actions: bool = True,
        show_camera: bool = True,
        max_steps: int = 20,
        command_pause_sec: float = 2.0,
    ):
        self.fsm = SpatialVLMFSM()
        self.camera = CameraStream(camera_id).start()
        self.vlm = QwenVLMClient(adapter_path=adapter_path, use_adapter=use_adapter)
        self.run_label = run_label or ("finetuned_qwen" if use_adapter else "base_qwen")
        if execute_actions:
            from BluetoothBot import BluetoothBot

            self.robot = BluetoothBot()
        else:
            self.robot = None
        self.execute_actions = execute_actions
        self.show_camera = show_camera
        self.max_steps = max_steps
        self.command_pause_sec = command_pause_sec
        run_name = datetime.now().strftime("nav_%Y%m%d_%H%M%S")
        self.logger = ExperimentLogger(Path(output_dir) / run_name)
        self.target_bench: int | None = None

    def open_robot(self):
        if self.robot is not None:
            self.robot.open_connection()

    def close(self):
        self.camera.stop()
        if self.robot is not None:
            self.robot.close_connection()

    def ask(self, frame, question_type: str, question: str, parser, step: int):
        result = self.vlm.ask(frame, question)
        parsed = parser(result["answer"])
        print(f"[step {step}] {question_type}: raw={result['answer']!r} parsed={parsed!r}")
        return result, parsed

    def send_commands(self, commands: list[str]):
        if not self.execute_actions or self.robot is None:
            return
        for command in commands:
            self.robot.send_message(command)
            if self.command_pause_sec > 0:
                time.sleep(self.command_pause_sec)

    def run(self):
        self.open_robot()
        try:
            for step in range(1, self.max_steps + 1):
                frame = self.camera.snapshot(flush_frames=3)
                if self.show_camera:
                    display(frame)
                image_path = self.logger.save_frame(frame, f"step_{step:03d}")

                if self.target_bench is None:
                    question_type = "ClosestBenchWithPerson"
                    question = self.fsm.question_dict[question_type]
                    result, parsed = self.ask(frame, question_type, question, parse_int, step)
                    self.target_bench = parsed if parsed and parsed > 0 else None
                    self.logger.log(
                        run_label=self.run_label,
                        step=step,
                        state="select_target",
                        image_path=image_path,
                        question_type=question_type,
                        question=question,
                        full_prompt=result["full_prompt"],
                        answer=result["answer"],
                        parsed_answer=parsed,
                        commands="",
                        latency_sec=result["latency_sec"],
                    )
                    if self.target_bench is None:
                        print("No occupied bench target found; stopping.")
                        break
                    continue

                arrived_question = self.fsm.question_dict["ArrivedAtBench"].format(
                    bench_number=self.target_bench
                )
                arrived_result, arrived = self.ask(
                    frame,
                    "ArrivedAtBench",
                    arrived_question,
                    parse_yes_no,
                    step,
                )
                if arrived == "yes":
                    self.logger.log(
                        run_label=self.run_label,
                        step=step,
                        state="arrived",
                        image_path=image_path,
                        question_type="ArrivedAtBench",
                        question=arrived_question,
                        full_prompt=arrived_result["full_prompt"],
                        answer=arrived_result["answer"],
                        parsed_answer=arrived,
                        commands="",
                        latency_sec=arrived_result["latency_sec"],
                    )
                    print("Navigation succeeded: arrived at target bench.")
                    break

                turn_question = self.fsm.question_dict["TurnDirectionToBench"].format(
                    bench_number=self.target_bench
                )
                turn_result, turn = self.ask(
                    frame,
                    "TurnDirectionToBench",
                    turn_question,
                    parse_turn,
                    step,
                )
                turn_commands = turn_answer_to_commands(str(turn))
                if turn_commands:
                    self.send_commands(turn_commands)
                    commands = turn_commands
                    state = "turn_to_target"
                else:
                    avoid_question = self.fsm.question_dict["AvoidObstacleToReachBench"].format(
                        bench_number=self.target_bench
                    )
                    avoid_result, action = self.ask(
                        frame,
                        "AvoidObstacleToReachBench",
                        avoid_question,
                        parse_action,
                        step,
                    )
                    commands = action_to_commands(str(action))
                    self.send_commands(commands)
                    state = "move_or_avoid"
                    result = avoid_result
                    parsed = action
                    self.logger.log(
                        run_label=self.run_label,
                        step=step,
                        state=state,
                        image_path=image_path,
                        question_type="AvoidObstacleToReachBench",
                        question=avoid_question,
                        full_prompt=result["full_prompt"],
                        answer=result["answer"],
                        parsed_answer=parsed,
                        commands=";".join(commands),
                        latency_sec=result["latency_sec"],
                    )
                    continue

                self.logger.log(
                    run_label=self.run_label,
                    step=step,
                    state=state,
                    image_path=image_path,
                    question_type="TurnDirectionToBench",
                    question=turn_question,
                    full_prompt=turn_result["full_prompt"],
                    answer=turn_result["answer"],
                    parsed_answer=turn,
                    commands=";".join(commands),
                    latency_sec=turn_result["latency_sec"],
                )
        finally:
            self.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a real-robot closed-loop navigation FSM.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--no-adapter", action="store_true", help="Run the base Qwen model without the fine-tuned LoRA adapter.")
    parser.add_argument("--run-label", default=None, help="Label written into result rows, e.g. base_qwen or finetuned_qwen.")
    parser.add_argument("--output-dir", default="real_robot_results/navigation")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--command-pause-sec",
        type=float,
        default=2.0,
        help="Seconds to wait after each Bluetooth command so the robot can complete the primitive.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not send Bluetooth commands.")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = BenchNavigationFSMRunner(
        camera_id=args.camera_id,
        adapter_path=args.adapter_path,
        use_adapter=not args.no_adapter,
        run_label=args.run_label,
        output_dir=args.output_dir,
        execute_actions=not args.dry_run,
        show_camera=not args.no_display,
        max_steps=args.max_steps,
        command_pause_sec=args.command_pause_sec,
    )
    runner.run()
