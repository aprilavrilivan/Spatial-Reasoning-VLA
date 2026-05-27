import argparse
import time
from datetime import datetime
from pathlib import Path

from FSM import SpatialVLMFSM
from answer_parsing import parse_action, parse_compass, parse_egocentric, parse_int, parse_turn, parse_yes_no
from camera_stream import CameraStream, display
from experiment_logger import ExperimentLogger
from robot_actions import action_to_commands, turn_answer_to_commands
from vlm_client import QwenVLMClient


UNIT_TESTS = {
    "CountPeople": {"parser": parse_int, "question_kwargs": {}},
    "CountAnimals": {"parser": parse_int, "question_kwargs": {}},
    "ClosestBench": {"parser": parse_int, "question_kwargs": {}},
    "ClosestStopSign": {"parser": parse_int, "question_kwargs": {}},
    "ClosestBenchWithPerson": {"parser": parse_int, "question_kwargs": {}},
    "DirectionToClosestBench": {"parser": parse_compass, "question_kwargs": {}},
    "DirectionToClosestStopSign": {"parser": parse_compass, "question_kwargs": {}},
    "BusHeadingDirection": {"parser": parse_compass, "question_kwargs": {}},
    "TurnDirectionToBench": {
        "parser": parse_turn,
        "question_kwargs": {"bench_number": 1},
        "command_fn": turn_answer_to_commands,
    },
    "TurnDirectionToStopSign": {
        "parser": parse_turn,
        "question_kwargs": {"stop_sign_number": 1},
        "command_fn": turn_answer_to_commands,
    },
    "BenchRelativeToHeading": {
        "parser": parse_egocentric,
        "question_kwargs": {"bench_number": 1},
    },
    "StopSignRelativeToHeading": {
        "parser": parse_egocentric,
        "question_kwargs": {"stop_sign_number": 1},
    },
    "AvoidObstacleToReachClosestBench": {
        "parser": parse_action,
        "question_kwargs": {},
        "command_fn": action_to_commands,
    },
    "AvoidObstacleToReachClosestStopSign": {
        "parser": parse_action,
        "question_kwargs": {},
        "command_fn": action_to_commands,
    },
    "ArrivedAtBench": {
        "parser": parse_yes_no,
        "question_kwargs": {"bench_number": 1},
    },
    "ArrivedAtAnimalsAroundStopSigns": {
        "parser": parse_yes_no,
        "question_kwargs": {"stop_sign_number": 1},
    },
}


class UnitTestRunner:
    def __init__(
        self,
        camera_id: int = 0,
        adapter_path: str | None = None,
        use_adapter: bool = True,
        run_label: str | None = None,
        output_dir: str | Path = "real_robot_results/unit_tests",
        execute_actions: bool = False,
        show_camera: bool = True,
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
        self.command_pause_sec = command_pause_sec
        run_name = datetime.now().strftime("unit_%Y%m%d_%H%M%S")
        self.logger = ExperimentLogger(Path(output_dir) / run_name)

    def open_robot(self):
        if self.robot is not None:
            self.robot.open_connection()

    def close(self):
        self.camera.stop()
        if self.robot is not None:
            self.robot.close_connection()

    def question_for(self, test_name: str) -> str:
        spec = UNIT_TESTS[test_name]
        template = self.fsm.question_dict[test_name]
        return template.format(**spec.get("question_kwargs", {}))

    def run_trial(self, test_name: str, trial_idx: int):
        spec = UNIT_TESTS[test_name]
        question = self.question_for(test_name)
        frame = self.camera.snapshot(flush_frames=3)
        if self.show_camera:
            display(frame)
        image_path = self.logger.save_frame(frame, f"{test_name}_{trial_idx:03d}")

        result = self.vlm.ask(frame, question)
        parsed = spec["parser"](result["answer"])
        commands = []
        if self.execute_actions and parsed is not None and "command_fn" in spec:
            commands = spec["command_fn"](str(parsed))
            for command in commands:
                self.robot.send_message(command)
                if self.command_pause_sec > 0:
                    time.sleep(self.command_pause_sec)

        print(f"[{test_name} #{trial_idx}] raw={result['answer']!r} parsed={parsed!r} commands={commands}")
        expected = input("Expected answer / success note (optional, Enter to skip): ").strip()
        is_correct = ""
        if expected:
            correct_input = input("Correct? [y/n] ").strip().lower()
            is_correct = correct_input in {"y", "yes", "1", "true"}

        self.logger.log(
            run_label=self.run_label,
            test_name=test_name,
            trial=trial_idx,
            image_path=image_path,
            question=question,
            full_prompt=result["full_prompt"],
            raw_answer=result["raw_answer"],
            answer=result["answer"],
            parsed_answer=parsed,
            expected=expected,
            is_correct=is_correct,
            commands=";".join(commands),
            latency_sec=result["latency_sec"],
        )

    def run(self, test_name: str, trials: int):
        if test_name not in UNIT_TESTS:
            raise ValueError(f"Unknown unit test {test_name!r}. Options: {sorted(UNIT_TESTS)}")
        self.open_robot()
        try:
            for idx in range(1, trials + 1):
                self.run_trial(test_name, idx)
        finally:
            self.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-robot unit tests for Zoo-Bus-VQA question types.")
    parser.add_argument("--test", choices=sorted(UNIT_TESTS), required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--no-adapter", action="store_true", help="Run the base Qwen model without the fine-tuned LoRA adapter.")
    parser.add_argument("--run-label", default=None, help="Label written into result rows, e.g. base_qwen or finetuned_qwen.")
    parser.add_argument("--output-dir", default="real_robot_results/unit_tests")
    parser.add_argument("--execute-actions", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--command-pause-sec",
        type=float,
        default=0.0,
        help="Seconds to wait after each Bluetooth command so the robot can complete the primitive.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = UnitTestRunner(
        camera_id=args.camera_id,
        adapter_path=args.adapter_path,
        use_adapter=not args.no_adapter,
        run_label=args.run_label,
        output_dir=args.output_dir,
        execute_actions=args.execute_actions,
        show_camera=not args.no_display,
        command_pause_sec=args.command_pause_sec,
    )
    runner.run(args.test, args.trials)
