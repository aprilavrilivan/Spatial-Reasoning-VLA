import argparse
import time
from datetime import datetime
from pathlib import Path

from FSM import SpatialVLMFSM
from answer_parsing import (
    parse_action,
    parse_compass,
    parse_egocentric,
    parse_id_list,
    parse_int,
    parse_turn,
    parse_yes_no,
)
from dynamic_test_specs import DYNAMIC_TEST_SPECS, spec_summary
from robot_actions import DRIVE_SPEED, TURN_SPEED, stop_command


COMPASS_DEGREES = {
    "North": 0,
    "Northeast": 45,
    "East": 90,
    "Southeast": 135,
    "South": 180,
    "Southwest": 225,
    "West": 270,
    "Northwest": 315,
}


class DynamicUnitTestRunner:
    def __init__(
        self,
        camera_id: int = 0,
        adapter_path: str | None = None,
        use_adapter: bool = True,
        run_label: str | None = None,
        output_dir: str | Path = "real_robot_results/dynamic_unit_tests",
        execute_actions: bool = False,
        show_camera: bool = True,
        max_steps: int = 30,
        max_turn_steps: int = 24,
        step_cm: int = 5,
        turn_deg: int = 10,
        realign_every: int = 2,
        command_pause_sec: float = 2.0,
        max_targets: int = 3,
        manual_label: bool = True,
    ):
        from camera_stream import CameraStream, display
        from experiment_logger import ExperimentLogger
        from vlm_client import QwenVLMClient

        self.fsm = SpatialVLMFSM()
        self.camera = CameraStream(camera_id).start()
        self.vlm = QwenVLMClient(adapter_path=adapter_path, use_adapter=use_adapter)
        self.run_label = run_label or ("finetuned_qwen" if use_adapter else "base_qwen")
        self.display = display
        if execute_actions:
            from BluetoothBot import BluetoothBot

            self.robot = BluetoothBot()
        else:
            self.robot = None
        self.execute_actions = execute_actions
        self.show_camera = show_camera
        self.max_steps = max_steps
        self.max_turn_steps = max_turn_steps
        self.step_cm = step_cm
        self.turn_deg = turn_deg
        self.realign_every = realign_every
        self.command_pause_sec = command_pause_sec
        self.max_targets = max_targets
        self.manual_label = manual_label
        run_name = datetime.now().strftime("dynamic_%Y%m%d_%H%M%S")
        self.logger = ExperimentLogger(Path(output_dir) / run_name)

    def open_robot(self):
        if self.robot is not None:
            self.robot.open_connection()

    def close(self):
        self.camera.stop()
        if self.robot is not None:
            self.robot.close_connection()

    def forward_command(self) -> str:
        return f"f,{DRIVE_SPEED},{self.step_cm}"

    def left_command(self) -> str:
        return f"l,{TURN_SPEED},{self.turn_deg}"

    def right_command(self) -> str:
        return f"r,{TURN_SPEED},{self.turn_deg}"

    def obstacle_left_command(self) -> str:
        return f"l,{TURN_SPEED},{int(round(self.turn_deg * 1.5))}"

    def obstacle_right_command(self) -> str:
        return f"r,{TURN_SPEED},{int(round(self.turn_deg * 1.5))}"

    def send_commands(self, commands: list[str], test_name: str, trial: int, step: int, phase: str):
        sent = bool(self.execute_actions and commands)
        if self.robot is not None and sent:
            for command in commands:
                self.robot.send_message(command)
                if self.command_pause_sec > 0:
                    time.sleep(self.command_pause_sec)
        self.logger.log(
            run_label=self.run_label,
            test_name=test_name,
            trial=trial,
            step=step,
            phase=phase,
            row_type="command",
            commands=";".join(commands),
            sent=sent,
        )

    def ask(self, test_name: str, trial: int, step: int, phase: str, question_type: str, question: str, parser):
        frame = self.camera.snapshot(flush_frames=3)
        if self.show_camera:
            self.display(frame)
        image_path = self.logger.save_frame(frame, f"{test_name}_trial{trial:02d}_step{step:03d}_{phase}")
        result = self.vlm.ask(frame, question)
        parsed = parser(result["answer"]) if parser is not None else result["answer"]
        print(
            f"[{test_name} trial {trial} step {step} {phase}] "
            f"{question_type}: raw={result['answer']!r} parsed={parsed!r}"
        )
        self.logger.log(
            run_label=self.run_label,
            test_name=test_name,
            trial=trial,
            step=step,
            phase=phase,
            row_type="vlm",
            image_path=image_path,
            question_type=question_type,
            question=question,
            full_prompt=result["full_prompt"],
            raw_answer=result["raw_answer"],
            answer=result["answer"],
            parsed_answer=parsed,
            latency_sec=result["latency_sec"],
        )
        return parsed

    def question(self, question_type: str, **kwargs) -> str:
        return self.fsm.question_dict[question_type].format(**kwargs)

    def arrival_question_type(self, target_kind: str) -> str:
        return "ArrivedAtBench" if target_kind == "bench" else "ArrivedAtAnimalsAroundStopSigns"

    def turn_question_type(self, target_kind: str) -> str:
        return "TurnDirectionToBench" if target_kind == "bench" else "TurnDirectionToStopSign"

    def relative_question_type(self, target_kind: str) -> str:
        return "BenchRelativeToHeading" if target_kind == "bench" else "StopSignRelativeToHeading"

    def closest_question_type(self, target_kind: str) -> str:
        return "ClosestBench" if target_kind == "bench" else "ClosestStopSign"

    def closest_direction_question_type(self, target_kind: str) -> str:
        return "DirectionToClosestBench" if target_kind == "bench" else "DirectionToClosestStopSign"

    def geometric_question_type(self, target_kind: str) -> str:
        return "GeometricDirectionToBench" if target_kind == "bench" else "GeometricDirectionToStopSign"

    def target_kwargs(self, target_kind: str, target_id: int) -> dict:
        return {"bench_number": target_id} if target_kind == "bench" else {"stop_sign_number": target_id}

    def target_label(self, target_kind: str, target_id: int) -> str:
        return f"{target_kind} #{target_id}"

    def ask_arrived(self, test_name: str, trial: int, step: int, target_kind: str, target_id: int):
        qtype = self.arrival_question_type(target_kind)
        question = self.question(qtype, **self.target_kwargs(target_kind, target_id))
        return self.ask(test_name, trial, step, "arrival_check", qtype, question, parse_yes_no)

    def ask_turn(self, test_name: str, trial: int, step: int, target_kind: str, target_id: int):
        qtype = self.turn_question_type(target_kind)
        question = self.question(qtype, **self.target_kwargs(target_kind, target_id))
        return self.ask(test_name, trial, step, "turn_check", qtype, question, parse_turn)

    def action_for_turn(self, turn_answer: str | None) -> list[str]:
        if turn_answer == "turn left":
            return [self.left_command()]
        if turn_answer == "turn right":
            return [self.right_command()]
        return []

    def action_for_egocentric(self, relative_answer: str | None) -> list[str]:
        if relative_answer == "front":
            return [self.forward_command()]
        if relative_answer in {"front-left", "left", "back-left", "back"}:
            return [self.left_command()]
        if relative_answer in {"front-right", "right", "back-right"}:
            return [self.right_command()]
        return []

    def action_for_compass_alignment(self, heading: str | None, target_direction: str | None) -> list[str]:
        if heading not in COMPASS_DEGREES or target_direction not in COMPASS_DEGREES:
            return []
        delta = (COMPASS_DEGREES[target_direction] - COMPASS_DEGREES[heading]) % 360
        if delta == 0:
            return [self.forward_command()]
        if delta <= 180:
            return [self.right_command()]
        return [self.left_command()]

    def face_target(self, test_name: str, trial: int, step: int, target_kind: str, target_id: int) -> tuple[bool, int]:
        for _ in range(self.max_turn_steps):
            turn = self.ask_turn(test_name, trial, step, target_kind, target_id)
            if turn == "already facing":
                return True, step + 1
            commands = self.action_for_turn(turn)
            if not commands:
                return False, step + 1
            self.send_commands(commands, test_name, trial, step, "turn_action")
            step += 1
        return False, step

    def approach_until_arrived(self, test_name: str, trial: int, step: int, target_kind: str, target_id: int) -> tuple[bool, int]:
        forward_steps = 0
        for _ in range(self.max_steps):
            arrived = self.ask_arrived(test_name, trial, step, target_kind, target_id)
            if arrived == "yes":
                self.send_commands([stop_command()], test_name, trial, step, "stop_on_arrival")
                return True, step + 1
            if arrived != "no":
                return False, step + 1
            if self.realign_every > 0 and forward_steps > 0 and forward_steps % self.realign_every == 0:
                faced, step = self.face_target(test_name, trial, step, target_kind, target_id)
                if not faced:
                    return False, step
            self.send_commands([self.forward_command()], test_name, trial, step, "forward_until_arrival")
            forward_steps += 1
            step += 1
        return False, step

    def face_and_arrive(
        self,
        test_name: str,
        trial: int,
        target_kind: str,
        target_id: int,
        log_final: bool = True,
    ) -> bool:
        step = 1
        print(f"Target: {self.target_label(target_kind, target_id)}")
        faced, step = self.face_target(test_name, trial, step, target_kind, target_id)
        if not faced:
            if log_final:
                self.log_final(test_name, trial, False, "failed_to_face_target")
            return False
        arrived, step = self.approach_until_arrived(test_name, trial, step, target_kind, target_id)
        if log_final:
            self.log_final(test_name, trial, arrived, "arrived" if arrived else "failed_to_arrive")
        return arrived

    def run_face_arrive(self, test_name: str, trial: int, spec: dict, args) -> bool:
        target_id = self.required_target_id(spec["target_kind"], args)
        return self.face_and_arrive(test_name, trial, spec["target_kind"], target_id)

    def run_egocentric(self, test_name: str, trial: int, spec: dict, args) -> bool:
        target_kind = spec["target_kind"]
        target_id = self.required_target_id(target_kind, args)
        step = 1
        for _ in range(self.max_steps):
            arrived = self.ask_arrived(test_name, trial, step, target_kind, target_id)
            if arrived == "yes":
                self.send_commands([stop_command()], test_name, trial, step, "stop_on_arrival")
                self.log_final(test_name, trial, True, "arrived")
                return True
            heading_question = self.fsm.question_dict["BusHeadingDirection"]
            self.ask(test_name, trial, step, "heading_probe", "BusHeadingDirection", heading_question, parse_compass)
            qtype = self.relative_question_type(target_kind)
            question = self.question(qtype, **self.target_kwargs(target_kind, target_id))
            relative = self.ask(test_name, trial, step, "relative_heading_control", qtype, question, parse_egocentric)
            commands = self.action_for_egocentric(relative)
            if not commands:
                self.log_final(test_name, trial, False, "parse_failure_or_no_action")
                return False
            self.send_commands(commands, test_name, trial, step, "relative_heading_action")
            step += 1
        self.log_final(test_name, trial, False, "timeout")
        return False

    def run_closest_navigation(self, test_name: str, trial: int, spec: dict, args) -> bool:
        target_kind = spec["target_kind"]
        step = 1
        direction_qtype = self.closest_direction_question_type(target_kind)
        self.ask(test_name, trial, step, "closest_direction_probe", direction_qtype, self.fsm.question_dict[direction_qtype], parse_compass)
        qtype = self.closest_question_type(target_kind)
        target_id = self.ask(test_name, trial, step, "closest_target_selection", qtype, self.fsm.question_dict[qtype], parse_int)
        if not target_id or target_id <= 0:
            self.log_final(test_name, trial, False, "no_valid_target")
            return False
        return self.face_and_arrive(test_name, trial, target_kind, target_id)

    def run_conditioned_closest_occupied_bench(self, test_name: str, trial: int, spec: dict, args) -> bool:
        step = 1
        self.ask(test_name, trial, step, "people_count_probe", "CountPeople", self.fsm.question_dict["CountPeople"], parse_int)
        target_id = self.ask(
            test_name,
            trial,
            step,
            "closest_occupied_bench_selection",
            "ClosestBenchWithPerson",
            self.fsm.question_dict["ClosestBenchWithPerson"],
            parse_int,
        )
        if not target_id or target_id <= 0:
            self.log_final(test_name, trial, False, "no_occupied_bench")
            return False
        self.ask(
            test_name,
            trial,
            step,
            "closest_bench_person_count_probe",
            "CountPersonAtClosestBench",
            self.fsm.question_dict["CountPersonAtClosestBench"],
            parse_int,
        )
        return self.face_and_arrive(test_name, trial, "bench", target_id)

    def run_conditioned_list(self, test_name: str, trial: int, spec: dict, args) -> bool:
        target_kind = spec["target_kind"]
        step = 1
        if target_kind == "bench":
            list_qtype = "ListBenchesWithAtLeastKPeople"
            count_qtype = "CountPeopleAtBench"
            list_question = self.question(list_qtype, k=args.k)
        else:
            self.ask(test_name, trial, step, "animal_count_probe", "CountAnimals", self.fsm.question_dict["CountAnimals"], parse_int)
            list_qtype = "ListStopSignsWithAtLeastKAnimals"
            count_qtype = "CountAnimalsAtStopSign"
            list_question = self.question(list_qtype, k=args.k)

        target_ids = self.ask(test_name, trial, step, "conditioned_target_list", list_qtype, list_question, parse_id_list)
        target_ids = [target_id for target_id in target_ids if target_id > 0]
        if not target_ids:
            self.log_final(test_name, trial, False, "empty_target_list")
            return False

        target_id = target_ids[0]
        count_question = self.question(count_qtype, **self.target_kwargs(target_kind, target_id))
        self.ask(test_name, trial, step, "target_count_verification", count_qtype, count_question, parse_int)
        return self.face_and_arrive(test_name, trial, target_kind, target_id)

    def run_obstacle_aware(self, test_name: str, trial: int, spec: dict, args) -> bool:
        target_kind = spec["target_kind"]
        target_id = self.required_target_id(target_kind, args)
        step = 1
        for _ in range(self.max_steps):
            arrived = self.ask_arrived(test_name, trial, step, target_kind, target_id)
            if arrived == "yes":
                self.send_commands([stop_command()], test_name, trial, step, "stop_on_arrival")
                self.log_final(test_name, trial, True, "arrived")
                return True
            turn = self.ask_turn(test_name, trial, step, target_kind, target_id)
            if turn != "already facing":
                commands = self.action_for_turn(turn)
                if not commands:
                    self.log_final(test_name, trial, False, "turn_parse_failure")
                    return False
                self.send_commands(commands, test_name, trial, step, "turn_to_target")
            else:
                qtype = "AvoidObstacleToReachBench" if target_kind == "bench" else "AvoidObstacleToReachStopSign"
                question = self.question(qtype, **self.target_kwargs(target_kind, target_id))
                action = self.ask(test_name, trial, step, "obstacle_action", qtype, question, parse_action)
                commands = self.commands_for_action(action)
                if not commands:
                    self.log_final(test_name, trial, False, "obstacle_action_parse_failure")
                    return False
                self.send_commands(commands, test_name, trial, step, "obstacle_action")
            step += 1
        self.log_final(test_name, trial, False, "timeout")
        return False

    def run_obstacle_aware_closest(self, test_name: str, trial: int, spec: dict, args) -> bool:
        target_kind = spec["target_kind"]
        step = 1
        for _ in range(self.max_steps):
            closest_qtype = self.closest_question_type(target_kind)
            target_id = self.ask(test_name, trial, step, "closest_target_probe", closest_qtype, self.fsm.question_dict[closest_qtype], parse_int)
            if not target_id or target_id <= 0:
                self.log_final(test_name, trial, False, "no_valid_closest_target")
                return False
            arrived = self.ask_arrived(test_name, trial, step, target_kind, target_id)
            if arrived == "yes":
                self.send_commands([stop_command()], test_name, trial, step, "stop_on_arrival")
                self.log_final(test_name, trial, True, "arrived")
                return True

            heading = self.ask(test_name, trial, step, "heading_probe", "BusHeadingDirection", self.fsm.question_dict["BusHeadingDirection"], parse_compass)
            direction_qtype = self.closest_direction_question_type(target_kind)
            direction = self.ask(test_name, trial, step, "closest_direction_probe", direction_qtype, self.fsm.question_dict[direction_qtype], parse_compass)
            align_commands = self.action_for_compass_alignment(heading, direction)
            if align_commands and align_commands != [self.forward_command()]:
                self.send_commands(align_commands, test_name, trial, step, "align_to_closest")
            else:
                qtype = "AvoidObstacleToReachClosestBench" if target_kind == "bench" else "AvoidObstacleToReachClosestStopSign"
                action = self.ask(test_name, trial, step, "closest_obstacle_action", qtype, self.fsm.question_dict[qtype], parse_action)
                commands = self.commands_for_action(action)
                if not commands:
                    self.log_final(test_name, trial, False, "obstacle_action_parse_failure")
                    return False
                self.send_commands(commands, test_name, trial, step, "closest_obstacle_action")
            step += 1
        self.log_final(test_name, trial, False, "timeout")
        return False

    def run_ordered_visit(self, test_name: str, trial: int, spec: dict, args) -> bool:
        target_kind = spec["target_kind"]
        qtype = "ClosestToFurthestBenches" if target_kind == "bench" else "ClosestToFurthestStopSigns"
        target_ids = self.ask(test_name, trial, 1, "ordered_target_selection", qtype, self.fsm.question_dict[qtype], parse_id_list)
        target_ids = [target_id for target_id in target_ids if target_id > 0][: self.max_targets]
        if not target_ids:
            self.log_final(test_name, trial, False, "empty_ordered_list")
            return False
        all_arrived = True
        for index, target_id in enumerate(target_ids, start=1):
            print(f"Ordered target {index}/{len(target_ids)}: {self.target_label(target_kind, target_id)}")
            arrived = self.face_and_arrive(test_name, trial, target_kind, target_id, log_final=False)
            all_arrived = all_arrived and arrived
            if not arrived:
                break
        self.log_final(test_name, trial, all_arrived, "ordered_visit_complete" if all_arrived else "ordered_visit_failed")
        return all_arrived

    def run_pairwise_then_visit(self, test_name: str, trial: int, spec: dict, args) -> bool:
        target_kind = spec["target_kind"]
        step = 1
        if target_kind == "bench":
            i, j = args.bench_i, args.bench_j
            qtype = "PairwiseCloserBench"
            question = self.question(qtype, bench_i=i, bench_j=j)
        else:
            i, j = args.stop_i, args.stop_j
            qtype = "PairwiseCloserStopSign"
            question = self.question(qtype, stop_i=i, stop_j=j)
        if i is None or j is None:
            raise ValueError(f"{test_name} requires --bench-i/--bench-j or --stop-i/--stop-j.")
        target_id = self.ask(test_name, trial, step, "pairwise_target_choice", qtype, question, parse_int)
        if target_id not in {i, j}:
            self.log_final(test_name, trial, False, "invalid_pairwise_choice")
            return False
        return self.face_and_arrive(test_name, trial, target_kind, target_id)

    def commands_for_action(self, action: str | None) -> list[str]:
        if action == "keep straight":
            return [self.forward_command()]
        if action == "turn left":
            return [self.obstacle_left_command(), self.forward_command()]
        if action == "turn right":
            return [self.obstacle_right_command(), self.forward_command()]
        return []

    def required_target_id(self, target_kind: str, args) -> int:
        target_id = args.bench_number if target_kind == "bench" else args.stop_sign_number
        if target_id is None:
            flag = "--bench-number" if target_kind == "bench" else "--stop-sign-number"
            raise ValueError(f"This test requires {flag}.")
        return target_id

    def log_final(self, test_name: str, trial: int, success: bool, reason: str):
        human_success = ""
        failure_reason = reason
        notes = ""
        if self.manual_label:
            value = input("External success label from operator [y/n/skip]: ").strip().lower()
            if value in {"y", "yes", "1", "true"}:
                human_success = True
            elif value in {"n", "no", "0", "false"}:
                human_success = False
                failure_reason = input("Failure reason (optional): ").strip() or reason
            notes = input("Notes (optional): ").strip()
        self.logger.log(
            run_label=self.run_label,
            test_name=test_name,
            trial=trial,
            row_type="final",
            model_loop_success=success,
            human_success=human_success,
            failure_reason=failure_reason,
            notes=notes,
        )

    def run_trial(self, test_name: str, trial: int, args) -> bool:
        spec = DYNAMIC_TEST_SPECS[test_name]
        protocol = spec["protocol"]
        if protocol == "face_arrive":
            return self.run_face_arrive(test_name, trial, spec, args)
        if protocol == "egocentric":
            return self.run_egocentric(test_name, trial, spec, args)
        if protocol == "closest_navigation":
            return self.run_closest_navigation(test_name, trial, spec, args)
        if protocol == "conditioned_closest_occupied_bench":
            return self.run_conditioned_closest_occupied_bench(test_name, trial, spec, args)
        if protocol == "conditioned_list":
            return self.run_conditioned_list(test_name, trial, spec, args)
        if protocol == "obstacle_aware":
            return self.run_obstacle_aware(test_name, trial, spec, args)
        if protocol == "obstacle_aware_closest":
            return self.run_obstacle_aware_closest(test_name, trial, spec, args)
        if protocol == "ordered_visit":
            return self.run_ordered_visit(test_name, trial, spec, args)
        if protocol == "pairwise_then_visit":
            return self.run_pairwise_then_visit(test_name, trial, spec, args)
        raise ValueError(f"Unknown dynamic test protocol: {protocol}")

    def run(self, test_name: str, trials: int, args):
        if test_name not in DYNAMIC_TEST_SPECS:
            raise ValueError(f"Unknown dynamic test {test_name!r}. Options: {sorted(DYNAMIC_TEST_SPECS)}")
        print(DYNAMIC_TEST_SPECS[test_name]["description"])
        print("Questions:", ", ".join(DYNAMIC_TEST_SPECS[test_name]["questions"]))
        self.open_robot()
        try:
            for trial in range(1, trials + 1):
                print(f"\n=== {test_name} trial {trial}/{trials} ===")
                self.run_trial(test_name, trial, args)
        finally:
            self.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run closed-loop dynamic unit tests for real-robot Spatial-VLA deployment."
    )
    parser.add_argument("--list-tests", action="store_true", help="Print dynamic test protocols and exit.")
    parser.add_argument("--test", choices=sorted(DYNAMIC_TEST_SPECS))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--no-adapter", action="store_true", help="Run the base Qwen model without the fine-tuned LoRA adapter.")
    parser.add_argument("--run-label", default=None, help="Label written into result rows, e.g. base_qwen or finetuned_qwen.")
    parser.add_argument("--output-dir", default="real_robot_results/dynamic_unit_tests")
    parser.add_argument("--execute-actions", action="store_true", help="Actually send Bluetooth commands to the robot.")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-manual-label", action="store_true", help="Do not prompt the operator for final success labels.")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-turn-steps", type=int, default=15)
    parser.add_argument("--step-cm", type=int, default=5)
    parser.add_argument("--turn-deg", type=int, default=10)
    parser.add_argument(
        "--realign-every",
        type=int,
        default=2,
        help="During approach, re-run target-facing correction after this many forward steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--command-pause-sec",
        type=float,
        default=2.0,
        help="Seconds to wait after each Bluetooth command so the robot can complete the primitive.",
    )
    parser.add_argument("--max-targets", type=int, default=3)
    parser.add_argument("--bench-number", type=int, default=None)
    parser.add_argument("--stop-sign-number", type=int, default=None)
    parser.add_argument("--bench-i", type=int, default=None)
    parser.add_argument("--bench-j", type=int, default=None)
    parser.add_argument("--stop-i", type=int, default=None)
    parser.add_argument("--stop-j", type=int, default=None)
    parser.add_argument("--k", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list_tests:
        print(spec_summary())
        raise SystemExit(0)
    if args.test is None:
        raise SystemExit("--test is required unless --list-tests is used.")
    runner = DynamicUnitTestRunner(
        camera_id=args.camera_id,
        adapter_path=args.adapter_path,
        use_adapter=not args.no_adapter,
        run_label=args.run_label,
        output_dir=args.output_dir,
        execute_actions=args.execute_actions,
        show_camera=not args.no_display,
        max_steps=args.max_steps,
        max_turn_steps=args.max_turn_steps,
        step_cm=args.step_cm,
        turn_deg=args.turn_deg,
        realign_every=args.realign_every,
        command_pause_sec=args.command_pause_sec,
        max_targets=args.max_targets,
        manual_label=not args.no_manual_label,
    )
    runner.run(args.test, args.trials, args)
    runner.close()
    
