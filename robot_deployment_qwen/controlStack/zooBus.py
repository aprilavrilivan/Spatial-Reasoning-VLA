"""Closed-loop Zoo-Bus complex navigation task.

The task has two phases:
1. Pick up all visible people by repeatedly visiting the nearest occupied bench.
2. Visit animal stop signs without revisiting stops that have already been served.

The runner intentionally decomposes difficult goals into reliable Zoo-Bus-VQA
primitives. For example, instead of directly asking ClosestBenchWithPerson, it
asks for benches from closest to furthest and then checks each bench's person
count. All physical navigation uses the obstacle-aware action questions.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

from answer_parsing import parse_action, parse_id_list, parse_int
from dynamic_unit_test_runner import DynamicUnitTestRunner
from robot_actions import stop_command


TEST_NAME = "zoo_bus_pickup_and_tour"


class ZooBusComplexRunner(DynamicUnitTestRunner):
    def __init__(
        self,
        *args,
        pickup_pause_sec: float = 0.0,
        stop_visit_pause_sec: float = 2.0,
        auto_continue: bool = False,
        max_pickups: int = 8,
        max_stop_visits: int = 6,
        parse_retries: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pickup_pause_sec = pickup_pause_sec
        self.stop_visit_pause_sec = stop_visit_pause_sec
        self.auto_continue = auto_continue
        self.max_pickups = max_pickups
        self.max_stop_visits = max_stop_visits
        self.parse_retries = parse_retries
        self.step = 1
        self.visited_stop_ids: set[int] = set()

    def log_state(self, phase: str, **fields):
        self.logger.log(
            run_label=self.run_label,
            test_name=TEST_NAME,
            trial=1,
            step=self.step,
            phase=phase,
            row_type="state",
            **fields,
        )

    def ask_retry(self, phase: str, question_type: str, question: str, parser, validator=None):
        last_value = None
        attempts = max(1, self.parse_retries + 1)
        for attempt in range(1, attempts + 1):
            value = self.ask(TEST_NAME, 1, self.step, phase, question_type, question, parser)
            last_value = value
            if validator is None or validator(value):
                return value
            self.log_state(
                f"{phase}_parse_retry",
                question_type=question_type,
                attempt=attempt,
                parsed_answer=value,
            )
            self.step += 1
        return last_value

    def ask_count_people(self) -> int | None:
        count = self.ask_retry(
            "people_remaining_check",
            "CountPeople",
            self.fsm.question_dict["CountPeople"],
            parse_int,
            lambda value: isinstance(value, int) and value >= 0,
        )
        self.step += 1
        return count

    def ask_ordered_benches(self) -> list[int]:
        ids = self.ask_retry(
            "bench_distance_order",
            "ClosestToFurthestBenches",
            self.fsm.question_dict["ClosestToFurthestBenches"],
            parse_id_list,
            lambda value: isinstance(value, list),
        )
        self.step += 1
        return self.clean_ids(ids)

    def ask_ordered_stops(self) -> list[int]:
        ids = self.ask_retry(
            "stop_distance_order",
            "ClosestToFurthestStopSigns",
            self.fsm.question_dict["ClosestToFurthestStopSigns"],
            parse_id_list,
            lambda value: isinstance(value, list),
        )
        self.step += 1
        return self.clean_ids(ids)

    def ask_bench_person_count(self, bench_id: int, phase: str = "bench_person_count") -> int | None:
        question = self.question("CountPeopleAtBench", bench_number=bench_id)
        count = self.ask_retry(
            f"{phase}_bench_{bench_id}",
            "CountPeopleAtBench",
            question,
            parse_int,
            lambda value: isinstance(value, int) and value >= 0,
        )
        self.step += 1
        return count

    def ask_stop_animal_count(self, stop_id: int, phase: str = "stop_animal_count") -> int | None:
        question = self.question("CountAnimalsAtStopSign", stop_sign_number=stop_id)
        count = self.ask_retry(
            f"{phase}_stop_{stop_id}",
            "CountAnimalsAtStopSign",
            question,
            parse_int,
            lambda value: isinstance(value, int) and value >= 0,
        )
        self.step += 1
        return count

    def clean_ids(self, ids: Iterable[int] | None) -> list[int]:
        clean: list[int] = []
        for value in ids or []:
            if isinstance(value, int) and value > 0 and value not in clean:
                clean.append(value)
        return clean

    def list_occupied_benches_fallback(self) -> list[int]:
        question = self.question("ListBenchesWithAtLeastKPeople", k=1)
        ids = self.ask_retry(
            "occupied_bench_list_fallback",
            "ListBenchesWithAtLeastKPeople",
            question,
            parse_id_list,
            lambda value: isinstance(value, list),
        )
        self.step += 1
        return self.clean_ids(ids)

    def list_animal_stops(self) -> list[int]:
        question = self.question("ListStopSignsWithAtLeastKAnimals", k=1)
        ids = self.ask_retry(
            "animal_stop_list",
            "ListStopSignsWithAtLeastKAnimals",
            question,
            parse_id_list,
            lambda value: isinstance(value, list),
        )
        self.step += 1
        ids = self.clean_ids(ids)
        if ids:
            return ids

        ordered = self.ask_ordered_stops()
        verified = []
        for stop_id in ordered[: self.max_stop_visits]:
            count = self.ask_stop_animal_count(stop_id, "animal_stop_fallback_count")
            if count is not None and count > 0:
                verified.append(stop_id)
        return verified

    def choose_nearest_with_pairwise_tournament(self, target_kind: str, candidate_ids: list[int]) -> int | None:
        candidate_ids = self.clean_ids(candidate_ids)
        if not candidate_ids:
            return None
        winner = candidate_ids[0]
        for challenger in candidate_ids[1:]:
            if target_kind == "bench":
                qtype = "PairwiseCloserBench"
                question = self.question(qtype, bench_i=winner, bench_j=challenger)
            else:
                qtype = "PairwiseCloserStopSign"
                question = self.question(qtype, stop_i=winner, stop_j=challenger)
            answer = self.ask_retry(
                f"pairwise_nearest_{target_kind}_{winner}_vs_{challenger}",
                qtype,
                question,
                parse_int,
                lambda value: value in {winner, challenger},
            )
            self.step += 1
            if answer in {winner, challenger}:
                winner = answer
        return winner

    def select_nearest_occupied_bench(self) -> int | None:
        ordered_benches = self.ask_ordered_benches()
        self.log_state("ordered_benches_for_pickup", ordered_benches=",".join(map(str, ordered_benches)))
        for bench_id in ordered_benches:
            count = self.ask_bench_person_count(bench_id, "scan_ordered_bench")
            if count is not None and count > 0:
                self.log_state("selected_pickup_bench", target_bench=bench_id, person_count=count)
                return bench_id

        fallback_ids = self.list_occupied_benches_fallback()
        if not fallback_ids:
            return None
        target_id = self.choose_nearest_with_pairwise_tournament("bench", fallback_ids)
        if target_id is not None:
            self.log_state(
                "selected_pickup_bench_from_fallback",
                target_bench=target_id,
                fallback_candidates=",".join(map(str, fallback_ids)),
            )
        return target_id

    def select_nearest_unvisited_stop(self, stop_ids: list[int]) -> int | None:
        remaining = [stop_id for stop_id in self.clean_ids(stop_ids) if stop_id not in self.visited_stop_ids]
        if not remaining:
            return None
        target_id = self.choose_nearest_with_pairwise_tournament("stop", remaining)
        if target_id is not None:
            self.log_state(
                "selected_unvisited_stop",
                target_stop=target_id,
                remaining_stops=",".join(map(str, remaining)),
                visited_stops=",".join(map(str, sorted(self.visited_stop_ids))),
            )
            return target_id

        ordered_stops = self.ask_ordered_stops()
        for stop_id in ordered_stops:
            if stop_id in remaining:
                self.log_state(
                    "selected_unvisited_stop_from_order_fallback",
                    target_stop=stop_id,
                    remaining_stops=",".join(map(str, remaining)),
                )
                return stop_id
        return None

    def navigate_obstacle_aware_target(self, target_kind: str, target_id: int, phase_prefix: str) -> bool:
        print(f"Obstacle-aware navigation target: {self.target_label(target_kind, target_id)}")
        for _ in range(self.max_steps):
            arrived = self.ask_arrived(TEST_NAME, 1, self.step, target_kind, target_id)
            self.step += 1
            if arrived == "yes":
                self.send_commands([stop_command()], TEST_NAME, 1, self.step, f"{phase_prefix}_stop_on_arrival")
                self.step += 1
                return True
            if arrived != "no":
                self.log_state(f"{phase_prefix}_arrival_parse_failure", target_kind=target_kind, target_id=target_id)
                return False

            turn = self.ask_turn(TEST_NAME, 1, self.step, target_kind, target_id)
            self.step += 1
            if turn != "already facing":
                commands = self.action_for_turn(turn)
                if not commands:
                    self.log_state(
                        f"{phase_prefix}_turn_parse_failure",
                        target_kind=target_kind,
                        target_id=target_id,
                        turn_answer=turn,
                    )
                    return False
                self.send_commands(commands, TEST_NAME, 1, self.step, f"{phase_prefix}_turn_to_target")
                self.step += 1
                continue

            qtype = "AvoidObstacleToReachBench" if target_kind == "bench" else "AvoidObstacleToReachStopSign"
            question = self.question(qtype, **self.target_kwargs(target_kind, target_id))
            action = self.ask_retry(
                f"{phase_prefix}_obstacle_action",
                qtype,
                question,
                parse_action,
                lambda value: value in {"keep straight", "turn left", "turn right"},
            )
            self.step += 1
            commands = self.commands_for_action(action)
            if not commands:
                self.log_state(
                    f"{phase_prefix}_obstacle_action_parse_failure",
                    target_kind=target_kind,
                    target_id=target_id,
                    action_answer=action,
                )
                return False
            self.send_commands(commands, TEST_NAME, 1, self.step, f"{phase_prefix}_obstacle_action")
            self.step += 1
        return False

    def wait_for_pickup(self, bench_id: int):
        self.send_commands([stop_command()], TEST_NAME, 1, self.step, "pause_for_manual_pickup")
        self.log_state("manual_pickup_required", bench_id=bench_id)
        print(f"\nPickup step: remove people from bench #{bench_id}.")
        if self.auto_continue:
            if self.pickup_pause_sec > 0:
                time.sleep(self.pickup_pause_sec)
        else:
            input("Press Enter after the people have been removed from the bench...")
        self.step += 1

    def wait_at_stop(self, stop_id: int):
        self.send_commands([stop_command()], TEST_NAME, 1, self.step, "pause_at_stop")
        self.log_state("stop_visited", stop_id=stop_id)
        print(f"\nVisited stop sign #{stop_id}.")
        if self.stop_visit_pause_sec > 0:
            time.sleep(self.stop_visit_pause_sec)
        self.step += 1

    def run_passenger_pickup_phase(self) -> bool:
        for pickup_index in range(1, self.max_pickups + 1):
            people_count = self.ask_count_people()
            if people_count is None:
                self.log_final(TEST_NAME, 1, False, "count_people_parse_failure")
                return False
            if people_count == 0:
                self.log_state("pickup_phase_complete", pickup_rounds=pickup_index - 1)
                return True

            target_bench = self.select_nearest_occupied_bench()
            if target_bench is None:
                self.log_final(TEST_NAME, 1, False, "no_occupied_bench_found")
                return False

            arrived = self.navigate_obstacle_aware_target("bench", target_bench, "pickup_bench")
            if not arrived:
                self.log_final(TEST_NAME, 1, False, "failed_to_reach_pickup_bench")
                return False
            self.wait_for_pickup(target_bench)

        self.log_final(TEST_NAME, 1, False, "max_pickups_exceeded")
        return False

    def run_animal_stop_tour_phase(self) -> bool:
        stop_ids = self.list_animal_stops()
        stop_ids = stop_ids[: self.max_stop_visits]
        self.log_state("animal_stop_targets_initialized", stop_targets=",".join(map(str, stop_ids)))
        if not stop_ids:
            self.log_final(TEST_NAME, 1, False, "no_animal_stops_found")
            return False

        while len(self.visited_stop_ids) < len(stop_ids):
            target_stop = self.select_nearest_unvisited_stop(stop_ids)
            if target_stop is None:
                self.log_final(TEST_NAME, 1, False, "no_unvisited_stop_found")
                return False

            arrived = self.navigate_obstacle_aware_target("stop", target_stop, "animal_stop")
            if not arrived:
                self.log_final(TEST_NAME, 1, False, "failed_to_reach_animal_stop")
                return False

            self.visited_stop_ids.add(target_stop)
            self.wait_at_stop(target_stop)

        self.log_state("animal_stop_tour_complete", visited_stops=",".join(map(str, sorted(self.visited_stop_ids))))
        return True

    def run_complex_task(self) -> bool:
        print("Zoo-Bus complex navigation task")
        print("Phase 1: pick up people from occupied benches.")
        print("Phase 2: visit animal stop signs without revisiting stops.")
        self.open_robot()
        try:
            if not self.run_passenger_pickup_phase():
                return False
            success = self.run_animal_stop_tour_phase()
            self.log_final(TEST_NAME, 1, success, "task_complete" if success else "task_failed")
            return success
        except KeyboardInterrupt:
            self.log_final(TEST_NAME, 1, False, "manual_interrupt")
            raise
        finally:
            self.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full Zoo-Bus pickup-and-tour complex navigation task.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--no-adapter", action="store_true", help="Run the base Qwen model without the fine-tuned LoRA adapter.")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--output-dir", default="real_robot_results/complex_navigation")
    parser.add_argument("--execute-actions", action="store_true", help="Actually send Bluetooth commands to the robot.")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-manual-label", action="store_true")
    parser.add_argument("--auto-continue", action="store_true", help="Do not block for manual pickup confirmation.")
    parser.add_argument("--pickup-pause-sec", type=float, default=0.0)
    parser.add_argument("--stop-visit-pause-sec", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-turn-steps", type=int, default=18)
    parser.add_argument("--step-cm", type=int, default=5)
    parser.add_argument("--turn-deg", type=int, default=10)
    parser.add_argument("--command-pause-sec", type=float, default=2.0)
    parser.add_argument("--max-pickups", type=int, default=8)
    parser.add_argument("--max-stop-visits", type=int, default=6)
    parser.add_argument("--parse-retries", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = ZooBusComplexRunner(
        camera_id=args.camera_id,
        adapter_path=args.adapter_path,
        use_adapter=not args.no_adapter,
        run_label=args.run_label,
        output_dir=Path(args.output_dir),
        execute_actions=args.execute_actions,
        show_camera=not args.no_display,
        max_steps=args.max_steps,
        max_turn_steps=args.max_turn_steps,
        step_cm=args.step_cm,
        turn_deg=args.turn_deg,
        realign_every=0,
        command_pause_sec=args.command_pause_sec,
        manual_label=not args.no_manual_label,
        pickup_pause_sec=args.pickup_pause_sec,
        stop_visit_pause_sec=args.stop_visit_pause_sec,
        auto_continue=args.auto_continue,
        max_pickups=args.max_pickups,
        max_stop_visits=args.max_stop_visits,
        parse_retries=args.parse_retries,
    )
    runner.run_complex_task()
