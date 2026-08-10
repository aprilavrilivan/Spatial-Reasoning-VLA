"""Registry for closed-loop dynamic unit tests.

These tests are navigation-skill protocols, not static VQA checks. Each test
uses one or more Zoo-Bus-VQA question types as closed-loop control signals.
"""


DYNAMIC_TEST_SPECS = {
    "face_arrive_bench": {
        "protocol": "face_arrive",
        "target_kind": "bench",
        "description": "Turn until facing a specified bench, then drive until ArrivedAtBench returns yes.",
        "questions": ["TurnDirectionToBench", "ArrivedAtBench"],
    },
    "face_arrive_stop": {
        "protocol": "face_arrive",
        "target_kind": "stop",
        "description": "Turn until facing a specified stop sign, then drive until arrival at its animal zone.",
        "questions": ["TurnDirectionToStopSign", "ArrivedAtAnimalsAroundStopSigns"],
    },
    "egocentric_bench": {
        "protocol": "egocentric",
        "target_kind": "bench",
        "description": "Use BenchRelativeToHeading and arrival checks as a local controller.",
        "questions": ["BusHeadingDirection", "BenchRelativeToHeading", "ArrivedAtBench"],
    },
    "egocentric_stop": {
        "protocol": "egocentric",
        "target_kind": "stop",
        "description": "Use StopSignRelativeToHeading and arrival checks as a local controller.",
        "questions": ["BusHeadingDirection", "StopSignRelativeToHeading", "ArrivedAtAnimalsAroundStopSigns"],
    },
    "closest_bench_navigation": {
        "protocol": "closest_navigation",
        "target_kind": "bench",
        "description": "Select the closest bench, log its direction, then navigate to it.",
        "questions": ["ClosestBench", "DirectionToClosestBench", "TurnDirectionToBench", "ArrivedAtBench"],
    },
    "closest_stop_navigation": {
        "protocol": "closest_navigation",
        "target_kind": "stop",
        "description": "Select the closest stop sign, log its direction, then navigate to its animal zone.",
        "questions": ["ClosestStopSign", "DirectionToClosestStopSign", "TurnDirectionToStopSign", "ArrivedAtAnimalsAroundStopSigns"],
    },
    "closest_occupied_bench_navigation": {
        "protocol": "conditioned_closest_occupied_bench",
        "target_kind": "bench",
        "description": "Find the closest bench with at least one person, verify count, then navigate to it.",
        "questions": ["CountPeople", "ClosestBenchWithPerson", "CountPersonAtClosestBench", "TurnDirectionToBench", "ArrivedAtBench"],
    },
    "bench_with_at_least_k_people": {
        "protocol": "conditioned_list",
        "target_kind": "bench",
        "description": "Find benches with at least k people, verify one target count, then navigate to it.",
        "questions": ["ListBenchesWithAtLeastKPeople", "CountPeopleAtBench", "TurnDirectionToBench", "ArrivedAtBench"],
    },
    "stop_with_at_least_k_animals": {
        "protocol": "conditioned_list",
        "target_kind": "stop",
        "description": "Find stop signs with at least k animals, verify one target count, then navigate to it.",
        "questions": ["CountAnimals", "ListStopSignsWithAtLeastKAnimals", "CountAnimalsAtStopSign", "TurnDirectionToStopSign", "ArrivedAtAnimalsAroundStopSigns"],
    },
    "obstacle_aware_bench": {
        "protocol": "obstacle_aware",
        "target_kind": "bench",
        "description": "Face a specified bench, use obstacle-avoidance answers to approach it, and stop on arrival.",
        "questions": ["TurnDirectionToBench", "AvoidObstacleToReachBench", "ArrivedAtBench"],
    },
    "obstacle_aware_stop": {
        "protocol": "obstacle_aware",
        "target_kind": "stop",
        "description": "Face a specified stop sign, use obstacle-avoidance answers to approach it, and stop on arrival.",
        "questions": ["TurnDirectionToStopSign", "AvoidObstacleToReachStopSign", "ArrivedAtAnimalsAroundStopSigns"],
    },
    "obstacle_aware_closest_bench": {
        "protocol": "obstacle_aware_closest",
        "target_kind": "bench",
        "description": "Use closest-bench direction and obstacle-avoidance answers to approach the closest bench.",
        "questions": ["DirectionToClosestBench", "BusHeadingDirection", "AvoidObstacleToReachClosestBench", "ArrivedAtBench"],
    },
    "obstacle_aware_closest_stop": {
        "protocol": "obstacle_aware_closest",
        "target_kind": "stop",
        "description": "Use closest-stop direction and obstacle-avoidance answers to approach the closest stop sign.",
        "questions": ["DirectionToClosestStopSign", "BusHeadingDirection", "AvoidObstacleToReachClosestStopSign", "ArrivedAtAnimalsAroundStopSigns"],
    },
    "ordered_visit_benches": {
        "protocol": "ordered_visit",
        "target_kind": "bench",
        "description": "Ask for benches from closest to furthest, then visit them in that order.",
        "questions": ["ClosestToFurthestBenches", "TurnDirectionToBench", "ArrivedAtBench"],
    },
    "ordered_visit_stops": {
        "protocol": "ordered_visit",
        "target_kind": "stop",
        "description": "Ask for stop signs from closest to furthest, then visit their animal zones in that order.",
        "questions": ["ClosestToFurthestStopSigns", "TurnDirectionToStopSign", "ArrivedAtAnimalsAroundStopSigns"],
    },
    "pairwise_bench_then_visit": {
        "protocol": "pairwise_then_visit",
        "target_kind": "bench",
        "description": "Choose the closer of two benches, then navigate to the chosen bench.",
        "questions": ["PairwiseCloserBench", "TurnDirectionToBench", "ArrivedAtBench"],
    },
    "pairwise_stop_then_visit": {
        "protocol": "pairwise_then_visit",
        "target_kind": "stop",
        "description": "Choose the closer of two stop signs, then navigate to the chosen stop animal zone.",
        "questions": ["PairwiseCloserStopSign", "TurnDirectionToStopSign", "ArrivedAtAnimalsAroundStopSigns"],
    },
}


def spec_summary() -> str:
    lines = []
    for name, spec in sorted(DYNAMIC_TEST_SPECS.items()):
        questions = ", ".join(spec["questions"])
        lines.append(f"{name}: {spec['description']} Questions: {questions}")
    return "\n".join(lines)
