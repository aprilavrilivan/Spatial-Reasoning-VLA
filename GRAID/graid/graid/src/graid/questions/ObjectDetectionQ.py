import logging
import math
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any

import torch
from PIL import Image

from graid.interfaces.ObjectDetectionI import (
    ObjectDetectionResultI,
    ObjectDetectionUtils,
)

logger = logging.getLogger(__name__)


class Question(ABC):
    @abstractmethod
    def __init__(
        self, question: str, variables: list[str], predicates: list[Callable[..., bool]]
    ) -> None:
        self.question = question
        self.variables = variables
        self.predicates = predicates
        self.other_question: Optional[str] = None

    def is_applicable(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> bool:
        """
        Check if the question is applicable to the given image and detections.

        Args:
            image: The image to check.
            detections: A list of ObjectDetectionResultI objects corresponding to the image.

        Returns:
            bool: True if all predicates return True, False otherwise.
        """
        return all(predicate(image, detections) for predicate in self.predicates)

    @abstractmethod
    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        """
        Apply the question to the image and detections.

        @precondition: is_applicable(image, detections) == True
        Args:
            image: The image to apply the question to.
            detections: A list of ObjectDetectionResultI objects corresponding to the image.

        Returns:
            A list of question-answer pairs where each pair with the substituted appropriate
            classes and the answer to that question.

            For example:
            Image: A person is sitting on a chair.
            Question: How many <object_class> are there in this image?
            apply() -> [
                ("How many person(s) are there in this image?", "1"),
                ("How many chair(s) are there in this image?", "1"),
            ]
        """
        pass

    def __repr__(self):
        representation = f"Question: {self.question}"
        # Safely check if 'other_question' is defined and not None
        if getattr(self, "other_question", None) is not None:
            representation += f"\nOther Question: {self.other_question}"

        return representation

    # New optional hook for questions that can benefit from a shared cache of expensive
    # per-image objects (e.g. depth map, SAM predictor).  By default this simply calls
    # the original `apply` implementation so subclasses are not required to override it.
    # Sub-classes that need the cache should implement their own version.
    def apply_with_cache(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        cache: dict[str, Any],
    ) -> list[tuple[str, str]]:  # noqa: D401 – simple wrapper
        return self.apply(image, detections)


class ObjectDetectionPredicates:
    @staticmethod
    def at_least_one_single_detection(
        image: Image.Image, detections: list[ObjectDetectionResultI]
    ) -> bool:
        if len(detections) <= 1:
            return len(detections) == 1
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1
        return any(c == 1 for c in counts.values())

    @staticmethod
    def at_least_x_many_class_detections(
        image: Image.Image, detections: list[ObjectDetectionResultI], x: int
    ) -> bool:
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1
        return len(counts) >= x

    @staticmethod
    def at_least_x_detections(
        image: Image.Image, detections: list[ObjectDetectionResultI], x: int
    ) -> bool:
        return len(detections) >= x

    @staticmethod
    def exists_non_overlapping_detections(
        image: Image.Image, detections: list[ObjectDetectionResultI]
    ) -> bool:
        for i, d1 in enumerate(detections):
            for j in range(i + 1, len(detections)):
                d2 = detections[j]
                if str(d1.label) != str(d2.label):
                    iou = ObjectDetectionUtils.pairwise_iou(d1, d2)
                    if iou.max() == 0:
                        return True
        return False

    @staticmethod
    def has_clusters(
        image: Image.Image, detections: list[ObjectDetectionResultI], threshold=50
    ) -> bool:
        import numpy as np

        if len(detections) < 2:
            return False

        # Compute centers from detections
        centers = []
        for det in detections:
            bbox = det.as_xyxy()[0]  # Get first bbox
            center_x = float((bbox[0] + bbox[2]) / 2.0)
            center_y = float((bbox[1] + bbox[3]) / 2.0)
            centers.append([center_x, center_y])

        centers = np.array(centers)
        # Simple O(n^2) proximity check; no heavy scipy
        n = centers.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                if (dx * dx + dy * dy) ** 0.5 < threshold:
                    return True
        return False

### <-- CUSTOM QUESTIONS HERE --> ###

# some helper functions
from typing import List, Tuple

VISUAL_ID_X_CLUSTER_TOL = 40.0

def sort_indices_strict_left_to_right(
    centroids: List[Tuple[float, float]],
) -> List[int]:
    if not centroids:
        return []
    by_x = sorted(range(len(centroids)), key=lambda i: centroids[i][0])
    groups: List[List[int]] = []
    cur_group: List[int] = [by_x[0]]
    cur_mean_x = centroids[by_x[0]][0]
    for idx in by_x[1:]:
        x = centroids[idx][0]
        if abs(x - cur_mean_x) <= VISUAL_ID_X_CLUSTER_TOL:
            cur_group.append(idx)
            cur_mean_x = sum(centroids[i][0] for i in cur_group) / len(cur_group)
        else:
            groups.append(cur_group)
            cur_group = [idx]
            cur_mean_x = x
    groups.append(cur_group)

    order: List[int] = []
    for group in groups:
        order.extend(sorted(group, key=lambda i: (centroids[i][1], centroids[i][0])))
    return order

def collect_detections_by_label(detections, target_labels):
    label_to_bboxes = {label: [] for label in target_labels}
    for det in detections:
        lbl = det.label
        bboxes = det.as_xyxy()
        if isinstance(lbl, torch.Tensor):
            for i, l in enumerate(lbl):
                str_l = str(l)
                if str_l in target_labels:
                    label_to_bboxes[str_l].append(bboxes[i])
        else:
            str_lbl = str(lbl)
            if str_lbl in target_labels:
                label_to_bboxes[str_lbl].append(bboxes[0])
    return label_to_bboxes

def centroid(bbox):
    x1, y1, x2, y2 = map(float, bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def assign_objects_to_location(object_centroids, location_centroids):
    # assign each object to nearest location
    location_occupancy = [0] * len(location_centroids)
    for oc in object_centroids:
        dists = [np.linalg.norm(np.array(oc) - np.array(lc)) for lc in location_centroids]
        nearest_location = np.argmin(dists)
        location_occupancy[nearest_location] += 1

    return location_occupancy


# These constants match the scene generator used for the zoo_bus dataset.
SCENE_BASE_LONGEST_SIDE = 4032.0
SCENE_HEADING_DOT_OFFSET = 215.0
SCENE_HEADING_DOT_RADIUS = 45.0
SCENE_ARRIVAL_STABILITY_MARGIN = 35.0
SCENE_PATH_MARGIN_STABILITY = 65.0
SCENE_ASSIGNMENT_STABILITY_MARGIN = 80.0
SCENE_DISTANCE_COMPARE_STABILITY = 80.0
PATH_ENTRY_T_STABILITY_MARGIN = 0.04
PATH_HEAD_ON_ANGLE_MARGIN_RAD = math.pi / 18.0
EIGHT_DIRECTION_BOUNDARY_MARGIN_RAD = math.pi / 36.0
TURN_DIRECTION_STABILITY_MARGIN_RAD = math.pi / 18.0
RELATIVE_DIRECTION_BOUNDARY_MARGIN_RAD = math.pi / 18.0


def scene_scale(image: Image.Image) -> float:
    """Return the resize factor from the original generated scene to `image`."""
    longest_side = float(max(image.size))
    if longest_side <= 0.0:
        return 1.0
    return longest_side / SCENE_BASE_LONGEST_SIDE


def scaled_scene_distance(image: Image.Image, distance: float) -> float:
    """Scale a pre-resize scene distance into the current image coordinates."""
    return float(distance) * scene_scale(image)


def arrival_stability_margin(image: Image.Image) -> float:
    """Return a no-ask buffer around the arrived/not-arrived threshold."""
    return max(8.0, scaled_scene_distance(image, SCENE_ARRIVAL_STABILITY_MARGIN))


def path_margin_stability(image: Image.Image) -> float:
    """Return a no-ask buffer around widened-path collision checks."""
    return max(8.0, scaled_scene_distance(image, SCENE_PATH_MARGIN_STABILITY))


def assignment_stability_margin(image: Image.Image) -> float:
    """Return a no-ask buffer for nearest-location assignments."""
    return max(8.0, scaled_scene_distance(image, SCENE_ASSIGNMENT_STABILITY_MARGIN))


def distance_compare_stability_margin(image: Image.Image) -> float:
    """Return a no-ask buffer for nearest/farthest distance comparisons."""
    return max(8.0, scaled_scene_distance(image, SCENE_DISTANCE_COMPARE_STABILITY))


def rounded_sector_is_stable(
    angle: float,
    sector_width: float,
    boundary_margin: float,
) -> bool:
    """Return False when `angle` lies too close to a round-to-nearest sector boundary."""
    remainder = (angle + sector_width / 2.0) % sector_width
    boundary_dist = min(remainder, sector_width - remainder)
    return boundary_dist > boundary_margin


def eight_direction_is_stable(dx: float, dy: float) -> bool:
    """Return False when the direction lies too close to an 8-way sector boundary."""
    angle = math.atan2(dy, dx)
    return rounded_sector_is_stable(
        angle, math.pi / 4, EIGHT_DIRECTION_BOUNDARY_MARGIN_RAD
    )


def stable_nearest_location_index(
    object_centroid: Tuple[float, float],
    location_centroids: List[Tuple[float, float]],
    margin: float,
) -> Optional[int]:
    """Return nearest location index, or None if the nearest-vs-second gap is unstable."""
    if not location_centroids:
        return None
    dists = [
        float(np.linalg.norm(np.array(object_centroid) - np.array(location_centroids[i])))
        for i in range(len(location_centroids))
    ]
    order = sorted(range(len(location_centroids)), key=lambda i: dists[i])
    if len(order) >= 2 and (dists[order[1]] - dists[order[0]]) <= margin:
        return None
    return int(order[0])


def stable_assign_objects_to_location(
    object_centroids: List[Tuple[float, float]],
    location_centroids: List[Tuple[float, float]],
    margin: float,
) -> Optional[Tuple[List[int], List[int]]]:
    """Assign objects to the nearest location, or return None if any assignment is unstable."""
    location_occupancy = [0] * len(location_centroids)
    assignments: List[int] = []
    for oc in object_centroids:
        nearest_location = stable_nearest_location_index(oc, location_centroids, margin)
        if nearest_location is None:
            return None
        location_occupancy[nearest_location] += 1
        assignments.append(nearest_location)
    return location_occupancy, assignments


def stable_distance_order(
    reference_point: Tuple[float, float],
    target_points: List[Tuple[float, float]],
    margin: float,
) -> Optional[Tuple[List[int], List[float]]]:
    """Return indices ordered by distance to `reference_point`, or None if any gap is unstable."""
    if not target_points:
        return None
    ref = np.array(reference_point, dtype=float)
    dists = [
        float(np.linalg.norm(np.array(tp, dtype=float) - ref))
        for tp in target_points
    ]
    order = sorted(range(len(target_points)), key=lambda i: dists[i])
    for i in range(len(order) - 1):
        if abs(dists[order[i + 1]] - dists[order[i]]) <= margin:
            return None
    return order, dists


def turn_direction_is_stable(delta: float) -> bool:
    """Return False when turn-vs-facing is too close to the decision boundary."""
    return abs(abs(delta) - (math.pi / 8)) > TURN_DIRECTION_STABILITY_MARGIN_RAD


def relative_direction_is_stable(delta: float) -> bool:
    """Return False when the target lies too close to a relative-direction sector boundary."""
    return rounded_sector_is_stable(
        delta, math.pi / 4, RELATIVE_DIRECTION_BOUNDARY_MARGIN_RAD
    )

'''
Group #1 Questions
'''

class CountPeople(Question):
    def __init__(self) -> None:
        super().__init__(
            question="How many people are currently visible in this scene? Respond with only an integer.",
            variables=[],
            predicates=[
                lambda image, detections: len(detections) > 0,
            ],
        )

    def apply(self, image, detections):
        count = 0
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    if str(l) == "person":
                        count += 1
            else:
                if str(lbl) == "person":
                    count += 1
        return [(self.question, str(count))]


class CountAnimals(Question):
    ANIMAL_LABELS = {"zebra", "elephant", "giraffe"}

    def __init__(self) -> None:
        super().__init__(
            question="How many animals are currently visible in this scene? Respond with only an integer.",
            variables=[],
            predicates=[
                lambda image, detections: len(detections) > 0,
            ],
        )

    def apply(self, image, detections):
        count = 0
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    if str(l) in self.ANIMAL_LABELS:
                        count += 1
            else:
                if str(lbl) in self.ANIMAL_LABELS:
                    count += 1
        return [(self.question, str(count))]


class CountPeopleAtBench(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "How many people are currently at bench #{bench_number}? Respond with only an integer."
            ),
            variables=["bench_number"],
            predicates=[
                lambda image, detections: len(detections) > 0,
            ],
        )

    def apply(self, image, detections):
        labeled_bboxes = collect_detections_by_label(detections, ["bench", "person"])
        benches = labeled_bboxes["bench"]
        persons = labeled_bboxes["person"]

        if len(benches) == 0:
            return []

        bench_centroids = [centroid(bbox) for bbox in benches]
        bench_order = sort_indices_strict_left_to_right(bench_centroids)

        if len(persons) == 0:
            return [
                (self.question.format(bench_number=rank + 1), "0")
                for rank in range(len(bench_order))
            ]

        person_centroids = [centroid(bbox) for bbox in persons]
        stable_assignment = stable_assign_objects_to_location(
            person_centroids,
            bench_centroids,
            assignment_stability_margin(image),
        )
        if stable_assignment is None:
            return []
        bench_occupancy, _ = stable_assignment

        qas = []
        for rank, bench_idx in enumerate(bench_order):
            question = self.question.format(bench_number=rank + 1)
            qas.append((question, str(bench_occupancy[bench_idx])))
        return qas


class CountAnimalsAtStopSign(Question):
    ANIMAL_LABELS = ["zebra", "elephant", "giraffe"]

    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "How many animals are currently around stop sign #{stop_sign_number}? Respond with only an integer."
            ),
            variables=["stop_sign_number"],
            predicates=[
                lambda image, detections: len(detections) > 0,
            ],
        )

    def apply(self, image, detections):
        labeled_bboxes = collect_detections_by_label(
            detections, ["stop sign"] + self.ANIMAL_LABELS
        )
        stop_signs = labeled_bboxes["stop sign"]
        animals = [
            bbox
            for label in self.ANIMAL_LABELS
            for bbox in labeled_bboxes[label]
        ]

        if len(stop_signs) == 0:
            return []

        stop_centroids = [centroid(bbox) for bbox in stop_signs]
        stop_order = sort_indices_strict_left_to_right(stop_centroids)

        if len(animals) == 0:
            return [
                (self.question.format(stop_sign_number=rank + 1), "0")
                for rank in range(len(stop_order))
            ]

        animal_centroids = [centroid(bbox) for bbox in animals]
        stable_assignment = stable_assign_objects_to_location(
            animal_centroids,
            stop_centroids,
            assignment_stability_margin(image),
        )
        if stable_assignment is None:
            return []
        stop_occupancy, _ = stable_assignment

        qas = []
        for rank, stop_idx in enumerate(stop_order):
            question = self.question.format(stop_sign_number=rank + 1)
            qas.append((question, str(stop_occupancy[stop_idx])))
        return qas


class ListBenchesWithAtLeastKPeople(Question):
    K_VALUES = [1, 2]

    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "List the IDs of all benches that have at least {k} people in ascending order, separated by commas. "
                "If none, respond with '0'."
            ),
            variables=["k"],
            predicates=[
                lambda image, detections: len(detections) > 0,
            ],
        )

    def apply(self, image, detections):
        labeled_bboxes = collect_detections_by_label(detections, ["bench", "person"])
        benches = labeled_bboxes["bench"]
        persons = labeled_bboxes["person"]

        if len(benches) == 0:
            return []

        bench_centroids = [centroid(bbox) for bbox in benches]
        bench_order = sort_indices_strict_left_to_right(bench_centroids)
        idx_to_number = {idx: num for num, idx in enumerate(bench_order, start=1)}

        if len(persons) == 0:
            bench_occupancy = [0] * len(benches)
        else:
            person_centroids = [centroid(bbox) for bbox in persons]
            stable_assignment = stable_assign_objects_to_location(
                person_centroids,
                bench_centroids,
                assignment_stability_margin(image),
            )
            if stable_assignment is None:
                return []
            bench_occupancy, _ = stable_assignment

        qas = []
        for k in self.K_VALUES:
            question = self.question.format(k=k)
            ids = sorted(
                idx_to_number[idx]
                for idx, occ in enumerate(bench_occupancy)
                if occ >= k
            )
            answer = ", ".join(str(i) for i in ids) if ids else "0"
            qas.append((question, answer))
        return qas


class ListStopSignsWithAtLeastKAnimals(Question):
    ANIMAL_LABELS = ["zebra", "elephant", "giraffe"]
    K_VALUES = [3, 4]

    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "List the IDs of all stop signs that have at least {k} animals around them in ascending order, separated by commas. "
                "If none, respond with '0'."
            ),
            variables=["k"],
            predicates=[
                lambda image, detections: len(detections) > 0,
            ],
        )

    def apply(self, image, detections):
        labeled_bboxes = collect_detections_by_label(
            detections, ["stop sign"] + self.ANIMAL_LABELS
        )
        stop_signs = labeled_bboxes["stop sign"]
        animals = [
            bbox
            for label in self.ANIMAL_LABELS
            for bbox in labeled_bboxes[label]
        ]

        if len(stop_signs) == 0:
            return []

        stop_centroids = [centroid(bbox) for bbox in stop_signs]
        stop_order = sort_indices_strict_left_to_right(stop_centroids)
        idx_to_number = {idx: num for num, idx in enumerate(stop_order, start=1)}

        if len(animals) == 0:
            stop_occupancy = [0] * len(stop_signs)
        else:
            animal_centroids = [centroid(bbox) for bbox in animals]
            stable_assignment = stable_assign_objects_to_location(
                animal_centroids,
                stop_centroids,
                assignment_stability_margin(image),
            )
            if stable_assignment is None:
                return []
            stop_occupancy, _ = stable_assignment

        qas = []
        for k in self.K_VALUES:
            question = self.question.format(k=k)
            ids = sorted(
                idx_to_number[idx]
                for idx, occ in enumerate(stop_occupancy)
                if occ >= k
            )
            answer = ", ".join(str(i) for i in ids) if ids else "0"
            qas.append((question, answer))
        return qas


'''
Group #2 Questions
'''

def bbox_edge_dist(b1, b2) -> float:
    """Axis-aligned bbox edge-to-edge distance (0.0 when touching or overlapping)."""
    x1_0, y1_0, x1_1, y1_1 = map(float, b1)
    x2_0, y2_0, x2_1, y2_1 = map(float, b2)
    dx = max(0.0, max(x2_0 - x1_1, x1_0 - x2_1))
    dy = max(0.0, max(y2_0 - y1_1, y1_0 - y2_1))
    return float(np.sqrt(dx * dx + dy * dy))


class ArrivedAtBench(Question):
    def __init__(self, dist_threshold: float) -> None:
        """Question: has the bus arrived at each bench (numbered left-to-right)?"""
        super().__init__(
            question=(
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "Is the clock close enough to be considered arrived at bench number {bench_number}? Respond with 'Yes' or 'No'."
            ),
            variables=["bench_number"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )
        self.dist_threshold: float = dist_threshold

    def apply(self, image, detections):
        """
        For each bench, numbered strictly left to right by x centroid,
        ask whether the bus (clock) has "arrived" at it.

        A bench is considered "arrived" if the bbox edge-to-edge distance
        between the bus and the bench is less than `self.dist_threshold`.
        """
        labeled_bboxes = collect_detections_by_label(detections, ["bench", "clock"])
        benches = labeled_bboxes.get("bench", [])
        clocks = labeled_bboxes.get("clock", [])

        if len(benches) == 0 or len(clocks) == 0:
            return []

        bus_bbox = clocks[0]
        bench_centroids = [centroid(bbox) for bbox in benches]
        bench_order = sort_indices_strict_left_to_right(bench_centroids)

        # Edge-to-edge distance between bus bbox and each bench bbox
        bench_distances = [bbox_edge_dist(bus_bbox, benches[i]) for i in range(len(benches))]
        effective_threshold = scaled_scene_distance(image, self.dist_threshold)
        stability_margin = arrival_stability_margin(image)

        qas: list[tuple[str, str]] = []
        for rank, bench_idx in enumerate(bench_order):
            if abs(bench_distances[bench_idx] - effective_threshold) <= stability_margin:
                continue
            question = self.question.format(bench_number=rank + 1)
            answer = "Yes" if bench_distances[bench_idx] < effective_threshold else "No"
            qas.append((question, answer))

        return qas


class ArrivedAtAnimalsAroundStopSigns(Question):
    def __init__(self, dist_threshold: float) -> None:
        """
        Does the bus (clock) arrive at the animals around each left-to-right
        numbered stop sign?
        """
        super().__init__(
            question=(
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "For each stop sign, consider all animals that are spatially closest to that stop sign. "
                "Is the clock close enough to be considered arrived at at least one of the animals around stop sign number {stop_sign_number}? Respond with 'Yes' or 'No'."
            ),
            variables=["stop_sign_number"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )
        # Distance threshold in pixels: distance from the clock to the nearest animal
        # in the group associated with a given stop sign.
        self.dist_threshold: float = dist_threshold

    def apply(self, image, detections):
        """
        1. Number all stop signs strictly left to right by x centroid.
        2. Assign each animal (elephant, giraffe, zebra) to its nearest stop sign,
           forming an animal group around each stop sign.
        3. For each stop sign, compute the minimum distance from the clock to
           any animal in its group. If this distance is below dist_threshold,
           answer 'Yes'; otherwise, answer 'No'.
        """
        # We only care about stop signs, the clock, and the three animal categories
        target_labels = ["stop sign", "clock", "elephant", "giraffe", "zebra"]
        labeled_bboxes = collect_detections_by_label(detections, target_labels)

        stop_signs = labeled_bboxes.get("stop sign", [])
        clocks = labeled_bboxes.get("clock", [])

        # Collect all animal bounding boxes (elephant, giraffe, zebra)
        animal_bboxes: List[Tuple[float, float, float, float]] = []
        for lbl in ("elephant", "giraffe", "zebra"):
            animal_bboxes.extend(labeled_bboxes.get(lbl, []))

        # If there are no stop signs, no clock, or no animals, do not generate questions
        if len(stop_signs) == 0 or len(clocks) == 0 or len(animal_bboxes) == 0:
            return []

        # Use the first clock as the bus position
        bus_bbox = clocks[0]

        # Compute centroids for all stop signs
        stop_sign_centroids = [centroid(bbox) for bbox in stop_signs]
        stop_sign_order = sort_indices_strict_left_to_right(
            stop_sign_centroids
        )

        # Compute centroids for all animals
        animal_centroids = [centroid(bbox) for bbox in animal_bboxes]

        # Assign each animal to its nearest stop sign
        stable_assignment = stable_assign_objects_to_location(
            animal_centroids,
            stop_sign_centroids,
            assignment_stability_margin(image),
        )
        if stable_assignment is None:
            return []
        _, animal_assignment = stable_assignment
        animals_per_stop_sign: List[List[int]] = [[] for _ in range(len(stop_signs))]
        for a_idx, nearest_stop in enumerate(animal_assignment):
            animals_per_stop_sign[nearest_stop].append(a_idx)

        # Precompute the minimum bbox edge-to-edge distance from the clock to
        # the animals around each stop sign. +inf if stop sign has no animals.
        min_dist_to_animals = [float("inf")] * len(stop_signs)
        for s_idx, animal_indices in enumerate(animals_per_stop_sign):
            if not animal_indices:
                continue
            dists = [
                bbox_edge_dist(bus_bbox, animal_bboxes[a_idx])
                for a_idx in animal_indices
            ]
            min_dist_to_animals[s_idx] = min(dists)
        effective_threshold = scaled_scene_distance(image, self.dist_threshold)
        stability_margin = arrival_stability_margin(image)

        # Generate QA pairs in the left-to-right order of stop signs
        qas: List[Tuple[str, str]] = []
        for rank, stop_sign_idx in enumerate(stop_sign_order):
            d = min_dist_to_animals[stop_sign_idx]
            if abs(d - effective_threshold) <= stability_margin:
                continue
            question = self.question.format(stop_sign_number=rank + 1)
            answer = "Yes" if d < effective_threshold else "No"
            qas.append((question, answer))

        return qas


class ClosestBench(Question):
    def __init__(self) -> None:
        """
        Ask: after numbering all benches strictly left to right by x centroid,
        which bench is closest to the bus (clock)? Answer is a single integer.
        """
        super().__init__(
            question=(
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "Which bench is closest to the clock? "
                "Answer with the bench ID."
            ),
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        # Only benches and the bus (clock) are needed
        labeled_bboxes = collect_detections_by_label(
            detections, ["bench", "clock"]
        )

        benches = labeled_bboxes.get("bench", [])
        clocks = labeled_bboxes.get("clock", [])

        # If no benches or no bus, do not generate this question
        if len(benches) == 0 or len(clocks) == 0:
            return []

        # Use the first clock as the bus position
        bus_bbox = clocks[0]
        bus_centroid = centroid(bus_bbox)

        # Geometry for benches
        bench_centroids = [centroid(bbox) for bbox in benches]
        # 1) Number benches strictly left to right by x centroid
        bench_order = sort_indices_strict_left_to_right(
            bench_centroids
        )

        # Map detection index -> bench number (1, 2, 3, ...)
        idx_to_number = {
            idx: num for num, idx in enumerate(bench_order, start=1)
        }

        # 2) Find the bench closest to the bus (clock)
        stable_order = stable_distance_order(
            bus_centroid,
            bench_centroids,
            distance_compare_stability_margin(image),
        )
        if stable_order is None:
            return []
        idx_by_distance, _ = stable_order
        closest_idx = idx_by_distance[0]

        # 3) Convert its detection index to the left-to-right bench number
        answer_number = idx_to_number[closest_idx]

        return [(self.question, str(answer_number))]


class ClosestStopSign(Question):
    def __init__(self) -> None:
        """
        Ask: after numbering all stop signs strictly left to right by x centroid,
        which stop sign is closest to the clock? Answer is a single integer.
        """
        super().__init__(
            question=(
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "Which stop sign is closest to the clock? "
                "Answer with its ID."
            ),
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        # Only stop signs and the bus (clock) are needed
        labeled_bboxes = collect_detections_by_label(
            detections, ["stop sign", "clock"]
        )

        stop_signs = labeled_bboxes.get("stop sign", [])
        clocks = labeled_bboxes.get("clock", [])

        # If no stop signs or no bus, do not generate this question
        if len(stop_signs) == 0 or len(clocks) == 0:
            return []

        # Use the first clock as the bus position
        bus_bbox = clocks[0]
        bus_centroid = centroid(bus_bbox)

        # Geometry for stop signs
        stop_sign_centroids = [centroid(bbox) for bbox in stop_signs]
        # 1) Number stop signs strictly left to right by x centroid
        stop_sign_order = sort_indices_strict_left_to_right(
            stop_sign_centroids
        )

        # Map detection index -> stop sign number (1, 2, 3, ...)
        idx_to_number = {
            idx: num for num, idx in enumerate(stop_sign_order, start=1)
        }

        # 2) Find the stop sign closest to the bus (clock)
        stable_order = stable_distance_order(
            bus_centroid,
            stop_sign_centroids,
            distance_compare_stability_margin(image),
        )
        if stable_order is None:
            return []
        idx_by_distance, _ = stable_order
        closest_idx = idx_by_distance[0]

        # 3) Convert its detection index to the left-to-right stop sign number
        answer_number = idx_to_number[closest_idx]

        return [(self.question, str(answer_number))]


class PairwiseCloserBench(Question):
    MAX_PAIRS = 4

    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "Which is closer to the clock, bench #{bench_i} or bench #{bench_j}? Respond with only the bench number."
            ),
            variables=["bench_i", "bench_j"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        labeled_bboxes = collect_detections_by_label(detections, ["bench", "clock"])
        benches = labeled_bboxes.get("bench", [])
        clocks = labeled_bboxes.get("clock", [])

        if len(benches) < 2 or len(clocks) == 0:
            return []

        bus_c = np.array(centroid(clocks[0]))
        bench_centroids = [centroid(bbox) for bbox in benches]
        bench_order = sort_indices_strict_left_to_right(bench_centroids)
        idx_to_number = {idx: num for num, idx in enumerate(bench_order, start=1)}
        number_to_idx = {num: idx for idx, num in idx_to_number.items()}
        bench_distances = [np.linalg.norm(np.array(c) - bus_c) for c in bench_centroids]
        distance_margin = distance_compare_stability_margin(image)

        bench_numbers = sorted(idx_to_number.values())
        all_pairs = [
            (bench_numbers[a], bench_numbers[b])
            for a in range(len(bench_numbers))
            for b in range(a + 1, len(bench_numbers))
        ]
        if len(all_pairs) > self.MAX_PAIRS:
            all_pairs = random.sample(all_pairs, self.MAX_PAIRS)

        qas = []
        for num_i, num_j in all_pairs:
            dist_i = bench_distances[number_to_idx[num_i]]
            dist_j = bench_distances[number_to_idx[num_j]]
            if abs(dist_i - dist_j) <= distance_margin:
                continue
            answer = str(num_i) if dist_i <= dist_j else str(num_j)
            qas.append((self.question.format(bench_i=num_i, bench_j=num_j), answer))
        return qas


class PairwiseCloserStopSign(Question):
    MAX_PAIRS = 4

    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "Which is closer to the clock, stop sign #{stop_i} or stop sign #{stop_j}? Respond with only the stop sign number."
            ),
            variables=["stop_i", "stop_j"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        labeled_bboxes = collect_detections_by_label(detections, ["stop sign", "clock"])
        stop_signs = labeled_bboxes.get("stop sign", [])
        clocks = labeled_bboxes.get("clock", [])

        if len(stop_signs) < 2 or len(clocks) == 0:
            return []

        bus_c = np.array(centroid(clocks[0]))
        stop_centroids = [centroid(bbox) for bbox in stop_signs]
        stop_order = sort_indices_strict_left_to_right(stop_centroids)
        idx_to_number = {idx: num for num, idx in enumerate(stop_order, start=1)}
        number_to_idx = {num: idx for idx, num in idx_to_number.items()}
        stop_distances = [np.linalg.norm(np.array(c) - bus_c) for c in stop_centroids]
        distance_margin = distance_compare_stability_margin(image)

        stop_numbers = sorted(idx_to_number.values())
        all_pairs = [
            (stop_numbers[a], stop_numbers[b])
            for a in range(len(stop_numbers))
            for b in range(a + 1, len(stop_numbers))
        ]
        if len(all_pairs) > self.MAX_PAIRS:
            all_pairs = random.sample(all_pairs, self.MAX_PAIRS)

        qas = []
        for num_i, num_j in all_pairs:
            dist_i = stop_distances[number_to_idx[num_i]]
            dist_j = stop_distances[number_to_idx[num_j]]
            if abs(dist_i - dist_j) <= distance_margin:
                continue
            answer = str(num_i) if dist_i <= dist_j else str(num_j)
            qas.append((self.question.format(stop_i=num_i, stop_j=num_j), answer))
        return qas


class ClosestToFurthestBenches(Question):
    def __init__(self) -> None:
        """Question: order benches by distance to the bus.

        Benches are first numbered strictly left to right by x centroid,
        and the answer lists these bench numbers from closest to farthest
        from the bus.
        """
        super().__init__(
            question=(
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "List the benches in order from closest to furthest from the clock, separated by commas. "
                "For example, '2, 1, 4, 3'. "
            ),
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        # Collect bench and clock (bus) detections
        labeled_bboxes = collect_detections_by_label(detections, ["bench", "clock"])
        benches = labeled_bboxes["bench"]
        bus_bboxes = labeled_bboxes["clock"]

        # If no benches or no bus are detected, skip this question
        if len(benches) == 0 or len(bus_bboxes) == 0:
            return []

        # We assume a single bus instance (first clock box)
        bus_bbox = bus_bboxes[0]

        # Compute centroids
        bench_centroids = [centroid(bbox) for bbox in benches]
        bus_centroid = centroid(bus_bbox)

        bench_order = sort_indices_strict_left_to_right(
            bench_centroids
        )

        # Map each bench index -> its left-to-right bench number (1, 2, 3, ...)
        bench_num = {idx: num for num, idx in enumerate(bench_order, start=1)}

        stable_order = stable_distance_order(
            bus_centroid,
            bench_centroids,
            distance_compare_stability_margin(image),
        )
        if stable_order is None:
            return []
        idx_by_distance, _ = stable_order
        idx_by_distance = [bench_num[idx] for idx in idx_by_distance]

        return [(self.question, ", ".join(map(str, idx_by_distance)))]


class ClosestToFurthestStopSigns(Question):
    def __init__(self) -> None:
        """Question: order stop signs by distance to the bus, using left-to-right numbering."""
        super().__init__(
            question=(
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "List the stop signs in order from closest to furthest from the clock, separated by commas. "
                "For example, '2, 1, 4, 3'. "
            ),
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        """
        Number stop signs strictly left to right by x centroid, then rank
        these numbered signs from closest to furthest from the bus (clock)
        by Euclidean distance between centroids.
        """
        labeled_bboxes = collect_detections_by_label(detections, ["stop sign", "clock"])
        stop_signs = labeled_bboxes.get("stop sign", [])
        clocks = labeled_bboxes.get("clock", [])

        # No applicable question if no stop signs or no bus detected
        if len(stop_signs) == 0 or len(clocks) == 0:
            return []

        # Use the first clock as the bus marker
        bus_bbox = clocks[0]

        # Compute centroids
        stop_sign_centroids = [centroid(bbox) for bbox in stop_signs]
        bus_centroid = centroid(bus_bbox)
        bus_c = np.array(bus_centroid)

        stop_sign_order = sort_indices_strict_left_to_right(
            stop_sign_centroids
        )

        # Map detection index -> left-to-right ID (1, 2, 3, ...)
        stop_sign_num = {idx: num for num, idx in enumerate(stop_sign_order, start=1)}

        stable_order = stable_distance_order(
            bus_centroid,
            stop_sign_centroids,
            distance_compare_stability_margin(image),
        )
        if stable_order is None:
            return []
        idx_by_distance, _ = stable_order
        ordered_ids = [stop_sign_num[idx] for idx in idx_by_distance]

        return [(self.question, ", ".join(map(str, ordered_ids)))]

'''
Group #3 Questions
'''

def check_path_blocked(bbox1, bbox2, bbox_other, margin: float = 0.0) -> bool:
    """
    Return True iff the segment between the centroids of bbox1 and bbox2
    intersects bbox_other (optionally expanded by `margin` pixels on each side).
    """
    c1 = np.array(centroid(bbox1))
    c2 = np.array(centroid(bbox2))
    if margin > 0.0:
        x0, y0, x1, y1 = map(float, bbox_other)
        expanded = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
        t = segment_box_entry_t(c1, c2, expanded)
    else:
        t = segment_box_entry_t(c1, c2, bbox_other)
    return t is not None


def _expand_bbox(bbox, margin: float):
    x0, y0, x1, y1 = map(float, bbox)
    return (x0 - margin, y0 - margin, x1 + margin, y1 + margin)


def _signed_angle_delta(base_dx: float, base_dy: float,
                        other_dx: float, other_dy: float) -> float:
    base_angle = math.atan2(base_dy, base_dx)
    other_angle = math.atan2(other_dy, other_dx)
    delta = other_angle - base_angle
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta <= -math.pi:
        delta += 2 * math.pi
    return delta

def segment_box_entry_t(
    p0: np.ndarray,
    p1: np.ndarray,
    bbox,
) -> Optional[float]:
    """
    Compute the entry parameter t (0 <= t <= 1) where the segment p0->p1
    enters the axis-aligned box `bbox`. If the segment does not intersect
    the box, return None.
    """
    x_min, y_min, x_max, y_max = map(float, bbox)
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    dx = x1 - x0
    dy = y1 - y0

    # Since we only care about the finite segment, start with [0, 1]
    t_min = 0.0
    t_max = 1.0

    # X slab
    if dx == 0.0:
        # Segment is vertical; if x is outside the slab, no intersection
        if x0 < x_min or x0 > x_max:
            return None
    else:
        t1 = (x_min - x0) / dx
        t2 = (x_max - x0) / dx
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_max < t_min:
            return None

    # Y slab
    if dy == 0.0:
        # Segment is horizontal; if y is outside the slab, no intersection
        if y0 < y_min or y0 > y_max:
            return None
    else:
        t1 = (y_min - y0) / dy
        t2 = (y_max - y0) / dy
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_max < t_min:
            return None

    return t_min


def stable_blocking_obstacles(
    image: Image.Image,
    bus_bbox,
    target_bbox,
    all_obstacles: list,
    bus_margin: float,
) -> Optional[list]:
    """Return ordered blocking obstacles, or None when the path geometry is unstable."""
    margin_buffer = path_margin_stability(image)
    lower_margin = max(0.0, bus_margin - margin_buffer)
    upper_margin = bus_margin + margin_buffer
    bus_c = np.array(centroid(bus_bbox), dtype=float)
    tgt_c = np.array(centroid(target_bbox), dtype=float)
    heading = tgt_c - bus_c

    entries: list[tuple[float, Any]] = []
    for obstacle in all_obstacles:
        blocked_lo = check_path_blocked(bus_bbox, target_bbox, obstacle, lower_margin)
        blocked_hi = check_path_blocked(bus_bbox, target_bbox, obstacle, upper_margin)
        if blocked_lo != blocked_hi:
            return None

        blocked_mid = check_path_blocked(bus_bbox, target_bbox, obstacle, bus_margin)
        if not blocked_mid:
            continue

        t = segment_box_entry_t(bus_c, tgt_c, _expand_bbox(obstacle, bus_margin))
        if t is None:
            return None
        entries.append((float(t), obstacle))

    if not entries:
        return []

    entries.sort(key=lambda item: item[0])
    if len(entries) >= 2 and (entries[1][0] - entries[0][0]) <= PATH_ENTRY_T_STABILITY_MARGIN:
        return None

    first_obstacle = entries[0][1]
    obstacle_c = np.array(centroid(first_obstacle), dtype=float)
    delta = abs(
        _signed_angle_delta(
            float(heading[0]),
            float(heading[1]),
            float(obstacle_c[0] - bus_c[0]),
            float(obstacle_c[1] - bus_c[1]),
        )
    )
    if delta <= PATH_HEAD_ON_ANGLE_MARGIN_RAD:
        return None

    return [obstacle for _, obstacle in entries]

def recommend_detour_direction(
    bus_bbox,
    target_bbox,
    blocking_obstacles: list,
    all_obstacles: list,
    image_width: int,
    image_height: int,
    block_margin: float = 0.0,
) -> str:
    # Returns "keep straight", "turn left", or "turn right".
    # Uses cross product of heading x v_obstacle to determine which side the
    # first-encountered blocking obstacle is on, then turns away from it.

    if not blocking_obstacles:
        return "keep straight"

    bus_c = np.array(centroid(bus_bbox), dtype=float)
    tgt_c = np.array(centroid(target_bbox), dtype=float)
    heading = tgt_c - bus_c

    # Find the first blocking obstacle along bus->target (smallest entry t)
    best_obstacle = None
    best_t = float("inf")
    for ob in blocking_obstacles:
        segment_target = _expand_bbox(ob, block_margin) if block_margin > 0.0 else ob
        t = segment_box_entry_t(bus_c, tgt_c, segment_target)
        if t is not None and t < best_t:
            best_t = t
            best_obstacle = ob

    if best_obstacle is None:
        # Fallback: use closest obstacle by centroid distance
        dists = [np.linalg.norm(np.array(centroid(ob), dtype=float) - bus_c) for ob in blocking_obstacles]
        best_obstacle = blocking_obstacles[int(np.argmin(dists))]

    obs_c = np.array(centroid(best_obstacle), dtype=float)
    v_obs = obs_c - bus_c

    # 2D cross product: heading x v_obs
    # Image coords have y pointing down, so:
    #   cross > 0 => obstacle is clockwise from heading (to the RIGHT) => turn left
    #   cross < 0 => obstacle is counter-clockwise (to the LEFT) => turn right
    cross = heading[0] * v_obs[1] - heading[1] * v_obs[0]

    if cross > 0:
        return "turn left"
    elif cross < 0:
        return "turn right"
    else:
        # Obstacle is directly on the heading line; pick the side with more room
        perp = np.array([-heading[1], heading[0]])  # 90 deg CCW of heading
        img_center = np.array([image_width / 2.0, image_height / 2.0])
        side = float(np.dot(img_center - bus_c, perp))
        return "turn left" if side >= 0 else "turn right"

_EIGHT_DIRS = [
    "East",
    "Southeast",
    "South",
    "Southwest",
    "West",
    "Northwest",
    "North",
    "Northeast",
]

def _eight_direction(dx: float, dy: float) -> str:
    """Return one of 8 compass directions for a displacement (dx, dy).
    Image coordinates: x+ is right (East), y+ is down (South).
    Uses equal 45-degree sectors via atan2.
    """
    import math
    sector = round(math.atan2(dy, dx) / (math.pi / 4)) % 8
    return _EIGHT_DIRS[sector]

class GeometricDirectionToBench(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "What is the relative direction of bench #{bench_number} to the clock? "
                "Answer with exactly one of: 'North', 'South', 'East', 'West', "
                "'Northeast', 'Northwest', 'Southeast', or 'Southwest'. "
            ),
            variables=["bench_number"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        labeled_bboxes = collect_detections_by_label(detections, ["bench", "clock"])
        benches = labeled_bboxes.get("bench", [])
        clocks = labeled_bboxes.get("clock", [])

        if len(benches) == 0 or len(clocks) == 0:
            return []

        bus_c = np.array(centroid(clocks[0]))
        bench_centroids = [centroid(b) for b in benches]
        bench_order = sort_indices_strict_left_to_right(bench_centroids)

        qas: list[tuple[str, str]] = []
        for rank, bench_idx in enumerate(bench_order):
            bench_c = np.array(bench_centroids[bench_idx])
            dx = float(bench_c[0] - bus_c[0])
            dy = float(bench_c[1] - bus_c[1])
            if not eight_direction_is_stable(dx, dy):
                continue
            direction = _eight_direction(
                dx,
                dy,
            )
            qas.append((self.question.format(bench_number=rank + 1), direction))
        return qas


class GeometricDirectionToStopSign(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "What is the relative direction of stop sign #{stop_sign_number} to the clock? "
                "Answer with exactly one of: 'North', 'South', 'East', 'West', "
                "'Northeast', 'Northwest', 'Southeast', or 'Southwest'. "
            ),
            variables=["stop_sign_number"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        labeled_bboxes = collect_detections_by_label(detections, ["stop sign", "clock"])
        stop_signs = labeled_bboxes.get("stop sign", [])
        clocks = labeled_bboxes.get("clock", [])

        if len(stop_signs) == 0 or len(clocks) == 0:
            return []

        bus_c = np.array(centroid(clocks[0]))
        stop_sign_centroids = [centroid(s) for s in stop_signs]
        stop_sign_order = sort_indices_strict_left_to_right(stop_sign_centroids)

        qas: list[tuple[str, str]] = []
        for rank, ss_idx in enumerate(stop_sign_order):
            ss_c = np.array(stop_sign_centroids[ss_idx])
            dx = float(ss_c[0] - bus_c[0])
            dy = float(ss_c[1] - bus_c[1])
            if not eight_direction_is_stable(dx, dy):
                continue
            direction = _eight_direction(
                dx,
                dy,
            )
            qas.append((self.question.format(stop_sign_number=rank + 1), direction))
        return qas


class AvoidObstacleToReachBench(Question):
    def __init__(self) -> None:
        """
        For each bench (numbered from left to right), recommend a direction
        for the bus to move in order to reach that bench while avoiding obstacles.

        Rules:
        - The bus is represented by a clock detection.
        - People sitting at the *target* bench do NOT count as obstacles.
        - Other benches, other people, stop signs, and animals
          (elephant, zebra, giraffe) are treated as potential obstacles.

        Answer format:
        - One of: "keep straight", "turn left", "turn right".
        """
        super().__init__(
            question=(
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "Imagine that the clock is now facing bench {bench_number} and wants to reach that bench. "
                "Ignore the people already at bench {bench_number}. "
                "If no other object blocks the straight path between the clock and bench {bench_number}, answer 'keep straight'. "
                "Otherwise, answer 'turn left' or 'turn right' to avoid the first blocking object along that path."
            ),
            variables=["bench_number"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        # Only these classes are relevant
        animal_labels = ["elephant", "zebra", "giraffe"]
        target_labels = ["bench", "person", "clock", "stop sign"] + animal_labels
        labeled_bboxes = collect_detections_by_label(detections, target_labels)

        benches = labeled_bboxes.get("bench", [])
        persons = labeled_bboxes.get("person", [])
        clocks = labeled_bboxes.get("clock", [])
        stop_signs = labeled_bboxes.get("stop sign", [])
        animal_bboxes = [
            bbox
            for lbl in animal_labels
            for bbox in labeled_bboxes.get(lbl, [])
        ]

        # If no benches or no bus, we don't generate this kind of question
        if len(benches) == 0 or len(clocks) == 0:
            return []

        bus_bbox = clocks[0]
        _bx0, _by0, _bx1, _by1 = map(float, bus_bbox)
        bus_margin = max((_bx1 - _bx0) / 2.0, (_by1 - _by0) / 2.0)

        # Bench geometry
        bench_centroids = [centroid(b) for b in benches]
        bench_order = sort_indices_strict_left_to_right(
            bench_centroids
        )

        # Assign each person to the nearest bench so that people
        # sitting on the *target* bench can be ignored as obstacles
        person_centroids = [centroid(b) for b in persons]
        stable_assignment = stable_assign_objects_to_location(
            person_centroids,
            bench_centroids,
            assignment_stability_margin(image),
        )
        if stable_assignment is None:
            return []
        _, person_assignment = stable_assignment

        qas: list[tuple[str, str]] = []
        for rank, bench_idx in enumerate(bench_order):
            bench_bbox = benches[bench_idx]

            # Build the full obstacle set for this bench
            all_obstacles = []

            # 1) Other benches (target bench itself is not an obstacle)
            for i, bbox in enumerate(benches):
                if i != bench_idx:
                    all_obstacles.append(bbox)

            # 2) People not assigned to this bench count as obstacles
            for i, bbox in enumerate(persons):
                assigned_idx = person_assignment[i] if i < len(person_assignment) else -1
                if assigned_idx != bench_idx:
                    all_obstacles.append(bbox)

            # 3) All stop signs
            all_obstacles.extend(stop_signs)

            # 4) All animals
            all_obstacles.extend(animal_bboxes)

            blocking_obstacles = stable_blocking_obstacles(
                image, bus_bbox, bench_bbox, all_obstacles, bus_margin
            )
            if blocking_obstacles is None:
                continue

            direction = recommend_detour_direction(
                bus_bbox=bus_bbox,
                target_bbox=bench_bbox,
                blocking_obstacles=blocking_obstacles,
                all_obstacles=all_obstacles,
                image_width=image.size[0],
                image_height=image.size[1],
                block_margin=bus_margin,
            )

            question = self.question.format(bench_number=rank + 1)
            answer = direction
            qas.append((question, answer))

        return qas


class AvoidObstacleToReachStopSign(Question):
    def __init__(self) -> None:
        """
        For each stop sign (numbered from left to right), recommend a direction
        for the bus to move in order to reach that stop sign while avoiding obstacles.

        Rules:
        - The bus is represented by a clock detection.
        - Animals that belong to the *target* stop sign's crowd
          (assigned to it as the nearest stop sign) do NOT count as obstacles.
        - Animals belonging to other stop signs, benches, persons, and other
          stop signs are treated as obstacles.

        Answer format:
        - One of: "keep straight", "turn left", "turn right".
        """
        super().__init__(
            question=(
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "Imagine that the clock is now facing stop sign {stop_sign_number} and wants to reach that stop sign. "
                "Ignore the animals already grouped with stop sign {stop_sign_number}. "
                "If no other object blocks the straight path between the clock and stop sign {stop_sign_number}, answer 'keep straight'. "
                "Otherwise, answer 'turn left' or 'turn right' to avoid the first blocking object along that path."
            ),
            variables=["stop_sign_number"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        animal_labels = ["elephant", "zebra", "giraffe"]
        target_labels = ["stop sign", "clock", "bench", "person"] + animal_labels
        labeled_bboxes = collect_detections_by_label(detections, target_labels)

        stop_signs = labeled_bboxes.get("stop sign", [])
        clocks = labeled_bboxes.get("clock", [])
        benches = labeled_bboxes.get("bench", [])
        persons = labeled_bboxes.get("person", [])
        animal_bboxes = [
            bbox
            for lbl in animal_labels
            for bbox in labeled_bboxes.get(lbl, [])
        ]

        # If no stop sign or no bus, do not generate questions
        if len(stop_signs) == 0 or len(clocks) == 0:
            return []

        bus_bbox = clocks[0]
        _bx0, _by0, _bx1, _by1 = map(float, bus_bbox)
        bus_margin = max((_bx1 - _bx0) / 2.0, (_by1 - _by0) / 2.0)

        # Stop sign geometry
        stop_sign_centroids = [centroid(b) for b in stop_signs]
        stop_sign_order = sort_indices_strict_left_to_right(
            stop_sign_centroids
        )

        # Assign animals to the nearest stop sign,
        # to decide which animals belong to which crowd.
        animal_centroids = [centroid(b) for b in animal_bboxes]
        stable_assignment = stable_assign_objects_to_location(
            animal_centroids,
            stop_sign_centroids,
            assignment_stability_margin(image),
        )
        if stable_assignment is None:
            return []
        _, animal_assignment = stable_assignment

        qas: list[tuple[str, str]] = []
        for rank, stop_idx in enumerate(stop_sign_order):
            target_stop_bbox = stop_signs[stop_idx]

            # Build the full obstacle set for this stop sign
            all_obstacles = []

            # 1) All benches
            all_obstacles.extend(benches)

            # 2) All persons
            all_obstacles.extend(persons)

            # 3) Other stop signs (not the target one)
            for i, bbox in enumerate(stop_signs):
                if i != stop_idx:
                    all_obstacles.append(bbox)

            # 4) Animals assigned to *other* stop signs are obstacles;
            #    animals assigned to this stop sign belong to the crowd, not obstacles.
            for i, bbox in enumerate(animal_bboxes):
                assigned_idx = animal_assignment[i] if i < len(animal_assignment) else -1
                if assigned_idx != stop_idx:
                    all_obstacles.append(bbox)

            blocking_obstacles = stable_blocking_obstacles(
                image, bus_bbox, target_stop_bbox, all_obstacles, bus_margin
            )
            if blocking_obstacles is None:
                continue

            direction = recommend_detour_direction(
                bus_bbox=bus_bbox,
                target_bbox=target_stop_bbox,
                blocking_obstacles=blocking_obstacles,
                all_obstacles=all_obstacles,
                image_width=image.size[0],
                image_height=image.size[1],
                block_margin=bus_margin,
            )

            question = self.question.format(stop_sign_number=rank + 1)
            answer = direction
            qas.append((question, answer))

        return qas


'''
Group #4 Questions
'''

def _find_red_dot(
    image: Image.Image,
    bus_bbox,
    search_radius: Optional[int] = None,
) -> Optional[tuple]:
    """
    Locate the vivid red heading dot near the bus using connected-component filtering.

    The heading dot is drawn at a fixed offset/radius in the pre-resize scene and then
    resized together with the full image. Stop signs are also red, so we keep only red
    components whose size and distance from the clock match the scaled heading dot.

    Returns (cx, cy) in full-image coordinates, or None if not found.
    """
    import cv2
    cx, cy = centroid(bus_bbox)
    scale = scene_scale(image)
    expected_offset = max(8.0, scaled_scene_distance(image, SCENE_HEADING_DOT_OFFSET))
    expected_radius = max(4.0, scaled_scene_distance(image, SCENE_HEADING_DOT_RADIUS))
    if search_radius is None:
        search_radius = int(round(expected_offset + 4.0 * expected_radius + max(12.0, 40.0 * scale)))
    img_w, img_h = image.size
    x1 = max(0, int(cx - search_radius))
    y1 = max(0, int(cy - search_radius))
    x2 = min(img_w, int(cx + search_radius))
    y2 = min(img_h, int(cy + search_radius))

    crop = np.array(image.convert("RGB"))[y1:y2, x1:x2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    # Red wraps around hue=0/180; two ranges cover both ends
    mask1 = cv2.inRange(hsv, np.array([0,   150, 150]), np.array([10,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 150, 150]), np.array([180, 255, 255]))
    mask  = cv2.bitwise_or(mask1, mask2)

    # --- Blob filtering: find the component closest to the expected heading distance ---
    expected_area = math.pi * expected_radius * expected_radius
    dot_area_min = max(40.0, expected_area * 0.20)
    dot_area_max = max(dot_area_min + 1.0, expected_area * 8.0)
    dist_tolerance = max(18.0, expected_radius * 2.5)

    num_labels, _, stats, comp_centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    best: Optional[tuple] = None
    best_err = float("inf")
    for i in range(1, num_labels):          # label 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (dot_area_min <= area <= dot_area_max):
            continue
        width = max(1, int(stats[i, cv2.CC_STAT_WIDTH]))
        height = max(1, int(stats[i, cv2.CC_STAT_HEIGHT]))
        aspect_ratio = width / float(height)
        if not (0.55 <= aspect_ratio <= 1.80):
            continue
        if width > expected_radius * 4.5 or height > expected_radius * 4.5:
            continue
        dot_cx = comp_centroids[i][0] + x1
        dot_cy = comp_centroids[i][1] + y1
        dist = math.hypot(dot_cx - cx, dot_cy - cy)
        if dist <= expected_radius:
            continue
        err  = abs(dist - expected_offset)
        if err < dist_tolerance and err < best_err:
            best_err = err
            best = (dot_cx, dot_cy)

    return best

class BusHeadingDirection(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "A red circle is placed in front of the clock in the image to indicate its current heading direction. "
                "Based on the position of the red circle relative to the clock, in which direction is the clock currently heading? "
                "Answer with exactly one of: 'North', 'South', 'East', 'West', "
                "'Northeast', 'Northwest', 'Southeast', or 'Southwest'."
            ),
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        clocks = collect_detections_by_label(detections, ["clock"]).get("clock", [])
        if not clocks:
            return []

        bus_bbox = clocks[0]
        red_dot  = _find_red_dot(image, bus_bbox)
        if red_dot is None:
            return []

        bus_c = np.array(centroid(bus_bbox))
        if not eight_direction_is_stable(
            float(red_dot[0] - bus_c[0]),
            float(red_dot[1] - bus_c[1]),
        ):
            return []
        direction = _eight_direction(
            float(red_dot[0] - bus_c[0]),
            float(red_dot[1] - bus_c[1]),
        )
        return [(self.question, direction)]


_RELATIVE_DIRS = [
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
]

def _relative_direction(heading_dx: float, heading_dy: float,
                         target_dx: float, target_dy: float) -> str:
    """Return one of 8 relative positions of the target in the robot's frame.
    Image coords: x+ right, y+ down.  Positive delta = clockwise = right side.
    """
    delta = _signed_angle_delta(heading_dx, heading_dy, target_dx, target_dy)
    sector = round(delta / (math.pi / 4)) % 8
    return _RELATIVE_DIRS[sector]

def _turn_direction(heading_dx: float, heading_dy: float,
                    target_dx: float, target_dy: float) -> str:
    """Return 'turn left', 'turn right', or 'already facing'.
    In image coords, positive delta (clockwise) = turn right.
    'already facing' when |delta| <= pi/8 (within the front sector).
    """
    delta = _signed_angle_delta(heading_dx, heading_dy, target_dx, target_dy)
    if abs(delta) <= math.pi / 8:
        return "already facing"
    return "turn right" if delta > 0 else "turn left"

def _get_bus_heading(image, detections):
    """Return (bus_bbox, heading_dx, heading_dy, bus_c) or None."""
    clocks = collect_detections_by_label(detections, ["clock"]).get("clock", [])
    if not clocks:
        return None
    bus_bbox = clocks[0]
    red_dot  = _find_red_dot(image, bus_bbox)
    if red_dot is None:
        return None
    bus_c = np.array(centroid(bus_bbox))
    return bus_bbox, float(red_dot[0] - bus_c[0]), float(red_dot[1] - bus_c[1]), bus_c

class TurnDirectionToBench(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "A red circle is placed in front of the clock to indicate its current heading direction. "
                "To face bench #{bench_number}, should the clock turn left, turn right, or is it already facing that bench? "
                "Answer with exactly one of: 'turn left', 'turn right', or 'already facing'."
            ),
            variables=["bench_number"],
            predicates=[ObjectDetectionPredicates.at_least_one_single_detection],
        )

    def apply(self, image, detections):
        info = _get_bus_heading(image, detections)
        if info is None:
            return []
        _, heading_dx, heading_dy, bus_c = info

        benches = collect_detections_by_label(detections, ["bench"]).get("bench", [])
        if not benches:
            return []
        bench_centroids = [centroid(b) for b in benches]
        bench_order = sort_indices_strict_left_to_right(bench_centroids)

        qas: list[tuple[str, str]] = []
        for rank, bench_idx in enumerate(bench_order):
            bench_c = np.array(bench_centroids[bench_idx])
            delta = _signed_angle_delta(
                heading_dx, heading_dy,
                float(bench_c[0] - bus_c[0]), float(bench_c[1] - bus_c[1]),
            )
            if not turn_direction_is_stable(delta):
                continue
            answer = _turn_direction(
                heading_dx, heading_dy,
                float(bench_c[0] - bus_c[0]), float(bench_c[1] - bus_c[1]),
            )
            qas.append((self.question.format(bench_number=rank + 1), answer))
        return qas


class TurnDirectionToStopSign(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "A red circle is placed in front of the clock to indicate its current heading direction. "
                "To face stop sign #{stop_sign_number}, should the clock turn left, turn right, or is it already facing that stop sign? "
                "Answer with exactly one of: 'turn left', 'turn right', or 'already facing'."
            ),
            variables=["stop_sign_number"],
            predicates=[ObjectDetectionPredicates.at_least_one_single_detection],
        )

    def apply(self, image, detections):
        info = _get_bus_heading(image, detections)
        if info is None:
            return []
        _, heading_dx, heading_dy, bus_c = info

        stop_signs = collect_detections_by_label(detections, ["stop sign"]).get("stop sign", [])
        if not stop_signs:
            return []
        ss_centroids = [centroid(s) for s in stop_signs]
        ss_order = sort_indices_strict_left_to_right(ss_centroids)

        qas: list[tuple[str, str]] = []
        for rank, ss_idx in enumerate(ss_order):
            ss_c = np.array(ss_centroids[ss_idx])
            delta = _signed_angle_delta(
                heading_dx, heading_dy,
                float(ss_c[0] - bus_c[0]), float(ss_c[1] - bus_c[1]),
            )
            if not turn_direction_is_stable(delta):
                continue
            answer = _turn_direction(
                heading_dx, heading_dy,
                float(ss_c[0] - bus_c[0]), float(ss_c[1] - bus_c[1]),
            )
            qas.append((self.question.format(stop_sign_number=rank + 1), answer))
        return qas


class BenchRelativeToHeading(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "A red circle is placed in front of the clock to indicate its current heading direction. "
                "Where is bench #{bench_number} relative to the clock's current heading direction? "
                "Answer with exactly one of: 'front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', or 'front-left'."
            ),
            variables=["bench_number"],
            predicates=[ObjectDetectionPredicates.at_least_one_single_detection],
        )

    def apply(self, image, detections):
        info = _get_bus_heading(image, detections)
        if info is None:
            return []
        _, heading_dx, heading_dy, bus_c = info

        benches = collect_detections_by_label(detections, ["bench"]).get("bench", [])
        if not benches:
            return []
        bench_centroids = [centroid(b) for b in benches]
        bench_order = sort_indices_strict_left_to_right(bench_centroids)

        qas: list[tuple[str, str]] = []
        for rank, bench_idx in enumerate(bench_order):
            bench_c = np.array(bench_centroids[bench_idx])
            delta = _signed_angle_delta(
                heading_dx, heading_dy,
                float(bench_c[0] - bus_c[0]), float(bench_c[1] - bus_c[1]),
            )
            if not relative_direction_is_stable(delta):
                continue
            answer = _relative_direction(
                heading_dx, heading_dy,
                float(bench_c[0] - bus_c[0]), float(bench_c[1] - bus_c[1]),
            )
            qas.append((self.question.format(bench_number=rank + 1), answer))
        return qas


class StopSignRelativeToHeading(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "A red circle is placed in front of the clock to indicate its current heading direction. "
                "Where is stop sign #{stop_sign_number} relative to the clock's current heading direction? "
                "Answer with exactly one of: 'front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', or 'front-left'."
            ),
            variables=["stop_sign_number"],
            predicates=[ObjectDetectionPredicates.at_least_one_single_detection],
        )

    def apply(self, image, detections):
        info = _get_bus_heading(image, detections)
        if info is None:
            return []
        _, heading_dx, heading_dy, bus_c = info

        stop_signs = collect_detections_by_label(detections, ["stop sign"]).get("stop sign", [])
        if not stop_signs:
            return []
        ss_centroids = [centroid(s) for s in stop_signs]
        ss_order = sort_indices_strict_left_to_right(ss_centroids)

        qas: list[tuple[str, str]] = []
        for rank, ss_idx in enumerate(ss_order):
            ss_c = np.array(ss_centroids[ss_idx])
            delta = _signed_angle_delta(
                heading_dx, heading_dy,
                float(ss_c[0] - bus_c[0]), float(ss_c[1] - bus_c[1]),
            )
            if not relative_direction_is_stable(delta):
                continue
            answer = _relative_direction(
                heading_dx, heading_dy,
                float(ss_c[0] - bus_c[0]), float(ss_c[1] - bus_c[1]),
            )
            qas.append((self.question.format(stop_sign_number=rank + 1), answer))
        return qas

'''
Group #5 Questions
'''

class CountPersonAtClosestBench(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "How many people are at the bench closest to the clock? Respond with only an integer."
            ),
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        labeled_bboxes = collect_detections_by_label(
            detections, ["bench", "person", "clock"]
        )

        benches = labeled_bboxes.get("bench", [])
        persons = labeled_bboxes.get("person", [])
        clocks = labeled_bboxes.get("clock", [])

        if len(benches) == 0 or len(clocks) == 0:
            return []

        bus_bbox = clocks[0]

        bench_centroids = [centroid(bbox) for bbox in benches]
        person_centroids = [centroid(bbox) for bbox in persons]
        bus_centroid = centroid(bus_bbox)

        # Find the bench closest to the bus
        stable_order = stable_distance_order(
            bus_centroid,
            bench_centroids,
            distance_compare_stability_margin(image),
        )
        if stable_order is None:
            return []
        idx_by_distance, _ = stable_order
        closest_bench_idx = idx_by_distance[0]

        # Count people assigned to the closest bench
        stable_assignment = stable_assign_objects_to_location(
            person_centroids,
            bench_centroids,
            assignment_stability_margin(image),
        )
        if stable_assignment is None:
            return []
        bench_occupancy, _ = stable_assignment

        count = bench_occupancy[closest_bench_idx]
        return [(self.question, str(count))]


class ClosestBenchWithPerson(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "Which bench is closest to the clock that has at least one person at it? Answer with the bench ID. "
                "If no benches have people, respond with '0'. "
            ),
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        labeled_bboxes = collect_detections_by_label(
            detections, ["bench", "person", "clock"]
        )

        benches = labeled_bboxes.get("bench", [])
        persons = labeled_bboxes.get("person", [])
        clocks = labeled_bboxes.get("clock", [])

        if len(benches) == 0 or len(clocks) == 0:
            return []
        if len(persons) == 0:
            return [(self.question, "0")]

        bus_bbox = clocks[0]

        bench_centroids = [centroid(bbox) for bbox in benches]
        person_centroids = [centroid(bbox) for bbox in persons]
        bus_centroid = centroid(bus_bbox)

        # 1) Number benches strictly left to right by x centroid.
        bench_order = sort_indices_strict_left_to_right(
            bench_centroids
        )
        # Map detection index -> bench number (1, 2, 3, ...)
        bench_num = {idx: num for num, idx in enumerate(bench_order, start=1)}

        # 2) Order benches by distance to the bus (closest -> furthest)
        stable_order = stable_distance_order(
            bus_centroid,
            bench_centroids,
            distance_compare_stability_margin(image),
        )
        if stable_order is None:
            return []
        bench_order_by_distance, _ = stable_order

        # 3) Assign each person to the nearest bench (by centroid distance)
        stable_assignment = stable_assign_objects_to_location(
            person_centroids,
            bench_centroids,
            assignment_stability_margin(image),
        )
        if stable_assignment is None:
            return []
        bench_occupancy, _ = stable_assignment

        # 4) Scan benches in distance order to find the closest bench with people
        for idx in bench_order_by_distance:
            if bench_occupancy[idx] > 0:
                # Return the left-to-right bench number (by x centroid)
                return [(self.question, str(bench_num[idx]))]

        # No benches have people
        return [(self.question, "0")]


class AvoidObstacleToReachClosestBench(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "Imagine that the clock is now facing the closest bench and wants to reach it. "
                "Ignore the people already at that bench. "
                "If no other object blocks the straight path between the clock and the closest bench, answer 'keep straight'. "
                "Otherwise, answer 'turn left' or 'turn right' to avoid the first blocking object along that path."
            ),
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        animal_labels = ["elephant", "zebra", "giraffe"]
        target_labels = ["bench", "person", "clock", "stop sign"] + animal_labels
        labeled_bboxes = collect_detections_by_label(detections, target_labels)

        benches = labeled_bboxes.get("bench", [])
        persons = labeled_bboxes.get("person", [])
        clocks = labeled_bboxes.get("clock", [])
        stop_signs = labeled_bboxes.get("stop sign", [])
        animal_bboxes = [
            bbox
            for lbl in animal_labels
            for bbox in labeled_bboxes.get(lbl, [])
        ]

        if len(benches) == 0 or len(clocks) == 0:
            return []

        bus_bbox = clocks[0]
        _bx0, _by0, _bx1, _by1 = map(float, bus_bbox)
        bus_margin = max((_bx1 - _bx0) / 2.0, (_by1 - _by0) / 2.0)
        bus_c = np.array(centroid(bus_bbox))

        bench_centroids = [centroid(b) for b in benches]

        # Find the closest bench to the bus
        stable_order = stable_distance_order(
            tuple(bus_c.tolist()),
            bench_centroids,
            distance_compare_stability_margin(image),
        )
        if stable_order is None:
            return []
        idx_by_distance, _ = stable_order
        closest_idx = idx_by_distance[0]
        bench_bbox = benches[closest_idx]

        # Assign each person to the nearest bench; people at target bench are not obstacles
        person_centroids = [centroid(p) for p in persons]
        stable_assignment = stable_assign_objects_to_location(
            person_centroids,
            bench_centroids,
            assignment_stability_margin(image),
        )
        if stable_assignment is None:
            return []
        _, person_assignment = stable_assignment

        all_obstacles = []
        for i, bbox in enumerate(benches):
            if i != closest_idx:
                all_obstacles.append(bbox)
        for i, bbox in enumerate(persons):
            if person_assignment[i] != closest_idx:
                all_obstacles.append(bbox)
        all_obstacles.extend(stop_signs)
        all_obstacles.extend(animal_bboxes)

        blocking_obstacles = stable_blocking_obstacles(
            image, bus_bbox, bench_bbox, all_obstacles, bus_margin
        )
        if blocking_obstacles is None:
            return []

        direction = recommend_detour_direction(
            bus_bbox=bus_bbox,
            target_bbox=bench_bbox,
            blocking_obstacles=blocking_obstacles,
            all_obstacles=all_obstacles,
            image_width=image.size[0],
            image_height=image.size[1],
            block_margin=bus_margin,
        )
        return [(self.question, direction)]


class AvoidObstacleToReachClosestStopSign(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "Imagine that the clock is now facing the closest stop sign and wants to reach it. "
                "Ignore the animals already grouped with that stop sign. "
                "If no other object blocks the straight path between the clock and the closest stop sign, answer 'keep straight'. "
                "Otherwise, answer 'turn left' or 'turn right' to avoid the first blocking object along that path."
            ),
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        animal_labels = ["elephant", "zebra", "giraffe"]
        target_labels = ["stop sign", "clock", "bench", "person"] + animal_labels
        labeled_bboxes = collect_detections_by_label(detections, target_labels)

        stop_signs = labeled_bboxes.get("stop sign", [])
        clocks = labeled_bboxes.get("clock", [])
        benches = labeled_bboxes.get("bench", [])
        persons = labeled_bboxes.get("person", [])
        animal_bboxes = [
            bbox
            for lbl in animal_labels
            for bbox in labeled_bboxes.get(lbl, [])
        ]

        if len(stop_signs) == 0 or len(clocks) == 0:
            return []

        bus_bbox = clocks[0]
        _bx0, _by0, _bx1, _by1 = map(float, bus_bbox)
        bus_margin = max((_bx1 - _bx0) / 2.0, (_by1 - _by0) / 2.0)
        bus_c = np.array(centroid(bus_bbox))

        stop_sign_centroids = [centroid(s) for s in stop_signs]

        # Find the closest stop sign to the bus
        stable_order = stable_distance_order(
            tuple(bus_c.tolist()),
            stop_sign_centroids,
            distance_compare_stability_margin(image),
        )
        if stable_order is None:
            return []
        idx_by_distance, _ = stable_order
        closest_idx = idx_by_distance[0]
        target_bbox = stop_signs[closest_idx]

        # Assign animals to the nearest stop sign; animals at target are not obstacles
        animal_centroids = [centroid(b) for b in animal_bboxes]
        stable_assignment = stable_assign_objects_to_location(
            animal_centroids,
            stop_sign_centroids,
            assignment_stability_margin(image),
        )
        if stable_assignment is None:
            return []
        _, animal_assignment = stable_assignment

        all_obstacles = []
        all_obstacles.extend(benches)
        all_obstacles.extend(persons)
        for i, bbox in enumerate(stop_signs):
            if i != closest_idx:
                all_obstacles.append(bbox)
        for i, bbox in enumerate(animal_bboxes):
            if animal_assignment[i] != closest_idx:
                all_obstacles.append(bbox)

        blocking_obstacles = stable_blocking_obstacles(
            image, bus_bbox, target_bbox, all_obstacles, bus_margin
        )
        if blocking_obstacles is None:
            return []

        direction = recommend_detour_direction(
            bus_bbox=bus_bbox,
            target_bbox=target_bbox,
            blocking_obstacles=blocking_obstacles,
            all_obstacles=all_obstacles,
            image_width=image.size[0],
            image_height=image.size[1],
            block_margin=bus_margin,
        )
        return [(self.question, direction)]


class DirectionToClosestBench(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "What is the relative direction of the closest bench to the clock? "
                "Answer with exactly one of: 'North', 'South', 'East', 'West', "
                "'Northeast', 'Northwest', 'Southeast', or 'Southwest'."
            ),
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        labeled_bboxes = collect_detections_by_label(detections, ["bench", "clock"])
        benches = labeled_bboxes.get("bench", [])
        clocks = labeled_bboxes.get("clock", [])

        if len(benches) == 0 or len(clocks) == 0:
            return []

        bus_c = np.array(centroid(clocks[0]), dtype=float)
        bench_centroids = [centroid(b) for b in benches]

        stable_order = stable_distance_order(
            tuple(bus_c.tolist()),
            bench_centroids,
            distance_compare_stability_margin(image),
        )
        if stable_order is None:
            return []
        idx_by_distance, _ = stable_order
        closest_idx = idx_by_distance[0]
        bench_c = np.array(bench_centroids[closest_idx], dtype=float)
        dx = float(bench_c[0] - bus_c[0])
        dy = float(bench_c[1] - bus_c[1])
        if not eight_direction_is_stable(dx, dy):
            return []

        direction = _eight_direction(
            dx,
            dy,
        )
        return [(self.question, direction)]


class DirectionToClosestStopSign(Question):
    def __init__(self) -> None:
        super().__init__(
            question=(
                "What is the relative direction of the closest stop sign to the clock? "
                "Answer with exactly one of: 'North', 'South', 'East', 'West', "
                "'Northeast', 'Northwest', 'Southeast', or 'Southwest'."
            ),
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(self, image, detections):
        labeled_bboxes = collect_detections_by_label(detections, ["stop sign", "clock"])
        stop_signs = labeled_bboxes.get("stop sign", [])
        clocks = labeled_bboxes.get("clock", [])

        if len(stop_signs) == 0 or len(clocks) == 0:
            return []

        bus_c = np.array(centroid(clocks[0]), dtype=float)
        ss_centroids = [centroid(s) for s in stop_signs]

        stable_order = stable_distance_order(
            tuple(bus_c.tolist()),
            ss_centroids,
            distance_compare_stability_margin(image),
        )
        if stable_order is None:
            return []
        idx_by_distance, _ = stable_order
        closest_idx = idx_by_distance[0]
        ss_c = np.array(ss_centroids[closest_idx], dtype=float)
        dx = float(ss_c[0] - bus_c[0])
        dy = float(ss_c[1] - bus_c[1])
        if not eight_direction_is_stable(dx, dy):
            return []

        direction = _eight_direction(
            dx,
            dy,
        )
        return [(self.question, direction)]

### <-- ORIGINAL GRAID QUESTIONS --> ###

class IsObjectCentered(Question):
    def __init__(self, buffer_ratio: float = 0.05) -> None:
        """Create an *Is-Object-Centered* question.

        Args:
            buffer_ratio: Fraction of the image width to treat as a no-ask buffer
                around the one-third and two-third vertical lines. A value such as
                ``0.05`` means 5 % of the image width on either side of the grid
                boundary will be treated as *ambiguous* – if any side of the
                bounding box falls in that zone, the question is skipped for
                that object.
        """
        super().__init__(
            question=(
                "Divide the image into thirds. In which third does the "
                "{object_1} primarily appear? Respond with the letter only: "
                "A) left third, B) middle third, C) right third."
            ),
            variables=["object_1"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )
        if buffer_ratio < 0 or buffer_ratio > 0.5:
            raise ValueError(
                "Buffer ratio provided does not make sense. Must be between 0 (no buffer) and 0.5 (half the image width)"
            )
        self.buffer_ratio: float = buffer_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        # classes with single instance
        single_classes = {k for k, v in counts.items() if v == 1}

        image_width, _ = image.size

        question_answer_pairs = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if key not in single_classes:
                        continue
                    bbox = det.as_xyxy()[i]
                    x_min, x_max = float(bbox[0]), float(bbox[2])
                    self._process_single_detection(
                        key, x_min, x_max, image_width, question_answer_pairs
                    )
            else:
                key = str(lbl)
                if key not in single_classes:
                    continue
                bbox = det.as_xyxy()[0]
                x_min, x_max = float(bbox[0]), float(bbox[2])
                self._process_single_detection(
                    key, x_min, x_max, image_width, question_answer_pairs
                )

        return question_answer_pairs

    def _process_single_detection(
        self,
        class_name: str,
        x_min: float,
        x_max: float,
        image_width: int,
        question_answer_pairs: list,
    ):
        question = self.question.format(object_1=class_name)

        left_line = image_width / 3
        right_line = 2 * image_width / 3
        buffer = self.buffer_ratio * image_width

        # Discard if bbox is too close to a boundary (ambiguous)
        if (
            abs(x_min - left_line) < buffer
            or abs(x_max - left_line) < buffer
            or abs(x_min - right_line) < buffer
            or abs(x_max - right_line) < buffer
        ):
            logger.debug("IsObjectCentered skipped due to ambiguity buffer")
            return

        # Determine third based on buffered grid
        if x_max < left_line - buffer:
            answer = "A"
        elif x_min > left_line + buffer and x_max < right_line - buffer:
            answer = "B"
        elif x_min > right_line + buffer:
            answer = "C"
        else:
            # Large object spans multiple thirds – ambiguous
            return
        question_answer_pairs.append((question, answer))

class WidthVsHeight(Question):
    def __init__(
        self,
        threshold: float = 0.75,
        non_articulated_classes: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            question="Is the width of the {object_1} appear to be larger than the height?",
            variables=["object_1"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )
        # ask recall. if object is detected, then ask for unique description
        if non_articulated_classes is not None and len(non_articulated_classes) == 0:
            raise ValueError(
                "non_articulated_classes must be a non-empty list of class names"
            )
        self.non_articulated_classes: Optional[list[str]] = non_articulated_classes
        self.threshold: float = threshold
        self.other_question: Optional[str] = (
            "Is the height of the {object_1} larger than the width?"
        )

    def __repr__(self):
        return f"Question: {self.question} (threshold: {self.threshold})"

    def _question_answer_ratio(
        self, class_name: str, ratio_wh: float, reverse: bool = False
    ) -> Optional[tuple[str, str]]:
        # Skip if near-square within threshold band
        if abs(ratio_wh - 1.0) < self.threshold:
            return None
        answer = "Yes" if ratio_wh > 1.0 else "No"
        if reverse:
            if self.other_question is not None:
                question = self.other_question.format(object_1=class_name)
                answer = "No" if answer == "Yes" else "Yes"
            else:
                return None
        else:
            question = self.question.format(object_1=class_name)
        return (question, answer)

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        reverse: bool = False,
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        qa: list[tuple[str, str]] = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if counts.get(key, 0) != 1:
                        continue
                    if (
                        self.non_articulated_classes is not None
                        and key not in self.non_articulated_classes
                    ):
                        continue
                    bbox = det.as_xyxy()[i]
                    w = float(bbox[2] - bbox[0])
                    h = float(bbox[3] - bbox[1])
                    ratio = w / max(h, 1e-6)
                    qa_pair = self._question_answer_ratio(key, ratio, reverse=reverse)
                    if qa_pair is not None:
                        qa.append(qa_pair)
            else:
                key = str(lbl)
                if counts.get(key, 0) != 1:
                    continue
                if (
                    self.non_articulated_classes is not None
                    and key not in self.non_articulated_classes
                ):
                    continue
                bbox = det.as_xyxy()[0]
                w = float(bbox[2] - bbox[0])
                h = float(bbox[3] - bbox[1])
                ratio = w / max(h, 1e-6)
                qa_pair = self._question_answer_ratio(key, ratio, reverse=reverse)
                if qa_pair is not None:
                    qa.append(qa_pair)
        return qa

class Quadrants(Question):
    def __init__(self, N: int, M: int, margin_ratio: float = 0.1) -> None:
        if N <= 0 or M <= 0:
            raise ValueError("N and M must be positive integers")
        if N * M > 12:
            raise ValueError("N * M must be less than or equal to 12")
        if margin_ratio < 0 or margin_ratio > 0.5:
            raise ValueError(
                "Margin ratio must be between 0 (no margin) and 0.5 (half the quadrant width/height)"
            )
        self.rows: int = N
        self.cols: int = M
        self.margin_ratio: float = margin_ratio
        super().__init__(
            question="Divide the image into a grid of {N} rows x {M} columns. Number the cells from left to right, then top to bottom, starting with 1. In what cell does the {object_1} appear?",
            variables=["object_1", "N", "M"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def _question_answer(
        self, image: Image.Image, class_name: str, bbox: torch.Tensor
    ) -> Optional[tuple[str, str]]:
        x_min, y_min, x_max, y_max = bbox
        detection_width = x_max - x_min
        detection_height = y_max - y_min

        image_width, image_height = image.size

        quadrant_width = image_width / self.cols
        quadrant_height = image_height / self.rows

        # Margin inside each quadrant that the bbox must fully respect
        margin_x = self.margin_ratio * quadrant_width
        margin_y = self.margin_ratio * quadrant_height

        # Require bbox to fit wholly inside a quadrant with the margin buffer
        if not (
            detection_width < quadrant_width - 2 * margin_x
            and detection_height < quadrant_height - 2 * margin_y
        ):
            return None

        # calculate the quadrant the object is in
        # if it is in multiple quadrants, ignore that object
        row = math.floor(float(y_min) / quadrant_height)
        if row != math.floor(float(y_max) / quadrant_height):
            logger.debug("Object spans multiple rows")
            return None
        col = math.floor(float(x_min) / quadrant_width)
        if col != math.floor(float(x_max) / quadrant_width):
            logger.debug("Object spans multiple columns")
            return None

        # Ensure bbox respects margin inside the identified quadrant
        if not (
            x_min >= col * quadrant_width + margin_x
            and x_max <= (col + 1) * quadrant_width - margin_x
            and y_min >= row * quadrant_height + margin_y
            and y_max <= (row + 1) * quadrant_height - margin_y
        ):
            logger.debug("Quadrants skipped due to margin ambiguity")
            return None

        quadrant = row * self.cols + col + 1

        question = self.question.format(
            object_1=class_name,
            N=self.rows,
            M=self.cols,
        )
        answer = str(quadrant)
        return (question, answer)

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        single_classes = {k for k, v in counts.items() if v == 1}

        question_answer_pairs = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if key not in single_classes:
                        continue
                    bbox = det.as_xyxy()[i]
                    qa = self._question_answer(image, key, bbox)
                    if qa is not None:
                        question_answer_pairs.append(qa)
            else:
                key = str(lbl)
                if key not in single_classes:
                    continue
                bbox = det.as_xyxy()[0]
                qa = self._question_answer(image, key, bbox)
                if qa is not None:
                    question_answer_pairs.append(qa)

        return question_answer_pairs

class LargestAppearance(Question):
    def __init__(self, threshold: float = 0.3) -> None:
        super().__init__(
            question="If you were to draw a tight box around each object in the image, which type of object would have the biggest box?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )
        # in the R.O.S. verifier, black out every single box then ask
        self.threshold = threshold

    def __repr__(self):
        return f"Question: {self.question} (threshold: {self.threshold})"

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 2) == True
        # Calculate areas for all detections
        areas = []
        labels = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    bbox = det.as_xyxy()[i]
                    area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    areas.append(area)
                    labels.append(str(l))
            else:
                bbox = det.as_xyxy()[0]
                area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                areas.append(area)
                labels.append(str(lbl))

        if len(areas) == 0:
            return []

        # Sort by area descending
        area_label_pairs = list(zip(areas, labels))
        area_label_pairs.sort(key=lambda x: x[0], reverse=True)

        if len(area_label_pairs) < 2:
            return []

        largest_area, largest_label = area_label_pairs[0]
        second_area, _ = area_label_pairs[1]

        if largest_area <= (1 + self.threshold) * second_area:
            return []

        return [(self.question, largest_label)]

class RankLargestK(Question):
    """Rank the *k* object classes that have the largest single-instance area.

    Example question (for k=3):

        "Rank the 3 kinds of objects that appear the largest in the image from
        largest to smallest. Provide your answer as a comma-separated list of
        object names only."
    """

    def __init__(self, k: int, margin_ratio: float = 0.3) -> None:
        """Create a RankLargestK question.

        Args:
            k: number of classes to rank.
            margin_ratio: required multiplicative margin between consecutive
                ranked areas. For class *i* to be considered larger than class
                *i+1*, its area must be at least ``(1 + margin_ratio)`` times
                larger. If any consecutive pair fails this criterion, the
                question will be skipped for that image.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if margin_ratio < 0:
            raise ValueError("margin_ratio must be non-negative")

        self.k: int = k
        self.margin_ratio: float = margin_ratio
        super().__init__(
            question=(
                "Rank the {k} kinds of objects that appear the largest (by pixel area) in the "
                "image from largest to smallest. Provide your answer as a "
                "comma-separated list of object names only."
            ),
            variables=["k"],
            predicates=[
                # Need at least k different classes detected
                lambda image, detections, k=k: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, k
                ),
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # Calculate max area per class
        class_max_area: dict[str, float] = {}

        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    bbox = det.as_xyxy()[i]
                    area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    class_max_area[key] = max(class_max_area.get(key, 0), area)
            else:
                key = str(lbl)
                bbox = det.as_xyxy()[0]
                area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                class_max_area[key] = max(class_max_area.get(key, 0), area)

        if len(class_max_area) < self.k:
            logger.debug("Not enough unique classes for RankLargestK question")
            return []

        # Sort classes by their largest instance area
        sorted_classes = sorted(
            class_max_area.items(), key=lambda item: item[1], reverse=True
        )

        # Verify margin criterion among top-k areas
        top_k = sorted_classes[: self.k]
        for i in range(len(top_k) - 1):
            area_i = top_k[i][1]
            area_next = top_k[i + 1][1]
            if area_i < (1 + self.margin_ratio) * area_next:
                logger.debug(
                    "RankLargestK margin threshold not met between %s and %s",
                    top_k[i][0],
                    top_k[i + 1][0],
                )
                return []

        top_k_labels = [cls for cls, _ in top_k]

        question = self.question.format(k=self.k)
        answer = ", ".join(map(str, top_k_labels))
        return [(question, answer)]

class MostAppearance(Question):
    def __init__(self, margin_ratio: float = 0.2) -> None:
        super().__init__(
            question="What kind of object appears the most frequently in the image?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )
        if margin_ratio < 0 or margin_ratio >= 1:
            raise ValueError(
                "The margin ratio between the classes that appear most frequently must be non-negative and less than 1"
            )
        self.margin_ratio: float = margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 2) == True
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        if len(counts) < 2:
            return []
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_count = sorted_counts[0][1]
        second_count = sorted_counts[1][1]
        if top_count < (1 + self.margin_ratio) * second_count:
            return []
        most = sorted_counts[0][0]
        return [(self.question, str(most))]

class LeastAppearance(Question):
    def __init__(self, margin_ratio: float = 0.2) -> None:
        super().__init__(
            question="What kind of object appears the least frequently in the image?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )
        if margin_ratio < 0 or margin_ratio >= 1:
            raise ValueError(
                "The margin ratio between the classes that appear least frequently must be non-negative and less than 1"
            )
        self.margin_ratio: float = margin_ratio

    def apply(
        self, image: Image.Image, detections: list[ObjectDetectionResultI]
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 2) == True
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        if len(counts) < 2:
            return []
        sorted_counts = sorted(counts.items(), key=lambda x: x[1])
        lowest = sorted_counts[0][1]
        second_lowest = sorted_counts[1][1]
        if second_lowest < (1 + self.margin_ratio) * lowest:
            return []
        least = sorted_counts[0][0]
        return [(self.question, str(least))]

class LeftOf(Question):
    def __init__(self) -> None:
        super().__init__(
            question="Is there at least one {object_1} to the left of any {object_2}?",
            variables=["object_1", "object_2"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
                ObjectDetectionPredicates.exists_non_overlapping_detections,
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 2) == True
        # @precondition: exists_non_overlapping_detections(image, detections) == True

        # Group detections by class
        class_detections: dict[str, list[tuple[ObjectDetectionResultI, int]]] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if key not in class_detections:
                        class_detections[key] = []
                    class_detections[key].append((det, i))
            else:
                key = str(lbl)
                if key not in class_detections:
                    class_detections[key] = []
                class_detections[key].append((det, 0))

        qa = []
        classes = list(class_detections.keys())
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i == j:
                    continue
                c1, c2 = classes[i], classes[j]
                found_yes = False

                for det1, idx1 in class_detections[c1]:
                    bbox1 = det1.as_xyxy()[idx1]
                    x2_1 = float(bbox1[2])  # right edge of object 1

                    for det2, idx2 in class_detections[c2]:
                        bbox2 = det2.as_xyxy()[idx2]
                        x1_2 = float(bbox2[0])  # left edge of object 2

                        if x2_1 < x1_2:  # object 1 is to the left of object 2
                            # Check non-overlap via IOU
                            # Create single detection objects for IOU check
                            single_det1 = ObjectDetectionResultI(
                                score=det1.score,
                                cls=det1.cls,
                                label=c1,
                                bbox=bbox1.unsqueeze(0),
                                image_hw=(image.height, image.width),
                            )
                            single_det2 = ObjectDetectionResultI(
                                score=det2.score,
                                cls=det2.cls,
                                label=c2,
                                bbox=bbox2.unsqueeze(0),
                                image_hw=(image.height, image.width),
                            )
                            if (
                                ObjectDetectionUtils.pairwise_iou(
                                    single_det1, single_det2
                                ).max()
                                == 0
                            ):
                                qa.append(
                                    (
                                        self.question.format(object_1=c1, object_2=c2),
                                        "Yes",
                                    )
                                )
                                found_yes = True
                                break
                    if found_yes:
                        break

                if not found_yes:
                    qa.append((self.question.format(object_1=c1, object_2=c2), "No"))
        return qa

class RightOf(Question):
    def __init__(self) -> None:
        super().__init__(
            question="Is there at least one {object_1} to the right of any {object_2}?",
            variables=["object_1", "object_2"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
                ObjectDetectionPredicates.exists_non_overlapping_detections,
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 2) == True
        # @precondition: exists_non_overlapping_detections(image, detections) == True

        # Group detections by class
        class_detections: dict[str, list[tuple[ObjectDetectionResultI, int]]] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if key not in class_detections:
                        class_detections[key] = []
                    class_detections[key].append((det, i))
            else:
                key = str(lbl)
                if key not in class_detections:
                    class_detections[key] = []
                class_detections[key].append((det, 0))

        qa = []
        classes = list(class_detections.keys())
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i == j:
                    continue
                c1, c2 = classes[i], classes[j]
                found_yes = False

                for det1, idx1 in class_detections[c1]:
                    bbox1 = det1.as_xyxy()[idx1]
                    x1_1 = float(bbox1[0])  # left edge of object 1

                    for det2, idx2 in class_detections[c2]:
                        bbox2 = det2.as_xyxy()[idx2]
                        x2_2 = float(bbox2[2])  # right edge of object 2

                        if x1_1 > x2_2:  # object 1 is to the right of object 2
                            # Check non-overlap via IOU
                            # Create single detection objects for IOU check
                            single_det1 = ObjectDetectionResultI(
                                score=det1.score,
                                cls=det1.cls,
                                label=c1,
                                bbox=bbox1.unsqueeze(0),
                                image_hw=(image.height, image.width),
                            )
                            single_det2 = ObjectDetectionResultI(
                                score=det2.score,
                                cls=det2.cls,
                                label=c2,
                                bbox=bbox2.unsqueeze(0),
                                image_hw=(image.height, image.width),
                            )
                            if (
                                ObjectDetectionUtils.pairwise_iou(
                                    single_det1, single_det2
                                ).max()
                                == 0
                            ):
                                qa.append(
                                    (
                                        self.question.format(object_1=c1, object_2=c2),
                                        "Yes",
                                    )
                                )
                                found_yes = True
                                break
                    if found_yes:
                        break

                if not found_yes:
                    qa.append((self.question.format(object_1=c1, object_2=c2), "No"))
        return qa


# One can image an AboveOf and BelowOf question as well
# However, these are actually not a good idea
# When you look at an image, what appears as a higher or lower
# y-coordinate may not necessarily translate to a higher or lower object
# This is especially true of perspective images (i.e. images taken from a distance)
# An object that is further away from the camera may appear at a higher
# y-coordinate than an object that is closer to the camera but they are
# in fact on the same plane


class LeftMost(Question):
    def __init__(self, margin_ratio: float = 0.05) -> None:
        super().__init__(
            question="What is the leftmost object in the image?",
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )
        if margin_ratio < 0 or margin_ratio > 1:
            raise ValueError("margin_ratio must be between 0 and 1")
        self.margin_ratio = margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True
        # Create list of all individual detections with their positions
        all_detections = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    bbox = det.as_xyxy()[i]
                    all_detections.append((bbox, str(l), det, i))
            else:
                bbox = det.as_xyxy()[0]
                all_detections.append((bbox, str(lbl), det, 0))

        if len(all_detections) == 0:
            return []

        # Sort by left edge (x1)
        all_detections.sort(key=lambda x: float(x[0][0]))

        if len(all_detections) < 2:
            # Single detection case: ensure it's on the left half fully
            bbox, label, _, _ = all_detections[0]
            x1, x2 = float(bbox[0]), float(bbox[2])
            if x1 < image.size[0] / 2 and x2 < image.size[0] / 2:
                return [(self.question, label)]
            return []

        # Check overlap between first two leftmost
        bbox1, label1, det1, idx1 = all_detections[0]
        bbox2, label2, det2, idx2 = all_detections[1]

        im_width = image.size[0]
        margin = self.margin_ratio * im_width
        right_edge_of_left_most = float(bbox1[2])
        left_edge_of_second_left_most = float(bbox2[0])
        overlap = right_edge_of_left_most + margin > left_edge_of_second_left_most
        if overlap:
            # Not enough horizontal gap – ambiguous
            return []

        # Ensure leftmost is on left half fully
        left_edge_of_left_most, right_edge_of_left_most = float(bbox1[0]), float(
            bbox1[2]
        )
        if not (
            left_edge_of_left_most < image.size[0] / 2
            and right_edge_of_left_most < image.size[0] / 2
        ):
            return []
        return [(self.question, label1)]

class RightMost(Question):
    def __init__(self, margin_ratio: float = 0.05) -> None:
        super().__init__(
            question="What is the rightmost object in the image?",
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )
        if margin_ratio < 0 or margin_ratio > 1:
            raise ValueError("margin_ratio must be between 0 and 1")
        self.margin_ratio = margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True
        # Create list of all individual detections with their positions
        all_detections = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    bbox = det.as_xyxy()[i]
                    all_detections.append((bbox, str(l), det, i))
            else:
                bbox = det.as_xyxy()[0]
                all_detections.append((bbox, str(lbl), det, 0))

        if len(all_detections) == 0:
            return []

        # Sort by right edge (x2) descending
        all_detections.sort(key=lambda x: float(x[0][2]), reverse=True)

        if len(all_detections) < 2:
            # Single detection case: ensure it's on the right half fully
            bbox, label, _, _ = all_detections[0]
            left_edge_of_right_most, right_edge_of_right_most = float(bbox[0]), float(
                bbox[2]
            )
            if (
                left_edge_of_right_most > image.size[0] / 2
                and right_edge_of_right_most > image.size[0] / 2
            ):
                return [(self.question, label)]
            return []

        # Check overlap between first two rightmost
        bbox1, label1, det1, idx1 = all_detections[0]
        bbox2, label2, det2, idx2 = all_detections[1]

        im_width = image.size[0]
        margin = self.margin_ratio * im_width
        left_edge_of_right_most = float(bbox1[0])
        right_edge_of_second_right_most = float(bbox2[2])
        overlap = left_edge_of_right_most - margin < right_edge_of_second_right_most
        if overlap:
            # Not enough horizontal gap – ambiguous
            return []

        # Ensure rightmost is on right half fully
        left_edge_of_right_most, right_edge_of_right_most = float(bbox1[0]), float(
            bbox1[2]
        )
        if not (
            left_edge_of_right_most > image.size[0] / 2
            and right_edge_of_right_most > image.size[0] / 2
        ):
            return []
        return [(self.question, label1)]

class HowMany(Question):
    # TODO: Create a version of this question that is multiple choice
    def __init__(self) -> None:
        super().__init__(
            question="How many {object_1}(s) are there in this image?",
            variables=["object_1"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 1
                ),
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 1) == True
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        return [
            (self.question.format(object_1=cls), str(cnt))
            for cls, cnt in counts.items()
        ]

class AreMore(Question):
    # TODO: Create a version of this question that is multiple choice
    def __init__(self, margin_ratio: float = 0.2) -> None:
        """AreMore question with margin-based count filtering.

        Args:
            margin_ratio: Required margin between counts. Only asks question if
                the larger count exceeds the smaller by at least this ratio.
                E.g., margin_ratio=0.2 means count_1 must be ≥ 1.2 * count_2.
        """
        super().__init__(
            question="Are there more {object_1}(s) than {object_2}(s) in this image?",
            variables=["object_1", "object_2"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )
        if margin_ratio < 0 or margin_ratio > 1:
            raise ValueError("margin_ratio must be between 0 and 1")
        self.margin_ratio = margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        classes = list(counts.keys())
        qa: list[tuple[str, str]] = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                o1, o2 = classes[i], classes[j]
                c1, c2 = counts[o1], counts[o2]
                if c1 > c2:
                    if c1 >= (1 + self.margin_ratio) * c2:
                        qa.append(
                            (self.question.format(object_1=o1, object_2=o2), "Yes")
                        )
                elif c2 > c1:
                    if c2 >= (1 + self.margin_ratio) * c1:
                        qa.append(
                            (self.question.format(object_1=o1, object_2=o2), "No")
                        )
        return qa

class WhichMore(Question):
    def __init__(self, margin_ratio: float = 0.2) -> None:
        """WhichMore question with margin-based count filtering.

        Args:
            margin_ratio: Required margin for clear winner. Only asks question if
                the winning count exceeds the second-highest by at least this ratio.
        """
        super().__init__(
            question="What appears the most in this image: {object_1}s, {object_2}s, or {object_3}s?",
            variables=["object_1", "object_2", "objejct_3"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )
        if margin_ratio < 0 or margin_ratio > 1:
            raise ValueError("margin_ratio must be between 0 and 1")
        self.margin_ratio = margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:

        detection_counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:
                for single_class_name in class_name:
                    detection_counts[single_class_name] = (
                        detection_counts.get(single_class_name, 0) + 1
                    )
            else:
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
        question_answer_pairs = []
        detected_classes = list(detection_counts.keys())

        for i in range(len(detected_classes)):
            for j in range(i + 1, len(detected_classes)):
                for k in range(j + 1, len(detected_classes)):
                    object_1, object_2, object_3 = (
                        detected_classes[i],
                        detected_classes[j],
                        detected_classes[k],
                    )
                    count_1, count_2, count_3 = (
                        detection_counts[object_1],
                        detection_counts[object_2],
                        detection_counts[object_3],
                    )

                    max_count = max(count_1, count_2, count_3)
                    # Sort counts to find second highest
                    sorted_counts = sorted([count_1, count_2, count_3], reverse=True)
                    second_highest_count = sorted_counts[1]

                    # Check if winner has significant margin over second place
                    if max_count < (1 + self.margin_ratio) * second_highest_count:
                        # Winner not clear enough - skip question
                        continue

                    max_objects = []
                    if count_1 == max_count:
                        max_objects.append(object_1)
                    if count_2 == max_count:
                        max_objects.append(object_2)
                    if count_3 == max_count:
                        max_objects.append(object_3)

                    if len(max_objects) == 1:
                        answer = max_objects[0]
                        question_answer_pairs.append(
                            (
                                self.question.format(
                                    object_1=object_1,
                                    object_2=object_2,
                                    object_3=object_3,
                                ),
                                answer + "s",
                            )
                        )
        return question_answer_pairs

class LeftMostWidthVsHeight(WidthVsHeight):
    def __init__(
        self, threshold: float = 0.75, spatial_margin_ratio: float = 0.05
    ) -> None:
        """LeftMostWidthVsHeight with spatial stability checks.

        Args:
            threshold: Aspect ratio threshold
            spatial_margin_ratio: Required spatial separation as fraction of image width.
                The leftmost object must be separated from the second-leftmost by at least
                this margin to ensure stable positioning.
        """
        super().__init__(threshold=threshold)
        self.question = (
            "Does the leftmost object in the image appear to be wider than it is tall?"
        )
        self.other_question = (
            "Does the leftmost object in the image appear to be taller than it is wide?"
        )
        if spatial_margin_ratio < 0 or spatial_margin_ratio > 1:
            raise ValueError("spatial_margin_ratio must be between 0 and 1")
        self.spatial_margin_ratio = spatial_margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        reverse: bool = False,
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        # Create list of all individual detections with their positions, filtered for single instances
        all_detections = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if counts.get(key, 0) == 1:
                        bbox = det.as_xyxy()[i]
                        all_detections.append((bbox, key, det, i))
            else:
                key = str(lbl)
                if counts.get(key, 0) == 1:
                    bbox = det.as_xyxy()[0]
                    all_detections.append((bbox, key, det, 0))

        # Sort by left edge (x1)
        all_detections.sort(key=lambda x: float(x[0][0]))

        im_width, _ = image.size

        for pos, (bbox, label, det, idx) in enumerate(all_detections):
            x1, x2 = float(bbox[0]), float(bbox[2])
            if not (x1 < im_width / 2 and x2 < im_width / 2):
                continue  # Must be in left half

            # Check spatial separation if there's a second leftmost
            if pos + 1 < len(all_detections):
                second_bbox, second_label, second_det, second_idx = all_detections[
                    pos + 1
                ]
                second_x1 = float(second_bbox[0])
                required_margin = self.spatial_margin_ratio * im_width
                if (second_x1 - x2) < required_margin:
                    continue  # Not enough separation

                # Check for overlap
                single_det1 = ObjectDetectionResultI(
                    score=(
                        det.score[idx]
                        if isinstance(det.score, torch.Tensor)
                        else det.score
                    ),
                    cls=det.cls[idx] if isinstance(det.cls, torch.Tensor) else det.cls,
                    label=label,
                    bbox=bbox.unsqueeze(0),  # Ensure 2D shape (1, 4)
                    image_hw=(image.height, image.width),
                )
                single_det2 = ObjectDetectionResultI(
                    score=(
                        second_det.score[second_idx]
                        if isinstance(second_det.score, torch.Tensor)
                        else second_det.score
                    ),
                    cls=(
                        second_det.cls[second_idx]
                        if isinstance(second_det.cls, torch.Tensor)
                        else second_det.cls
                    ),
                    label=second_label,
                    bbox=second_bbox.unsqueeze(0),  # Ensure 2D shape (1, 4)
                    image_hw=(image.height, image.width),
                )

                if (
                    ObjectDetectionUtils.pairwise_iou(single_det1, single_det2).max()
                    > 0
                ):
                    logger.debug("Leftmost object overlaps with second-leftmost object")
                    continue

            # Calculate aspect ratio
            w = float(bbox[2] - bbox[0])
            h = float(bbox[3] - bbox[1])
            ratio = w / max(h, 1e-6)

            qa = self._question_answer_ratio(label, ratio, reverse=reverse)
            return [qa] if qa is not None else []

        return []

class RightMostWidthVsHeight(WidthVsHeight):
    def __init__(
        self, threshold: float = 0.75, spatial_margin_ratio: float = 0.05
    ) -> None:
        """RightMostWidthVsHeight with spatial stability checks.

        Args:
            threshold: Aspect ratio threshold (inherited from WidthVsHeight)
            spatial_margin_ratio: Required spatial separation as fraction of image width.
                The rightmost object must be separated from the second-rightmost by at least
                this margin to ensure stable positioning.
        """
        super().__init__(threshold=threshold)
        self.question = (
            "Does the rightmost object in the image appear to be wider than it is tall?"
        )
        self.other_question = "Does the rightmost object in the image appear to be taller than it is wide?"
        if spatial_margin_ratio < 0 or spatial_margin_ratio > 1:
            raise ValueError("spatial_margin_ratio must be between 0 and 1")
        self.spatial_margin_ratio = spatial_margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        reverse: bool = False,
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        # Create list of all individual detections with their positions, filtered for single instances
        all_detections = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if counts.get(key, 0) == 1:
                        bbox = det.as_xyxy()[i]
                        all_detections.append((bbox, key, det, i))
            else:
                key = str(lbl)
                if counts.get(key, 0) == 1:
                    bbox = det.as_xyxy()[0]
                    all_detections.append((bbox, key, det, 0))

        # Sort by right edge (x2) descending
        all_detections.sort(key=lambda x: float(x[0][2]), reverse=True)

        im_width, _ = image.size

        for pos, (bbox, label, det, idx) in enumerate(all_detections):
            x1, x2 = float(bbox[0]), float(bbox[2])
            if not (x1 > im_width / 2 and x2 > im_width / 2):
                continue  # Must be in right half

            # Check spatial separation if there's a second rightmost
            if pos + 1 < len(all_detections):
                second_bbox, second_label, second_det, second_idx = all_detections[
                    pos + 1
                ]
                second_x2 = float(second_bbox[2])
                required_margin = self.spatial_margin_ratio * im_width
                if (x1 - second_x2) < required_margin:
                    continue  # Not enough separation

                # Check for overlap
                single_det1 = ObjectDetectionResultI(
                    score=(
                        det.score[idx]
                        if isinstance(det.score, torch.Tensor)
                        else det.score
                    ),
                    cls=det.cls[idx] if isinstance(det.cls, torch.Tensor) else det.cls,
                    label=label,
                    bbox=bbox.unsqueeze(0),  # Ensure 2D shape (1, 4)
                    image_hw=(image.height, image.width),
                )
                single_det2 = ObjectDetectionResultI(
                    score=(
                        second_det.score[second_idx]
                        if isinstance(second_det.score, torch.Tensor)
                        else second_det.score
                    ),
                    cls=(
                        second_det.cls[second_idx]
                        if isinstance(second_det.cls, torch.Tensor)
                        else second_det.cls
                    ),
                    label=second_label,
                    bbox=second_bbox.unsqueeze(0),  # Ensure 2D shape (1, 4)
                    image_hw=(image.height, image.width),
                )

                if (
                    ObjectDetectionUtils.pairwise_iou(single_det1, single_det2).max()
                    > 0
                ):
                    logger.debug(
                        "Rightmost object overlaps with second-rightmost object"
                    )
                    continue

            # Calculate aspect ratio
            w = float(bbox[2] - bbox[0])
            h = float(bbox[3] - bbox[1])
            ratio = w / max(h, 1e-6)

            qa = self._question_answer_ratio(label, ratio, reverse=reverse)
            return [qa] if qa is not None else []

        return []

class MoreThanThresholdHowMany(Question):
    """More-than count question with built-in Yes/No balance.

    For each detected object class with count *N* we generate two prompts:

    1. *Yes case*   – target = ⌊N / threshold⌋.
       The detector's count is safely above the target, so the correct answer is **Yes**.

    2. *No case*    – target = ⌈N × threshold⌉.
       The detector's count is well below the target, so the correct answer is **No**.

    The gap created by the multiplicative buffer acts as a hedge against recall / precision noise
    while keeping the overall Yes/No distribution roughly balanced.
    """

    def __init__(self, threshold: float = 2.0):
        if threshold <= 1.0:
            raise ValueError("threshold should be > 1.0 for 'more than' questions")

        self.threshold: float = threshold
        super().__init__(
            question="Are there {target} or more {object_1}(s) in this image? Respond Yes/No.",
            variables=["object_1", "target"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 1
                ),
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:

        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    counts[str(l)] = counts.get(str(l), 0) + 1
            else:
                counts[str(lbl)] = counts.get(str(lbl), 0) + 1

        qa_pairs: list[tuple[str, str]] = []
        for cls, n in counts.items():
            if n == 0:
                continue

            # Question that should be answered "Yes" (target below n)
            target_yes = max(1, math.floor(n / self.threshold))
            if target_yes == n:
                target_yes = max(1, target_yes - 1)

            q_yes = self.question.format(object_1=cls, target=target_yes)
            qa_pairs.append((q_yes, "Yes"))

            # Question that should be answered "No" (target well above n)
            target_no = math.ceil(n * self.threshold)
            if target_no == n:
                target_no += 1

            q_no = self.question.format(object_1=cls, target=target_no)
            qa_pairs.append((q_no, "No"))

        return qa_pairs

class LessThanThresholdHowMany(Question):
    """Less-than count question with symmetric Yes/No balance.

    For detected count *N* we generate:

    1. *Yes case* – target = ⌈N / threshold⌉ (> N), so the answer **Yes** is correct.
    2. *No case*  – target = ⌊N × threshold⌋ (< N), so **No** is correct.

    This mirrors the more-than version and maintains balanced answer keys while
    providing a tolerance band for detector errors.
    """

    def __init__(self, threshold: float = 0.5):
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be between 0 and 1 for 'less than'")

        self.threshold: float = threshold
        super().__init__(
            question="Are there less than {target} {object_1}(s) in this image? Respond Yes/No.",
            variables=["object_1", "target"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 1
                ),
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:

        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    counts[str(l)] = counts.get(str(l), 0) + 1
            else:
                counts[str(lbl)] = counts.get(str(lbl), 0) + 1

        qa_pairs: list[tuple[str, str]] = []
        for cls, n in counts.items():
            if n == 0:
                continue

            # Question that should be answered "Yes" (target above n)
            target_yes = math.ceil(n / self.threshold)
            if target_yes == n:
                target_yes += 1

            q_yes = self.question.format(object_1=cls, target=target_yes)
            qa_pairs.append((q_yes, "Yes"))

            # Question that should be answered "No" (target well below n)
            target_no = max(1, math.floor(n * self.threshold))
            if target_no == n:
                target_no = max(1, target_no - 1)

            # If target is 1, switch to grammatically correct presence question
            if target_no == 1:
                q_no = (
                    "Are there any {object_1}(s) in this image? Respond Yes/No."
                ).format(object_1=cls)
                # Since n > 0 (we skipped n == 0 above), the correct answer is "Yes"
                qa_pairs.append((q_no, "Yes"))
            else:
                q_no = self.question.format(object_1=cls, target=target_no)
                qa_pairs.append((q_no, "No"))

        return qa_pairs

class MultiChoiceHowMany(Question):
    """Noise-tolerant *How Many* as a 3-way multiple-choice question.

    Workflow per detected object class with count *N*:

    1.  Build **contiguous** numeric buckets based on *N* (and confidence variance):
        • *low*  :   `0 – ⌊α · N⌋`
        • *mid*  :   `⌈α · N⌉ – ⌊β · N⌋`
        • *high* :   `⌈β · N⌉ – ⌈β · N⌋+w`  (finite width so all three look alike)
       where `(α, β) = (0.5, 1.5)` by default or `(0.4, 1.8)` when per-class
       confidence variance > 0.05, and *w* equals the width of the mid bucket.

    2.  Randomly **shuffle** which bucket is labelled A, B, or C.  This removes
        the positional/letter bias while the LLM still sees all ranges.

    3.  The correct answer letter is determined after the shuffle so that the
        dataset remains balanced across A/B/C over time.

    4.  A fourth option **D) Unsure / Not Visible** is always listed to allow a
        graceful fallback when the model feels uncertain.

    Questions are only generated when `N ≥ 4`; for very small counts, the
    buckets become too narrow to be useful.
    """

    def __init__(self):
        super().__init__(
            question="How many {object_1}(s) are in the image? Choose one: "
            "A) {range_a}, B) {range_b}, C) {range_c}, D) Unsure / Not Visible. "
            "Respond with the letter only.",
            variables=["object_1", "range_a", "range_b", "range_c"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 1
                ),
            ],
        )

    def _bucket_ranges(self, n: int, var: float) -> tuple[dict[str, str], str]:
        """Return bucket description dict and the *semantic* correct bucket key.

        Keys: "low", "mid", "high" → string description "x–y" (inclusive).
        Also returns which *bucket key* contains ``n`` so we can map it to the
        shuffled letter later.
        """

        # Variance-based adjustment of coefficients
        low_coef, mid_high_coef = (0.4, 1.8) if var > 0.05 else (0.5, 1.5)

        # Bucket boundaries (inclusive)
        low_max = max(0, int((low_coef * n) - 1e-6))
        mid_min = low_max + 1
        mid_max = int(mid_high_coef * n)
        high_min = mid_max + 1

        # Make the high bucket a finite *range* with similar width to mid bucket
        mid_width = mid_max - mid_min
        high_max = high_min + max(2, mid_width)  # ensure non-zero width

        buckets: dict[str, str] = {
            "low": f"0-{low_max}" if low_max > 0 else "0-{mid_min-1}",
            "mid": f"{mid_min}-{mid_max}",
            "high": f"{high_min}-{high_max}",
        }

        # With fixed α/β the detected count N always lands in the mid bucket,
        # so we can simply hard-code it instead of checking.
        correct_bucket = "mid"

        return buckets, correct_bucket

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:

        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    counts[str(l)] = counts.get(str(l), 0) + 1
            else:
                counts[str(lbl)] = counts.get(str(lbl), 0) + 1

        qa_pairs: list[tuple[str, str]] = []
        for cls, n in counts.items():
            if n < 4:
                continue
            # extract per-detection confidences for this class
            scores: list[float] = []
            for det in detections:
                lbl = det.label
                conf = getattr(det, "score", getattr(det, "confidence", 1.0))
                if isinstance(lbl, torch.Tensor):
                    for idx in range(lbl.shape[0]):
                        if str(lbl[idx]) == cls:
                            scores.append(
                                float(conf[idx])
                                if isinstance(conf, torch.Tensor)
                                else float(conf)
                            )
                else:
                    if str(lbl) == cls:
                        scores.append(float(conf))

            var = float(np.var(scores)) if len(scores) > 1 else 0.0

            buckets, correct_bucket = self._bucket_ranges(n, var)

            # Randomly permute letter → bucket mapping to avoid letter bias
            letters = ["A", "B", "C"]
            random.shuffle(letters)
            bucket_keys = ["low", "mid", "high"]

            letter_to_bucket = {
                letter: bucket for letter, bucket in zip(letters, bucket_keys)
            }

            # Build question text in A/B/C order after permutation
            q = self.question.format(
                object_1=cls,
                range_a=buckets[letter_to_bucket["A"].lower()],
                range_b=buckets[letter_to_bucket["B"].lower()],
                range_c=buckets[letter_to_bucket["C"].lower()],
            )

            # Identify the letter assigned to the correct bucket
            correct_letter = {bkey: ltr for ltr, bkey in letter_to_bucket.items()}[
                correct_bucket
            ]

            qa_pairs.append((q, correct_letter))

        return qa_pairs

class Closer(Question):
    def __init__(self, margin_ratio: float = 0.1) -> None:
        """
        Closer question using depth perception and SAM segmentation.

        Args:
            margin_ratio: Required relative depth difference for reliable comparison.
                Objects must differ by at least this fraction of the closer object's depth.
        """
        super().__init__(
            question="Is there at least one {object_1} that appears closer to the camera than any {object_2}?",
            variables=["object_1", "object_2"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
                ObjectDetectionPredicates.exists_non_overlapping_detections,
            ],
        )
        if margin_ratio <= 0 or margin_ratio >= 1:
            raise ValueError("margin_ratio must be between 0 and 1")
        self.margin_ratio = margin_ratio

        # Initialize SAM and DepthPro models lazily
        self._sam_predictor = None
        self._depth_model = None

    def _get_sam_predictor(self, cache: Optional[dict[str, Any]] = None):
        """Get SAM predictor, preferring shared cache if available."""
        if cache and "sam_predictor" in cache:
            return cache["sam_predictor"]   
        if self._sam_predictor is None:
            from graid.utilities.sam_utils import SAMPredictor
            from graid.utilities.common import get_default_device            
            device = get_default_device()
            self._sam_predictor = SAMPredictor(device=device)
        return self._sam_predictor

    def _get_depth_model(self):
        """Lazy initialization of DepthPro model."""
        if self._depth_model is None:
            from graid.models.DepthPro import DepthPro
            from graid.utilities.common import get_default_device
            device = get_default_device()
            self._depth_model = DepthPro(device=device)
        return self._depth_model

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        return self.apply_with_cache(image, detections, {})

    # New method using shared cache to reuse precomputed depth maps
    def apply_with_cache(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        cache: dict,
    ) -> list[tuple[str, str]]:
        import time
        from graid.utilities.sam_utils import compare_object_depths

        start_time = time.time()
        depth_model = cache.get("depth_model") if cache else None
        if "depth_map" not in cache:
            if depth_model is None:
                depth_model = self._get_depth_model()
            try:
                dm = depth_model.predict_depth(image).depth_prediction
                cache["depth_map"] = dm
            except Exception as e:
                logger.debug(f"Closer depth calculation failed: {e}")
                return []
        depth_map = cache["depth_map"]
        sam_predictor = self._get_sam_predictor(cache)

        # 1. Group detections by class and build a flat list
        class_detections: dict[str, list[ObjectDetectionResultI]] = {}
        for det in detections:
            key = str(det.label)
            class_detections.setdefault(key, []).append(det)

        # 2. Build a mask cache using the batched SAM helper
        mask_cache: dict[int, Optional[torch.Tensor]] = cache.get("sam_masks", {})
        missing = [d for d in detections if id(d) not in mask_cache]
        if missing:
            try:
                new_masks = sam_predictor.get_masks_from_bboxes(image, detections)
            except Exception:
                return []
            if len(new_masks) != len(detections):
                return []
            for d, m in new_masks:
                mask_cache[id(d)] = m
            cache["sam_masks"] = mask_cache

        qa_pairs = []
        classes = list(class_detections.keys())
        main_loop_start_time = time.time()

        for i in range(len(classes)):
            for j in range(len(classes)):
                if i == j:
                                continue
                c1, c2 = classes[i], classes[j]

                found_yes = False
                for det1 in class_detections[c1]:
                    for det2 in class_detections[c2]:
                        if ObjectDetectionUtils.pairwise_iou(det1, det2).max() > 0:
                            continue

                        mask1 = mask_cache.get(id(det1))
                        mask2 = mask_cache.get(id(det2))
                        if mask1 is None or mask2 is None:
                            continue

                        comparison, _, _ = compare_object_depths(
                            depth_map, det1, mask1, det2, mask2, self.margin_ratio
                        )

                        if comparison == "object1_front":
                            qa_pairs.append(
                                (self.question.format(object_1=c1, object_2=c2), "Yes")
                            )
                            found_yes = True
                            break
                    if found_yes:
                        break

                if not found_yes:
                    qa_pairs.append(
                        (self.question.format(object_1=c1, object_2=c2), "No")
                    )

        logger.debug(f"Closer: Main question loop took {time.time() - main_loop_start_time:.4f}s")
        logger.debug(f"Closer: Total apply_with_cache took {time.time() - start_time:.4f}s")
        return qa_pairs

class Farther(Question):
    def __init__(self, margin_ratio: float = 0.1) -> None:
        """
        Farther question using depth perception and SAM segmentation.

        Args:
            margin_ratio: Required relative depth difference for reliable comparison.
        """
        super().__init__(
            question="Is there at least one {object_1} that appears farther from the camera than any {object_2}?",
            variables=["object_1", "object_2"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
                ObjectDetectionPredicates.exists_non_overlapping_detections,
            ],
        )
        if margin_ratio <= 0 or margin_ratio >= 1:
            raise ValueError("margin_ratio must be between 0 and 1")
        self.margin_ratio = margin_ratio

        # Initialize SAM and DepthPro models lazily
        self._sam_predictor = None
        self._depth_model = None

    def _get_sam_predictor(self):
        """Lazy initialization of SAM predictor."""
        if self._sam_predictor is None:
            from graid.utilities.sam_utils import SAMPredictor
            from graid.utilities.common import get_default_device
            device = get_default_device()
            self._sam_predictor = SAMPredictor(device=device)
        return self._sam_predictor

    def _get_depth_model(self):
        """Lazy initialization of DepthPro model."""
        if self._depth_model is None:
            from graid.models.DepthPro import DepthPro
            from graid.utilities.common import get_default_device
            device = get_default_device()
            self._depth_model = DepthPro(device=device)
        return self._depth_model

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        return self.apply_with_cache(image, detections, {})

    def apply_with_cache(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        cache: dict,
    ) -> list[tuple[str, str]]:
        import time
        from graid.utilities.sam_utils import compare_object_depths

        start_time = time.time()
        depth_model = cache.get("depth_model") if cache else None
        if "depth_map" not in cache:
            if depth_model is None:
                depth_model = self._get_depth_model()
            try:
                dm = depth_model.predict_depth(image).depth_prediction
                cache["depth_map"] = dm
            except Exception as e:
                logger.debug(f"Farther depth calculation failed: {e}")
                return []
        depth_map = cache["depth_map"]
        

        sam_predictor = self._get_sam_predictor()
        

        # Group detections by class and flat list
        class_detections: dict[str, list[ObjectDetectionResultI]] = {}
        for det in detections:
            for d_single in det.flatten():
                key = str(d_single.label)
                class_detections.setdefault(key, []).append(d_single)

        # Build mask cache via batched SAM
        # Reuse per-image SAM mask cache if present
        mask_cache: dict[int, Optional[torch.Tensor]] = cache.get("sam_masks", {})
        missing = [d for d in detections if id(d) not in mask_cache]
        if missing:
            try:
                new_masks = sam_predictor.get_masks_from_bboxes(image, detections)
            except Exception:
                return []
            if len(new_masks) != len(detections):
                return []
            for d, m in new_masks:
                mask_cache[id(d)] = m
            cache["sam_masks"] = mask_cache

        qa_pairs = []
        classes = list(class_detections.keys())
        main_loop_start_time = time.time()

        for i in range(len(classes)):
            for j in range(len(classes)):
                if i == j:
                    continue
                c1, c2 = classes[i], classes[j]

                found_yes = False
                for det1 in class_detections[c1]:
                    for det2 in class_detections[c2]:
                        if ObjectDetectionUtils.pairwise_iou(det1, det2).max() > 0:
                            continue

                        mask1 = mask_cache.get(id(det1))
                        mask2 = mask_cache.get(id(det2))
                        if mask1 is None or mask2 is None:
                            continue

                        comparison, _, _ = compare_object_depths(
                            depth_map, det1, mask1, det2, mask2, self.margin_ratio
                        )

                        if comparison == "object2_front":
                            qa_pairs.append(
                                (self.question.format(object_1=c1, object_2=c2), "Yes")
                            )
                            found_yes = True
                            break
                    if found_yes:
                        break

                if not found_yes:
                    qa_pairs.append(
                        (self.question.format(object_1=c1, object_2=c2), "No")
                    )

        logger.debug(f"Farther: Main question loop took {time.time() - main_loop_start_time:.4f}s")
        logger.debug(f"Farther: Total apply_with_cache took {time.time() - start_time:.4f}s")
        return qa_pairs

class DepthRanking(Question):
    """Rank the *k* object classes that are closest to the camera.

    Example question (for k=3):

        "Rank the 3 kinds of objects that appear the closest in the image from
        closest to farthest. Provide your answer as a comma-separated list of
        object names only."
    """

    def __init__(self, k: int, margin_ratio: float = 0.2) -> None:
        """Create a DepthRanking question.

        Args:
            k: number of classes to rank.
            margin_ratio: required multiplicative margin between consecutive
                ranked depths. For class *i* to be considered closer than class
                *i+1*, its depth must be at most `(1 - margin_ratio)` times
                the depth of i+1. If any consecutive pair fails this criterion, the
                question will be skipped for that image.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if not (0 < margin_ratio < 1):
            raise ValueError("margin_ratio must be between 0 and 1")

        self.k: int = k
        self.margin_ratio: float = margin_ratio
        super().__init__(
            question=(
                "Rank the {k} kinds of objects that appear the closest to the camera in the "
                "image from closest to farthest. Provide your answer as a "
                "comma-separated list of object names only."
            ),
            variables=["k"],
            predicates=[
                # Need at least k different classes detected
                lambda image, detections, k=k: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, k
                ),
            ],
        )

        # Initialize SAM and DepthPro models lazily
        self._sam_predictor = None
        self._depth_model = None

    def _get_sam_predictor(self):
        """Lazy initialization of SAM predictor."""
        if self._sam_predictor is None:
            from graid.utilities.sam_utils import SAMPredictor
            from graid.utilities.common import get_default_device
            device = get_default_device()
            self._sam_predictor = SAMPredictor(device=device)
        return self._sam_predictor

    def _get_depth_model(self):
        """Lazy initialization of DepthPro model."""
        if self._depth_model is None:
            from graid.models.DepthPro import DepthPro
            from graid.utilities.common import get_default_device
            device = get_default_device()
            self._depth_model = DepthPro(device=device)
        return self._depth_model

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        if len(detections) == 0:
            logger.debug("No detections for DepthRanking question")
            return []
        return self.apply_with_cache(image, detections, {})

    def apply_with_cache(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        cache: dict,
    ) -> list[tuple[str, str]]:
        depth_model = cache.get("depth_model") if cache else None
        if "depth_map" not in cache:
            if depth_model is None:
                depth_model = self._get_depth_model()
            try:
                cache["depth_map"] = depth_model.predict_depth(image).depth_prediction
            except Exception as e:
                logger.debug(f"DepthRanking depth calc failed: {e}")
                return []
        depth_map = cache["depth_map"]

        from graid.utilities.sam_utils import extract_average_depth_from_mask

        sam_predictor = self._get_sam_predictor()

        try:
            if len(detections) < self.k:
                return []

            # Use batched SAM to get masks for all detections
            # Reuse per-image SAM mask cache if available
            mask_cache: dict[int, Optional[torch.Tensor]] = cache.get("sam_masks", {})
            missing = [d for d in detections if id(d) not in mask_cache]
            if missing:
                for d, m in sam_predictor.get_masks_from_bboxes(image, detections):
                    mask_cache[id(d)] = m
                cache["sam_masks"] = mask_cache

            # Compute refined depth per detection
            class_min_depth: dict[str, float] = {}
            for det in detections:
                mask = mask_cache.get(id(det))
                if mask is None:
                    continue
                avg_depth = extract_average_depth_from_mask(depth_map, mask)
                if avg_depth is None:
                    continue
                cls = str(det.label)
                if cls not in class_min_depth or avg_depth < class_min_depth[cls]:
                    class_min_depth[cls] = avg_depth

            if len(class_min_depth) < self.k:
                logger.debug("Not enough classes with valid depth for DepthRanking")
                return []

            sorted_classes = sorted(class_min_depth.items(), key=lambda kv: kv[1])
            top_k = sorted_classes[: self.k]

            # margin check
            for i in range(len(top_k) - 1):
                if top_k[i][1] > (1 - self.margin_ratio) * top_k[i + 1][1]:
                    logger.debug("DepthRanking margin threshold not met between %s and %s", top_k[i][0], top_k[i + 1][0])
                    return []

            labels_ordered = [cls for cls, _ in top_k]
            return [(self.question.format(k=self.k), ", ".join(labels_ordered))]
        except Exception as e:
            logger.debug(f"DepthRanking failed with cache: {e}")
            return []

class ObjectsInRow(Question):
    def __init__(self, variance_threshold: float = 0.1) -> None:
        """Linear regression-based row detection.

        Args:
            variance_threshold: Maximum normalized variance for y-centers to be
                considered in a row. Lower values = stricter row detection.
        """
        super().__init__(
            question="Are there any objects arranged in a row?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_detections(
                    image, detections, 3
                ),
            ],
        )
        self.variance_threshold = variance_threshold

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        from sklearn.linear_model import LinearRegression

        if len(detections) < 3:
            return [(self.question, "No")]

        # Get center points - normalized detections have bbox shape (1, 4)
        centers = []
        for detection in detections:
            bbox = detection.as_xyxy()[0]  # Shape (4,) after indexing
            x_center = float((bbox[0] + bbox[2]) / 2)
            y_center = float((bbox[1] + bbox[3]) / 2)
            centers.append((x_center, y_center))

        # Sort by x-coordinate
        centers = sorted(centers, key=lambda p: p[0])

        # Try sliding windows of 3+ objects
        image_height = image.size[1]

        for window_size in range(3, len(centers) + 1):
            for start in range(len(centers) - window_size + 1):
                window = centers[start : start + window_size]

                # Extract x and y coordinates
                x_coords = np.array([p[0] for p in window]).reshape(-1, 1)
                y_coords = np.array([p[1] for p in window])

                # Fit linear regression
                reg = LinearRegression().fit(x_coords, y_coords)
                y_pred = reg.predict(x_coords)

                # Calculate normalized variance (by image height)
                variance = np.var(y_coords - y_pred)
                normalized_variance = variance / (image_height**2)

                if normalized_variance < self.variance_threshold:
                    return [(self.question, "Yes")]

        return [(self.question, "No")]

class ObjectsInLine(Question):
    def __init__(self, variance_threshold: float = 0.1) -> None:
        """Multiple choice question about which objects are in a row.

        Args:
            variance_threshold: Same as ObjectsInRow for consistency.
        """
        super().__init__(
            question="Which objects appear to be arranged in a row? A) {option_a}, B) {option_b}, C) {option_c}, D) No clear row arrangement. Respond with the letter only.",
            variables=["option_a", "option_b", "option_c"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_detections(
                    image, detections, 3
                ),
            ],
        )
        self.variance_threshold = variance_threshold

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        from sklearn.linear_model import LinearRegression

        if len(detections) < 3:
            return []

        # Get centers with labels - normalized detections have bbox shape (1, 4)
        centers_with_labels = []
        for detection in detections:
            bbox = detection.as_xyxy()[0]  # Shape (4,) after indexing
            x_center = float((bbox[0] + bbox[2]) / 2)
            y_center = float((bbox[1] + bbox[3]) / 2)
            label = str(detection.label)
            centers_with_labels.append((x_center, y_center, label))

        # Sort by x-coordinate
        centers_with_labels = sorted(centers_with_labels, key=lambda p: p[0])

        # Find best row arrangement
        best_row = None
        best_variance = float("inf")
        image_height = image.size[1]

        for window_size in range(3, len(centers_with_labels) + 1):
            for start in range(len(centers_with_labels) - window_size + 1):
                window = centers_with_labels[start : start + window_size]

                x_coords = np.array([p[0] for p in window]).reshape(-1, 1)
                y_coords = np.array([p[1] for p in window])

                reg = LinearRegression().fit(x_coords, y_coords)
                y_pred = reg.predict(x_coords)

                variance = np.var(y_coords - y_pred)
                normalized_variance = variance / (image_height**2)

                if (
                    normalized_variance < self.variance_threshold
                    and normalized_variance < best_variance
                ):
                    best_variance = normalized_variance
                    best_row = [p[2] for p in window]  # Extract labels

        if best_row is None:
            return []  # No valid row found

        # Create multiple choice options
        correct_text = ", ".join(sorted(set(best_row)))

        # Generate distractors
        all_labels = [str(d.label) for d in detections]
        unique_labels = list(set(all_labels))
        random.shuffle(unique_labels)

        # Create distinct distractors
        distractor1 = ", ".join(unique_labels[: min(3, len(unique_labels))])
        distractor2 = ", ".join(unique_labels[-min(2, len(unique_labels)) :])

        # Ensure distractors are different from correct answer
        max_attempts = 10
        attempt = 0
        while (
            distractor1 == correct_text
            or distractor2 == correct_text
            or distractor1 == distractor2
        ) and attempt < max_attempts:
            random.shuffle(unique_labels)
            distractor1 = ", ".join(unique_labels[: min(3, len(unique_labels))])
            distractor2 = ", ".join(unique_labels[-min(2, len(unique_labels)) :])
            attempt += 1

        # If still duplicates, skip this question
        if (
            distractor1 == correct_text
            or distractor2 == correct_text
            or distractor1 == distractor2
        ):
            return []

        # Randomly assign correct answer to A, B, or C
        options = [correct_text, distractor1, distractor2]
        random.shuffle(options)
        correct_letter = ["A", "B", "C"][options.index(correct_text)]

        q = self.question.format(
            option_a=options[0], option_b=options[1], option_c=options[2]
        )

        return [(q, correct_letter)]

class MostClusteredObjects(Question):
    def __init__(self, eps_ratio: float = 0.05, min_samples: int = 3) -> None:
        """DBSCAN-based clustering with multiple choice answers.

        Args:
            eps_ratio: Maximum distance between points in a cluster as a fraction
                of the image diagonal. Default 0.05 means 5% of image diagonal.
            min_samples: Minimum points required to form a cluster.
        """
        super().__init__(
            question="Which group of objects appears most tightly clustered? A) {option_a}, B) {option_b}, C) {option_c}, D) No clear clusters. Respond with the letter only.",
            variables=["option_a", "option_b", "option_c"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_detections(
                    image, detections, 9  # Need at least 3 clusters × 3 objects each
                ),
            ],
        )
        self.eps_ratio = eps_ratio
        self.min_samples = min_samples

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        from sklearn.cluster import DBSCAN

        if len(detections) < 9:
            return []

        # Get centers and labels - normalized detections have bbox shape (1, 4)
        centers = []
        labels = []
        for detection in detections:
            bbox = detection.as_xyxy()[0]  # Shape (4,) after indexing
            x_center = float((bbox[0] + bbox[2]) / 2)
            y_center = float((bbox[1] + bbox[3]) / 2)
            centers.append([x_center, y_center])
            labels.append(str(detection.label))

        centers = np.array(centers)

        # Calculate eps as a fraction of image diagonal
        image_width, image_height = image.size
        image_diagonal = math.sqrt(image_width**2 + image_height**2)
        eps = self.eps_ratio * image_diagonal

        # Apply DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=self.min_samples).fit(centers)
        cluster_labels = clustering.labels_

        # Group objects by cluster (ignore noise points with label -1)
        clusters = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id != -1:  # Not noise
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(labels[i])

        if len(clusters) < 2:
            return []  # Need at least 2 clusters to compare

        # Find most compact cluster
        def cluster_compactness(cluster_id):
            cluster_points = centers[cluster_labels == cluster_id]
            if len(cluster_points) < 2:
                return float("inf")
            return np.mean(np.var(cluster_points, axis=0))

        most_compact_id = min(clusters.keys(), key=cluster_compactness)
        most_compact_objects = list(set(clusters[most_compact_id]))  # Remove duplicates

        # Create multiple choice options
        correct_text = ", ".join(sorted(most_compact_objects))

        # Generate distractors from other clusters or random combinations
        all_unique_labels = list(set(labels))
        random.shuffle(all_unique_labels)

        # Create distractors ensuring they're different from correct answer
        distractor1 = ", ".join(all_unique_labels[: min(3, len(all_unique_labels))])
        distractor2 = ", ".join(all_unique_labels[-min(2, len(all_unique_labels)) :])

        # Ensure distractors are different from correct answer
        max_attempts = 10
        attempt = 0
        while (
            distractor1 == correct_text
            or distractor2 == correct_text
            or distractor1 == distractor2
        ) and attempt < max_attempts:
            random.shuffle(all_unique_labels)
            distractor1 = ", ".join(all_unique_labels[: min(3, len(all_unique_labels))])
            distractor2 = ", ".join(
                all_unique_labels[-min(2, len(all_unique_labels)) :]
            )
            attempt += 1

        # If still duplicates after attempts, skip this question
        if (
            distractor1 == correct_text
            or distractor2 == correct_text
            or distractor1 == distractor2
        ):
            return []

        # Randomly assign correct answer
        options = [correct_text, distractor1, distractor2]
        random.shuffle(options)
        correct_letter = ["A", "B", "C"][options.index(correct_text)]

        q = self.question.format(
            option_a=options[0], option_b=options[1], option_c=options[2]
        )

        return [(q, correct_letter)]

# Dynamically discover all Question classes in this module
import inspect
import sys


def _build_all_questions():
    """Build ALL_QUESTIONS list by discovering all Question subclasses in this module."""
    current_module = sys.modules[__name__]
    question_classes = {}

    # Find all classes that inherit from Question
    for name, obj in inspect.getmembers(current_module, inspect.isclass):
        if (
            issubclass(obj, Question)
            and obj != Question  # Exclude the base class
            and hasattr(obj, "is_applicable")
        ):  # Ensure it's a concrete question class
            question_classes[name] = obj

    return question_classes


# Build the dictionary of available question classes
ALL_QUESTION_CLASSES = _build_all_questions()

# Keep the old ALL_QUESTIONS for backward compatibility, but it's no longer used
ALL_QUESTIONS = []

DEPTH_QUESTIONS = [
    Closer,
    Farther,
    DepthRanking,
]
