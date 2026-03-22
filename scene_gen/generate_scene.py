import random
import math
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont
from ultralytics import YOLO

# ========= TUNABLE PARAMETERS =========

SCRIPT_DIR = Path(__file__).resolve().parent

# Paths
IMG_DIR = SCRIPT_DIR / "src"            # Directory containing sprites
OUTPUT_DIR = SCRIPT_DIR / "output"      # Output directory
BACKGROUND_FILE_NAME = "background.jpeg"

# Base-scene sprite counts (placed once, fixed per base scene)
STOP_SIGN_COUNT_RANGE = (2, 3)      # 2-3 stop signs placed in base scene
BENCH_COUNT_RANGE     = (3, 5)      # 3-5 benches placed in base scene

# Per-variant counts (re-randomized for every variant)
ANIMALS_PER_STOP_RANGE  = (2, 5)    # 2-5 animals added around each stop sign per variant
PEOPLE_PER_BENCH_RANGE  = (0, 3)    # 0-3 people added in front of each bench per variant

# Radii of rings (in pixels)
ANIMAL_RING_RADIUS = 370               # Animals around stop sign
PEOPLE_RING_RADIUS = 300               # People around bench

# Overlap control on rings (closer to 1.0 = less overlap allowed)
ANIMAL_WIDTH_SCALE = 0.9
PEOPLE_WIDTH_SCALE = 0.9

# Max attempts to sample a valid angle on the ring
ANIMAL_MAX_TRIES = 100
PEOPLE_MAX_TRIES = 100

# Bench-front arc in radians (fractions of pi)
BENCH_ANGLE_MIN_FRAC = 0.2             # Left-front of bench
BENCH_ANGLE_MAX_FRAC = 0.8             # Right-front of bench

# Canvas margin (objects will stay inside [MARGIN, W-MARGIN])
MARGIN = 30

# Grid layout for placing sprites
GRID_ROWS = 3
GRID_COLS = 3
GRID_JITTER_SCALE = 0.25

# Base-scene sprite overlap check (small safety pad between sprite bboxes)
GROUP_PAD = 30
MAX_SCENE_PLACEMENT_TRIES = 30

# Sprite sizes (px) for boundary clamping so people don't go off-screen.
# person: (308, 574)  bench: (568, 338)
PERSON_SPRITE_W, PERSON_SPRITE_H = 308, 574
BENCH_SPRITE_W,  BENCH_SPRITE_H  = 568, 338
# Extra canvas margins for bench placement so that people on the arc stay on-screen:
#   X: RING * cos(arc_angle) + pw/2 - bw/2 ≈ 113 px
#   Y (bottom): RING + ph/2 - bh/2 = 300+287-169 = 418 px
BENCH_PLACE_X_PAD     = 113
BENCH_PLACE_Y_BOT_PAD = 418
# Extra canvas margin for stop sign placement so that animals on the full 360° ring
# stay on-screen.  Pad ≈ ANIMAL_RING_RADIUS + half of the widest/tallest animal sprite.
STOP_PLACE_PAD = ANIMAL_RING_RADIUS + 200   # 570 px on all four sides

# Max retries per variant slot to find a valid animal+people layout
MAX_VARIANT_TRIES = 5

# Number of base layouts and bus variants
NUM_BASE_SCENES = 420
BUS_VARIANTS_PER_SCENE = 25

# Bus augmentation parameters
BUS_SPRITE_FILE_NAME = "clock.png"
BUS_ROTATE = True
BUS_DIFF_THRESHOLD = 8
BUS_MAX_PLACEMENT_TRIES = 200
BUS_MIN_CENTER_DIST = 150.0

# Arrived-at placement: some variants intentionally place the bus near a target
# so the ArrivedAt questions have a healthy mix of Yes/No answers after resize.
ARRIVED_AT_PROB       = 0.40   # fraction of variants that try proximity placement
ARRIVED_DIST_MAX      = 190    # max pre-resize edge-to-edge gap (px)
ARRIVED_MAX_TRIES     = 60     # attempts before falling back to random placement
ARRIVED_JITTER        = 60     # ±px jitter along the axis parallel to the chosen side

# Heading dot (red circle drawn in front of the bus)
BUS_HEADING_DOT_OFFSET = 215
BUS_HEADING_DOT_RADIUS = 45
BUS_HEADING_DOT_COLOR = (220, 30, 30, 255)

# YOLO filtering
YOLO_WEIGHTS = SCRIPT_DIR / "yolo12s.pt"
YOLO_CONF = 0.20
YOLO_IMGSZ = 1280
YOLO_IOU_MATCH_THR = 0.30
OUTPUT_JPEG_QUALITY = 95

# =====================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BACKGROUND_FILE = IMG_DIR / BACKGROUND_FILE_NAME
BUS_FILE = IMG_DIR / BUS_SPRITE_FILE_NAME

SPRITE_FILES = {
    "bench":    IMG_DIR / "bench.png",
    "person":   IMG_DIR / "person.png",
    "stopSign": IMG_DIR / "stopSign.png",
    "zebra":    IMG_DIR / "zebra.png",
    "elephant": IMG_DIR / "elephant.png",
    "giraffe":  IMG_DIR / "giraffe.png",
}

ANIMAL_NAMES = ["zebra", "elephant", "giraffe"]


def load_sprites() -> Dict[str, Image.Image]:
    """Load all main sprites as RGBA images."""
    sprites: Dict[str, Image.Image] = {}
    for name, path in SPRITE_FILES.items():
        img = Image.open(path).convert("RGBA")
        sprites[name] = img
    return sprites


def paste_sprite(bg: Image.Image, sprite: Image.Image, x: int, y: int) -> None:
    """Paste sprite onto bg at (x, y) with alpha compositing, clipping at canvas edges."""
    w, h = sprite.size
    bg_w, bg_h = bg.size
    if x >= bg_w or y >= bg_h:
        return
    crop_x0 = max(0, -x)
    crop_y0 = max(0, -y)
    crop_x1 = min(w, bg_w - x)
    crop_y1 = min(h, bg_h - y)
    if crop_x0 >= crop_x1 or crop_y0 >= crop_y1:
        return
    cropped = sprite.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    bg.alpha_composite(cropped, (x + crop_x0, y + crop_y0))


# ========= Geometry helpers =========

def _rect_overlap(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> bool:
    """Return True if two (x0,y0,x1,y1) rectangles overlap (strictly)."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _bbox_edge_dist(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """Return axis-aligned edge-to-edge distance between two bboxes."""
    ax0, ay0, ax1, ay1 = map(float, a)
    bx0, by0, bx1, by1 = map(float, b)
    dx = max(0.0, max(bx0 - ax1, ax0 - bx1))
    dy = max(0.0, max(by0 - ay1, ay0 - by1))
    return math.hypot(dx, dy)


def _union_bbox(bboxes: List[Tuple[int,int,int,int]]) -> Tuple[int,int,int,int]:
    """Return the bounding box that covers all given bboxes."""
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)


def _any_sprite_overlap(
    placements: List[Tuple[int,int,int,int,str]],
    pad: int = GROUP_PAD,
) -> bool:
    """Return True if any two (x,y,gw,gh,kind) sprite bboxes overlap (with pad)."""
    expanded = [(x-pad, y-pad, x+gw+pad, y+gh+pad) for (x, y, gw, gh, _) in placements]
    for i in range(len(expanded)):
        for j in range(i+1, len(expanded)):
            if _rect_overlap(expanded[i], expanded[j]):
                return True
    return False


def _stop_cells_non_adjacent(
    cell_indices: List[int],
    candidate_items: List[Tuple[str, "Image.Image"]],
    cols: int = GRID_COLS,
) -> bool:
    """
    Return True if stop-sign grid cells are visually well separated.

    Constraints:
    - no two stop signs may be adjacent (8-connectivity)
    - no two stop signs may share the same grid column

    The unique-column rule matters because stop signs are labeled left-to-right
    in the final image. If two stop signs sit in the same column, tiny detector
    jitter can swap their apparent x-order and poison multiple ID-based QA pairs.
    """
    stop_cells = [
        ci for ci, (kind, _) in zip(cell_indices, candidate_items)
        if kind == "stop"
    ]
    stop_cols = set()
    for i in range(len(stop_cells)):
        r1, c1 = stop_cells[i] // cols, stop_cells[i] % cols
        if c1 in stop_cols:
            return False
        stop_cols.add(c1)
        for j in range(i + 1, len(stop_cells)):
            r2, c2 = stop_cells[j] // cols, stop_cells[j] % cols
            if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                return False
    return True


def _groups_valid(
    group_bbox_list: List[Tuple[int,int,int,int]],
    img_w: int,
    img_h: int,
    pad: int = 10,
) -> bool:
    """
    Return True iff all group bboxes are within canvas bounds and no two overlap.
    """
    for i, a in enumerate(group_bbox_list):
        # Bounds check
        if a[0] < MARGIN or a[1] < MARGIN or a[2] > img_w - MARGIN or a[3] > img_h - MARGIN:
            return False
        # Pairwise overlap check
        ea = (a[0]-pad, a[1]-pad, a[2]+pad, a[3]+pad)
        for j in range(i+1, len(group_bbox_list)):
            b = group_bbox_list[j]
            eb = (b[0]-pad, b[1]-pad, b[2]+pad, b[3]+pad)
            if _rect_overlap(ea, eb):
                return False
    return True


# ========= Per-variant sprite placement helpers =========

def _add_animals_around_stop(
    img: Image.Image,
    stop_bbox: Tuple[int,int,int,int],
    n_animals: int,
    sprites: Dict[str, Image.Image],
) -> Tuple[List[Tuple[str, int, int, int, int]], Image.Image]:
    """
    Place n_animals randomly on a ring of radius ANIMAL_RING_RADIUS around the
    stop sign center. Angular collision avoidance prevents heavy overlap on the ring.
    Returns (animal_entries, modified_img) where each entry is (species, x0, y0, x1, y1).
    If n_animals==0, returns ([], img) unchanged.
    """
    if n_animals == 0:
        return [], img

    out = img.copy()
    cx = (stop_bbox[0] + stop_bbox[2]) / 2.0
    cy = (stop_bbox[1] + stop_bbox[3]) / 2.0
    R = ANIMAL_RING_RADIUS

    placed: List[Tuple[float, int, int]] = []   # (angle, sprite_w, sprite_h)
    animal_entries: List[Tuple[str, int, int, int, int]] = []

    for _ in range(n_animals):
        name = random.choice(ANIMAL_NAMES)
        spr = sprites[name]
        w, h = spr.size
        half_angle_w = (ANIMAL_WIDTH_SCALE * w) / (2.0 * R)

        chosen: Optional[float] = None
        for _ in range(ANIMAL_MAX_TRIES):
            angle = random.uniform(0.0, 2.0 * math.pi)
            ok = True
            for a_prev, w_prev, _ in placed:
                half_prev = (ANIMAL_WIDTH_SCALE * w_prev) / (2.0 * R)
                dtheta = abs(angle - a_prev)
                dtheta = min(dtheta, 2.0 * math.pi - dtheta)
                if dtheta < (half_angle_w + half_prev):
                    ok = False
                    break
            if ok:
                chosen = angle
                break
        if chosen is None:
            chosen = random.uniform(0.0, 2.0 * math.pi)

        placed.append((chosen, w, h))
        x = int(cx + R * math.cos(chosen) - w / 2)
        y = int(cy + R * math.sin(chosen) + h / 2 - h)
        paste_sprite(out, spr, x, y)
        animal_entries.append((name, x, y, x + w, y + h))

    return animal_entries, out


def _add_people_around_bench(
    img: Image.Image,
    bench_bbox: Tuple[int,int,int,int],
    n_people: int,
    sprites: Dict[str, Image.Image],
    avoid_bboxes: Optional[List[Tuple[int,int,int,int]]] = None,
) -> Tuple[List[Tuple[int,int,int,int]], Image.Image]:
    """
    Place n_people on the front arc of the bench (angle range [pi*MIN, pi*MAX]).
    Angular collision avoidance prevents person-person overlap on the arc.
    Positions overlapping any bbox in avoid_bboxes (e.g. animal bboxes) are rejected.
    Returns (people_bboxes, modified_img). If n_people==0, returns ([], img) unchanged.
    """
    if n_people == 0:
        return [], img

    if avoid_bboxes is None:
        avoid_bboxes = []

    out = img.copy()
    person_spr = sprites["person"]
    pw, ph = person_spr.size
    cx = (bench_bbox[0] + bench_bbox[2]) / 2.0
    cy = (bench_bbox[1] + bench_bbox[3]) / 2.0
    R = PEOPLE_RING_RADIUS
    angle_min = math.pi * BENCH_ANGLE_MIN_FRAC
    angle_max = math.pi * BENCH_ANGLE_MAX_FRAC
    half_w = (PEOPLE_WIDTH_SCALE * pw) / (2.0 * R)

    placed_angles: List[float] = []
    people_bboxes: List[Tuple[int,int,int,int]] = []

    for _ in range(n_people):
        chosen: Optional[float] = None
        for _ in range(PEOPLE_MAX_TRIES):
            a = random.uniform(angle_min, angle_max)
            # Angular collision between people on same bench
            if not all(abs(a - p) >= 2 * half_w for p in placed_angles):
                continue
            # Pixel collision with avoid_bboxes (animals from all stop signs)
            px_try = int(cx + R * math.cos(a) - pw / 2)
            py_try = int(cy + R * math.sin(a) + ph / 2 - ph)
            person_box = (px_try, py_try, px_try + pw, py_try + ph)
            if any(_rect_overlap(person_box, ab) for ab in avoid_bboxes):
                continue
            chosen = a
            break
        if chosen is None:
            continue   # no valid angle found — skip this person slot

        placed_angles.append(chosen)
        px = int(cx + R * math.cos(chosen) - pw / 2)
        py = int(cy + R * math.sin(chosen) + ph / 2 - ph)
        paste_sprite(out, person_spr, px, py)
        people_bboxes.append((px, py, px + pw, py + ph))

    return people_bboxes, out


# ========= Base scene generation =========

def generate_scene(
    seed: Optional[int] = None, index: int = 0
) -> Optional[Tuple[Image.Image, List[Tuple[int,int,int,int]], List[Tuple[int,int,int,int]]]]:
    """
    Place [3,5] bench sprites and [2,3] stop sign sprites on the background.
    No animals or people are placed here — those are added per-variant.
    Returns (bg, bench_bboxes, stop_bboxes).
    """
    if seed is not None:
        random.seed(seed)

    bg = Image.open(BACKGROUND_FILE).convert("RGBA")
    bg_w, bg_h = bg.size
    sprites = load_sprites()

    n_stops   = random.randint(*STOP_SIGN_COUNT_RANGE)
    n_benches = random.randint(*BENCH_COUNT_RANGE)

    # Build list of (kind, sprite) pairs and shuffle for random grid assignment
    items: List[Tuple[str, Image.Image]] = []
    for _ in range(n_stops):
        items.append(("stop", sprites["stopSign"]))
    for _ in range(n_benches):
        items.append(("bench", sprites["bench"]))
    random.shuffle(items)

    n_items     = len(items)
    total_cells = GRID_ROWS * GRID_COLS
    n_place     = min(n_items, total_cells)
    cell_w      = (bg_w - 2 * MARGIN) / GRID_COLS
    cell_h      = (bg_h - 2 * MARGIN) / GRID_ROWS

    candidate: List[Tuple[int,int,int,int,str]] = []
    candidate_items: List[Tuple[str, Image.Image]] = []
    placements: List[Tuple[int,int,int,int,str]] = []
    chosen_items: List[Tuple[str, Image.Image]] = []

    for _ in range(MAX_SCENE_PLACEMENT_TRIES):
        random.shuffle(items)
        cell_indices = random.sample(range(total_cells), n_place)
        candidate = []
        candidate_items = list(items[:n_place])

        for ci, (kind, spr) in zip(cell_indices, candidate_items):
            gw, gh = spr.size
            row, col = ci // GRID_COLS, ci % GRID_COLS
            cx_base = MARGIN + (col + 0.5) * cell_w
            cy_base = MARGIN + (row + 0.5) * cell_h
            jx = random.uniform(-GRID_JITTER_SCALE, GRID_JITTER_SCALE) * cell_w
            jy = random.uniform(-GRID_JITTER_SCALE, GRID_JITTER_SCALE) * cell_h
            cx = cx_base + jx
            cy = cy_base + jy
            x = int(cx - gw / 2)
            y = int(cy - gh / 2)
            if kind == "bench":
                x = max(MARGIN + BENCH_PLACE_X_PAD,
                        min(x, bg_w - MARGIN - gw - BENCH_PLACE_X_PAD))
                y = max(MARGIN, min(y, bg_h - MARGIN - gh - BENCH_PLACE_Y_BOT_PAD))
            else:  # stop sign: pad for full 360° animal ring
                x = max(MARGIN + STOP_PLACE_PAD,
                        min(x, bg_w - MARGIN - gw - STOP_PLACE_PAD))
                y = max(MARGIN + STOP_PLACE_PAD,
                        min(y, bg_h - MARGIN - gh - STOP_PLACE_PAD))
            candidate.append((x, y, gw, gh, kind))

        if (not _any_sprite_overlap(candidate)
                and _stop_cells_non_adjacent(cell_indices, candidate_items)):
            placements = candidate
            chosen_items = candidate_items
            break

    if not placements:
        # All retries failed — skip this scene to avoid bad layouts
        return None

    bench_bboxes: List[Tuple[int,int,int,int]] = []
    stop_bboxes:  List[Tuple[int,int,int,int]] = []

    for (x, y, gw, gh, kind), (_, spr) in zip(placements, chosen_items):
        paste_sprite(bg, spr, x, y)
        if kind == "bench":
            bench_bboxes.append((x, y, x + gw, y + gh))
        else:
            stop_bboxes.append((x, y, x + gw, y + gh))

    return bg, bench_bboxes, stop_bboxes


# ========= Bus placement helpers =========

def is_patch_blank(
    scene: Image.Image,
    plain_bg: Image.Image,
    x: int, y: int, w: int, h: int,
    diff_threshold: int = BUS_DIFF_THRESHOLD,
) -> bool:
    """Return True if the scene patch at (x,y,w,h) is indistinguishable from the plain background."""
    patch_scene = scene.crop((x, y, x+w, y+h)).convert("RGB")
    patch_bg    = plain_bg.crop((x, y, x+w, y+h)).convert("RGB")
    diff = ImageChops.difference(patch_scene, patch_bg)
    if diff.getbbox() is None:
        return True
    return max(ch[1] for ch in diff.getextrema()) <= diff_threshold


# ========= YOLO quality filter =========

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bboxes_match(
    placed: List[Tuple[int,int,int,int]],
    detected: List[Tuple[int,int,int,int]],
    iou_thr: float = YOLO_IOU_MATCH_THR,
) -> bool:
    """Greedy bijection: every placed bbox must match exactly one detected bbox by IoU."""
    if len(placed) != len(detected):
        return False
    used = [False] * len(detected)
    for p in placed:
        best_iou, best_j = 0.0, -1
        for j, d in enumerate(detected):
            if used[j]:
                continue
            v = _iou(p, d)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j == -1 or best_iou < iou_thr:
            return False
        used[best_j] = True
    return True


def yolo_placement_ok(
    model: YOLO,
    pil_img: Image.Image,
    expected_benches: List[Tuple[int,int,int,int]],
    expected_stops: List[Tuple[int,int,int,int]],
    expected_people: List[Tuple[int,int,int,int]],
    expected_animals: Dict[str, List[Tuple[int,int,int,int]]],
    need_clock: int = 1,
    conf: float = YOLO_CONF,
    imgsz: int = YOLO_IMGSZ,
) -> bool:
    """
    Run YOLO and verify:
    - bench / stop sign / person / each animal species: IoU bijection match
    - exactly need_clock clocks detected
    expected_animals: {"zebra": [...], "elephant": [...], "giraffe": [...]}
    """
    img = np.array(pil_img.convert("RGB"))
    results = model.predict(source=img, verbose=False, conf=conf, imgsz=imgsz)
    r = results[0]

    det_benches: List[Tuple[int,int,int,int]] = []
    det_stops:   List[Tuple[int,int,int,int]] = []
    det_people:  List[Tuple[int,int,int,int]] = []
    det_animals: Dict[str, List[Tuple[int,int,int,int]]] = {sp: [] for sp in ANIMAL_NAMES}
    clock_cnt = 0

    for box in r.boxes:
        cls  = int(box.cls[0])
        name = model.names.get(cls, str(cls))
        x0, y0, x1, y1 = [int(round(v)) for v in box.xyxy[0].tolist()]
        if name == "bench":
            det_benches.append((x0, y0, x1, y1))
        elif name == "stop sign":
            det_stops.append((x0, y0, x1, y1))
        elif name == "person":
            det_people.append((x0, y0, x1, y1))
        elif name in ANIMAL_NAMES:
            det_animals[name].append((x0, y0, x1, y1))
        elif name == "clock":
            clock_cnt += 1

    if clock_cnt != need_clock:
        return False
    if not _bboxes_match(expected_benches, det_benches):
        return False
    if not _bboxes_match(expected_stops, det_stops):
        return False
    if not _bboxes_match(expected_people, det_people):
        return False
    for sp in ANIMAL_NAMES:
        if not _bboxes_match(expected_animals.get(sp, []), det_animals[sp]):
            return False
    return True


def _finalize_output_jpeg(
    img: Image.Image,
    quality: int = OUTPUT_JPEG_QUALITY,
) -> tuple[Image.Image, bytes]:
    """Encode the exact JPEG artifact that will be saved and reopen it for validation."""
    rgb_img = img.convert("RGB")
    buffer = BytesIO()
    rgb_img.save(buffer, format="JPEG", quality=quality)
    jpeg_bytes = buffer.getvalue()
    with Image.open(BytesIO(jpeg_bytes)) as jpeg_img:
        validated_img = jpeg_img.convert("RGB")
    return validated_img, jpeg_bytes


# ========= ID annotation helpers =========

def _sort_lr_tiebreak(
    bboxes: List[Tuple[int,int,int,int]],
    image_w: int,
    x_eps_ratio: float = 0.03,
) -> List[int]:
    """
    Sort bbox indices in a visually stable reading order.

    Objects whose x-centroids are very close are treated as belonging to the
    same visual column and are then ordered top-to-bottom inside that column.
    This keeps printed IDs stable when detector jitter slightly perturbs x.
    """
    centers = [((b[0]+b[2])/2.0, (b[1]+b[3])/2.0) for b in bboxes]
    if not centers:
        return []

    x_tol = max(24.0, image_w * x_eps_ratio)
    by_x = sorted(range(len(bboxes)), key=lambda i: centers[i][0])

    groups: List[List[int]] = []
    cur_group: List[int] = [by_x[0]]
    cur_mean_x = centers[by_x[0]][0]
    for idx in by_x[1:]:
        x = centers[idx][0]
        if abs(x - cur_mean_x) <= x_tol:
            cur_group.append(idx)
            cur_mean_x = sum(centers[i][0] for i in cur_group) / len(cur_group)
        else:
            groups.append(cur_group)
            cur_group = [idx]
            cur_mean_x = x
    groups.append(cur_group)

    order: List[int] = []
    for group in groups:
        order.extend(sorted(group, key=lambda i: (centers[i][1], centers[i][0])))
    return order


def _load_tag_font(size: int = 28) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_tag_pil(
    img: Image.Image,
    text: str,
    anchor_bbox: Tuple[int,int,int,int],
    fill: Tuple[int,int,int,int],
    border: Tuple[int,int,int,int],
    text_color: Tuple[int,int,int,int],
    prefer_right: bool = True,
    pad: int = 7,
    border_w: int = 3,
    font_size: int = 28,
) -> None:
    """Draw a rectangular ID tag next to anchor_bbox using PIL (in-place on RGBA img)."""
    W, H = img.size
    x0, y0, x1, y1 = map(int, anchor_bbox)
    cy = (y0 + y1) // 2

    font = _load_tag_font(font_size)
    _d = ImageDraw.Draw(img)
    tb = _d.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]

    tag_w = tw + 2 * pad
    tag_h = max(th + 2 * pad, int(1.2 * th) + 2 * pad)

    right_x = x1 + 6
    left_x  = x0 - 6 - tag_w
    if prefer_right and right_x + tag_w <= W - 1:
        tx0 = right_x
    elif left_x >= 0:
        tx0 = left_x
    else:
        tx0 = max(0, min(right_x, W - tag_w))
    ty0 = max(0, min(cy - tag_h // 2, H - tag_h))
    tx1, ty1 = tx0 + tag_w, ty0 + tag_h

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle((tx0, ty0, tx1, ty1), fill=fill, outline=border, width=border_w)

    text_x = tx0 + (tag_w - tw) // 2 - tb[0]
    text_y = ty0 + (tag_h - th) // 2 - tb[1]

    stroke = max(2, font_size // 12)
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            if dx == 0 and dy == 0:
                continue
            d.text((text_x+dx, text_y+dy), text, font=font, fill=(255, 255, 255, 255))
    d.text((text_x, text_y), text, font=font, fill=text_color)
    img.alpha_composite(overlay)


def _annotate_ids_pil(img: Image.Image, benches: List[Tuple], stops: List[Tuple]) -> None:
    """Draw bench/stop-sign ID tags onto a PIL RGBA image in-place."""
    W, _ = img.size
    bench_order = _sort_lr_tiebreak(benches, W)
    for k, bi in enumerate(bench_order):
        _draw_tag_pil(img, str(k+1), benches[bi],
                      fill=(255, 230, 0, 240), border=(0,0,0,255), text_color=(0,0,0,255))
    stop_order = _sort_lr_tiebreak(stops, W)
    for k, si in enumerate(stop_order):
        _draw_tag_pil(img, str(k+1), stops[si],
                      fill=(0, 220, 0, 240), border=(0,0,0,255), text_color=(0,0,0,255))


# ========= Variant generation =========

def add_bus_variants_for_one_scene(
    model: YOLO,
    base_no_person: Image.Image,
    bench_bboxes: List[Tuple[int,int,int,int]],
    stop_bboxes: List[Tuple[int,int,int,int]],
    stem: str,
    num_variants: int = BUS_VARIANTS_PER_SCENE,
) -> None:
    """
    Generate variants from a base scene (bench + stop sign sprites only).
    For each variant:
      1. Add 2-5 animals per stop sign and 0-3 people per bench (independently randomized).
      2. Check that all realized group bboxes are non-overlapping and within canvas bounds.
         Retry up to MAX_VARIANT_TRIES times; discard if still invalid.
      3. Place a randomly rotated bus either near a target object or on a blank region.
      4. Annotate IDs, encode the final JPEG, and run YOLO quality check on that exact artifact.
      5. Save the validated JPEG bytes.
    """
    sprites    = load_sprites()
    bus_sprite = Image.open(BUS_FILE).convert("RGBA")
    plain_bg   = Image.open(BACKGROUND_FILE).convert("RGB")

    W, H = base_no_person.size
    used_centers: List[Tuple[float, float]] = []

    for k in range(num_variants):
        # ---- Step 1 & 2: try to build a valid animal+people layout ----
        variant_img: Optional[Image.Image] = None
        all_animal_entries: List[Tuple[str,int,int,int,int]] = []
        all_people_bboxes: List[Tuple[int,int,int,int]] = []

        for _ in range(MAX_VARIANT_TRIES):
            cur = base_no_person.copy()
            _all_animals: List[Tuple[str,int,int,int,int]] = []
            _stop_animals: List[List[Tuple[str,int,int,int,int]]] = []

            # Add animals around each stop sign
            for stop_bbox in stop_bboxes:
                n = random.randint(*ANIMALS_PER_STOP_RANGE)
                aentries, cur = _add_animals_around_stop(cur, stop_bbox, n, sprites)
                _stop_animals.append(aentries)
                _all_animals.extend(aentries)

            # Flat bbox list (no species) for collision avoidance with people
            _all_animal_bboxes = [(x0,y0,x1,y1) for _,x0,y0,x1,y1 in _all_animals]

            # Add people around each bench, avoiding all placed animals
            _all_people: List[Tuple[int,int,int,int]] = []
            _bench_people: List[List[Tuple[int,int,int,int]]] = []
            for bench_bbox in bench_bboxes:
                n = random.randint(*PEOPLE_PER_BENCH_RANGE)
                pbbs, cur = _add_people_around_bench(cur, bench_bbox, n, sprites, _all_animal_bboxes)
                _bench_people.append(pbbs)
                _all_people.extend(pbbs)

            # Compute realized group bboxes (sprite + its satellites)
            group_bboxes: List[Tuple[int,int,int,int]] = []
            for i, stop_bbox in enumerate(stop_bboxes):
                animal_bboxes_i = [(x0,y0,x1,y1) for _,x0,y0,x1,y1 in _stop_animals[i]]
                members = [stop_bbox] + animal_bboxes_i
                group_bboxes.append(_union_bbox(members))
            for i, bench_bbox in enumerate(bench_bboxes):
                members = [bench_bbox] + _bench_people[i]
                group_bboxes.append(_union_bbox(members))

            if _groups_valid(group_bboxes, W, H):
                variant_img        = cur
                all_animal_entries = _all_animals
                all_people_bboxes  = _all_people
                break

        if variant_img is None:
            print(f"[warn] {stem} variant {k}: no valid layout after {MAX_VARIANT_TRIES} tries, skipping")
            continue

        # ---- Step 3: place bus ----
        if BUS_ROTATE:
            angle   = random.uniform(0.0, 360.0)
            rotated = bus_sprite.rotate(angle, expand=True)
        else:
            angle   = 0.0
            rotated = bus_sprite.copy()

        ow, oh = rotated.size
        placed = False
        cx = cy = 0.0

        def _try_place(
            px: int,
            py: int,
            target_bbox: Optional[Tuple[int,int,int,int]] = None,
            target_max_gap: Optional[float] = None,
        ) -> bool:
            """Return True and commit placement if (px, py) is a valid bus top-left."""
            nonlocal cx, cy, placed
            px = max(MARGIN, min(W - MARGIN - ow, px))
            py = max(MARGIN, min(H - MARGIN - oh, py))
            bus_bbox = (px, py, px + ow, py + oh)
            if target_bbox is not None and target_max_gap is not None:
                if _bbox_edge_dist(bus_bbox, target_bbox) > target_max_gap:
                    return False
            if not is_patch_blank(variant_img, plain_bg, px, py, ow, oh):
                return False
            cx_try = px + ow / 2.0
            cy_try = py + oh / 2.0
            if any(math.hypot(cx_try - bpx, cy_try - bpy) < BUS_MIN_CENTER_DIST
                   for bpx, bpy in used_centers):
                return False
            cx, cy = cx_try, cy_try
            used_centers.append((cx, cy))
            placed = True
            return True

        # Collect arrived-at target bboxes: benches + individual animals
        arrived_targets = list(bench_bboxes) + [
            (x0, y0, x1, y1) for _, x0, y0, x1, y1 in all_animal_entries
        ]

        if arrived_targets and random.random() < ARRIVED_AT_PROB:
            # Proximity placement: keep the actual post-clamp gap within ARRIVED_DIST_MAX.
            target = random.choice(arrived_targets)
            tx0, ty0, tx1, ty1 = target
            t_cx = (tx0 + tx1) // 2
            t_cy = (ty0 + ty1) // 2
            for _ in range(ARRIVED_MAX_TRIES):
                side = random.choice(('left', 'right', 'top', 'bottom'))
                gap  = random.randint(0, ARRIVED_DIST_MAX)
                j    = random.randint(-ARRIVED_JITTER, ARRIVED_JITTER)
                if side == 'right':
                    bx = tx1 + gap
                    by = t_cy - oh // 2 + j
                elif side == 'left':
                    bx = tx0 - gap - ow
                    by = t_cy - oh // 2 + j
                elif side == 'bottom':
                    by = ty1 + gap
                    bx = t_cx - ow // 2 + j
                else:  # top
                    by = ty0 - gap - oh
                    bx = t_cx - ow // 2 + j
                if _try_place(bx, by, target_bbox=target, target_max_gap=ARRIVED_DIST_MAX):
                    break

        if not placed:
            # Fall back to random blank-region placement
            for _ in range(BUS_MAX_PLACEMENT_TRIES):
                x = random.randint(MARGIN, max(MARGIN, W - MARGIN - ow))
                y = random.randint(MARGIN, max(MARGIN, H - MARGIN - oh))
                if _try_place(x, y):
                    break

        if not placed:
            print(f"[warn] cannot find blank region for {stem} variant {k}")
            continue

        out = variant_img.copy()
        out.alpha_composite(rotated, (int(cx - ow / 2.0), int(cy - oh / 2.0)))

        # Heading dot: red circle in front of the bus
        theta_rad = math.radians(angle)
        front_dx  = -math.sin(theta_rad)
        front_dy  = -math.cos(theta_rad)
        dot_cx = cx + front_dx * BUS_HEADING_DOT_OFFSET
        dot_cy = cy + front_dy * BUS_HEADING_DOT_OFFSET
        r = BUS_HEADING_DOT_RADIUS
        ImageDraw.Draw(out).ellipse(
            [dot_cx - r, dot_cy - r, dot_cx + r, dot_cy + r],
            fill=BUS_HEADING_DOT_COLOR,
        )

        # Resize: long side → 1280 px
        w, h = out.size
        scale = 1280 / max(w, h)
        out = out.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)

        def _scale(bboxes: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
            return [(int(b[0]*scale), int(b[1]*scale), int(b[2]*scale), int(b[3]*scale))
                    for b in bboxes]

        scaled_benches = _scale(bench_bboxes)
        scaled_stops   = _scale(stop_bboxes)
        scaled_people  = _scale(all_people_bboxes)

        # Build per-species bbox dicts (scaled) for IoU bijection per species
        species_bboxes: Dict[str, List[Tuple[int,int,int,int]]] = {
            sp: [] for sp in ANIMAL_NAMES
        }
        for sp, x0, y0, x1, y1 in all_animal_entries:
            species_bboxes[sp].append((int(x0*scale), int(y0*scale),
                                       int(x1*scale), int(y1*scale)))

        # ---- Step 4: annotate and validate the exact final JPEG artifact ----
        _annotate_ids_pil(out, scaled_benches, scaled_stops)
        final_img, final_jpeg_bytes = _finalize_output_jpeg(out)
        if not yolo_placement_ok(
            model,
            final_img,
            scaled_benches,
            scaled_stops,
            scaled_people,
            species_bboxes,
        ):
            continue

        # ---- Step 5: save the validated JPEG bytes ----
        out_path = OUTPUT_DIR / f"{stem}_bus_{k:02d}.jpg"
        out_path.write_bytes(final_jpeg_bytes)
        print("Saved", out_path)


# ========= Entry point =========

if __name__ == "__main__":
    model = YOLO(YOLO_WEIGHTS)

    for i in range(NUM_BASE_SCENES):
        result = generate_scene(index=i)
        if result is None:
            print(f"[warn] scene_{i:03d}: placement failed after {MAX_SCENE_PLACEMENT_TRIES} retries, skipping")
            continue
        base_no_person, bench_bboxes, stop_bboxes = result
        add_bus_variants_for_one_scene(
            model=model,
            base_no_person=base_no_person,
            bench_bboxes=bench_bboxes,
            stop_bboxes=stop_bboxes,
            stem=f"scene_{i:03d}",
            num_variants=BUS_VARIANTS_PER_SCENE,
        )
