import re


YES = {"yes", "true", "y"}
NO = {"no", "false", "n"}
ACTIONS = {
    "keep straight": "keep straight",
    "straight": "keep straight",
    "go straight": "keep straight",
    "forward": "keep straight",
    "turn left": "turn left",
    "left": "turn left",
    "go left": "turn left",
    "turn right": "turn right",
    "right": "turn right",
    "go right": "turn right",
}
TURN_LABELS = {
    **ACTIONS,
    "already facing": "already facing",
    "facing": "already facing",
}
COMPASS = {
    "north",
    "south",
    "east",
    "west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
}
EGOCENTRIC = {
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
}


def clean_text(text) -> str:
    text = "" if text is None else str(text)
    text = text.strip()
    for prefix in ("ASSISTANT:", "Assistant:", "assistant:", "[INST]", "</s>", "<s>"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    if "</s>" in text:
        text = text.split("</s>", 1)[0].strip()
    return text.strip().strip("\"'`")


def normalize_space(text) -> str:
    return re.sub(r"\s+", " ", clean_text(text).lower()).strip()


def parse_yes_no(text) -> str | None:
    norm = normalize_space(text).rstrip(".")
    if norm in YES:
        return "yes"
    if norm in NO:
        return "no"
    match = re.search(r"\b(yes|no|true|false)\b", norm)
    if not match:
        return None
    return "yes" if match.group(1) in YES else "no"


def parse_int(text) -> int | None:
    norm = normalize_space(text)
    match = re.search(r"-?\d+", norm)
    return int(match.group(0)) if match else None


def parse_id_list(text) -> list[int]:
    norm = normalize_space(text)
    if norm in {"", "none", "no", "n/a"}:
        return []
    return [int(x) for x in re.findall(r"\d+", norm)]


def parse_action(text) -> str | None:
    norm = normalize_space(text).replace("_", " ").rstrip(".")
    for key, value in ACTIONS.items():
        if norm == key or key in norm:
            return value
    return None


def parse_turn(text) -> str | None:
    norm = normalize_space(text).replace("_", " ").rstrip(".")
    for key, value in TURN_LABELS.items():
        if norm == key or key in norm:
            return value
    return None


def parse_compass(text) -> str | None:
    norm = normalize_space(text).replace("-", "").replace(" ", "")
    for label in COMPASS:
        if norm == label:
            return label.capitalize() if len(label) <= 5 else label.title()
    return None


def parse_egocentric(text) -> str | None:
    norm = normalize_space(text).replace(" ", "-")
    for label in EGOCENTRIC:
        if norm == label:
            return label
    return None
