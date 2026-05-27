from answer_parsing import parse_action, parse_turn


DRIVE_SPEED = 10
TURN_SPEED = 10
STEP_CM = 5
TURN_DEG = 15


def action_to_commands(answer: str) -> list[str]:
    """Convert a VLM action answer into the robot command protocol."""
    action = parse_action(answer)
    if action == "keep straight":
        return [f"f,{DRIVE_SPEED},{STEP_CM}"]
    if action == "turn left":
        return [f"l,{TURN_SPEED},{TURN_DEG}", f"f,{DRIVE_SPEED},{STEP_CM}"]
    if action == "turn right":
        return [f"r,{TURN_SPEED},{TURN_DEG}", f"f,{DRIVE_SPEED},{STEP_CM}"]
    return []


def turn_answer_to_commands(answer: str) -> list[str]:
    turn = parse_turn(answer)
    if turn == "turn left":
        return [f"l,{TURN_SPEED},{TURN_DEG}"]
    if turn == "turn right":
        return [f"r,{TURN_SPEED},{TURN_DEG}"]
    return []


def stop_command() -> str:
    return "s,0,0"
