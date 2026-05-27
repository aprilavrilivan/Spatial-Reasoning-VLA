from FSM import SpatialVLMFSM


def main():
    fsm = SpatialVLMFSM()
    print("Interactive FSM smoke test. Enter raw VLM-style answers when prompted.")
    print("Press Enter to leave an observation unchanged.\n")

    while True:
        print(f"Current state: {fsm.get_current_state()}")
        print(f"Current target: {fsm.get_target()}")
        if fsm.get_current_state() == "END":
            print("FSM reached END.")
            break

        keys = fsm.get_relevant_observation_keys()
        questions = fsm.get_relevant_questions(keys)
        observations = {}
        for key in keys:
            if key not in questions:
                continue
            print(f"\n[{key}] {questions[key]}")
            value = input("Raw answer: ").strip()
            if value:
                observations[key] = value

        if observations:
            fsm.update_observations(observations)
        else:
            print("No updates provided.")

        print("\nObservations:")
        for key, value in fsm.observations.items():
            print(f"  {key}: {value}")
        print("-" * 60)


if __name__ == "__main__":
    main()
