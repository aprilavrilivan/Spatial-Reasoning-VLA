from vlm_client import QwenVLMClient
from experiment_logger import ExperimentLogger
from BluetoothBot import BluetoothBot
from FSM import question_dict
from answer_parsing import parse_int, parse_id_list, parse_action, parse_compass, parse_egocentric, parse_turn, parse_yes_no
from camera_stream import CameraStream, display
import time
import argparse
class Runner:
    def __init__(self,run_label,adapter_path,use_adapter):
        self.logger = ExperimentLogger()
        self.vlm = QwenVLMClient(adapter_path=adapter_path, use_adapter=use_adapter)
        self.run_label = run_label or ("finetuned_qwen" if use_adapter else "base_qwen")

    def log_final(self, test_name: str,  success: bool, reason: str):
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
            # trial=trial,
            row_type="final",
            model_loop_success=success,
            human_success=human_success,
            failure_reason=failure_reason,
            notes=notes,
        )

    def ask(self, test_name: str, step: int, phase: str, question_type: str, question: str, parser,frame):
        # frame = self.camera.snapshot(flush_frames=3)
        # if self.show_camera:
            # self.display(frame)
        image_path = self.logger.save_frame(frame, f"{test_name}_step{step:03d}_{phase}")
        result = self.vlm.ask(frame, question)
        parsed = parser(result["answer"]) if parser is not None else result["answer"]
        print(
            f"[{test_name} step {step} {phase}] "
            f"{question_type}: raw={result['answer']!r} parsed={parsed!r}"
        )
        self.logger.log(
            run_label=self.run_label,
            test_name=test_name,
            # trial=trial,
            step=step,
            phase=phase,
            row_type="vlm",
            image_path=image_path,
            # question_type=question_type,
            question=question,
            full_prompt=result["full_prompt"],
            raw_answer=result["raw_answer"],
            answer=result["answer"],
            parsed_answer=parsed,
            latency_sec=result["latency_sec"],
        )
        return parsed

if __name__ == "__main__":
    #Steps:
    #1. get ordering of benches with people closest to furthest
    #2. visit each bus until no more people are on the field
    #3. list each stop sign closest to furthest
    #4. visit each stop sign at a set time (no temporal logic, either sleep system or go instantly)
    #5. terminate loop
    parser = argparse.ArgumentParser(description="Zoo bus full complex path")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--no-adapter", action="store_true", help="Run the base Qwen model without the fine-tuned LoRA adapter.")
    args = parser.parse_args()

    run = Runner(adapter_path=args.adapter_path, use_adapter=args.no_adapter)
    bot = BluetoothBot()
    camera = CameraStream()
    bot.open_connection()
    camera.start()

    #Step 1:
    camera.update()
    step = 0
    img = camera.read()
    bus_ordering = run.ask(step=step,img=img,question=f"{question_dict['ListBenchesWithAtLeastKPeople']} k=1",question_type="Bus_Ordering",phase="Find_benches")
    bus_ordering= parse_id_list(bus_ordering)
    #Step 2:
    
    display(img)
    exit_ans = False
    exit_ans_1 = False
    exit_ans_2 = ""
    if(bus_ordering[0] != "N/A"):
            
        while(exit_ans):
            #Get closest bench with person
            target_bus = int(parse_int(run.ask(img=img,question=f"{question_dict['ClosestBenchWithPerson']}",phase="Go_to_closest_bench" ,question_type="ClosestBench")))
            #Are we at closest bench?
            while(exit_ans_1):
                if(exit_ans_2 == "yes"):
                    #Yes, wait for people to onboard
                    camera.update()
                    img = camera.read()
                    display(img)
                    exit_ans_1 = run.ask(step=step,img=img,question=f"{question_dict["CountPersonAtClosestBench"]}",phase="Onboarding",question_type="Count_People")
                    exit_ans_1 = int(parse_int(exit_ans_1)) != 0
                    step += 1
                else:
                    #No, driver closer
                    camera.update()
                    img = camera.read()
                    display(img)
                    robot_controls = run.ask(step=step, img=img,question=f"{question_dict['AvoidObstacleToReachBench']} bench_number = {target_bus}",phase="Go_to_closest_bench",question_type="ArrivedAtBench")
                    robot_controls = parse_action(robot_controls)
                    bot.send_message(robot_controls)
                    camera.update()
                    img = camera.read()
                    exit_ans_2 = run.ask(step=step,img=img,question=f"{question_dict['ArrivedAtBench']} bench_number = {target_bus}",question_type="ArrivedAtBench",phase="Go_to_closest_bench")
                    exit_ans_2 = parse_yes_no(exit_ans_2)
                    step += 1
            #Are all the people on the field picked up?
            camera.update()
            img = camera.read()
            display(img)
            exit_ans = int(parse_int(run.ask(step=step,img=img,question=question_dict["CountPeople"],phase="Go_to_closest_bench",question_type="CountPeople"))) != 0
            step += 1

    #Step 3
    camera.update()
    img = camera.read()
    stopSign_list = run.ask(step=step,img=img,question=f"{question_dict['ListStopSignsWithAtLeastKAnimals']} k=1",question_type="ListStopSignsWithAtLeastKAnimals",phase="Go_to_closest_zoo")
    stopSign_list= parse_id_list(bus_ordering)
    step += 1
    #Step 2:
    
    display(img)
    exit_ans = False
    exit_ans_1 = False
    exit_ans_2 = ""
    if(stopSign_list[0] != "N/A"):
        while(exit_ans):
            #Get closest stop sign 
            target_bus = int(parse_int(run.ask(step=step,img=img,question=f"{question_dict['ClosestStopSign']}",question_type="ClosestStopSign",phase="Go_to_closest_zoo")))
            #Are we at closest stop sign?
            step += 1
            while(exit_ans_1):
                if(exit_ans_2 == "yes"):
                    #Yes, sleep for a little bit
                    time.sleep(3)
                    step +=1
                    
                else:
                    #No, driver closer
                    #We dont have anyway to prevent revisiting zoos at this point
                    camera.update()
                    img = camera.read()
                    display(img)
                    robot_controls = run.ask(step=step,img=img,question=f"{question_dict['AvoidObstacleToReachClosestBench']}",question_type="AvoidObstacleToReachClosestBench",phase="Go_to_closest_zoo")
                    robot_controls = parse_action(robot_controls)
                    bot.send_message(robot_controls)
                    camera.update()
                    img = camera.read()
                    exit_ans_2 = run.ask(step=step,img=img,question=f"{question_dict['ArrivedAtAnimalsAroundStopSigns']}",question_type="ArrivedAtAnimalsAroundStopSigns",phase="Go_to_closest_zoo")
                    exit_ans_2 = parse_yes_no(exit_ans_2)
                    step +=1
            #Are all the animals on the field picked up?
            camera.update()
            img = camera.read()
            display(img)
            exit_ans = int(parse_int(run.ask(step=step,img=img,question=question_dict["ListStopSignsWithAtLeastKAnimals"],question_type="ListStopSignsWithAtLeastKAnimals",phase="Go_to_closest_zoo"))) != 0
            step += 1
        