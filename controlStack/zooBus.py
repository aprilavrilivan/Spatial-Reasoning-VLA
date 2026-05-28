from vlm_client import ask
from BluetoothBot import BluetoothBot
from FSM import question_dict
from answer_parsing import parse_int, parse_id_list, parse_action, parse_compass, parse_egocentric, parse_turn, parse_yes_no
from camera_stream import CameraStream, display
import time
if __name__ == "__main__":
    #Steps:
    #1. get ordering of benches with people closest to furthest
    #2. visit each bus until no more people are on the field
    #3. list each stop sign closest to furthest
    #4. visit each stop sign at a set time (no temporal logic, either sleep system or go instantly)
    #5. terminate loop
    bot = BluetoothBot()
    camera = CameraStream()
    bot.open_connection()
    camera.start()
    #Step 1:
    camera.update()
    img = camera.read()
    bus_ordering = ask(img,f"{question_dict['ListBenchesWithAtLeastKPeople']} k=1")
    bus_ordering= parse_id_list(bus_ordering)
    #Step 2:
    
    display(img)
    exit_ans = False
    exit_ans_1 = False
    exit_ans_2 = ""
    if(bus_ordering[0] != "N/A"):
            
        while(exit_ans):
            #Get closest bench with person
            target_bus = int(parse_int(ask(img,f"{question_dict['ClosestBenchWithPerson']}")))
            #Are we at closest bench?
            while(exit_ans_1):
                if(exit_ans_2 == "yes"):
                    #Yes, wait for people to onboard
                    camera.update()
                    img = camera.read()
                    display(img)
                    exit_ans_1 = ask(img,f"{question_dict["CountPersonAtClosestBench"]}")
                    exit_ans_1 = int(parse_int(exit_ans_1)) != 0
                else:
                    #No, driver closer
                    camera.update()
                    img = camera.read()
                    display(img)
                    robot_controls = ask(img,f"{question_dict['AvoidObstacleToReachBench']} bench_number = {target_bus}")
                    robot_controls = parse_action(robot_controls)
                    bot.send_message(robot_controls)
                    camera.update()
                    img = camera.read()
                    exit_ans_2 = ask(img,f"{question_dict['ArrivedAtBench']} bench_number = {target_bus}")
                    exit_ans_2 = parse_yes_no(exit_ans_2)
            #Are all the people on the field picked up?
            camera.update()
            img = camera.read()
            display(img)
            exit_ans = int(parse_int(ask(img,question_dict["CountPeople"]))) != 0

    #Step 3
    camera.update()
    img = camera.read()
    stopSign_list = ask(img,f"{question_dict['ListStopSignsWithAtLeastKAnimals']} k=1")
    stopSign_list= parse_id_list(bus_ordering)
    #Step 2:
    
    display(img)
    exit_ans = False
    exit_ans_1 = False
    exit_ans_2 = ""
    if(stopSign_list[0] != "N/A"):
        while(exit_ans):
            #Get closest stop sign 
            target_bus = int(parse_int(ask(img,f"{question_dict['ClosestStopSign']}")))
            #Are we at closest stop sign?
            while(exit_ans_1):
                if(exit_ans_2 == "yes"):
                    #Yes, sleep for a little bit
                    time.sleep(3)
                    
                else:
                    #No, driver closer
                    #We dont have anyway to prevent revisiting zoos at this point
                    camera.update()
                    img = camera.read()
                    display(img)
                    robot_controls = ask(img,f"{question_dict['AvoidObstacleToReachClosestBench']} ")
                    robot_controls = parse_action(robot_controls)
                    bot.send_message(robot_controls)
                    camera.update()
                    img = camera.read()
                    exit_ans_2 = ask(img,f"{question_dict['ArrivedAtAnimalsAroundStopSigns']}")
                    exit_ans_2 = parse_yes_no(exit_ans_2)
            #Are all the people on the field picked up?
            camera.update()
            img = camera.read()
            display(img)
            exit_ans = int(parse_int(ask(img,question_dict["CountPeople"]))) != 0
