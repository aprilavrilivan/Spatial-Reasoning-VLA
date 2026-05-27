"""
Wrapper around the serial connection to the bluetooth robot. You must pair with
the robot first, check the correct outgoing COM port number in Device Manager, 
and ***REPLACE THE PORT NAME VARIABLE*** before using this class to open the
serial connection. 

 * Bluetooth device name: SVLM-bot
 * Bluetooth device address: 00:14:03:06:75:c2
 * Bluetooth connection PIN: 1234

"""
import time
import serial

PORT = '/dev/rfcomm1' # REPLACE WITH CORRECT OUTGOING PORT NUMBER
BAUD_RATE = 9600

class BluetoothBot:
    ser = None

    # establish the serial connection
    def open_connection(self):
        try:
            print(f"Connecting to {PORT}...")
            self.ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
            time.sleep(2) # Stabilize connection
            print(f"Connected to {PORT}!")

        except serial.SerialException as e:
            print(f"Could not connect to {PORT}: {e}. Check Bluetooth settings.")

    # close the serial connection
    def close_connection(self):
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
                print("Connection closed.")
                self.ser = None
            else: 
                print("No open serial connection")
        except serial.SerialException as e:
            print(f"Error: {e}")

    # send message through serial connection
    def send_message(self, message):
        try: 
            if self.ser.is_open:
                outgoing = str(message) + "\n"
                self.ser.write(outgoing.encode('utf-8'))
            else:
                print("No open serial connection")
        except serial.SerialException as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    from LowLevelFSM import *
    
    try:
        print(f"Opening Bluetooth connection...")
        robot = BluetoothBot()
        robot.open_connection()
        
        start_pos = Point(0,0)
        print(f"Start robot at {start_pos}")
        ll_fsm = LowLevelFSM(start_pos)
        print(f"Robot pos: {ll_fsm.robot_state.cur_pos}")

        new_pos = Point(10,0)
        print(f"Robot goes to {new_pos}")
        commands = ll_fsm.go_forward(10)
        ll_fsm.update_robot_state(new_pos)

        for command in commands:
            print(f"Sending command: {command}")
            robot.send_message(command)
            time.sleep(5) # wait for robot to process command

        print(f"Robot new pos: {ll_fsm.robot_state.cur_pos}")
        print(f"Robot new heading: {ll_fsm.robot_state.cur_heading}")

        list_of_destinations = [Point(10,10), Point(0,10), Point(0,0)]
        for dest in list_of_destinations:
            print(f"Robot heads to {dest}")
            commands = ll_fsm.go_to_dest(dest)
            ll_fsm.update_robot_state(dest)

            for command in commands:
                print(f"Sending command: {command}")
                robot.send_message(command)
                time.sleep(5) # wait for robot to process command

            print(f"Robot new pos: {ll_fsm.robot_state.cur_pos}")
            print(f"Robot new heading: {ll_fsm.robot_state.cur_heading}")

        print("Done with navigation test.")
        

    except serial.SerialException as e:
        print(f"Could not connect to {PORT}: {e}. Check Bluetooth settings.")
    except KeyboardInterrupt:
        print("\nForce closed.")
    finally:
        if 'robot' in locals() and robot.ser.is_open:
            robot.close_connection()
    