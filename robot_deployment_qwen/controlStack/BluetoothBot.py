"""
Wrapper around the serial connection to the Bluetooth robot. Pair the robot
first, identify its outgoing serial device, and set SPATIAL_VLA_BLUETOOTH_PORT
before opening the connection.

Pair the HC-05 module using the address and PIN configured on your own device.
On Linux, bind it to a serial device before running this module.

"""
import os
import time
import serial

PORT = os.environ.get("SPATIAL_VLA_BLUETOOTH_PORT", "/dev/rfcomm0")
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
    
