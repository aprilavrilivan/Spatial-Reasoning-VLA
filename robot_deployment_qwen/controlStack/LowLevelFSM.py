import datetime
import math

TURN_SPEED = 10
DRIVE_SPEED = 10

class LowLevelFSM:
    def __init__(self, pos = None):
        self.robot_state = RobotState(pos)
    
    def update_robot_state(self, new_pos):
        self.robot_state.update_pos(new_pos)
    
    def go_to_dest(self, dest):
        # get a list of directions to dest (point)
        dist_to_dest = self.robot_state.cur_pos.get_dist(dest)
        dest_heading = self.robot_state.cur_pos.get_heading(dest)

        return self.turn_to_heading(dest_heading) + self.go_forward(dist_to_dest)
    
    def turn_to_heading(self, new_heading):
        heading_diff = get_heading_diff(self.robot_state.cur_heading, new_heading)
        turn_direction = 'r' if heading_diff < 0 else 'l'
        
        return [f"{turn_direction},{TURN_SPEED},{int(abs(heading_diff))}",]
    
    def go_forward(self, dist):
        return [f"f,{DRIVE_SPEED},{int(dist)}"]


def get_heading_diff(cur_heading, new_heading):
    # helper to get smallest angle between 2 headings
    diff = new_heading - cur_heading
    return (diff + 180) % 360 - 180

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def get_dist(self, point2):
        # returns Euclidean distance in cm
        return ( (point2.x-self.x)**2 + (point2.y-self.y)**2 )**0.5
    
    def get_heading(self, point2):
        # returns heading from +x axis in degrees (0 to 360)
        heading_rad = math.atan2((point2.y-self.y), (point2.x-self.x))
        return (math.degrees(heading_rad) + 360) % 360
    
    def __str__(self):
        return str((self.x, self.y))

class RobotState:
    cur_pos = Point(0, 0) # current position relative to arena bottom left corner [cm] x [cm]
    cur_speed = 0 # current calulated speed [cm/s]
    cur_heading = 0 # current heading relative to +x axis [deg]
    cur_time = None

    def __init__(self, pos = None):
        if pos is not None:
            self.update_pos(pos)
        
    def update_pos(self, new_pos):
        """
        Docstring for update
        
        :param new_pos: new position captured [cm] x [cm]
        """
        
        old_pos = self.cur_pos
        self.cur_pos = new_pos
        if self.cur_time is not None:
            # update speed and heading
            old_time = self.cur_time
            self.cur_time = datetime.datetime.now()
            dist = old_pos.get_dist(new_pos)
            time_interval = (self.cur_time-old_time).total_seconds()
            self.speed = dist / time_interval
            self.cur_heading = old_pos.get_heading(new_pos)
        else:
            # no previous position
            self.cur_time = datetime.datetime.now()

    
# FOR TESTING
if __name__ == "__main__":
    start_pos = Point(10,10)
    print(f"Start robot at {start_pos}")
    ll_fsm = LowLevelFSM(Point(10,10))
    print(f"Robot new pos: {ll_fsm.robot_state.cur_pos}")
    new_pos = Point(5,10)
    print(f"Robot heads to {new_pos}")
    ll_fsm.update_robot_state(Point(5,10))
    print(f"Robot new pos: {ll_fsm.robot_state.cur_pos}")
    print(f"Robot new heading: {ll_fsm.robot_state.cur_heading}")
    new_heading = 90
    commands = ll_fsm.turn_to_heading(new_heading)
    print(f"Commands to turn to heading {new_heading}: {commands}")
    dest_pos = Point(0,10)
    commands = ll_fsm.go_to_dest(dest_pos)
    print(f"Directions to point {dest_pos}: {commands}")