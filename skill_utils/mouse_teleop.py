'''
classes to control robot pose via mouse connected to linux computer
'''
from threading import Thread, Event
import numpy as np
from spatialmath import SE3, UnitQuaternion
import struct
from skill_utils.format_pose import to_format, from_format

def parse_mouse_msg(msg):
    if msg & 0b10000000 == 0b10000000:
        return 1
    elif msg & 0b00000001 == 0b00000001:
        return -1
    else:
        return 0

class MouseTeleop(Thread):
    def __init__(self, pose):
        super().__init__()
        self.stop_event=Event()
        self.pose=pose
        self.moveeps=0.01

    @property
    def formatted_pose(self):
        return to_format(self.pose)

    @formatted_pose.setter
    def formatted_pose(self, val):
        self.pose=from_format(val)

    def stop(self):
        self.stop_event.set()
        
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        with open("/dev/input/mice", mode="rb") as f:
            while not self.stop_event.is_set():
                buttonraw, xraw, yraw = struct.unpack("BBB", f.read(3))
                button, x, y = map(parse_mouse_msg, [buttonraw, xraw, yraw])
                self.pose=SE3.Trans(x*self.moveeps,y*self.moveeps,0) * self.pose
                
