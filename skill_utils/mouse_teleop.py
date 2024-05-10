'''
classes to control robot pose via mouse connected to linux computer
'''
from threading import Thread, Event
import numpy as np
from spatialmath import SE3, UnitQuaternion
import struct
from skill_utils.format_pose import to_format, from_format

def clip(val,uplim):
    if abs(val)>uplim:
        return (val/abs(val))*uplim
    else:
        return val

class MouseTeleop(Thread):
    def __init__(self, pose):
        super().__init__()
        self.stop_event=Event()
        self.pose=pose
        self.moveeps=0.0005
        self.uplim=15

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
                buttonraw, yraw, xraw = struct.unpack("Bbb", f.read(3))
                y=clip(yraw,self.uplim)
                x=clip(xraw,self.uplim)
                self.pose=SE3.Trans(x*self.moveeps,y*self.moveeps,0) * self.pose
                
