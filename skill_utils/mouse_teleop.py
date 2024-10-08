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

class MousePose(Thread):
    def __init__(self, pose):
        super().__init__()
        self.stop_event=Event()
        self.pose=SE3(pose)
        self.moveeps=0.0005
        self.uplim=7

    @property
    def formatted_pose(self):
        return to_format(self.pose.data[0])

    @formatted_pose.setter
    def formatted_pose(self, val):
        self.pose=SE3(from_format(val))

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
                buttonraw, xraw, yraw = struct.unpack("Bbb", f.read(3))
                y=-clip(yraw,self.uplim) # invert sign as well to match left-right
                x=clip(xraw,self.uplim)
                # the axes seem to be mixed up as
                # well, so we give the mouse x and y inputs to the
                # robot y and x inputs
                self.pose=SE3.Trans(y*self.moveeps,x*self.moveeps,0) * self.pose
                
