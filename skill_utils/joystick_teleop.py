'''
classes to control robot pose via gamepad connected to linux computer
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

class JoystickTeleop(Thread):
    def __init__(self, pose):
        super().__init__()
        self.stop_event=Event()
        self.pose=pose
        self.moveeps=0.0005
        self.scale=1e-5

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
        cmds=[0,0,0]
        with open("/dev/input/js0", mode="rb") as f:
            while not self.stop_event.is_set():
                ev_time, ev_val, ev_type, ev_num= struct.unpack('IhBB', f.read(8))
                if ev_num == 0b0000_0001:
                    # axis="y"
                    cmds[1]=self.scale*ev_val*self.moveeps
                elif ev_num == 0b0000_0010:
                    # axis="x"
                    cmds[0]=self.scale*ev_val*self.moveeps
                self.pose=SE3.Trans(*cmds) * self.pose
                cmds=[0,0,0]
                
