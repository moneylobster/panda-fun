'''
2d versions of classes in teleop.py
'''
from threading import Thread, Event
import numpy as np
from spatialmath import SE3, UnitQuaternion

from skill_utils.teleop import KeyboardCommandHandler
from skill_utils.format_pose import to_format, from_format

class KeyboardPoseController(KeyboardCommandHandler):
    def __init__(self, pose, homeq=None):
        self.moveeps=0.01
        self.pose=pose
        self.homeq=homeq
        
        super().__init__()

    @property
    def formatted_pose(self):
        return to_format(self.pose)

    @formatted_pose.setter
    def formatted_pose(self, val):
        self.pose=from_format(val)

    def w(self):
        # forward
        self.pose=SE3.Trans(-self.moveeps,0,0) * self.pose

    def s(self):
        # backward
        self.pose=SE3.Trans(self.moveeps,0,0) * self.pose

    def a(self):
        # right
        self.pose=SE3.Trans(0,-self.moveeps,0) * self.pose

    def d(self):
        # left
        self.pose=SE3.Trans(0,self.moveeps,0) * self.pose

    def r(self):
        # home
        if self.real:
            #real
            if self.homeq==None:
                self.panda.move_to_start()
            else:
                self.panda.move_to_joint_position(self.homeq)
            self.update_endeff()
            self.panda.start_controller(self.ctrl)
        else:
            #sim
            self.panda.q=self.panda.qr
            self.update_endeff()
            self.env.step(0.05)

    def m(self):
        # vacuum
        # is vacuum on?
        state=self.gripper.read_once()
        if state.part_present:
            try:
                self.gripper.drop_off(self.gripeps)
            except:
                # if unsuccessful
                self.gripper.stop()
        else:
            try:
                self.gripper.vacuum(3,self.gripeps)
            except:
                # if unsuccessful
                self.gripper.stop()

