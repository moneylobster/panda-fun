'''
2d versions of classes in teleop.py
'''
from threading import Thread, Event
import numpy as np
from spatialmath import SE3, UnitQuaternion

from skill_utils.teleop import KeyboardHandler
from skill_utils.format_pose import to_format, from_format

class KeyboardPoseController(KeyboardHandler):
    def __init__(self):
        self.moveeps=0.01
        self.startingjoints=np.array([ 0.        ,
                                       -0.78539816,
                                       0.        ,
                                       -2.35619449,
                                       0.        ,
                                       1.57079633,
                                       0.78539816])
        self.pose=None
        self.endevent=Event()
        self.policyevent=Event()
        self.startevent=Event()
        self.stopevent=Event()
        self.delevent=Event()
        super().__init__()

    @property
    def formatted_pose(self):
        return to_format(self.pose)

    @formatted_pose.setter
    def formatted_pose(self, val):
        self.pose=from_format(val)

    def w(self):
        print(self.pose) #debug print
        # forward
        self.pose=SE3.Trans(self.moveeps,0,0) * self.pose

    def s(self):
        # backward
        self.pose=SE3.Trans(-self.moveeps,0,0) * self.pose

    def a(self):
        # right
        self.pose=SE3.Trans(0,-self.moveeps,0) * self.pose

    def d(self):
        # left
        self.pose=SE3.Trans(0,self.moveeps,0) * self.pose

    def q(self):
        # quit program
        self.endevent.set()

    def c(self):
        # hand control to policy
        self.policyevent.set()

    def g(self):
        # start recording
        self.startevent.set()

    def h(self):
        # stop recording
        self.stopevent.set()

    def j(self):
        # delete most recent episode
        self.delevent.set()
