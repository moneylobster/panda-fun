'''
2d versions of classes in teleop.py
'''
from threading import Thread, Event
from skill_utils.teleop import Teleop, KeyboardHandler
from skill_utils.format_pose import to_format, from_format

class Teleop2d(Teleop):
    def __init__(self, mode="real", ip="10.0.0.2", record=False):
        super().__init__(mode, ip, record)
        self.homeq=None # TODO set this

    # unbind the up and down functionality to prevent z-level movement
    def up(self):
        pass
    def down(self):
        pass

class KeyboardPoseController(KeyboardHandler):
    def __init__(self):
        self.moveeps=0.01
        self.pose=np.array([ 0.        ,
                             -0.78539816,
                             0.        ,
                             -2.35619449,
                             0.        ,
                             1.57079633,
                             0.78539816])
        self.endevent=Event()
        self.policyevent=Event()
        super().__init__()

    @property
    def formatted_pose(self):
        return to_format(self.pose)

    @formatted_pose.setter
    def formatted_pose(self, val):
        self.pose=from_format(val)

    def w(self):
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
