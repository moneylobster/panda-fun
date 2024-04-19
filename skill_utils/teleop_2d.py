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
        raise NotImplementedError
        self.moveeps=0.01
        self.startingpose=None #TODO!!!!
        super().__init__(self.moveeps, self.startingpose)

    @property
    def formatted_pose(self):
        return to_format(self.pose)

    @formatted_pose.setter
    def formatted_pose(self, val):
        self.pose=from_format(val)
