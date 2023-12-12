'''
simple teleoperation with the end-effector orientation staying constant.
'''

from utils.getch import getch
from spatialmath import SE3

def translate_input(char):
    '''
    turn keypress into appropriate command.
    '''
    keymap={
        b'w': "forward",
        b's': "backward",
        b'a': "left",
        b'd': "right",
        b'q': "down",
        b'e': "up",
        b' ': "gripper",
        b'h': "home",
    }
    return keymap[char]

class Teleop():
    def __init__(self, mode="real", ip="10.0.0.2"):
        """
        
        Parameters
        ----------
        mode : string: "real" or "sim"
            whether to run teleop in real life or in simulation
        ip : string
            ip of robot for "real" option

        """
        if mode=="real":
            # run in real life
            import panda_py
            import panda_py.libfranka
            self.panda=panda_py.Panda(ip)
            self.gripper=panda_py.libfranka.VacuumGripper(ip)
            self.endeff=panda.get_pose()
            self.real=True
        elif mode=="sim":
            # run in simulation maybe?
            # TODO complete
            import swift
            self.env=swift.Swift()
            self.real=False

    @endeff.setter
    def endeff_set(self, val):
        """
        set endeff pose and move robot to new pose
        """
        self.endeff=val
        if self.real:
            # TODO change into CartesianImpedanceController sometime
            self.panda.move_to_pose(self.endeff)
        else:
            #TODO implement movement in simulator
            pass

    def process_key(self,char):
        """
        Process the keypress and turn it into a movement command

        Parameters
        ----------
        char : string, single char
            input keypress
        """
        cmd=translate_input(char)
        if cmd=="forward":
            self.forward()
        elif cmd=="backward":
            self.backward()
        elif cmd=="left":
            self.left()
        elif cmd=="right":
            self.right()
        elif cmd=="down":
            self.down()
        elif cmd=="up":
            self.up()
        elif cmd=="gripper":
            self.gripper()
        elif cmd=="home":
            self.home()
        else:
            print(f"Invalid command {cmd}")

    def forward(self):
        """
        Move endeff forward. (in x direction)
        """
        self.endeff=self.endeff @ SE3.Trans(0.01,0,0)
        
        
        
# init
teleop=Teleop("real", "10.0.0.2")
while True:
    teleop.process_key(getch())
