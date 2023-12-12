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
        'w': "forward",
        's': "backward",
        'a': "left",
        'd': "right",
        'q': "down",
        'e': "up",
        ' ': "gripper",
        'h': "home",
        't': "quit",
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
            self._endeff=SE3(self.panda.get_pose()) #store as SE3
            self.real=True
        else:
            # run in simulation maybe?
            # TODO complete
            import swift
            self.env=swift.Swift()
            # self.panda=False
            # self.endeff=False
            self.real=False

        # how much to move by in each press
        self.moveeps=0.01

    @property
    def endeff(self):
        return self._endeff
    
    @endeff.setter
    def endeff(self, val):
        """
        set endeff pose and move robot to new pose
        """
        print("BEFORE:")
        print(self._endeff)
        self._endeff=val
        print("AFTER:")
        print(self._endeff)
        if self.real:
            # TODO change into CartesianImpedanceController sometime
            self.panda.move_to_pose(self._endeff.data[0])
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
        elif cmd=="quit":
            quit()
        else:
            print(f"Invalid command {cmd}")

    def forward(self):
        self.endeff=self.endeff @ SE3.Trans(0,self.moveeps,0)

    def backward(self):
        self.endeff=self.endeff @ SE3.Trans(0,-self.moveeps,0)

    def right(self):
        self.endeff=self.endeff @ SE3.Trans(self.moveeps,0,0)

    def left(self):
        self.endeff=self.endeff @ SE3.Trans(-self.moveeps,0,0)

    def up(self):
        self.endeff=self.endeff @ SE3.Trans(0,0,self.moveeps)

    def down(self):
        self.endeff=self.endeff @ SE3.Trans(0,0,-self.moveeps)
        
        
        
# init
teleop=Teleop("real", "10.0.0.2")
while True:
    teleop.process_key(getch())
