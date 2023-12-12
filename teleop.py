'''
simple teleoperation with the end-effector orientation staying constant.
'''

from utils.getch import getch
from spatialmath import SE3,UnitQuaternion
import panda_py
from panda_py import controllers
import panda_py.libfranka

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
            self.real=True
            self.panda=panda_py.Panda(ip)
            self.gripper=panda_py.libfranka.VacuumGripper(ip)
            self._endeff=SE3(self.panda.get_pose()) #store as SE3
            # use a cartesianimpedance controller
            self.ctrl=controllers.CartesianImpedance(filter_coeff=1.0)
            self.panda.start_controller(self.ctrl)
        else:
            # run in simulation maybe?
            # TODO complete
            import swift
            self.real=False
            self.env=swift.Swift()
            # self.panda=False
            # self.endeff=False

        # how much to move by in each press
        self.moveeps=0.01

    def update_endeff(self):
        self._endeff=SE3(self.panda.get_pose())

    @property
    def endeff(self):
        return self._endeff
    
    @endeff.setter
    def endeff(self, val):
        """
        set endeff pose and move robot to new pose
        """
        self._endeff=val
        if self.real:
            # self.panda.move_to_joint_position(panda_py.ik(self._endeff.data[0]))
            self.ctrl.set_control(self._endeff.t, UnitQuaternion(self._endeff))
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

    def home(self):
        try:
            self.panda.move_to_start()
            self.panda.update_endeff()
        except:
            print("Can't home!")
        
        
        
# init
teleop=Teleop("real", "10.0.0.2")
with teleop.panda.create_context(frequency=100) as ctx:
    while ctx.ok():
        teleop.process_key(getch())
