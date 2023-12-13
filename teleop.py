'''
simple teleoperation with the end-effector orientation staying constant.
'''

from utils.getch import getch
from spatialmath import SE3,UnitQuaternion
from datetime import timedelta

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
        ' ': "vacuum",
        'h': "home",
        'u': "update",
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
            from panda_py import controllers
            import panda_py.libfranka
            self.real=True
            self.panda=panda_py.Panda(ip)
            self.gripper=panda_py.libfranka.VacuumGripper(ip)
            self._endeff=SE3(self.panda.get_pose()) #store as SE3
            # use a cartesianimpedance controller
            self.ctrl=controllers.CartesianImpedance(nullspace_stiffness=0.1, filter_coeff=1.0)
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
        # how long to try to vacuum for
        self.gripeps=timedelta(seconds=0.5)

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
            self.ctrl.set_control(self._endeff.t, UnitQuaternion(self._endeff).vec_xyzs)
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
        elif cmd=="vacuum":
            self.vacuum()
        elif cmd=="home":
            self.home()
        elif cmd=="update":
            self.update_endeff()
        elif cmd=="quit":
            self.gripper.stop()
            quit()
        else:
            print(f"Invalid command {cmd}")

    def forward(self):
        self.endeff=SE3.Trans(self.moveeps,0,0) * self._endeff

    def backward(self):
        self.endeff=SE3.Trans(-self.moveeps,0,0) * self._endeff

    def right(self):
        self.endeff=SE3.Trans(0,-self.moveeps,0) * self._endeff

    def left(self):
        self.endeff=SE3.Trans(0,self.moveeps,0) * self._endeff

    def up(self):
        self.endeff=SE3.Trans(0,0,self.moveeps) * self._endeff

    def down(self):
        self.endeff=SE3.Trans(0,0,-self.moveeps) * self._endeff

    def home(self):
        try:
            self.panda.move_to_start()
            self.update_endeff()
            self.panda.start_controller(self.ctrl)
        except Exception as e:
            # will give an error if in sim, likely
            print(e)

    def vacuum(self):
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
        
        
# init
teleop=Teleop("real", "10.0.0.2")
with teleop.panda.create_context(frequency=1000) as ctx:
    while ctx.ok():
        teleop.process_key(getch())
