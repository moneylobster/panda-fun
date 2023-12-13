'''
simple teleoperation with the end-effector orientation staying constant.
'''
from datetime import timedelta, datetime

import numpy as np
from spatialmath import SE3, UnitQuaternion

from utils.getch import getch


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
    def __init__(self, mode="real", ip="10.0.0.2", record=False):
        """
        
        Parameters
        ----------
        mode : string: "real" or "sim"
            whether to run teleop in real life or in simulation
        ip : string
            ip of robot for "real" option

        """
        self.record=record
        
        if mode=="real":
            # run in real life
            self.real=True
            self.panda=panda_py.Panda(ip)
            self.gripper=panda_py.libfranka.VacuumGripper(ip)
            self._endeff=SE3(self.panda.get_pose()) #store as SE3
            # use a cartesianimpedance controller
            self.ctrl=controllers.CartesianImpedance(nullspace_stiffness=0.1, filter_coeff=1.0)
            self.panda.start_controller(self.ctrl)
            if self.record:
                # start logging
                self.panda.enable_logging()
        else:
            # run in simulation maybe?
            self.real=False
            self.env=swift.Swift()
            self.env.launch(realtime=True)
            self.panda=rtb.models.Panda()
            self._endeff=self.panda.fkine(self.panda.q)
            self.env.add(self.panda)
            if self.record:
                print("WARNING: recording in sim not implemented yet")

        # how much to move by in each press
        self.moveeps=0.01
        # how long to try to vacuum for
        self.gripeps=timedelta(seconds=0.5)

    def update_endeff(self):
        if self.real:
            #real
            self._endeff=SE3(self.panda.get_pose())
        else:
            #sim
            self._endeff=self.panda.fkine(self.panda.q)

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
            #real
            # self.panda.move_to_joint_position(panda_py.ik(self._endeff.data[0]))
            self.ctrl.set_control(self._endeff.t, UnitQuaternion(self._endeff).vec_xyzs)
        else:
            #sim
            arrived=False
            while not arrived:
                v,arrived=rtb.p_servo(self.panda.fkine(self.panda.q),self._endeff,1)
                self.panda.qd = np.linalg.pinv(self.panda.jacobe(self.panda.q)) @ v
                self.env.step(0.05)
            # self.panda.q=self.panda.ikine_LM(self._endeff, q0=self.panda.q).q
            

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
            self.quit()
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
        if self.real:
            #real
            self.panda.move_to_start()
            self.update_endeff()
            self.panda.start_controller(self.ctrl)
        else:
            #sim
            self.panda.q=self.panda.qr
            self.update_endeff()
            self.env.step(0.05)

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

    def quit(self):
        if self.real:
            self.gripper.stop()
            if self.record:
                # stop logging, save log
                self.panda.disable_logging()
                log=self.panda.get_log()
                filename=f"log_{datetime.now().isoformat()}.npy"
                np.save(filename,log)
                print(f"Saved log as {filename}")
        quit()
        

if __name__=="__main__":
    # read second argument to see if we run it in sim or real
    import sys
    if len(sys.argv)==1:
        print("Specify sim or real.")
    elif sys.argv[1]=="real":
        # real
        import panda_py
        import panda_py.libfranka
        from panda_py import controllers
        teleop=Teleop("real", "10.0.0.2", True)
        with teleop.panda.create_context(frequency=1000) as ctx:
            while ctx.ok():
                teleop.process_key(getch())
    elif sys.argv[1]=="sim":
        # sim
        import roboticstoolbox as rtb
        import swift
        teleop=Teleop("sim")
        while True:
            teleop.process_key(getch())
