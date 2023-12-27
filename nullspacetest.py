'''
operate arm and record as binarized np array
'''
import numpy as np
import time
import sys

import panda_py
from panda_py import controllers
import spatialmath.base as smb

import matplotlib.pyplot as plt


if len(sys.argv)!=3:
  raise RuntimeError("Please specify trans_stiff and rot_stiff")

tra_stiff=sys.argv[1] #default is 200
rot_stiff=sys.argv[2] #default is 10

panda=panda_py.Panda("10.0.0.2")
panda.move_to_start()

LEN = 8

input(f'Next, teach a trajectory for {LEN} seconds. Press enter to begin.')
panda.teaching_mode(True)
panda.enable_logging(LEN * 1000)
time.sleep(LEN)
panda.teaching_mode(False)

log=panda.get_log()
print(log.keys())

# for cartesian ctrl
endeffs=np.array(log["O_T_EE"])
endeffs=endeffs.reshape((-1, 4, 4)).transpose((0, 2, 1))
poss=[endeff[:3,3] for endeff in endeffs]
rots=[smb.r2q(endeff[:3,:3],order="xyzs") for endeff in endeffs]

print(f"endeffs: {endeffs[0:2]} total size:{endeffs.shape}")
print(f"pos/rot for first: {poss[0]} {rots[0]}")

# for joint ctrl
q = panda.get_log()['q']
dq = panda.get_log()['dq']

##########################################################
input('Press enter to replay: JOINT POS')
panda.move_to_joint_position(q[0])
i = 0

ctrl = controllers.JointPosition()

panda.enable_logging(LEN * 1000)
panda.start_controller(ctrl)
with panda.create_context(frequency=1000, max_runtime=LEN) as ctx:
    while ctx.ok():
        ctrl.set_control(q[i], dq[i]) #joint pos.
        i += 1
jointposlog=panda.get_log()
jointposq=jointposlog['q']
np.save("jointposq.npy",jointposq)
##########################################################
input('Press enter to replay: CARTESIAN IMP., DEFAULT NULLSPACE')
panda.move_to_joint_position(q[0])
i = 0

ctrl = controllers.CartesianImpedance(
    impedance=np.array(
        [[tra_stiff,0,0,0,0,0],
         [0,tra_stiff,0,0,0,0],
         [0,0,tra_stiff,0,0,0],
         [0,0,0,rot_stiff,0,0],
         [0,0,0,0,rot_stiff,0],
         [0,0,0,0,0,rot_stiff]]),
    filter_coeff=1.0)

panda.enable_logging(LEN * 1000)
panda.start_controller(ctrl)
with panda.create_context(frequency=1000, max_runtime=LEN) as ctx:
    while ctx.ok():
        ctrl.set_control(poss[i], rots[i]) #cartesian imp.
        i += 1
cartimpdeflog=panda.get_log()
cartimpdefq=cartimpdeflog['q']
np.save("cartimpdefq.npy",cartimpdefq)
##########################################################
input('Press enter to replay: CARTESIAN IMP., NULLSPACE=q')
panda.move_to_joint_position(q[0])
i = 0

ctrl = controllers.CartesianImpedance(
  impedance=np.array(
                    [[tra_stiff,0,0,0,0,0],
                     [0,tra_stiff,0,0,0,0],
                     [0,0,tra_stiff,0,0,0],
                     [0,0,0,rot_stiff,0,0],
                     [0,0,0,0,rot_stiff,0],
                     [0,0,0,0,0,rot_stiff]]),
  filter_coeff=1.0)

panda.enable_logging(LEN * 1000)
panda.start_controller(ctrl)      
with panda.create_context(frequency=1000, max_runtime=LEN) as ctx:
    while ctx.ok():
        ctrl.set_control(poss[i], rots[i], q[i]) #cartesian imp.
        i += 1
cartimplog=panda.get_log()
cartimpq=cartimplog['q']
np.save("cartimpdefq.npy",cartimpq)

plt.plot(jointposq)
plt.savefig("jointposq.png")

plt.plot(cartimpdefq)
plt.savefig("cartimpdefq.png")

plt.plot(cartimpq)
plt.savefig("cartimpq.png")
