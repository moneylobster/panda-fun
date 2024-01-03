'''
play back a recording made in real life
'''
import numpy as np
import time
import sys

import panda_py
from panda_py import controllers
import spatialmath.base as smb

# parse args
if len(sys.argv)<2:
  raise RuntimeError("Please supply the name of the .npy file to play. Optionally specify control mode: cartesian, joint or jointpos (default)")

if len(sys.argv)>=3:
    ctrl_mode=sys.argv[2]
else:
    ctrl_mode="jointpos"
    
logname=sys.argv[1]

# load log
log=np.load(logname, allow_pickle=True)
log=log.item()

# connect to panda
panda=panda_py.Panda("10.0.0.2")
panda.move_to_start()

# for cartesian ctrl
endeffs=np.array(log["O_T_EE"])
endeffs=endeffs.reshape((-1, 4, 4)).transpose((0, 2, 1))
poss=[endeff[:3,3] for endeff in endeffs]
rots=[smb.r2q(endeff[:3,:3],order="xyzs") for endeff in endeffs]

# for joint ctrl
q = log['q']
dq = log['dq']

input('Ready. Press enter to play trajectory.')
panda.move_to_joint_position(q[0])
i = 0

# configure controller
if ctrl_mode=="jointpos":
  ctrl = controllers.JointPosition()
  arg1=q
  arg2=dq
elif ctrl_mode=="joint":
  raise NotImplementedError("Joint Impedance isn't implemented yet")
  ctrl = controllers.JointImpedance()
  arg1=q
elif ctrl_mode=="cartesian":
  ctrl = controllers.CartesianImpedance(filter_coeff=1.0)
  arg1=poss
  arg2=rots
else:
  raise Exception(f"Unknown controller mode {ctrl_mode}")

# Start controller  
panda.start_controller(ctrl)
with panda.create_context(frequency=1000, max_runtime=LEN) as ctx:
  while ctx.ok():
    ctrl.set_control(arg1[i], arg2[i])
    i += 1
