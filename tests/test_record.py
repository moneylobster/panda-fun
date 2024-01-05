'''
operate arm and record as binarized np array
'''
import numpy as np
import time
import sys

import panda_py
from panda_py import controllers
import spatialmath.base as smb


if len(sys.argv)==1:
  raise RuntimeError("Please specify cartesian, joint or jointpos")
ctrl_mode=sys.argv[1]

panda=panda_py.Panda("10.0.0.2")

panda.move_to_start()

# print('Please teach three poses to the robot.')
# positions = []
# panda.teaching_mode(True)
# for i in range(3):
#   print(f'Move the robot into pose {i+1} and press enter to continue.')
#   input()
#   positions.append(panda.q)
#   simpanda.fkine(panda.q)

# panda.teaching_mode(False)
# input('Press enter to move through the three poses.')
# panda.move_to_joint_position(positions)

LEN = 4
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

input('Press enter to replay trajectory')
panda.move_to_joint_position(q[0])
i = 0

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
  
panda.start_controller(ctrl)
with panda.create_context(frequency=1000, max_runtime=LEN) as ctx:
  while ctx.ok():
    # ctrl.set_control(poss[i], rots[i]) #cartesian imp.
    # ctrl.set_control(q[i], dq[i]) #joint pos.
    ctrl.set_control(arg1[i], arg2[i])
    i += 1
