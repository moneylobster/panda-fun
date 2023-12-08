import numpy as np
import time

import panda_py
from panda_py import controllers
import spatialmath.base as smb
# import swift


panda=panda_py.Panda("10.0.0.2")

# env = swift.Swift()
# env.launch(realtime=True)
# simpanda = rtb.models.Panda()

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

LEN = 10
input(f'Next, teach a trajectory for {LEN} seconds. Press enter to begin.')
panda.teaching_mode(True)
panda.enable_logging(LEN * 1000)
time.sleep(LEN)
panda.teaching_mode(False)

log=panda.get_log()
print(log.keys())
endeffs=np.array(log["O_T_EE"])
endeffs=endeffs.reshape((-1, 4, 4)).transpose((0, 2, 1))
poss=[endeff[:3,3] for endeff in endeffs]
rots=[smb.r2q(endeff[:3,:3],order="xyzs") for endeff in endeffs]

print(f"endeffs: {endeffs[0:2]} total size:{endeffs.shape}")
print(f"pos/rot for first: {poss[0]} {rots[0]}")

q = panda.get_log()['q']
# dq = panda.get_log()['dq']

input('Press enter to replay trajectory')
panda.move_to_joint_position(q[0])
i = 0
# ctrl = controllers.JointPosition()
ctrl = controllers.CartesianImpedance(filter_coeff=1.0)
panda.start_controller(ctrl)
with panda.create_context(frequency=1000, max_runtime=LEN) as ctx:
  while ctx.ok():
    ctrl.set_control(poss[i], rots[i])
    # ctrl.set_control(q[i], dq[i])
    i += 1
    # simpanda.fkine(panda.q)

# ctrl = controllers.CartesianImpedance(filter_coeff=1.0)
# x0 = panda.get_position()
# q0 = panda.get_orientation()
# runtime = np.pi * 4.0
# panda.start_controller(ctrl)

# with panda.create_context(frequency=1e3, max_runtime=runtime) as ctx:
#     while ctx.ok():
#         x_d = x0.copy()
#         x_d[1] += 0.1 * np.sin(ctrl.get_time())
#         ctrl.set_control(x_d, q0)
