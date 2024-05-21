import panda_py
from spatialmath import SE3
import scipy.spatial.transform as st

panda=panda_py.Panda("10.0.0.2")
current_pose=panda.get_pose()

print(f"Current pose is {current_pose}")

newpose=SE3(current_pose)+SE3(0.1,0,0)

print(newpose.data)
newrot=newpose.data[0][:3,:3]

# panda.move_to_pose(newpose.t,st.Rotation.from_matrix(newrot).as_quat())

### ALTERNATE APPROACH:

newq=panda_py.ik(newpose.t,st.Rotation.from_matrix(newrot).as_quat(), q_init=panda.q)
print(f"old q:{panda.q} new q:{newq}")

panda.move_to_joint_position(newq)
