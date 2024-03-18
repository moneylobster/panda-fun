import numpy as np
import time

from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.panda_interpolation_controller import PandaInterpolationController

from skill_utils.format_pose import to_format, from_format

max_pos_speed=0.25
max_rot_speed=0.6
cube_diag = np.linalg.norm([1,1,1])
max_obs_buffer_size=30

shm_manager=SharedMemoryManager()
shm_manager.start()

robot = PandaInterpolationController(
    shm_manager=shm_manager,
    robot_ip="10.0.0.2",
    frequency=125, # UR5 CB3 RTDE
    lookahead_time=0.1,
    gain=300,
    max_pos_speed=max_pos_speed*cube_diag,
    max_rot_speed=max_rot_speed*cube_diag,
    launch_timeout=10,
    tcp_offset_pose=None,
    payload_mass=None,
    payload_cog=None,
    joints_init=None,
    joints_init_speed=1.05,
    soft_real_time=False,
    verbose=True,
    receive_keys=None,
    get_max_k=max_obs_buffer_size
    )

robot.start(wait=True)

state=robot.get_all_state()
print(state)
print(np.reshape(state["ActualTCPPose"],(4,4)).T)
a=to_format(np.reshape(state["ActualTCPPose"],(4,4)).T)
print(a)
b=a[:]
b[1]+=0.01
print(b)
robot.schedule_waypoint(b, time.time()+5)
