from diffusion_policy.real_world.panda_interpolation_controller import PandaInterpolationController

robot = PandaInterpolationController(
    shm_manager=shm_manager,
    robot_ip=robot_ip,
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

print(robot.get_all_state())
