import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
# import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
import panda_py
from panda_py import controllers
# from rtde_control import RTDEControlInterface
# from rtde_receive import RTDEReceiveInterface
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from skill_utils.format_pose import to_format, from_format
import roboticstoolbox as rtb
from spatialmath import SE3
from datetime import timedelta, datetime

def format_to_6d(pose):
    """
    turn a 9D vector in (xyz)+(2 columns of SO3) format into
    the format that the posetrajectoryinterpolator takes:
    (xyz)+(rotvec)
    """
    pose = from_format(np.array(pose))
    pose = np.reshape(pose, (4,4))
    pose_6d=np.hstack((pose[:3,3],st.Rotation.from_matrix(pose[:3,:3]).as_rotvec()))
    return pose_6d


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    HOME = 3
    VACUUM = 4


class PandaInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """

    def __init__(self,
            shm_manager: SharedMemoryManager,
            robot_ip,
            frequency=125,
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,
            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="PandaPositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                ['ActualTCPPose',"O_T_EE"],
                ['ActualQ',"q"],
                ['ActualQd',"dq"],

                ['TargetTCPPose',"O_T_EE_d"],
                ['TargetQ',"q_d"],
                ['TargetQd',"dq_d"]
            ]
        # rtde_r = RTDEReceiveInterface(hostname=robot_ip)
        panda=panda_py.Panda(robot_ip)
        example = dict()
        pstate=panda.get_state()
        for key in receive_keys:
            example[key[0]] = np.array(getattr(pstate, key[1]))
        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

        # how long to try to vacuum for
        self.gripeps=timedelta(seconds=0.5)

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[PandaPositionalController] Controller process spawned at {self.pid}")


    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose=format_to_6d(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose=format_to_6d(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def home(self):
        """
        home the robot
        """
        message = {
            'cmd' : Command.HOME.value
        }
        self.input_queue.put(message)

    def vacuum(self):
        """
        toggle vacuum
        """
        message = {
            'cmd' : Command.VACUUM.value
        }
        self.input_queue.put(message)


    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= controller implementations ============
    def controller_setup(self):
        '''
        set up and return a controller.
        '''
        # use a cartesianimpedance controller
        tra_stiff=210 #default is 200
        rot_stiff=50 #default is 10
        ctrl=controllers.CartesianImpedance(
            impedance=np.array(
                [[tra_stiff,0,0,0,0,0],
                 [0,tra_stiff,0,0,0,0],
                 [0,0,tra_stiff,0,0,0],
                 [0,0,0,rot_stiff,0,0],
                 [0,0,0,0,rot_stiff,0],
                 [0,0,0,0,0,rot_stiff]]),
            nullspace_stiffness=0.1,
            filter_coeff=1.0)
        return ctrl, {}

    def controller_run(self, ctrl, pose_command, misc):
        '''
        send a command using the controller.
        '''
        angsquat=st.Rotation.from_rotvec(pose_command[3:]).as_quat()
        ctrl.set_control(pose_command[:3],angsquat)

    # ========= convenience func ============
    def setup(self, panda):
        '''
        do all the setup stuff
        '''
        # create controller
        ctrl, misc=self.controller_setup()
        # add panda to misc
        misc["panda"]=panda

        if ctrl:
            # start controller
            panda.start_controller(ctrl)

        return ctrl, misc

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start rtde
        robot_ip = self.robot_ip
        panda=panda_py.Panda(robot_ip)
        gripper=panda_py.libfranka.VacuumGripper(robot_ip)
        # rtde_c = RTDEControlInterface(hostname=robot_ip)
        # rtde_r = RTDEReceiveInterface(hostname=robot_ip)

        try:
            if self.verbose:
                print(f"[PandaPositionalController] Connect to robot: {robot_ip}")

            # set parameters
            if self.tcp_offset_pose is not None:
                raise NotImplementedError("Tool Center Point setting not done yet")
                # rtde_c.setTcp(self.tcp_offset_pose)
            if self.payload_mass is not None:
                raise NotImplementedError("Payload Mass setting is not done yet")
                # if self.payload_cog is not None:
                #     assert rtde_c.setPayload(self.payload_mass, self.payload_cog)
                # else:
                #     assert rtde_c.setPayload(self.payload_mass)

            # init pose
            if self.joints_init is not None:
                panda.move_to_joint_position(self.joints_init)
                # assert rtde_c.moveJ(self.joints_init, self.joints_init_speed, 1.4)

            # main loop
            dt = 1. / self.frequency
            # curr_pose = rtde_r.getActualTCPPose()
            # curr_pose=SE3(panda.get_pose())
            # curr_pose_6d=np.hstack((curr_pose.t,UnitQuaternion(curr_pose).eul()))
            curr_pose=format_to_6d(to_format(panda.get_pose()))
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            # create controller
            ctrl, misc=self.setup(panda)

            iter_idx = 0
            with panda.create_context(frequency=self.frequency) as ctx:
                while ctx.ok():
                    # start control iteration
                    # t_start = rtde_c.initPeriod()
                    t_start=time.time()

                    # send command to robot
                    t_now = time.monotonic()
                    # diff = t_now - pose_interp.times[-1]
                    # if diff > 0:
                    #     print('extrapolate', diff)
                    pose_command = pose_interp(t_now)
                    # vel = 0.5
                    # acc = 0.5

                    # execute command
                    self.controller_run(ctrl, pose_command, misc)

                    # assert rtde_c.servoL(pose_command,
                    #     vel, acc, # dummy, not used by ur5
                    #     dt,
                    #     self.lookahead_time,
                    #     self.gain)

                    # update robot state
                    state = dict()
                    pstate=panda.get_state()
                    for key in self.receive_keys:
                        state[key[0]] = np.array(getattr(pstate, key[1]))
                    state['robot_receive_timestamp'] = time.time()
                    self.ring_buffer.put(state)

                    # fetch command from queue
                    try:
                        commands = self.input_queue.get_all()
                        n_cmd = len(commands['cmd'])
                    except Empty:
                        n_cmd = 0

                    # execute commands
                    for i in range(n_cmd):
                        command = dict()
                        for key, value in commands.items():
                            command[key] = value[i]
                        cmd = command['cmd']

                        if cmd == Command.STOP.value:
                            # stop immediately, ignore later commands
                            break
                        elif cmd == Command.SERVOL.value:
                            # since curr_pose always lag behind curr_target_pose
                            # if we start the next interpolation with curr_pose
                            # the command robot receive will have discontinouity
                            # and cause jittery robot behavior.
                            target_pose = command['target_pose']
                            duration = float(command['duration'])
                            curr_time = t_now + dt
                            t_insert = curr_time + duration
                            pose_interp = pose_interp.drive_to_waypoint(
                                pose=target_pose,
                                time=t_insert,
                                curr_time=curr_time,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed
                            )
                            last_waypoint_time = t_insert
                            if self.verbose:
                                print("[PandaPositionalController] New pose target:{} duration:{}".format(
                                    target_pose, duration))
                        elif cmd == Command.SCHEDULE_WAYPOINT.value:
                            target_pose = command['target_pose']
                            target_time = float(command['target_time'])
                            # translate global time to monotonic time
                            target_time = time.monotonic() - time.time() + target_time
                            curr_time = t_now + dt
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=target_pose,
                                time=target_time,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                            last_waypoint_time = target_time
                        elif cmd == Command.HOME.value:
                            # home
                            panda.stop_controller()
                            if type(self.joints_init)==type(None):
                                panda.move_to_start()
                            else:
                                panda.move_to_joint_position(self.joints_init)
                            # recreate controller
                            ctrl, misc=self.setup(panda)
                        elif cmd == Command.VACUUM.value:
                            # vacuum
                            # is vacuum on?
                            state=gripper.read_once()
                            if state.part_present:
                                try:
                                    gripper.drop_off(self.gripeps)
                                except Exception:
                                    # if unsuccessful
                                    gripper.stop()
                            else:
                                try:
                                    gripper.vacuum(3,self.gripeps)
                                except Exception:
                                    # if unsuccessful
                                    gripper.stop()
                        else:
                            break

                    # regulate frequency
                    # rtde_c.waitPeriod(t_start)

                    # first loop successful, ready to receive command
                    if iter_idx == 0:
                        self.ready_event.set()
                    iter_idx += 1

                    if self.verbose:
                        print(f"[PandaPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # manditory cleanup
            # decelerate
            # rtde_c.servoStop()

            # terminate
            # rtde_c.stopScript()
            # rtde_c.disconnect()
            # rtde_r.disconnect()

            self.ready_event.set()

class PandaInterpolationControllerRRMC(PandaInterpolationController):
    '''
    Alternative controller that uses resolved rate motion control
    '''

    # ========= controller implementations ============
    def controller_setup(self):
        '''
        set up and return a controller.
        '''
        # use a velocity controller
        ctrl=controllers.IntegratedVelocity()
        # Initialize roboticstoolbox model
        panda_rtb = rtb.models.Panda()
        misc={}
        misc["panda_rtb"]=panda_rtb
        return ctrl, misc

    def controller_run(self, ctrl, pose_command, misc):
        '''
        send a command using the controller.
        '''
        # extract from misc
        panda=misc["panda"]
        panda_rtb=misc["panda_rtb"]

        angsmat=st.Rotation.from_rotvec(pose_command[3:]).as_matrix()
        T_goal=SE3.Rt(angsmat, t=pose_command[:3])
        curr_pose=panda.get_pose()
        v,arrived=rtb.p_servo(curr_pose, T_goal, 1.7, 0.1)
        qd=np.linalg.pinv(panda_rtb.jacobe(panda.q)) @ v
        ctrl.set_control(qd[:7])

class PandaInterpolationControllerStrict(PandaInterpolationController):
    '''
    Alternative panda controller that uses move_to_pose
    '''
    # ========= controller implementations ============
    def controller_setup(self):
        '''
        set up and return a controller.
        '''
        # no controller here.
        return None, {}

    def controller_run(self, ctrl, pose_command, misc):
        '''
        send a command using the controller.
        '''
        # extract from misc
        panda=misc["panda"]

        #impedance cart. and rot. params
        param1=800
        param2=160

        angsmat=st.Rotation.from_rotvec(pose_command[3:]).as_matrix()
        T_goal=SE3.Rt(angsmat, t=pose_command[:3])
        panda.move_to_pose(T_goal,
                           impedance=np.diag([param1,param1,param1,param2,param2,param2]))

class PandaInterpolationControllerIK(PandaInterpolationController):
    '''
    Alternative panda controller that uses inverse kinematics
    '''
    # ========= controller implementations ============
    def controller_setup(self):
        '''
        set up and return a controller.
        '''
        # no controller here.
        return None, {}

    def controller_run(self, ctrl, pose_command, misc):
        '''
        send a command using the controller.
        '''
        # extract from misc
        panda=misc["panda"]

        angsquat=st.Rotation.from_rotvec(pose_command[3:]).as_quat()

        newq=panda_py.ik(pose_command[:3], angsquat, q_init=panda.q)
        panda.move_to_joint_position(newq)


