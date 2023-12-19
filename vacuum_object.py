"""
Uses a vacuum gripper to pick up an object, then drops it.
"""
import sys
import time
from datetime import timedelta

import panda_py
import panda_py.libfranka

if __name__=='__main__':
    if len(sys.argv) < 2:
        raise RuntimeError(f'Usage: python {sys.argv[0]} <robot-hostname>')

    # Connect to the gripper via the same IP as the robot arm
    # This doesn't prevent you from connecting to the arm
    gripper=panda_py.libfranka.VacuumGripper(sys.argv[1])

    try:
        # Print gripper state
        state=gripper.read_once()
        print(f"""Gripper State:
        is vacuum within setpoint: {state.in_control_range}
        part detached: {state.part_detached}
        part present: {state.part_present}
        device status: {state.device_status}
        actual power: {state.actual_power}
        vacuum: {state.vacuum}""")
        
        # Vacuum the object
        # The first argument is the vacuum pressure level
        # The second argument is how long to try vacuuming for before giving an error
        try:
            gripper.vacuum(3, timedelta(seconds=1))
        except:
            raise RuntimeError("Failed to grasp object.")
            
        # Wait 3 seconds and check if the object is still grasped.
        time.sleep(3)
        # This works by checking if the pressure level of the vacuum
        # is within a specified range
        state=gripper.read_once()
        if not state.in_control_range:
            print("Object lost.")
        else:
            print("Releasing object.")
            # Release the object
            # The time argument specifies when to time-out.
            try:
                gripper.drop_off(timedelta(seconds=1))
            except:
                raise RuntimeError("Failed to drop object off.")
    finally:
        # Stop whatever the gripper is doing when program terminates.
        gripper.stop()
