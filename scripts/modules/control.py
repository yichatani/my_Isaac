import numpy as np
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats # type: ignore
from omni.isaac.core.utils.types import ArticulationAction # type: ignore
from omni.isaac.dynamic_control import _dynamic_control   # type: ignore
from omni.isaac.core.utils.stage import open_stage, get_current_stage, add_reference_to_stage # type: ignore
from pxr import UsdPhysics  # type: ignore
import time

# robot_path = "/ur10e"

def interpolate_joint_positions(start_positions, target_positions, steps=50):
    """
    Interpolate between start and target joint positions.
    """
    interpolated_positions = np.linspace(start_positions, target_positions, steps)
    return interpolated_positions

def width_to_finger_angle(width):
    """Transfer from width to angle"""
    max_width = 0.140  # For 2F-140
    max_angle = 0.7    # Maximum finger_joint angle in radians
    scale = max_width / max_angle

    if width < 0 or width > max_width:
        raise ValueError(f"Width {width} out of range [0, {max_width}]")

    # Convert width to finger joint angle
    finger_angle = (max_width - width) / scale
    return finger_angle

def finger_angle_to_width(finger_angle):
    """Transfer from angle to width"""
    max_width = 0.140  # For 2F-140
    max_angle = 0.7    # Maximum finger_joint angle in radians
    scale = max_width / max_angle

    if finger_angle > max_angle:
        finger_angle = max_angle

    if finger_angle < 0:
        raise ValueError(f"Finger angle {finger_angle} < 0")

    # Convert finger joint angle to width
    width = max_width - (scale * finger_angle)
    return width

def control_gripper(robot, finger_start, finger_target,   # finger_start is width
                    complete_joint_positions, simulation_context,recording_event):
    """
        To control gripper open and close by width
        By position control
    """
    finger_start = width_to_finger_angle(finger_start)
    finger_target = width_to_finger_angle(finger_target)
    finger_moves = interpolate_joint_positions(finger_start, finger_target, steps=50)
    for position in finger_moves:
        action = ArticulationAction(joint_positions=np.array([position]), joint_indices=np.array([6]))
        robot.apply_action(action)
        # complete_joint_positions[6] = position
        # complete_joint_positions[7:10] = [-position] * 3  # left_inner_knuckle_joint, right_inner_knuckle_joint, right_outer_knuckle_joint
        # complete_joint_positions[10:12] = [position] * 2  # left_inner_finger_joint, right_inner_finger_joint
        # robot.set_joint_positions(complete_joint_positions)
        simulation_context.step(render=True)
        # recording_event.set()
        if not recording_event.is_set():
            recording_event.set()
    return complete_joint_positions


def set_joint_stiffness_damping(stage, joint_path, stiffness, damping):
    """ Set stiffness and damping for a specific joint using UsdPhysics.DriveAPI """
    joint_path = "/ur10e/robotiq_140_base_link/finger_joint"
    joint_prim = stage.GetPrimAtPath(joint_path)
    # joint_prim = "/ur10e/robotiq_140_base_link/finger_joint"
    if not joint_prim or not joint_prim.IsValid():
        print(f"Joint '{joint_path}' not found in the stage.")
        return False
    
    # Apply the Drive API for the joint
    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")

    # Set stiffness and damping values
    drive_api.GetStiffnessAttr().Set(stiffness)
    drive_api.GetDampingAttr().Set(damping)

    print(f"Drive API applied to '{joint_path}' with stiffness={stiffness}, damping={damping}")
    return True

def start_force_control_gripper(robot, simulation_context,recording_event):
    gripper_dof_name = "finger_joint"
    gripper_dof_path = "/ur10e/robotiq_140_base_link/finger_joint"
    gripper_dof_index = robot.dof_names.index(gripper_dof_name)
    stage = get_current_stage()

    print("robot_dof_names:",robot.dof_names)
    # exit()

    set_joint_stiffness_damping(stage, gripper_dof_path, stiffness=0.0, damping=0.0)
    simulation_context.step(render=True)
    # recording_event.set()
    if not recording_event.is_set():
        recording_event.set()

    torques = robot.get_applied_joint_efforts()
    torques[gripper_dof_index] = 4     # max torque
    robot.set_joint_efforts(torques)
    simulation_context.step(render=True)
    # recording_event.set()
    if not recording_event.is_set():
        recording_event.set()

def stop_force_control_gripper(robot,simulation_context,recording_event):
    gripper_dof_name = "finger_joint"
    gripper_dof_path = "/ur10e/robotiq_140_base_link/finger_joint"
    gripper_dof_index = robot.dof_names.index(gripper_dof_name)
    stage = get_current_stage()
    # complete_joint_positions = robot.get_joint_positions()
    # robot.set_joint_positions(complete_joint_positions)

    # Save original stiffness & damping
    original_stiffness = 10000.0  # Default, modify if needed
    original_damping = 1000.0  # Default, modify if needed

    set_joint_stiffness_damping(stage, gripper_dof_path, stiffness=original_stiffness, damping=original_damping)
    #robot.set_joint_positions(robot.get_joint_positions())
    # for _ in range(50):
    #     simulation_context.step(render=True)

    torques = robot.get_applied_joint_efforts()
    torques[gripper_dof_index] = 0.0
    robot.set_joint_efforts(torques)
    simulation_context.step(render=True)
    # recording_event.set()
    if not recording_event.is_set():
        recording_event.set()

    print("Gripper reset!")



def control_robot(robot, start_position, target_position, simulation_context,recording_event):
    """To control the robot by joint positions"""
    trajectory = interpolate_joint_positions(start_position, target_position, steps=50)
    for joint_positions in trajectory:
        complete_joint_positions = robot.get_joint_positions()
        complete_joint_positions[:6] = joint_positions
        # robot.set_joint_positions(complete_joint_positions)
        action = ArticulationAction(complete_joint_positions)
        robot.apply_action(action)
        simulation_context.step(render=True)
        # recording_event.set()
        if not recording_event.is_set():
            recording_event.set()
    return complete_joint_positions



