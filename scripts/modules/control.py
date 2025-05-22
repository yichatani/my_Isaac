import numpy as np
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats # type: ignore
from omni.isaac.core.utils.types import ArticulationAction # type: ignore
from omni.isaac.dynamic_control import _dynamic_control   # type: ignore
from omni.isaac.core.utils.stage import open_stage, get_current_stage, add_reference_to_stage # type: ignore
from pxr import UsdPhysics  # type: ignore
from modules.record_data import recording, observing

def interpolate_joint_positions(start_positions, target_positions, steps=50)-> np.ndarray:
    """
    Interpolate between start and target joint positions.
    """
    interpolated_positions = np.linspace(start_positions, target_positions, steps)
    return interpolated_positions

def width_to_finger_angle(width: float) -> float:
    """Transfer from width to angle"""
    max_width = 0.140  # For 2F-140. # Now open.
    max_angle = 0.785    # Maximum finger_joint angle in radians. # Now close.
    scale = max_width / max_angle

    if width < 0 or width > max_width:
        raise ValueError(f"Width {width} out of range [0, {max_width}]")

    # Convert width to finger joint angle
    finger_angle = (max_width - width) / scale # when width == max_width, finger_angle = 0
    return finger_angle

def finger_angle_to_width(finger_angle: float) -> float:
    """Transfer from angle to width"""
    max_width = 0.140  # For 2F-140
    max_angle = 0.785    # Maximum finger_joint angle in radians
    scale = max_width / max_angle

    if finger_angle > max_angle:
        finger_angle = max_angle

    if finger_angle < 0:
        # raise ValueError(f"Finger angle {finger_angle} < 0")
        finger_angle = 0

    # Convert finger joint angle to width
    width = max_width - (scale * finger_angle)
    return width

def control_gripper(robot, cameras, finger_start, finger_target,
                    simulation_context,episode_path, is_record=True, steps = 5)-> np.ndarray:
    """
        To control gripper open and close by angle
        By position control
        1 dimension: gripper
    """
    finger_moves = interpolate_joint_positions(finger_start, finger_target, steps)
    for position in finger_moves:
        action = ArticulationAction(joint_positions=np.array([position]), joint_indices=np.array([6]))
        robot.apply_action(action)
        for _ in range(1):
            simulation_context.step(render=True)
        if is_record:
            recording(robot,cameras,episode_path,simulation_context,is_compression=True)
    complete_joint_positions = robot.get_joint_positions()
    return complete_joint_positions


def set_joint_stiffness_damping(stage, joint_path, stiffness, damping):
    """ Set stiffness and damping for a specific joint using UsdPhysics.DriveAPI """
    joint_prim = stage.GetPrimAtPath(joint_path)
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


def start_force_control_gripper(robot):
    """
        Start force control for the gripper
    """
    gripper_dof_name = "finger_joint"
    gripper_dof_path = "/World/ur10e_robotiq2f_140_ROS/ur10e_robotiq2f_140/Robotiq_2F_140_config/finger_joint"
    gripper_dof_index = robot.dof_names.index(gripper_dof_name)
    stage = get_current_stage()

    set_joint_stiffness_damping(stage, gripper_dof_path, stiffness=0.0, damping=0.0)

    torques = robot.get_applied_joint_efforts()
    torques[gripper_dof_index] = 3     # max torque
    robot.set_joint_efforts(torques)


def stop_force_control_gripper(robot):
    """
        Stop force control for the gripper
    
    """
    gripper_dof_name = "finger_joint"
    gripper_dof_path = "/World/ur10e_robotiq2f_140_ROS/ur10e_robotiq2f_140/Robotiq_2F_140_config/finger_joint"
    gripper_dof_index = robot.dof_names.index(gripper_dof_name)
    stage = get_current_stage()

    original_stiffness = 10000.0  # Default, modify if needed
    original_damping = 1000.0  # Default, modify if needed

    set_joint_stiffness_damping(stage, gripper_dof_path, stiffness=original_stiffness, damping=original_damping)

    torques = robot.get_applied_joint_efforts()
    torques[gripper_dof_index] = 0.0
    robot.set_joint_efforts(torques)

    print("Gripper reset!")



def control_robot(robot, cameras, start_position, target_position, simulation_context, episode_path, is_record=True,steps=50)-> np.ndarray:
    """To control the robot by joint positions
        6 dimension: robot joints
    """
    trajectory = interpolate_joint_positions(start_position, target_position, steps)
    for joint_positions in trajectory:
        complete_joint_positions = robot.get_joint_positions()
        complete_joint_positions[:6] = joint_positions
        action = ArticulationAction(complete_joint_positions)
        robot.apply_action(action)
        for _ in range(1):
            simulation_context.step(render=True)
        if is_record:
            recording(robot, cameras,episode_path,simulation_context,is_compression=True)
    complete_joint_positions = robot.get_joint_positions()
    return complete_joint_positions


def control_robot_by_policy(robot, record_camera_dict:dict, actions:np.ndarray,simulation_context,data_sample,obs_steps)->dict:
    """To control the robot by policy
        7 dimension:
            6 dimension: robot joints
            1 dimension: gripper
    """
    assert actions.shape[1] == 7, "Expected 7 DoF action"
    for action in actions:
        complete_joint_positions = robot.get_joint_positions()
        complete_joint_positions[:7] = action
        complete_joint_positions[6] = complete_joint_positions[6] - 0.02
        robot.apply_action(ArticulationAction(joint_positions=complete_joint_positions))
        data_sample = observing(robot,record_camera_dict,simulation_context,data_sample,obs_steps=obs_steps)
        for _ in range(15):
            simulation_context.step(render=True)
    return data_sample

def control_both_robot_gripper(robot, cameras, start_joint_position, target_joint_position, simulation_context, episode_path, is_record=True,steps=50)-> np.ndarray:
    """"To control the robot and the gripper at the same time
        7 dimension:
            6 dimension: robot joints
            1 dimension: gripper
    """
    trajectory = interpolate_joint_positions(start_joint_position, target_joint_position, steps)
    for joint_positions in trajectory:
        complete_joint_positions = robot.get_joint_positions()
        complete_joint_positions[:7] = joint_positions
        action = ArticulationAction(complete_joint_positions)
        robot.apply_action(action)
        for _ in range(10):
            simulation_context.step(render=True)
        if is_record:
            recording(robot, cameras,episode_path,simulation_context,is_compression=True)
    complete_joint_positions = robot.get_joint_positions()
    return complete_joint_positions