import os
import yaml
import h5py
import math
import numpy as np
from termcolor import cprint
from omni.isaac.core.prims import XFormPrim # type: ignore
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats, euler_angles_to_quats,quats_to_euler_angles # type: ignore
from modules.control import control_gripper,control_robot,start_force_control_gripper, \
stop_force_control_gripper,width_to_finger_angle,control_both_robot_gripper
from modules.transform import transform_terminator,get_end_effector_pose,T_pose_2_joints
from modules.record_data import recording, create_episode_file

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR + "/../../episodes")

with open(ROOT_DIR + '/../prim_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
obj_prim_paths = [
    item['path'] for item in config['obj_prim_paths'] if item.get('enabled', False)
]

def calculate_slope(x, y):
    # Fit a linear regression model to the data
    slope, intercept = np.polyfit(x, y, 1)
    return slope

def remove_close_values(arr, threshold=1e-3):
    """Remove values that are very close to each other."""
    arr = np.sort(arr)  # Ensure sorted order
    diff = np.diff(arr)  # Compute differences between consecutive elements
    mask = np.insert(diff > threshold, 0, True)  # Keep first element, remove close ones
    return arr[mask]

def if_grasping_success(prim_paths:list) -> bool:
    """Can be used to check if the grasping is successful
        check the height of objects
    """
    for prim_path in prim_paths:
        obj = XFormPrim(prim_path)
        if obj.get_world_pose()[0][2] > 0.80: # this value should be checked
            return True
    return False


def planning_grasp_path(robot,cameras,any_data_dict,AKSolver,simulation_context,
                        initial_joint_positions:np.array=np.array([0, -1.447, 0.749, -0.873, -1.571, 0]),
                        ending_joint_positions:np.array=np.array([-0.85, -1.147, 0.549, -0.873, -1.571, 0]))-> bool:
    """
    Plan the grasp path for the robot.
    """
    T_target = transform_terminator(any_data_dict)
    target_translation = T_target[:3,3]
    target_rotation = T_target[:3,:3]

    target_translation_up20 = target_translation + np.array([0,0,0.2])
    target_rotation_up20 = target_rotation

    target_translation_end = np.array([0.5,-0.10,0.7])
    target_rotation_end = target_rotation

    target_joint_positions = T_pose_2_joints(target_translation, target_rotation, AKSolver)
    if target_joint_positions is None:
        print("No valid target joint positions found.")
        return False
    target_joint_positions_up20 = T_pose_2_joints(target_translation_up20, target_rotation_up20, AKSolver)
    if target_joint_positions_up20 is None:
        print("No valid target up20 joint positions found.")
        return False
    target_joint_positions_end = T_pose_2_joints(target_translation_end, target_rotation_end, AKSolver)
    if target_joint_positions_end is None:
        print("No valid target end joint positions found.")
        return False

    initial_width = any_data_dict["width"] + 0.02
    if initial_width > 0.14:
        initial_width = 0.14
    target_joint_positions_up20 = np.append(target_joint_positions_up20, width_to_finger_angle(initial_width))

    episode_path = create_episode_file(cameras)
    ###1 go to the up20 position
    complete_joint_positions = robot.get_joint_positions()
    complete_joint_positions = control_both_robot_gripper(robot,cameras,complete_joint_positions[:7],target_joint_positions_up20,
                                             simulation_context,episode_path,is_record=True,steps=50)
    
    target_joint_positions = np.append(target_joint_positions, width_to_finger_angle(initial_width))
    ###2 go to the target position
    complete_joint_positions = control_both_robot_gripper(robot,cameras,complete_joint_positions[:7],target_joint_positions,
                                             simulation_context,episode_path,is_record=True, steps=20)
    
    start_force_control_gripper(robot)
    # select 16 steps to record
    selected_steps = np.linspace(0, 19, 10).astype(int)
    for _ in range(20):
        simulation_context.step(render = True)
        ###3 close the gripper
        if _ in selected_steps:
            recording(robot, cameras, episode_path, simulation_context)
    ###4 go to the end joint position to check if success or not
    target_joint_positions_end = np.append(target_joint_positions_end, robot.get_joint_positions()[6])
    complete_joint_positions = control_both_robot_gripper(robot,cameras,complete_joint_positions[:7],target_joint_positions_end,
                                             simulation_context,episode_path,is_record=True,steps=30)
    with h5py.File(episode_path, "a") as f:
        if "label" not in f:
            # If dataset does not exist, create it with initial size (1,1) and allow resizing
            label_dataset = f.create_dataset("label", shape=(0,), dtype=np.int32, compression="gzip")
            label_dataset[0] = 0  # Default to negative
        else:
            label_dataset = f["label"]
            label_dataset[0] = 0  # Default to negative

        # check_width = np.array([])
        for _ in range(10):
            simulation_context.step(render = True)
            # check_width = np.append(check_width, robot.get_joint_positions()[6])
        
        if if_grasping_success(obj_prim_paths):
            label_dataset[0] = 1
            cprint("####Success!####","blue")
        else:
            cprint("<<<<Faile!>>>>>>","red")
            
    print("Updated label dataset in", episode_path)

    complete_joint_positions = control_robot(robot,cameras,complete_joint_positions[:6],ending_joint_positions,
                                             simulation_context,episode_path,is_record=False,steps=30)
    for _ in range(10):
        simulation_context.step(render = True)
    
    stop_force_control_gripper(robot)
    # reset the gripper
    complete_joint_positions = control_gripper(robot,cameras,complete_joint_positions[6],0,
                                               simulation_context,episode_path, is_record=False)
    # back to the initial position
    complete_joint_positions = control_robot(robot,cameras,complete_joint_positions[:6],initial_joint_positions,
                                             simulation_context, episode_path,is_record=False,steps=20)
    for _ in range(10):
        simulation_context.step(render = True)
    
    return True