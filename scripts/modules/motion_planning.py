import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
import yaml
import h5py
import math
import numpy as np
from omni.isaac.core.prims import XFormPrim # type: ignore
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats # type: ignore
from modules.control import control_gripper,control_robot,finger_angle_to_width, start_force_control_gripper, stop_force_control_gripper,width_to_finger_angle,control_both_robot_gripper
from modules.transform import transform_terminator
from modules.record_data import recording

DATA_DIR = os.path.join(ROOT_DIR + "/../../episodes")

with open(ROOT_DIR + '/../prim_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
robot_path = config['robot_path']
camera_paths = config['camera_paths']
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

def planning_grasp_path(robot,cameras, any_data_dict,AKSolver,simulation_context,episode_path):

    setting_joint_positions = np.array([0, -1.447, 0.749, -0.873, -1.571, 0])
    putting_joint_positions = np.array([-0.85, -1.147, 0.549, -0.873, -1.571, 0])
    complete_joint_positions = robot.get_joint_positions()
    # complete_joint_positions = control_gripper(robot, cameras, finger_angle_to_width(complete_joint_positions[6]),any_data_dict["width"],
    #                                                     complete_joint_positions,simulation_context,episode_path,is_record=True)
    T_target = transform_terminator(any_data_dict)
    target_translation = T_target[:3,3]
    target_rotation = T_target[:3,:3]
    # print(f">>target_position>>:\n{target_translation}\n>>target_rotation>>\n:{target_rotation}")

    target_translation_up20 = target_translation + np.array([0,0,0.4])
    target_rotation_up20 = target_rotation
    # print(f">>target_position_up10>>:\n{target_translation_up10}\n>>target_rotation_up10>>\n:{target_rotation_up10}")

    target_orientation = rot_matrices_to_quats(target_rotation)
    target_orientation_up20 = rot_matrices_to_quats(target_rotation_up20)
    target_joint_states,succ = AKSolver.compute_inverse_kinematics(target_translation,target_orientation)
    target_up20_joint_states,succ = AKSolver.compute_inverse_kinematics(target_translation_up20,target_orientation_up20)
    target_joint_positions = target_joint_states.joint_positions
    target_up20_joint_positions = target_up20_joint_states.joint_positions

    print(f"####Check the target joint_position####:{target_joint_positions}")
    print(f"####Check the up20 joint_position####:{target_up20_joint_positions}")

    target_joint_positions = target_joint_positions.copy()
    target_up20_joint_positions = target_up20_joint_positions.copy()

    # make sure the wrist don't rotate too much, to prevent collision
    if abs(target_joint_positions[5]) > math.pi/2:
        target_joint_positions[5] = abs(target_joint_positions[5]) - math.pi
    if abs(target_up20_joint_positions[5]) > math.pi/2:
        target_up20_joint_positions[5] = abs(target_up20_joint_positions[5]) - math.pi


    target_up20_joint_positions = np.append(target_up20_joint_positions, width_to_finger_angle(any_data_dict["width"]))
    complete_joint_positions = control_both_robot_gripper(robot,cameras,complete_joint_positions[:7],target_up20_joint_positions,
                                             simulation_context,episode_path,is_record=True,steps=40)
    target_joint_positions = np.append(target_joint_positions, width_to_finger_angle(any_data_dict["width"]))
    complete_joint_positions = control_both_robot_gripper(robot,cameras,complete_joint_positions[:7],target_joint_positions,
                                             simulation_context,episode_path,is_record=True, steps=30)

    start_force_control_gripper(robot)
    # select 16 steps to record
    selected_steps = np.linspace(0, 59, 16).astype(int)
    for _ in range(60):
        simulation_context.step(render = True)
        if _ in selected_steps:
            recording(robot, cameras, episode_path, simulation_context)
        

    complete_joint_positions = control_both_robot_gripper(robot,cameras,complete_joint_positions[:7],target_up20_joint_positions,
                                             simulation_context,episode_path,is_record=True,steps=30)
    
    with h5py.File(episode_path, "a") as f:
        if "label" not in f:
            # If dataset does not exist, create it with initial size (1,1) and allow resizing
            label_dataset = f.create_dataset("label", shape=(1,), dtype=np.int32, compression="gzip")

            label_dataset[0] = 0  # Default to negative
        else:
            label_dataset = f["label"]
            label_dataset[0] = 0  # Default to negative

        check_width = np.array([])
        for _ in range(50):
            simulation_context.step(render = True)
            check_width = np.append(check_width, robot.get_joint_positions()[6])
        
        # if calculate_slope(np.arange(2),check_width[-2:])< -0.031497 * check_width[-1] + 0.022048 and not math.isclose(
        #     robot.get_joint_positions()[6] * 0.14/0.725, 0.14, abs_tol=2e-3): 
        # and not robot.get_joint_positions()[6] > 0.7
        if if_grasping_success(obj_prim_paths):
            label_dataset[0] = 1
            print("################")
            print("####Success!####")
            print("################")
        else:
            print("!!!!!!!!!!!!!!!!")
            print("<<<<Faile!>>>>>>")
            print("!!!!!!!!!!!!!!!!")
            
    print("Updated label dataset in", episode_path)


    complete_joint_positions = control_robot(robot,cameras,complete_joint_positions[:6],putting_joint_positions,
                                             simulation_context,episode_path,is_record=False,steps=30)
    for _ in range(10):
        simulation_context.step(render = True)
    
    stop_force_control_gripper(robot)
    complete_joint_positions = robot.get_joint_positions()
    finger_joint_width = finger_angle_to_width(complete_joint_positions[6])
    complete_joint_positions = control_gripper(robot,cameras,finger_joint_width,0.14,complete_joint_positions,
                                               simulation_context,episode_path, is_record=False)

    complete_joint_positions = control_robot(robot,cameras,complete_joint_positions[:6],setting_joint_positions,
                                             simulation_context, episode_path,is_record=False,steps=30)
    for _ in range(50):
        simulation_context.step(render = True)
        # if not recording_event.is_set():
        #     recording_event.set()