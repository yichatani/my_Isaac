import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
import h5py
import math
import numpy as np
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats # type: ignore
from modules.control import control_gripper,control_robot,finger_angle_to_width, start_force_control_gripper, stop_force_control_gripper
from modules.transform import transform_terminator
from modules.record_data import recording

DATA_DIR = os.path.join(ROOT_DIR + "/../../episodes")

def planning_grasp_path(robot,cameras, any_data_dict,AKSolver,simulation_context,episode_path):

    setting_joint_positions = np.array([0, -1.447, 0.749, -0.873, -1.571, 0])
    putting_joint_positions = np.array([-0.85, -1.147, 0.549, -0.873, -1.571, 0])
    complete_joint_positions = robot.get_joint_positions()
    T_target = transform_terminator(any_data_dict)
    target_translation = T_target[:3,3]
    target_rotation = T_target[:3,:3]
    # print(f">>target_position>>:\n{target_translation}\n>>target_rotation>>\n:{target_rotation}")

    target_translation_up20 = target_translation + np.array([0,0,0.2])
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

    # target_joint_positions.setflags(write=True)
    # target_up20_joint_positions.setflags(write=True)
    target_joint_positions = target_joint_positions.copy()
    target_up20_joint_positions = target_up20_joint_positions.copy()

    # make sure the wrist don't rotate too much, to prevent collision
    if abs(target_joint_positions[5]) > math.pi/2:
        target_joint_positions[5] = abs(target_joint_positions[5]) - math.pi
    if abs(target_up20_joint_positions[5]) > math.pi/2:
        target_up20_joint_positions[5] = abs(target_up20_joint_positions[5]) - math.pi

    # exit()
    
    complete_joint_positions = control_robot(robot,cameras,complete_joint_positions[:6],target_up20_joint_positions,simulation_context,episode_path,is_record=True,steps=70)

    complete_joint_positions = control_robot(robot,cameras,complete_joint_positions[:6],target_joint_positions,simulation_context,episode_path,is_record=True, steps=70)
    #     simulation_context.step(render = True)
    # end_position,end_rotation = AKSolver.compute_end_effector_pose()
    # print(f"==end_position==:\n{end_position}\n==end_rotation==\n:{end_rotation}")
    
    start_force_control_gripper(robot)
    for _ in range(40):
        simulation_context.step(render = True)
        # if not recording_event.is_set():
        #     recording_event.set()
        recording(robot,cameras,episode_path,simulation_context)
        

    complete_joint_positions = control_robot(robot,cameras,complete_joint_positions[:6],target_up20_joint_positions,simulation_context,episode_path,is_record=True,steps=50)
    

    # stop_event.set()
    # record_thread.join()
    # print("Recording thread stopped.")


    # num_files = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
    # for _ in range(5):
    #     simulation_context.step(render = True)
    # complete_joint_positions = robot.get_joint_positions()
    # print("completele_joint_positions[6]",complete_joint_positions[6])
    # episode_path = os.path.join(DATA_DIR, f"episode_{num_files-1}.h5")
    # with h5py.File(episode_path, "a") as f:
    #     label_dataset = f["label"]
    #     label_dataset.resize((label_dataset.shape[0] + 1, 1))
    #     if round(complete_joint_positions[6],1)==0.7:
    #         label_dataset[-1] = 0   # 0 means negative samples
    #     else:
    #         label_dataset[-1] = 1   # 1 means positive samples
    with h5py.File(episode_path, "a") as f:
        if "label" not in f:
            # If dataset does not exist, create it with initial size (1,1) and allow resizing
            label_dataset = f.create_dataset("label", shape=(1,), dtype=np.int32, compression="gzip")

            label_dataset[0] = 1  # Default to positive
        else:
            label_dataset = f["label"]

        for _ in range(80):
            simulation_context.step(render = True)
            if math.isclose(complete_joint_positions[6] * 0.14/0.725, 0.14, abs_tol=1e-2):  # Tolerance of 0.01
                label_dataset[0] = 0  # Negative sample
            else:
                label_dataset[0] = 1  # Positive sample
            
    print("Updated label dataset in", episode_path)


    complete_joint_positions = control_robot(robot,cameras,complete_joint_positions[:6],putting_joint_positions,simulation_context,episode_path,is_record=False,steps=30)
    for _ in range(10):
        simulation_context.step(render = True)
        # if not recording_event.is_set():
        #     recording_event.set()
    
    stop_force_control_gripper(robot)
    complete_joint_positions = robot.get_joint_positions()
    finger_joint_width = finger_angle_to_width(complete_joint_positions[6])
    complete_joint_positions = control_gripper(robot,cameras,finger_joint_width,0.14,complete_joint_positions,simulation_context,episode_path, is_record=False)

    complete_joint_positions = control_robot(robot,cameras,complete_joint_positions[:6],setting_joint_positions,simulation_context, episode_path,is_record=False,steps=30)
    for _ in range(50):
        simulation_context.step(render = True)
        # if not recording_event.is_set():
        #     recording_event.set()