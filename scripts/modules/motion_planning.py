import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats # type: ignore
from modules.control import control_gripper,control_robot,finger_angle_to_width, start_force_control_gripper, stop_force_control_gripper
from modules.transform import transform_terminator


def planning_grasp_path(robot,any_data_dict,AKSolver,simulation_context):

    setting_joint_positions = np.array([0, -1.447, 0.749, -0.873, -1.571, 0])
    putting_joint_positions = np.array([-0.85, -1.147, 0.549, -0.873, -1.571, 0])
    complete_joint_positions = robot.get_joint_positions()
    T_target = transform_terminator(any_data_dict)
    target_translation = T_target[:3,3]
    target_rotation = T_target[:3,:3]
    print(f">>target_position>>:\n{target_translation}\n>>target_rotation>>\n:{target_rotation}")

    target_translation_up10 = target_translation + np.array([0,0,0.1])
    target_rotation_up10 = target_rotation
    print(f">>target_position_up10>>:\n{target_translation_up10}\n>>target_rotation_up10>>\n:{target_rotation_up10}")

    target_orientation = rot_matrices_to_quats(target_rotation)
    target_orientation_up10 = rot_matrices_to_quats(target_rotation_up10)
    target_joint_states,succ = AKSolver.compute_inverse_kinematics(target_translation,target_orientation)
    target_up10_joint_states,succ = AKSolver.compute_inverse_kinematics(target_translation_up10,target_orientation_up10)
    target_joint_positions = target_joint_states.joint_positions
    target_up10_joint_positions = target_up10_joint_states.joint_positions
    
    complete_joint_positions = control_robot(robot,complete_joint_positions[:6],target_up10_joint_positions,simulation_context)

    complete_joint_positions = control_robot(robot,complete_joint_positions[:6],target_joint_positions,simulation_context)
    # for _ in range(5):
    #     simulation_context.step(render = True)
    end_position,end_rotation = AKSolver.compute_end_effector_pose()
    print(f"==end_position==:\n{end_position}\n==end_rotation==\n:{end_rotation}")
    
    start_force_control_gripper(robot,simulation_context)
    for _ in range(40):
        simulation_context.step(render = True)

    complete_joint_positions = control_robot(robot,target_joint_positions,target_up10_joint_positions,simulation_context)

    complete_joint_positions = control_robot(robot,target_up10_joint_positions,putting_joint_positions,simulation_context)
    for _ in range(10):
        simulation_context.step(render = True)
    
    stop_force_control_gripper(robot,simulation_context)
    complete_joint_positions = robot.get_joint_positions()
    finger_joint_width = finger_angle_to_width(complete_joint_positions[6])
    complete_joint_positions = control_gripper(robot,finger_joint_width,0.14,complete_joint_positions,simulation_context)

    complete_joint_positions = control_robot(robot,putting_joint_positions,setting_joint_positions,simulation_context)
    for _ in range(50):
        simulation_context.step(render = True)