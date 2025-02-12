import numpy as np
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats # type: ignore
from omni.isaac.core.utils.types import ArticulationAction # type: ignore
from omni.isaac.dynamic_control import _dynamic_control   # type: ignore
from omni.isaac.core.utils.stage import open_stage, get_current_stage, add_reference_to_stage # type: ignore
from pxr import UsdPhysics  # type: ignore
import time

# robot_path = "/ur10e"

def get_data(simulation_context,rgb_annotator,depth_annotator):
    """Deprecated"""
    # Get data
    for _ in range(4):
        simulation_context.step(render=True)
    
    rgb_data = rgb_annotator.get_data()
    print("rgb_shape:",rgb_data.shape)
    depth_data = depth_annotator.get_data()

    print("Depth min:", np.min(depth_data), "max:", np.max(depth_data))

    if rgb_data is None or depth_data is None:
        raise RuntimeError("Failed to retrieve RGB or Depth data.")
    data_dict = {
        "rgb": rgb_data,
        "depth": depth_data
    }
    return data_dict

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
                    complete_joint_positions, simulation_context):
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



def start_force_control_gripper(robot, simulation_context):
    gripper_dof_name = "finger_joint"
    gripper_dof_path = "/ur10e/robotiq_140_base_link/finger_joint"
    gripper_dof_index = robot.dof_names.index(gripper_dof_name)
    stage = get_current_stage()

    print("robot_dof_names:",robot.dof_names)
    # exit()

    set_joint_stiffness_damping(stage, gripper_dof_path, stiffness=0.0, damping=0.0)
    simulation_context.step(render=True)

    torques = robot.get_applied_joint_efforts()
    torques[gripper_dof_index] = 4     # max torque
    robot.set_joint_efforts(torques)
    simulation_context.step(render=True)

def stop_force_control_gripper(robot,simulation_context):
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

    print("Gripper reset!")



def control_robot(robot, start_position, target_position, simulation_context):
    """To control the robot by joint positions"""
    trajectory = interpolate_joint_positions(start_position, target_position, steps=50)
    for joint_positions in trajectory:
        complete_joint_positions = robot.get_joint_positions()
        complete_joint_positions[:6] = joint_positions
        # robot.set_joint_positions(complete_joint_positions)
        action = ArticulationAction(complete_joint_positions)
        robot.apply_action(action)
        simulation_context.step(render=True)
    return complete_joint_positions


##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##

"""

    Deprecated now below. 
    
    Don't use.

"""

def force_control_gripper(robot, max_torque, simulation_context):

    gripper_dof_name = "finger_joint"
    gripper_dof_path = "/ur10e/robotiq_140_base_link/finger_joint"
    gripper_dof_index = robot.dof_names.index(gripper_dof_name)
    stage = get_current_stage()

    print("robot_dof_names:",robot.dof_names)
    # exit()

    # Save original stiffness & damping
    original_stiffness = 10000.0  # Default, modify if needed
    original_damping = 1000.0  # Default, modify if needed


    set_joint_stiffness_damping(stage, gripper_dof_path, stiffness=0.0, damping=0.0)
    simulation_context.step(render=True)

    max_time = 2.0
    start_time = time.time()

    # while time.time() - start_time < max_time:
    
    for _ in range(100):
        torques = robot.get_applied_joint_efforts()
        torques[gripper_dof_index] = 4     # max torque
        robot.set_joint_efforts(torques)
        simulation_context.step(render=True)

    # while time.time() - start_time < max_time:
    #     elapsed_time = time.time() - start_time
    #     torque = (elapsed_time / max_time) * max_torque

    #     torques = robot.get_applied_joint_efforts()
    #     torques[gripper_dof_index] = torque
    #     robot.set_joint_efforts(torques)

    #     simulation_context.step(render=True)

    # torque_step = 0.1
    # initial_torque = 0
    # current_torque = max_torque
    # while current_torque <= max_torque:
    #     torques = robot.get_applied_joint_efforts()
    #     torques[gripper_dof_index] = current_torque
    #     robot.set_joint_efforts(torques)
    #     current_torque += torque_step
    #     simulation_context.step(render=True)
    
    # while current_torque > 0:
    #     torques = robot.get_applied_joint_efforts()
    #     torques[gripper_dof_index] = current_torque
    #     robot.set_joint_efforts(torques)
    #     current_torque -= torque_step
    #     simulation_context.step(render=True)
    
    complete_joint_positions = robot.get_joint_positions()
    robot.set_joint_positions(complete_joint_positions)
    set_joint_stiffness_damping(stage, gripper_dof_path, stiffness=original_stiffness, damping=original_damping)
    #robot.set_joint_positions(robot.get_joint_positions())
    for _ in range(50):
        simulation_context.step(render=True)

    torques = robot.get_applied_joint_efforts()
    torques[gripper_dof_index] = 0.0
    robot.set_joint_efforts(torques)
    simulation_context.step(render=True)

    print("Gripper reset!")

def rrt_control_robot_by_joints(robot,rrt_planner,path_planner_visualizer,target_joint_positions,simulation_context):
    """
        Don't use. Not fixed well yet, maybe the auto inverse computation by rrt is different.
        Or maybe the restriction is too weak. 
    """
    rrt_planner.set_cspace_target(target_joint_positions)
    rrt_planner.update_world()
    plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)
    while plan:
        action = plan.pop(0)
        robot.apply_action(action)
        simulation_context.step(render = True)


def rrt_control_robot_by_endpose(robot,rrt_planner,path_planner_visualizer,T_target,simulation_context):
    """
        Don't use. Not fixed well yet, maybe the auto inverse computation by rrt is different.
        Or maybe the restriction is too weak. 
    """
    T_target_translation = T_target[:3,3]
    T_target_orientation = T_target[:3,:3]
    T_target_orientation = rot_matrices_to_quats(T_target_orientation)
    rrt_planner.set_end_effector_target(T_target_translation, T_target_orientation)
    rrt_planner.update_world()
    plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)
    while plan:
        action = plan.pop(0)
        robot.apply_action(action)
        simulation_context.step(render = True)


# def effort_control_gripper(robot,simulation_context):
#     """To control gripper open and close by effort"""
#     for _ in range(50):
#         robot.set_joint_efforts([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#         simulation_context.step(render=True)





# def force_control_gripper_v1(robot):
#     initial_torque = 0.0
#     max_torque = 5.0  
#     torque_step = 0.1
#     current_torque = initial_torque
#     while current_torque <= max_torque:
#         robot.set_joint_efforts(np.array([current_torque]), joint_indices=np.array([6]))
#         current_torque += torque_step
#         time.sleep(0.1)


# def force_control_gripper_v2(robot,simulation_context):
#     dc_interface = dc.acquire_dynamic_control_interface()
#     # robot = dc.get_articulation(robot_path)
#     # gripper_dof = dc.find_articulation_dof(robot, "finger_joint")
#     # initial_torque = 0.0
#     # max_torque = 5.0  
#     # torque_step = 0.1
#     # current_torque = initial_torque
#     # dc.set_dof_effort(6, 5)
#     for _ in range(50):
#         dof_index = 6
#         effort = 10.0  # 10 N·m
#         dc_interface.set_dof_effort(dof_index, effort)
#         simulation_context.step(render=True)
#     # while current_torque <= max_torque:
#     #     dc.set_dof_effort(6, 5)
#     #     current_torque += torque_step
#     #     time.sleep(0.1)

# def force_control_gripper_v3(robot_path, simulation_context):
#     dc_interface = _dynamic_control.acquire_dynamic_control_interface()
#     # dc_interface = dc.acquire_dynamic_control_interface()

#     # index
#     robot_handle = dc_interface.get_articulation(robot_path)

#     print(f"robot handle:{robot_handle}")

#     gripper_dof_index = dc_interface.find_articulation_dof(robot_handle, "finger_joint")
#     if gripper_dof_index == -1:
#         print("DC robot can't find")
#         return

#     # stiffness and damping to let it could be controles
#     dof_properties = dc_interface.get_dof_properties(gripper_dof_index)
#     dof_properties.stiffness = 0.0
#     dof_properties.damping = 0.0
#     dc_interface.set_dof_properties(gripper_dof_index, dof_properties)

#     print(f"DOF index: {gripper_dof_index}")
    
#     max_time = 2.0
#     start_time = simulation_context.get_current_time()
#     effort = 10.0  # 10 N·m

#     while simulation_context.get_current_time() - start_time < max_time:
#         dc_interface.set_dof_effort(gripper_dof_index, effort)
#         simulation_context.step(render=True)
#     dof_properties.stiffness = 10000
#     dof_properties.damping = 1000
#     print("Success!")


import os
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.extensions import get_extension_path_from_name # type: ignore
from omni.isaac.motion_generation.lula import RRT
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy, PathPlannerVisualizer

mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

urdf_path = rmp_config_dir + "/universal_robots/ur10e/ur10e.urdf"
robot_description_path = rmp_config_dir + "/universal_robots/ur10e/rmpflow/ur10e_robot_description.yaml"
rmpflow_config_path = rmp_config_dir + "/universal_robots/ur10e/rmpflow/ur10e_rmpflow_config.yaml"

def ur10e_RmpFlow(robot, Target ,steps=50):
    """
        Need to check.
    """
    rmpflow = RmpFlow(
            robot_description_path = robot_description_path,
            urdf_path = urdf_path,
            rmpflow_config_path = rmpflow_config_path,
            end_effector_frame_name = "tool0",
            maximum_substep_size = 0.00334
        )
    articulation_rmpflow = ArticulationMotionPolicy(robot, rmpflow)
    rmpflow.set_end_effector_target(
        target_position=Target[:3,3],
        target_orientation = R.from_matrix(Target[:3,:3]).as_quat()
    )
    action = articulation_rmpflow.get_next_articulation_action(steps)
    robot.apply_action(action)
    # print("action:",action)
    # exit()
    # return action
    
    

def setup_path_rrt_planner(yaml_path, urdf_path,rrt_config_path):
    """
    Setup RRT path planner.
    """
    rrt_planner = RRT(
        robot_description_path=yaml_path,
        urdf_path=urdf_path,
        rrt_config_path=rrt_config_path,
        end_effector_frame_name="tool0"
    )
    return rrt_planner


############ Initial rrt_planner
# rrt_planner = setup_path_rrt_planner(yaml_path, urdf_path,rrt_config_path)
# active_joints = rrt_planner.get_active_joints()
# watched_joints = rrt_planner.get_watched_joints()
# print(f"Active joints in C-space: {active_joints}")
# print(f"Watched joints in C-space:{watched_joints}")
# path_planner_visualizer = PathPlannerVisualizer(robot, rrt_planner)


def rrt_planning(rrt_planner, T, complete_joint_positions):
    ## Set target and planning
    rrt_planner.set_end_effector_target(
        target_translation=T[:3,3],
        target_orientation=R.from_matrix(T[:3,:3]).as_quat()
    )
    # Update
    rrt_planner.update_world()
    # motion planning
    plan = rrt_planner.compute_path(
        active_joint_positions=complete_joint_positions[:6],
        watched_joint_positions=None,
    )
    if plan is not None:
        print("Success, later try Excute Zero ...")
    else:
        print("plan_Zero planning Failed!!!")
    return plan


# def move_rrt_origin(robot, rrt_planner, simulation_context, any_data_dict):


#     # while True:
#     #     ur10e_RmpFlow(robot, T_tool0_2_baselink ,steps=0.5)
#     #     simulation_context.step(render = True)


#     ## Plan to Grasp
#     plan_0 = rrt_planning(rrt_planner=rrt_planner,   # T is the target Transfor
#                         T=T_tool0_2_baselink,
#                         complete_joint_positions=complete_joint_positions)    # complete_joint_positions is the current joints values
#     if plan_0 is not None:
#         for i in range(1, len(plan_0)):
#             complete_joint_positions = control_robot(robot,plan_0[i-1],plan_0[i],
#                                                         complete_joint_positions,simulation_context)
#         print("Successfully Reached Grasp! ^_^")
#         complete_joint_positions = control_gripper(robot=robot, 
#                                             # finger_start=0.14,
#                                             finger_start=any_data_dict["width"],
#                                             finger_target=0,
#                                             complete_joint_positions=complete_joint_positions,
#                                             simulation_context=simulation_context)
#         plan_0 = None

#     # transform_stage = Usd.Stage.Open(usd_file_path)
#     # frame_tool0 = transform_stage.GetPrimAtPath(tool0_path)
#     # xformable_tool0 = UsdGeom.Xformable(frame_tool0)
#     # T_tool0_2_baselink = xformable_tool0.GetLocalTransformation()
#     # T_tool0_2_baselink = np.array(T_tool0_2_baselink).T
#     # print("##tool0 to baselink##:\n",T_tool0_2_baselink)
    
#     ## Go to Putting
#     rrt_planner.set_cspace_target(putting_joint_positions)
#     plan_1 = rrt_planner.compute_path(
#                 active_joint_positions=complete_joint_positions[:6],
#                 watched_joint_positions=None
#         )
#     if plan_1 is not None:
#         for i in range(1, len(plan_1)):
#             complete_joint_positions = control_robot(robot,plan_1[i-1],plan_1[i],
#                                                         complete_joint_positions,simulation_context)
#         print("Successfully Reached Putting! ^_^")
#         complete_joint_positions = control_gripper(robot=robot, 
#                                             # finger_start=any_data_dict["width"],
#                                             finger_start=0,
#                                             finger_target=0.14,
#                                             complete_joint_positions=complete_joint_positions,
#                                             simulation_context=simulation_context)
#         plan_1 = None
    
#     ## Return Home
#     rrt_planner.set_cspace_target(active_joint_positions)
#     plan_2 = rrt_planner.compute_path(
#                 active_joint_positions=putting_joint_positions,
#                 watched_joint_positions=None
#         )
#     if plan_2 is not None:
#         for i in range(1, len(plan_2)):
#             complete_joint_positions = control_robot(robot,plan_2[i-1],plan_2[i],
#                                                         complete_joint_positions,simulation_context)
#         print("Successfully Reached Home! ^_^")
#         plan_2 = None