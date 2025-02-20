"""

    Deprecated now below. 
    
    Don't use.

"""

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


"""

    Deprecated below.

"""
def initial_camera__(camera_path):
    """Deprecated"""
    rep.new_layer()
    camera = rep.get.prim_at_path(camera_path)
    if not camera:
        raise RuntimeError(f"Camera not found at path: {camera_path}")
    print(f"Using camera at path: {camera_path}")
    # Turn off camera's physics
    camera_prim = get_current_stage().GetPrimAtPath(camera_path)
    physics_api = UsdPhysics.RigidBodyAPI.Apply(camera_prim)
    physics_api.GetRigidBodyEnabledAttr().Set(False)

    render_product = rep.create.render_product(camera_path, resolution=(1920, 1080))
    # Render_product = rep.create.render_product(camera_path)
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_annotator.attach([render_product])
    depth_annotator.attach([render_product])

    return rgb_annotator,depth_annotator

def set_camera_parameters__(camera):
    # OpenCV camera matrix and width and height of the camera sensor, from the calibration file
    width, height = 1920, 1080
    # camera_matrix = [[1662.77, 0.0, 970.94], [0.0, 1281.77, 600.37], [0.0, 0.0, 1.0]]
    camera_matrix = [[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]

    # Pixel size in microns, aperture and focus distance from the camera sensor specification
    # Note: to disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
    pixel_size = 3 * 1e-3   # in mm, 3 microns is a common pixel size for high resolution cameras
    f_stop = 1.8            # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
    focus_distance = 0.6    # in meters, the distance from the camera to the object plane

    # Calculate the focal length and aperture size from the camera matrix
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
    horizontal_aperture =  pixel_size * width                   # The aperture size in mm
    vertical_aperture =  pixel_size * height
    focal_length_x  = fx * pixel_size
    focal_length_y  = fy * pixel_size
    focal_length = (focal_length_x + focal_length_y) / 2         # The focal length in mm

    # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
    camera.set_focal_length(focal_length / 10.0)                # Convert from mm to cm (or 1/10th of a world unit)
    camera.set_focus_distance(focus_distance)                   # The focus distance in meters
    camera.set_lens_aperture(f_stop * 100.0)                    # Convert the f-stop to Isaac Sim units
    camera.set_horizontal_aperture(horizontal_aperture / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
    camera.set_vertical_aperture(vertical_aperture / 10.0)

    camera.set_clipping_range(0.05, 1.0e5)

    return camera
