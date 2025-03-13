"""Launch the simulation application."""
from omni.kit.app import get_app # type: ignore
from omni.isaac.kit import SimulationApp # type: ignore
simulation_app = SimulationApp({"headless": False})

def enable_extensions():
    extension_manager = get_app().get_extension_manager()
    extension_manager.set_extension_enabled("omni.isaac.motion_generation",True)
    extension_manager.set_extension_enabled("omni.physx", True)
enable_extensions()

"""Rest everything follows."""
# Import necessary libraries
import os
import sys
import signal
import time
import rospkg
from PIL import Image
import numpy as np
import argparse
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import open3d as o3d
from gsnet import AnyGrasp # type: ignore
from graspnetAPI import GraspGroup
from scipy.spatial.transform import Rotation as R
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from omni.isaac.core import World  # type: ignore
from omni.isaac.core.utils.stage import open_stage, get_current_stage, add_reference_to_stage # type: ignore
from omni.isaac.core.prims import RigidPrim, XFormPrim # type: ignore
from omni.isaac.core.articulations import Articulation # type: ignore
from omni.isaac.core.utils.prims import is_prim_path_valid # type: ignore
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver, PathPlannerVisualizer # type: ignore
from omni.isaac.core.simulation_context import SimulationContext # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path # type: ignore
from pxr import UsdPhysics, Sdf # type: ignore
#from omni.isaac.core.utils.collision import check_collision # type: ignore
from omni.isaac.motion_generation.lula import RRT  # type: ignore
import omni.replicator.core as rep # type: ignore

from omni.isaac.core.utils.nucleus import get_assets_root_path # type: ignore
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices # type: ignore
from omni.isaac.core.objects.cuboid import VisualCuboid # type: ignore
from omni.isaac.core.utils.extensions import get_extension_path_from_name # type: ignore
from omni.isaac.core.utils.distance_metrics import rotational_distance_angle # type: ignore

### Paths
usd_file_path = os.path.join(ROOT_DIR, "ur10e_grasp_set.usd")
mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
print("kinematics_config_dir:",kinematics_config_dir)
urdf_path = kinematics_config_dir + "/universal_robots/ur10e/ur10e.urdf"
yaml_path = kinematics_config_dir + "/universal_robots/ur10e/rmpflow/ur10e_robot_description.yaml"
rrt_config_path = os.path.join(ROOT_DIR, "rrt_config.yaml")

### 
robot_path = "/ur10e"
# camera_path = "/ur10e/cam_holder_with_onrobot_single/Camera"
camera_path = "/ur10e/tool0/Camera"
tool0_path = "/ur10e/tool0"
base_path = "/ur10e/base"
baselink_path = "/ur10e/base_link"
robotiqpad_R_path = "/ur10e/right_inner_finger_pad"
robotiqpad_L_path = "/ur10e/left_inner_finger_pad"
###

############ AnyGrasp Parts
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=os.path.join(ROOT_DIR, "log/checkpoint_detection.tar"), help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.14, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.14, cfgs.max_gripper_width))
cfgs.top_down_grasp = True
anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()
############

def create_rotation_matrix(axis, angle_degrees):
    """
    Create a 4x4 homogeneous transformation matrix for a rotation around a given axis.

    Parameters:
        axis (str): Axis of rotation ('x', 'y', or 'z').
        angle_degrees (float): Rotation angle in degrees.

    Returns:
        numpy.ndarray: A 4x4 homogeneous transformation matrix representing the rotation.
    """
    angle_radians = np.radians(angle_degrees)
    c, s = np.cos(angle_radians), np.sin(angle_radians)

    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    # Convert to a 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    return T

def define_grasp_pose(grasp_pose):      # Set for Anygrasp transform case
    """
    Define the grasp pose relative to the camera frame.

    Parameters:
        grasp_pose (object): An object containing `translation` (numpy array of shape (3,))
                             and `rotation_matrix` (numpy array of shape (3, 3)).

    Returns:
        numpy.ndarray: A 4x4 homogeneous transformation matrix representing the final grasp pose.
    """
    # Extract translation and rotation matrix from grasp_pose
    grasp_translation = grasp_pose.translation
    grasp_rotation_matrix = grasp_pose.rotation_matrix

    # Construct the initial homogeneous transformation matrix for the grasp pose
    T_grasp = np.eye(4)  # 4x4 identity matrix
    T_grasp[:3, :3] = grasp_rotation_matrix  # Top-left 3x3 block is the rotation matrix
    T_grasp[:3, 3] = grasp_translation  # Top-right 3x1 block is the translation vector

    
    # Create rotation matrices
    rotate_y_only = create_rotation_matrix('y', 90)  # Rotate 90 degrees around the Y-axis
    rotate_z_only = create_rotation_matrix('z', 90)  # Rotate 90 degrees around the Z-axis

    # Apply the transformations sequentially
    T_grasp = np.dot(T_grasp, rotate_y_only)  # Apply Y-axis rotation
    T_grasp = np.dot(T_grasp, rotate_z_only)  # Apply Z-axis rotation

    return T_grasp

def relative_pose(frame_A,frame_B):
    # Get A and B
    position_A, orientation_A = frame_A.get_world_pose()
    position_B, orientation_B = frame_B.get_world_pose()

    # Transform quat to matrix
    rotation_matrix_A = R.from_quat(orientation_A).as_matrix()
    rotation_matrix_B = R.from_quat(orientation_B).as_matrix()

    # Build the transform matrix
    T_A = np.eye(4)
    T_A[:3, :3] = rotation_matrix_A
    T_A[:3, 3] = position_A

    T_B = np.eye(4)
    T_B[:3, :3] = rotation_matrix_B
    T_B[:3, 3] = position_B

    # Compute B reverse pose
    T_B_inv = np.linalg.inv(T_B)

    # Compute the pose A relative B
    T_A_relative_to_B = np.dot(T_B_inv, T_A)

    return T_A_relative_to_B

def relative_to_world(frame):
    position_frame, orientation_baselink = frame.get_world_pose()
    rotation_frame = R.from_quat(orientation_baselink).as_matrix()
    T_frame = np.eye(4)
    T_frame[:3, :3] = rotation_frame
    T_frame[:3, 3] = position_frame
    return T_frame

def relative_pose_camera(frame_A,frame_B):
    # Get A and B
    position_A, orientation_A = frame_A.get_world_pose()
    position_B, orientation_B = frame_B.get_world_pose()

    # Transform quat to matrix
    rotation_matrix_A = R.from_quat(orientation_A).as_matrix()
    rotation_matrix_B = R.from_quat(orientation_B).as_matrix()

    rotate_y_only = create_rotation_matrix('y', 180)
    rotate_z_only = create_rotation_matrix('z', -90)
    
    # Build the transform matrix
    T_A = np.eye(4)
    T_A[:3, :3] = rotation_matrix_A
    T_A[:3, 3] = position_A

    # Apply the transformations sequentially
    T_A = np.dot(T_A, rotate_y_only)  # Apply Y-axis rotation
    T_A = np.dot(T_A, rotate_z_only)  # Apply Z-axis rotation

    T_B = np.eye(4)
    T_B[:3, :3] = rotation_matrix_B
    T_B[:3, 3] = position_B

    # Compute B reverse pose
    T_B_inv = np.linalg.inv(T_B)

    # Compute the pose A relative B
    T_A_relative_to_B = np.dot(T_B_inv, T_A)

    return T_A_relative_to_B

def T_relative_2_TCP(frame_tool0):

    robotiqpad_R = RigidPrim(prim_path = robotiqpad_R_path)
    robotiqpad_L = RigidPrim(prim_path = robotiqpad_L_path)
    position_R, orientation_R = robotiqpad_R.get_world_pose()
    position_L, orientation_L = robotiqpad_L.get_world_pose()

    end_pose_translation = (position_R + position_L)/2
    assert orientation_R.any() == orientation_L.any()
    end_pose_rotation = R.from_quat(orientation_R).as_matrix()
    
    position_tool0, orientation_tool0 = frame_tool0.get_world_pose()
    orientation_tool0 = R.from_quat(orientation_tool0).as_matrix()

    T_tool0 = np.eye(4)
    T_tool0[:3, :3] = orientation_tool0
    T_tool0[:3, 3] = position_tool0

    T_target = np.eye(4)
    T_target[:3, :3] = end_pose_rotation
    T_target[:3, 3] = end_pose_translation

    # T_adjust = np.array([
    #     [0, -1, 0, 0],
    #     [-1, 0, 0, 0],
    #     [0, 0, -1, 0],
    #     [0, 0, 0, 1]
    # ])

    # T_target = T_target @ T_adjust

    print("TCP:",T_target)

    T_target_inv = np.linalg.inv(T_target)

    # Compute the pose A relative B
    T_tool0_relative_to_target = np.dot(T_target_inv, T_tool0)

    # ##
    # T_tool0_relative_to_target[:3,:3] = np.eye(3)
    # ##
    return T_tool0_relative_to_target

def visualize_pose(position, orientation_matrix, name="/World/TargetMarker"):
    """
    Visualize a pose in Isaac Sim using an arrow or axis marker.
    
    Parameters:
        position (np.ndarray): Position of the target [x, y, z].
        orientation_matrix (np.ndarray): 3x3 orientation matrix.
        name (str): Name of the marker in the scene.
    """
    from pxr import Gf, UsdGeom

    # Convert orientation matrix to quaternion for visualization
    from scipy.spatial.transform import Rotation as R
    quat = R.from_matrix(orientation_matrix).as_quat()  # [x, y, z, w]

    # Create a marker (e.g., a coordinate axis or an arrow)
    marker = XFormPrim(prim_path=name, position=position, orientation=quat)
    
    # Initialize the marker in the simulation
    marker.initialize()

    # Optionally, customize the marker's appearance (e.g., scale, color)
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(name)
    if not prim:
        raise RuntimeError(f"Failed to create marker at {name}")

    # # Create a sphere or geometry for visibility
    # UsdGeom.Sphere.Define(stage, '/World' + "/Sphere")
    # sphere = UsdGeom.Sphere(stage.GetPrimAtPath('/World' + "/Sphere"))
    # sphere.GetRadiusAttr().Set(0.02)  # Set radius for visibility
    # sphere.AddTranslateOp().Set(Gf.Vec3d(*position))

def vis_grasps(gg,cloud):
    trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    cloud.transform(trans_mat)
    grippers = gg.to_open3d_geometry_list()
    for gripper in grippers:
        gripper.transform(trans_mat)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, 
        origin=[0, 0, 0]
    )
    # o3d.visualization.draw_geometries([*grippers, cloud, coord_frame])
    o3d.visualization.draw_geometries([grippers[0], cloud, coord_frame])

def any_grasp(data_dict):
    
    colors = data_dict["rgb"] / 255.0
    depths = data_dict["depth"]

    # get camera intrinsics
    fx, fy = 1662.77, 1281.77
    cx, cy = 970.94, 600.37
    scale = 1.0
    # set workspace to filter output grasps
    xmin, xmax = -0.19, 0.19
    ymin, ymax = -0.19, 0.19
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    colors = colors[:, :3]  # remove transparent Alpha channel
    
    print("points shape:", points.shape, "colors shape:", colors.shape)
    if points.shape[0] == 0:
        print("Warning: Empty point cloud!")
    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=False, dense_grasp=False, collision_detection=False)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    gg = gg[0:20]
    vis_grasps(gg,cloud)
    # exit()

    target_grasp_pose_to_cam = define_grasp_pose(gg[0])

    data_dict = {
        "width":gg[0].width,
        "T":target_grasp_pose_to_cam,
        "score":gg[0].score
    }

    return data_dict

def handle_signal(signum, frame):
    """Handle SIGINT for clean exit."""

    print("Simulation interrupted. Exiting...")
    if 'simulation_context' in globals():
        simulation_context.stop() # type: ignore
    if 'simulation_app' in globals():
        simulation_app.close()
    sys.exit(0)

def find_robot(robot_path):
    """Check if the robot exists in the scene."""
    if is_prim_path_valid(robot_path):
        print(f"Robot found at: {robot_path}")
    else:
        print(f"Robot not found at: {robot_path}")
        exit(1)

def initialize_robot(robot_path):
    """Initialize the robot articulation."""

    robot = Articulation(prim_path=robot_path)
    robot.initialize()

    # launch self_collision
    robot.set_enabled_self_collisions(True)
    print(f"Self-collision enabled: {robot.get_enabled_self_collisions()}")
    
    # enable continuous collision detection
    # enable_continuous_collision(robot_path)

    robot.set_solver_position_iteration_count(64)
    robot.set_solver_velocity_iteration_count(64)
    print("Available DOF Names:", robot.dof_names)
    return robot

def initialize_simulation_context():
    """Initialize and reset the simulation context."""
    simulation_context = SimulationContext()
    simulation_context.initialize_physics()

    simulation_context.reset()
    return simulation_context

def setup_kinematics_solver(yaml_path, urdf_path):
    """Set up the kinematics solver using Lula."""
    kinematics_solver = LulaKinematicsSolver(
        robot_description_path=yaml_path,
        urdf_path=urdf_path
    )
    return kinematics_solver

def setup_path_planner(yaml_path, urdf_path,rrt_config_path):
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

def interpolate_joint_positions(start_positions, target_positions, steps=50):
    """
    Interpolate between start and target joint positions.
    """
    interpolated_positions = np.linspace(start_positions, target_positions, steps)
    return interpolated_positions

def Get_data(simulation_context,rgb_annotator,depth_annotator):

    # Get data
    for _ in range(4):
        simulation_context.step(render=True)
    
    rgb_data = rgb_annotator.get_data()
    depth_data = depth_annotator.get_data()

    print("Depth min:", np.min(depth_data), "max:", np.max(depth_data))

    if rgb_data is None or depth_data is None:
        raise RuntimeError("Failed to retrieve RGB or Depth data.")
    data_dict = {
        "rgb": rgb_data,
        "depth": depth_data
    }

    return data_dict

def width_to_finger_angle(width):
    max_width = 0.140  # For 2F-140
    max_angle = 0.7    # Maximum finger_joint angle in radians
    scale = max_width / max_angle

    if width < 0 or width > max_width:
        raise ValueError(f"Width {width} out of range [0, {max_width}]")

    # Convert width to finger joint angle
    finger_angle = (max_width - width) / scale
    return finger_angle

def finger_angle_to_width(finger_angle):
    max_width = 0.140  # For 2F-140
    max_angle = 0.7    # Maximum finger_joint angle in radians
    scale = max_width / max_angle

    if finger_angle < 0 or finger_angle > max_angle:
        raise ValueError(f"Finger angle {finger_angle} out of range [0, {max_angle}]")

    # Convert finger joint angle to width
    width = max_width - (scale * finger_angle)
    return width

def control_gripper(robot, finger_start, finger_target,   # finger_start is finger_angle
                    complete_joint_positions, simulation_context):
    finger_start = width_to_finger_angle(finger_start)
    finger_target = width_to_finger_angle(finger_target)
    finger_moves = interpolate_joint_positions(finger_start, finger_target, steps=50)
    for position in finger_moves:
        complete_joint_positions[6] = position
        complete_joint_positions[7:10] = [-position] * 3  # left_inner_knuckle_joint, right_inner_knuckle_joint, right_outer_knuckle_joint
        complete_joint_positions[10:12] = [position] * 2  # left_inner_finger_joint, right_inner_finger_joint
        robot.set_joint_positions(complete_joint_positions)
        simulation_context.step(render=True)
    return complete_joint_positions

def control_robot(robot, start_position, target_position, 
                    complete_joint_positions, simulation_context):
    trajectory = interpolate_joint_positions(start_position, target_position, steps=50)
    for joint_positions in trajectory:
        complete_joint_positions = robot.get_joint_positions()
        complete_joint_positions[:6] = joint_positions
        robot.set_joint_positions(complete_joint_positions)
        simulation_context.step(render=True)
    return complete_joint_positions

def save_camera_data(data_dict, output_dir="./output_data"):
    """
    Save RGB and Depth data to files.
    
    Args:
        data_dict (dict): Dictionary with "rgb" and "depth" data.
        output_dir (str): Directory to save the files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save RGB image
    rgb_image = Image.fromarray(data_dict["rgb"])
    rgb_image.save(os.path.join(output_dir, "rgb_image.png"))
    print(f"RGB image saved to {os.path.join(output_dir, 'rgb_image.png')}")
    
    # Save Depth data as normalized grayscale image
    depth_data = data_dict["depth"]
    depth_normalized = ((depth_data - np.min(depth_data)) / np.ptp(depth_data) * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_normalized)
    depth_image.save(os.path.join(output_dir, "depth_image.png"))
    print(f"Depth image saved to {os.path.join(output_dir, 'depth_image.png')}")
    
    # Save Depth data as NumPy file
    np.save(os.path.join(output_dir, "depth_data.npy"), depth_data)
    print(f"Depth data saved to {os.path.join(output_dir, 'depth_data.npy')}")

def resolve_ros_path(path):
    if path.startswith("package://"):
        path = path[len("package://"):]
        package_name, relative_path = path.split("/", 1)
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(package_name)
        return os.path.join(package_path, relative_path)
    return path

def rotation_matrix_to_euler_zyx(R):
    """
    rotation to euler
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0

    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

def transform_terminator(any_data_dict):
    frame_cam = RigidPrim(camera_path)
    frame_tool0 = RigidPrim(tool0_path)
    frame_base = RigidPrim(base_path)
    frame_baselink = RigidPrim(baselink_path)
    
    T_cam_2_world = relative_to_world(frame_cam)
    T_base_2_world = relative_to_world(frame_base)
    T_baselink_2_world = relative_to_world(frame_baselink)
    T_tool0_2_world = relative_to_world(frame_tool0)
    print("tool0 to world:",T_tool0_2_world)
    # exit()
    T_cam_2_tool0 = np.linalg.inv(T_tool0_2_world) @ T_cam_2_world
    T_optic_2_cam = np.array([ # This is a settled value
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    print("T_optic_2_cam:",T_optic_2_cam)
    T_target_2_optic = any_data_dict["T"]

    T_tool0_2_cam = np.linalg.inv(T_cam_2_world) @ T_tool0_2_world
    print("T_tool0_2_cam:",T_tool0_2_cam)
    T_tool0_2_optic = np.linalg.inv(T_optic_2_cam) @ T_tool0_2_cam
    print("#####tool0 to optic#####:",T_tool0_2_optic)
    # exit()

    # exit()
    # T_tool0_2_target = np.linalg.inv(T_target_2_optic) @ T_tool0_2_optic

    # T_tool0_2_tool0  = np.eye(4)
    # T_tool0_2_tool0[:3,:3] = T_tool0_2_target[:3,:3]

    # # T_tool0_2_target = np.linalg.inv(T_target_2_optic) @ T_tool0_2_optic
    # T_tool0_2_target = T_tool0_relative_2_target(frame_tool0,robotiqpad_R,robotiqpad_L)
    # T_optic_2_target =  np.linalg.inv(T_target_2_optic)
    # # T_tool0_2_target[:3,3] = T_tool0_2_target_translation[:3,3]
    # T_tool0_2_target[:3,:3] = T_optic_2_target[:3,:3]


    # T_tool0_2_gripper = np.eye(4)
    # T_tool0_2_gripper[:3,:3] = T_tool0_2_optic[:3,:3]
    T_tool0_2_TCP = T_relative_2_TCP(frame_tool0)
    # T_tool0_2_gripper[:3,3] = T_tool0_2_gripper_translation 
    
    T_cam_2_baselink = relative_pose(frame_cam,frame_baselink)
    # T_tool0_2_baselink = T_cam_2_baselink @ T_optic_2_cam @ T_target_2_optic @ T_tool0_2_gripper
    # T_tool0_2_baselink = T_cam_2_baselink @ T_optic_2_cam @ T_target_2_optic @ T_tool0_2_optic
    T_cam_2_TCP = T_relative_2_TCP(frame_cam)
    T_target_2_TCP = T_cam_2_TCP @ T_optic_2_cam @ T_target_2_optic
    T_TCP_2_target = np.linalg.inv(T_target_2_TCP)
    T_TCP_2_optic = T_optic_2_cam @ T_cam_2_TCP

    T_tool0_2_target = T_cam_2_tool0 @ T_optic_2_cam
    T_tool0_2_target[:3,3] = T_tool0_2_TCP[:3,3]
    T_target_2_baselink = T_cam_2_baselink @ T_optic_2_cam @ T_target_2_optic

    T_adjust = np.array([
        [0, -1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    T_unit = np.eye(4)

    T_TCP_2_target[:3,3] = T_unit[:3,3]

    # T_align = T_target_2_optic @ np.linalg.inv(T_TCP_2_optic)
    T_align = np.eye(4)
    T_align[:3,:3] = T_target_2_optic[:3,:3]

    # T_align[:3,0] = T_target_2_optic[2,:3]
    # T_align[:3,1] = T_target_2_optic[0,:3]
    # T_align[:3,2] = T_target_2_optic[1,:3]

    # T_adjust = np.eye(4)
    # T_adjust[:3,:3] = T_tool0_2_optic[:3,:3]


    T_reori = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])




    T_tool0_2_baselink = T_target_2_baselink @ T_TCP_2_target @ T_tool0_2_TCP # tool0 relative to baselink
    
    
####################################################
    translation = T_tool0_2_baselink[:3, 3]
    R_target = T_target_2_optic[:3, :3]


    yaw_target, pitch_target, roll_target = rotation_matrix_to_euler_zyx(R_target)
    print(f": Yaw = {yaw_target:.2f}°, Pitch = {pitch_target:.2f}°, Roll = {roll_target:.2f}°")

    yaw_tool0 = roll_target
    pitch_tool0 = yaw_target
    roll_tool0 = pitch_target
    rotate_z_only = create_rotation_matrix('x',yaw_target)  # Z  SET
    rotate_y_only = create_rotation_matrix('y',pitch_target) # Y
    rotate_x_only = create_rotation_matrix('z',roll_target) # X

    T_tool0_2_baselink = T_tool0_2_baselink @ rotate_z_only       # Yaw
    
    T_tool0_2_baselink = T_tool0_2_baselink @ rotate_y_only       # Pitch

    T_tool0_2_baselink = T_tool0_2_baselink @ rotate_x_only       # Roll

    # T_tool0_2_baselink = T_tool0_2_baselink @ rotate_x_only       # pitch

    # exit()



    # # rotate_x_only = create_rotation_matrix('x', -90)
    # # T_tool0_2_baselink = T_tool0_2_baselink @ rotate_x_only
    # # rotate_y_only = create_rotation_matrix('y', -90)
    # # T_tool0_2_baselink = T_tool0_2_baselink @ rotate_y_only
    # # rotate_z_only = create_rotation_matrix('z', 90)
    # # T_tool0_2_baselink = T_tool0_2_baselink @ rotate_z_only



    # print("R_target:",R_target)

    # R_reorient = np.array([
    # [0, 0, 1],
    # [1, 0, 0],
    # [0, 1, 0]
    # ])

    # R_reorient_inv = np.linalg.inv(R_reorient)

    # R_tool0 = R_reorient_inv @ R_target @ R_reorient

    # print("R_tool0:",R_tool0)

    # T_tool0 = np.eye(4)
    # T_tool0[:3,:3] = R_tool0
    # T_tool0_2_baselink = T_tool0_2_baselink @ T_tool0
    
    
    
    
    
    # T_tool0_2_baselink = np.eye(4)
    # T_tool0_2_baselink[:3, :3] = R_tool0
    # T_tool0_2_baselink[:3, 3] = translation



    # T_tool0_2_baselink = T_tool0_2_baselink @ T_reori @ T_target_2_optic
    
    
    
    
    
    
    
    
    # T_tool0_2_baselink = T_tool0_2_baselink @ T_align


    # T_tool0_2_baselink = T_target_2_baselink @ T_tool0_2_TCP @ T_target_2_TCP 
    # T_tool0_2_baselink = T_target_2_baselink @ T_TCP_2_optic @ np.linalg.inv(T_tool0_2_TCP)


    # rotate_z_only = create_rotation_matrix('z', -90)
    # T_tool0_2_baselink = T_tool0_2_baselink @ rotate_z_only
    # rotate_y_only = create_rotation_matrix('y', 180)
    # T_tool0_2_baselink = T_tool0_2_baselink @ rotate_y_only


    # rotate_x_only = create_rotation_matrix('x', -90)
    # T_tool0_2_baselink = T_tool0_2_baselink @ rotate_x_only

    # # rotate_x_only = create_rotation_matrix('x', -90)
    # # T_tool0_2_baselink = T_tool0_2_baselink @ rotate_x_only
    # # rotate_y_only = create_rotation_matrix('y', -90)
    # # T_tool0_2_baselink = T_tool0_2_baselink @ rotate_y_only
    # # rotate_z_only = create_rotation_matrix('z', 90)
    # # T_tool0_2_baselink = T_tool0_2_baselink @ rotate_z_only


    T_tool0_2_world = T_baselink_2_world @ T_tool0_2_baselink
    visualize_pose(T_tool0_2_world[:3,3], T_tool0_2_world[:3,:3], name="/World/TargetPoseMarker")
    # T_tool0_2_base = T_tool0_2_world
    # T_tool0_2_base[2,3] = T_tool0_2_base[2,3] - T_baselink_2_world[2,3]
    T_tool0_2_base =  np.linalg.inv(T_base_2_world) @ T_tool0_2_world
    

    # print("camera to world:",T_cam_2_world)
    # print("baselink to world:",T_baselink_2_world)
    # print("tool0 to world:", T_tool0_2_world)

    # print("T_cam_2_tool0:", T_cam_2_tool0)
    # print("T_tool0_2_tool0:", T_tool0_2_tool0)
    # print("tool0 to target:",T_tool0_2_target)
    # print("target to optic:",T_target_2_optic)
    # print("optic to camera:",T_optic_2_cam)
    # print("camera to baselink:",T_cam_2_baselink)
    # print("tool0 to baselink:",T_tool0_2_baselink)
    # print("too0 to world:",T_tool0_2_world)
    # print("##tool0 to base##:",T_tool0_2_base)

    # return T_tool0_2_base
    return T_tool0_2_base

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


############ main
def main():
    
    ##############
    # Initialize #
    ##############
    # Open the stage
    open_stage(usd_path=usd_file_path)
    # Initialize the world and simulation context
    simulation_context = initialize_simulation_context()
    # Locate and initialize the robot
    robot = initialize_robot(robot_path)
    # Initialize camera
    rep.new_layer()
    camera = rep.get.prim_at_path(camera_path)
    if not camera:
        raise RuntimeError(f"Camera not found at path: {camera_path}")
    print(f"Using camera at path: {camera_path}")
    # Turn off camera's physics
    camera_prim = get_current_stage().GetPrimAtPath(camera_path)
    physics_api = UsdPhysics.RigidBodyAPI.Apply(camera_prim)
    physics_api.GetRigidBodyEnabledAttr().Set(False)

    # parent_path = get_prim_at_path(camera_path).GetParent().GetPath()
    # print(f"Camera parent: {parent_path}")

    render_product = rep.create.render_product(camera_path, resolution=(1920, 1080))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_annotator.attach([render_product])
    depth_annotator.attach([render_product])
    # Set Start and putting values(which is for sure)
    complete_joint_positions = robot.get_joint_positions()
    active_joint_positions = np.array([0, -1.447, 0.749, -0.873, -1.571, 0])
    putting_joint_positions = np.array([-0.749, -1.247, 0.549, -0.873, -1.571, 0])
    ############ Initial robot
    complete_joint_positions = control_robot(robot=robot,
                                             start_position=active_joint_positions,
                                             target_position=active_joint_positions,
                                             complete_joint_positions=complete_joint_positions,
                                             simulation_context=simulation_context)
    ############ Initial gripper  
    # complete_joint_positions = control_gripper(robot=robot, 
    #                                            finger_start=0.14,
    #                                            finger_target=0,
    #                                            complete_joint_positions=complete_joint_positions,
    #                                            simulation_context=simulation_context)
    # complete_joint_positions = control_gripper(robot=robot, 
    #                                            finger_start=0,
    #                                            finger_target=0.14,
    #                                            complete_joint_positions=complete_joint_positions,
    #                                            simulation_context=simulation_context)
    ############ Initial rrt_planner
    rrt_planner = setup_path_planner(yaml_path, urdf_path,rrt_config_path)
    active_joints = rrt_planner.get_active_joints()
    print(f"Active joints in C-space: {active_joints}")
    


    ########################
    # Main simulation loop #
    ########################
    signal.signal(signal.SIGINT, handle_signal)  # Graceful exit on Ctrl+C

    for _ in range(5):
        ############ camera
        data_dict = Get_data(simulation_context,rgb_annotator,depth_annotator)
        print(f"RGB Data shape: {data_dict['rgb'].shape}")
        print(f"Depth Data shape: {data_dict['depth'].shape}")
        # save_camera_data(data_dict, output_dir="./output_data")

        #############
        # Transform #
        #############
        any_data_dict = any_grasp(data_dict)
        T_tool0_2_base = transform_terminator(any_data_dict)

        ## Plan to Grasp
        plan_0 = rrt_planning(rrt_planner=rrt_planner,   # T is the target Transfor
                          T=T_tool0_2_base,
                          complete_joint_positions=complete_joint_positions)    # complete_joint_positions is the current joints values
        if plan_0 is not None:
            for i in range(1, len(plan_0)):
                complete_joint_positions = control_robot(robot,plan_0[i-1],plan_0[i],
                                                         complete_joint_positions,simulation_context)
            print("Successfully Reached Grasp! ^_^")
            complete_joint_positions = control_gripper(robot=robot, 
                                               finger_start=0.14,
                                            #    finger_target=any_data_dict["width"],
                                                finger_target=0,
                                               complete_joint_positions=complete_joint_positions,
                                               simulation_context=simulation_context)
            plan_0 = None
        

        ## Go to Putting
        rrt_planner.set_cspace_target(putting_joint_positions)
        plan_1 = rrt_planner.compute_path(
                    active_joint_positions=complete_joint_positions[:6],
                    watched_joint_positions=None
            )
        if plan_1 is not None:
            for i in range(1, len(plan_1)):
                complete_joint_positions = control_robot(robot,plan_1[i-1],plan_1[i],
                                                         complete_joint_positions,simulation_context)
            print("Successfully Reached Putting! ^_^")
            complete_joint_positions = control_gripper(robot=robot, 
                                               # finger_start=any_data_dict["width"],
                                                finger_start=0,
                                               finger_target=0.14,
                                               complete_joint_positions=complete_joint_positions,
                                               simulation_context=simulation_context)
            plan_1 = None


        ## Return Home
        rrt_planner.set_cspace_target(active_joint_positions)
        plan_2 = rrt_planner.compute_path(
                    active_joint_positions=putting_joint_positions,
                    watched_joint_positions=None
            )
        if plan_2 is not None:
            for i in range(1, len(plan_2)):
                complete_joint_positions = control_robot(robot,plan_2[i-1],plan_2[i],
                                                         complete_joint_positions,simulation_context)
            print("Successfully Reached Home! ^_^")
            plan_2 = None


        # # Step simulation
        for _ in range(5):
            simulation_context.step(render = True)


        # Clean
        torch.cuda.empty_cache()  # clean GPU

    # while True:
    #     if plan_0 is not None:
    #         for i in range(1, len(plan_0)):
    #             complete_joint_positions = control_robot(robot,plan_0[i-1],plan_0[i],
    #                                                      complete_joint_positions,simulation_context)
    #         print("Successfully Reached! ^_^")
    #         complete_joint_positions = control_gripper(robot=robot, 
    #                                            finger_start=0.14,
    #                                            finger_target=any_data_dict["width"],
    #                                            complete_joint_positions=complete_joint_positions,
    #                                            simulation_context=simulation_context)
    #         plan_0 = None

    #     # Step simulation
    #     simulation_context.step(render = True)


if __name__ == "__main__":
    main()



