import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.stage import get_current_stage# type: ignore
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats, euler_angles_to_quats,quats_to_euler_angles # type: ignore
from omni.isaac.core.prims import XFormPrim # type: ignore
import omni.usd # type: ignore
from pxr import Usd, UsdGeom # type: ignore

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(ROOT_DIR + '/../prim_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
camera_path = config['camera_paths']['sensor']
tool0_path = config['tool0_path']
baselink_path = config['baselink_path']
robotiqpad_R_path = config['robotiqpad_R_path']

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

def relative_pose(frame_A,frame_B):
    """
    Compute the relative pose of frame A with respect to frame B.
    The relative pose is defined as the transformation matrix that transforms points
    from frame B to frame A.
    The transformation matrix is a 4x4 matrix that represents the rotation and translation
    of frame A with respect to frame B.
    The rotation is represented as a 3x3 matrix and the translation is represented as a 3x1 vector.
    The transformation matrix is used to transform points from frame B to frame A.
    """
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

def visualize_pose(position, orientation_matrix, name="/World/TargetMarker"):
    """
    Visualize a pose in Isaac Sim using an arrow or axis marker.
    
    Parameters:
        position (np.ndarray): Position of the target [x, y, z].
        orientation_matrix (np.ndarray): 3x3 orientation matrix.
        name (str): Name of the marker in the scene.
    """

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


def camera_extrinsic(camera_path, baselink_path):
    """
    Get the extrinsic matrix of the camera with respect to the baselink.
    The extrinsic matrix is a 4x4 matrix that represents the rotation and translation
    of the camera with respect to the baselink.
    The rotation is represented as a 3x3 matrix and the translation is represented as a 3x1 vector.
    The extrinsic matrix is used to transform points from the camera coordinate system to the baselink coordinate system.
    """
    # Get the stage
    stage = omni.usd.get_context().get_stage()

    frame_baselink = stage.GetPrimAtPath(baselink_path)
    frame_front_camera = stage.GetPrimAtPath(camera_path)
    T_baselink_2_global = omni.usd.get_world_transform_matrix(frame_baselink)
    T_front_camera_2_global = omni.usd.get_world_transform_matrix(frame_front_camera)

    T_baselink_2_global = np.array(T_baselink_2_global).reshape(4,4).T
    print("T_baselink_2_global:\n",T_baselink_2_global)
    T_front_camera_2_global = np.array(T_front_camera_2_global).reshape(4,4).T
    print("T_front_camera_2_global:\n",T_front_camera_2_global)
    T_front_camera_2_baselink = np.linalg.inv(T_baselink_2_global) @ T_front_camera_2_global
    print("front camera to baselink:\n", T_front_camera_2_baselink)
    Rx_180 = np.array([ # only for Isaac camera 
    [1,  0,   0, 0],
    [0, -1,   0, 0],
    [0,  0,  -1, 0],
    [0,  0,   0, 1]
    ])
    T_front_camera_2_baselink_new = T_front_camera_2_baselink @ Rx_180
    print("extrinsic matrix:\n", T_front_camera_2_baselink_new)

    return T_front_camera_2_baselink_new


def get_end_effector_pose()-> np.ndarray[6]:
    """
    Get the end effector pose of the robot.
    The pose is represented as a 6D vector containing translation and rotation.
    The rotation is represented as Euler angles (roll, pitch, yaw).
    """
    transform_stage = get_current_stage()

    frame_tool0 = transform_stage.GetPrimAtPath(tool0_path)
    xformable_tool0 = UsdGeom.Xformable(frame_tool0)
    T_tool0_2_baselink = xformable_tool0.GetLocalTransformation()
    T_tool0_2_baselink = np.array(T_tool0_2_baselink).T

    T_end_effector_translation = T_tool0_2_baselink[:3,3]
    T_end_effector_rotation = T_tool0_2_baselink[:3,:3]
    T_end_effector_orientation = rot_matrices_to_quats(T_end_effector_rotation)
    T_end_effector_euler = quats_to_euler_angles(T_end_effector_orientation)
    
    T_end_pose = np.concatenate((T_end_effector_translation, T_end_effector_euler), axis=0)

    return T_end_pose


def get_local_transform(prim_path: str) -> np.ndarray:
    """
    Get the local transformation matrix of a given prim in the USD stage.

    Args:
        prim_path (str): The path to the prim in the USD stage.

    Returns:
        np.ndarray: A 4x4 transformation matrix representing the local transformation of the prim.
    """
    transform_stage = get_current_stage()
    frame = transform_stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(frame)
    T_local = xformable.GetLocalTransformation()
    T_local = np.array(T_local).T
    return T_local

def get_world_transform(prim_path: str) -> np.ndarray:
    """
    Get the world transformation matrix of a given prim in the USD stage.
    Args:
        prim_path (str): The path to the prim in the USD stage.
    Returns:
        np.ndarray: A 4x4 transformation matrix representing the world transformation of the prim.
    """
    transform_stage = get_current_stage()
    frame = transform_stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(frame)
    T_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    T_world = np.array(T_world).T
    return T_world


def gripper_width_to_openpoint_z(width_m: float) -> float:
    """
    Convert the gripper width in meters to the grasp center z-coordinate in meters.
    """
    width_mm = width_m * 1000
    # z_mm = -0.16786 * width_mm + 200.6
    z_mm = -0.167857 * width_mm + 233.3
    return z_mm / 1000


def T_pose_2_joints(T_translation:np.ndarray, T_rotation:np.ndarray, AKSolver) -> np.ndarray:
    """
    Process the translation and rotation to get the joint positions.
    """
    T_quats = rot_matrices_to_quats(T_rotation)
    T_joint_states, succ = AKSolver.compute_inverse_kinematics(T_translation, T_quats)
    T_joint_positions = T_joint_states.joint_positions
    if not succ:
        # raise ValueError("IK failed, skipping")
        print(">>>IK failed<<<")
        return None                  
    # make sure the wrist don't rotate too much, to prevent collision
    # if abs(T_joint_positions[5]) > math.pi/2:
    #     T_joint_positions = T_joint_positions.copy()
    #     T_joint_positions[5] = abs(T_joint_positions[5]) - math.pi
    return T_joint_positions 


def transform_terminator(any_data_dict):
    
    """
    Transforming function to transform
    the terminator to the tool0 frame.
    Args:
        any_data_dict (dict): A dictionary containing the data to be transformed.
            - "T": The transformation matrix of the target.
            - "depth": The depth value for the transformation.
    Returns:
        np.ndarray: The transformation matrix from the base link to the tool0 frame. 
    """
    # Get the transformation matrix from the base link to the tool0 frame
    T_baselink_2_tool0 = get_local_transform(tool0_path)
    T_tool0_2_camera = get_local_transform(camera_path)
    
    T_baselink_2_camera = T_baselink_2_tool0 @ T_tool0_2_camera
    
    T_tool0_2_TCP = np.eye(4)
    T_tool0_2_TCP[2,3] = gripper_width_to_openpoint_z(any_data_dict["width"]) - any_data_dict["depth"] - 0.01

    T_baselink_2_optic = T_baselink_2_camera
    T_baselink_2_optic[:3,:3] = T_baselink_2_tool0[:3,:3]

    T_optic_2_target = any_data_dict["T"]
    T_TCP_2_tool0 = np.linalg.inv(T_tool0_2_TCP)
    T_baselink_2_Tool0 = T_baselink_2_optic @ T_optic_2_target @ T_TCP_2_tool0

    return T_baselink_2_Tool0
   