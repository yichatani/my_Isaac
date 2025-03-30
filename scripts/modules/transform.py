import os
import numpy as np
from scipy.spatial.transform import Rotation as R
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from omni.isaac.core.utils.stage import get_current_stage# type: ignore
from omni.isaac.core.prims import XFormPrim # type: ignore
import omni.usd # type: ignore
from pxr import Usd, UsdGeom # type: ignore

usd_file_path = os.path.join(ROOT_DIR, "../../ur10e_grasp_set.usd")
robot_path = "/World/ur10e"
camera_path = "/World/ur10e/tool0/Camera"
front_camera_path = "/World/front"
tool0_path = "/World/ur10e/tool0"
flange_path = "/Wprld/ur10e/flange"
base_path = "/World/ur10e/base"
baselink_path = "/World/ur10e/base_link"
robotiqpad_R_path = "/World/ur10e/right_inner_finger_pad"
robotiqpad_L_path = "/World/ur10e/left_inner_finger_pad"

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


def transform_terminator(any_data_dict):
    
    """
    Transforming function to transform 
    from target tool0 to baselink
    """
    transform_stage = Usd.Stage.Open(usd_file_path)

    # stage = omni.usd.get_context().get_stage()

    # frame_baselink = stage.GetPrimAtPath(baselink_path)
    # frame_front_camera = stage.GetPrimAtPath(front_camera_path)

    # # 获取世界变换
    # T_baselink_2_global = omni.usd.get_world_transform_matrix(frame_baselink)
    # # print("T_baselink_2_global:\n",T_baselink_2_global)
    # T_front_camera_2_global = omni.usd.get_world_transform_matrix(frame_front_camera)
    # # print("T_front_camera_2_global:\n",T_front_camera_2_global)

    # # 转换为 numpy
    # T_baselink_2_global = np.array(T_baselink_2_global).reshape(4,4).T
    # print("T_baselink_2_global:\n",T_baselink_2_global)
    # T_front_camera_2_global = np.array(T_front_camera_2_global).reshape(4,4).T
    # print("T_front_camera_2_global:\n",T_front_camera_2_global)

    # # 计算相对变换
    # T_front_camera_2_baselink = np.linalg.inv(T_baselink_2_global) @ T_front_camera_2_global

    # print("front camera to baselink:\n", T_front_camera_2_baselink)

    # Rx_180 = np.array([
    # [1,  0,   0, 0],
    # [0, -1,   0, 0],
    # [0,  0,  -1, 0],
    # [0,  0,   0, 1]
    # ])

    # T_front_camera_2_baselink_new = T_front_camera_2_baselink @ Rx_180
    # print("front camera to baselink new:\n", T_front_camera_2_baselink_new)

    # exit()

    frame_tool0 = transform_stage.GetPrimAtPath(tool0_path)
    xformable_tool0 = UsdGeom.Xformable(frame_tool0)
    T_tool0_2_baselink = xformable_tool0.GetLocalTransformation()
    T_tool0_2_baselink = np.array(T_tool0_2_baselink).T
    # print("origin tool0 to baselink:\n",T_tool0_2_baselink)

    frame_right_pad = transform_stage.GetPrimAtPath(robotiqpad_R_path)
    xformable_TCP = UsdGeom.Xformable(frame_right_pad)
    T_Rpad_2_baselink = xformable_TCP.GetLocalTransformation()
    T_Rpad_2_baselink = np.array(T_Rpad_2_baselink).T

    frame_camera = transform_stage.GetPrimAtPath(camera_path)
    xformable_camera = UsdGeom.Xformable(frame_camera)
    T_camera_2_tool0 = xformable_camera.GetLocalTransformation()
    T_camera_2_tool0 = np.array(T_camera_2_tool0).T
    # print("camera to tool0:\n",T_camera_2_tool0)

    T_camera_2_baselink = T_tool0_2_baselink @ T_camera_2_tool0
    # T_camera_2_baselink = T_tool0_2_baselink
    # print("camera to tool0:",T_camera_2_tool0)

    T_Rpad_2_tool0 = np.linalg.inv(T_tool0_2_baselink) @ T_Rpad_2_baselink
    # print("Rpad to tool0:\n",T_Rpad_2_tool0)
    

    T_TCP_2_tool0 = np.eye(4)
    T_TCP_2_tool0[2,3] = T_Rpad_2_tool0[2,3] + 0.033 - any_data_dict["depth"]  ## 0.02

    T_optic_2_baselink = T_camera_2_baselink
    T_optic_2_baselink[:3,:3] = T_tool0_2_baselink[:3,:3]
    # print("optic to baselink:\n",T_optic_2_baselink)

    T_target_2_optic = any_data_dict["T"]

    # T_target_2_baselink = T_optic_2_baselink @ T_target_2_optic
    # print("target to optic:\n",T_target_2_optic)
    # print("target to baselink:\n",T_target_2_baselink)

    T_tool0_2_TCP = np.linalg.inv(T_TCP_2_tool0)
    # print("tool0 to TCP:\n",T_tool0_2_TCP)

    T_Tool0_2_baselink = T_optic_2_baselink @ T_target_2_optic @ T_tool0_2_TCP
    # print("target tool0 to baselink:\n",T_Tool0_2_baselink)

    return T_Tool0_2_baselink
   