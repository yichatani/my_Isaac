import os
import numpy as np
from PIL import Image
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera
from isaacsim.core.utils.types import ArticulationActions
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats, \
euler_angles_to_quats, quats_to_euler_angles, quats_to_rot_matrices
from scipy.spatial.transform import Rotation as R

import cv2
import threading
import queue
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "episodes")

asset_path = ROOT_DIR + "/franka_manipulation.usd"
urdf_path = "/home/ani/isaacsim/exts/isaacsim.robot_motion.motion_generation/" \
"motion_policy_configs/FR3/fr3.urdf"
robot_description_path = "/home/ani/isaacsim/exts/isaacsim.robot_motion.motion_generation/" \
"motion_policy_configs/FR3/rmpflow/fr3_robot_description.yaml"

# Prim path
marker_prim_path = "/_40_large_marker"
camera_prim_path = "/fr3/fr3_hand_tcp/hand"

# Recording settings
TARGET_HEIGHT = 448
TARGET_WIDTH = 448

# Thread-safe queue for recording data
recording_queue = queue.Queue()
recording_active = threading.Event()

def initial_camera(camera_path, frequency, resolution):
    """Initialize Camera"""
    camera = Camera(
        prim_path=camera_path,
        frequency=frequency,
        resolution=resolution,
    )

    camera.initialize()
    camera.add_motion_vectors_to_frame()
    camera.add_distance_to_image_plane_to_frame()
    camera = set_camera_parameters(camera)

    print("Camera Initialized!")

    return camera


def set_camera_parameters(camera):
    
    f_stop = 0
    focus_distance = 0.4

    horizontal_aperture = 20.955
    vertical_aperture = 15.2908
    focal_length = 18.14756

    camera.set_focal_length(focal_length / 10.0)
    camera.set_focus_distance(focus_distance)
    camera.set_lens_aperture(f_stop * 100.0)
    camera.set_horizontal_aperture(horizontal_aperture / 10.0)
    camera.set_vertical_aperture(vertical_aperture / 10.0)
    camera.set_clipping_range(0.1, 3.0)

    return camera


def rgb_and_depth(camera):
    """
    Get RGB and depth images from camera
    Note: simulation_context.step() should be called BEFORE this function
    """
    # Get camera data
    camera_data = camera.get_current_frame()
    
    if camera_data is None:
        return None
    
    # Extract RGB (convert from RGBA to RGB)
    rgba = camera_data.get("rgba")
    depth = camera_data.get("distance_to_image_plane")
    
    if rgba is None or depth is None:
        return None
    
    rgb = rgba[:, :, :3]
    
    return {"rgb": rgb, "depth": depth}


def resize_images(rgb, depth):
    """
    Resize both RGB and depth images to a fixed size.
    """
    rgb_resized = cv2.resize(rgb, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(depth, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return rgb_resized, depth_resized


def get_franka_end_effector_pose(art, ik):
    """
    Get Franka end effector pose (position + quaternion) + gripper width
    Returns: np.ndarray of shape (8,) - [x, y, z, qw, qx, qy, qz, gripper_width]
    """
    # Get current joint positions
    joint_positions = art.get_joint_positions().squeeze()
    
    # Compute forward kinematics for end effector
    tcp_position, tcp_rotation = ik.compute_forward_kinematics(
        "fr3_hand_tcp",
        joint_positions[:7]
    )
    
    # Ensure position is 1D array
    tcp_position = np.array(tcp_position).flatten()
    
    # Convert rotation matrix to quaternion
    # tcp_rotation is typically a 3x3 rotation matrix
    tcp_rotation = np.array(tcp_rotation)
    if tcp_rotation.shape == (3, 3):
        # Convert rotation matrix to quaternion [w, x, y, z]
        tcp_quat = rot_matrices_to_quats(tcp_rotation.reshape(1, 3, 3)).flatten()
    else:
        # If already quaternion, just flatten
        tcp_quat = tcp_rotation.flatten()
    
    # Combine position and rotation into 7D pose
    end_effector_pose = np.concatenate([tcp_position, tcp_quat])
    
    # Add gripper width (average of two finger joints)
    gripper_width = (joint_positions[7] + joint_positions[8])
    end_effector_pose = np.append(end_effector_pose, gripper_width)
    
    assert end_effector_pose.shape == (8,), f"Expected shape (8,), got {end_effector_pose.shape}"
    
    return end_effector_pose


# def recording_thread_worker(episode_dir):
#     """
#     Worker thread that continuously saves data from the queue
#     """
#     print(f"[Recording Thread] Started, saving to {episode_dir}")
    
#     poses_list = []
#     frame_index = 0
    
#     while recording_active.is_set() or not recording_queue.empty():
#         try:
#             # Get data from queue with timeout
#             data = recording_queue.get(timeout=0.5)
            
#             rgb = data['rgb']
#             depth = data['depth']
#             pose = data['pose']
#             timestamp = data['timestamp']
            
#             # Save RGB image
#             rgb_path = os.path.join(episode_dir, f"rgb_{frame_index:06d}.png")
#             rgb_img = Image.fromarray(rgb)
#             rgb_img.save(rgb_path)
            
#             # Save depth image (convert to uint16 for PNG, multiply by 1000 to preserve precision)
#             depth_path = os.path.join(episode_dir, f"depth_{frame_index:06d}.png")
#             depth_uint16 = (depth * 1000).astype(np.uint16)
#             depth_img = Image.fromarray(depth_uint16)
#             depth_img.save(depth_path)
            
#             # Collect pose data
#             poses_list.append({
#                 'frame_index': frame_index,
#                 'timestamp': timestamp,
#                 'pose': pose
#             })
            
#             frame_index += 1
            
#             if frame_index % 10 == 0:
#                 print(f"[Recording Thread] Saved {frame_index} frames")
            
#             recording_queue.task_done()
            
#         except queue.Empty:
#             continue
#         except Exception as e:
#             print(f"[Recording Thread] Error: {e}")
#             continue
    
#     # Save all poses to a single NPZ file
#     if len(poses_list) > 0:
#         poses_array = np.array([p['pose'] for p in poses_list])
#         timestamps_array = np.array([p['timestamp'] for p in poses_list])
#         indices_array = np.array([p['frame_index'] for p in poses_list])
        
#         npz_path = os.path.join(episode_dir, "poses.npz")
#         np.savez(
#             npz_path,
#             poses=poses_array,
#             timestamps=timestamps_array,
#             indices=indices_array
#         )
#         print(f"[Recording Thread] Saved {len(poses_list)} poses to {npz_path}")
    
#     print(f"[Recording Thread] Finished. Total frames: {frame_index}")

def recording_thread_worker(episode_dir):
    """
    Worker thread that continuously saves data from the queue
    """
    print(f"[Recording Thread] Started, saving to {episode_dir}")
    
    # Create subdirectories
    rgb_dir = os.path.join(episode_dir, "rgb")
    depth_dir = os.path.join(episode_dir, "depth")
    ee_pose_dir = os.path.join(episode_dir, "ee_pose")
    marker_pose_dir = os.path.join(episode_dir, "marker_pose")
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(ee_pose_dir, exist_ok=True)
    os.makedirs(marker_pose_dir, exist_ok=True)
    
    ee_poses_list = []
    marker_poses_list = []
    frame_index = 0
    
    while recording_active.is_set() or not recording_queue.empty():
        try:
            data = recording_queue.get(timeout=0.5)
            
            rgb = data['rgb']
            depth = data['depth']
            ee_pose = data['ee_pose']
            marker_in_camera_pos = data['marker_in_camera_pos']
            marker_in_camera_quat = data['marker_in_camera_quat']
            timestamp = data['timestamp']
            
            # Save RGB image
            rgb_path = os.path.join(rgb_dir, f"{frame_index:06d}.png")
            rgb_img = Image.fromarray(rgb)
            rgb_img.save(rgb_path)
            
            # Save depth image
            depth_path = os.path.join(depth_dir, f"{frame_index:06d}.png")
            depth_uint16 = (depth * 1000).astype(np.uint16)
            depth_img = Image.fromarray(depth_uint16)
            depth_img.save(depth_path)
            
            # Collect end effector pose data
            ee_poses_list.append({
                'frame_index': frame_index,
                'timestamp': timestamp,
                'pose': ee_pose
            })
            
            # Collect marker pose data (concatenate position and quaternion)
            marker_pose_7d = np.concatenate([marker_in_camera_pos, marker_in_camera_quat])
            marker_poses_list.append({
                'frame_index': frame_index,
                'timestamp': timestamp,
                'pose': marker_pose_7d
            })
            
            frame_index += 1
            
            if frame_index % 10 == 0:
                print(f"[Recording Thread] Saved {frame_index} frames")
            
            recording_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Recording Thread] Error: {e}")
            continue
    
    # Save end effector poses
    if len(ee_poses_list) > 0:
        ee_poses_array = np.array([p['pose'] for p in ee_poses_list])
        ee_timestamps_array = np.array([p['timestamp'] for p in ee_poses_list])
        ee_indices_array = np.array([p['frame_index'] for p in ee_poses_list])
        
        ee_npz_path = os.path.join(ee_pose_dir, "end_poses.npz")
        np.savez(
            ee_npz_path,
            poses=ee_poses_array,
            timestamps=ee_timestamps_array,
            indices=ee_indices_array
        )
        print(f"[Recording Thread] Saved {len(ee_poses_list)} end effector poses to {ee_npz_path}")
    
    # Save marker poses
    if len(marker_poses_list) > 0:
        marker_poses_array = np.array([p['pose'] for p in marker_poses_list])
        marker_timestamps_array = np.array([p['timestamp'] for p in marker_poses_list])
        marker_indices_array = np.array([p['frame_index'] for p in marker_poses_list])
        
        marker_npz_path = os.path.join(marker_pose_dir, "marker_poses.npz")
        np.savez(
            marker_npz_path,
            poses=marker_poses_array,
            timestamps=marker_timestamps_array,
            indices=marker_indices_array
        )
        print(f"[Recording Thread] Saved {len(marker_poses_list)} marker poses to {marker_npz_path}")
    
    print(f"[Recording Thread] Finished. Total frames: {frame_index}")


def create_episode_directory():
    """
    Create a new episode directory
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    num_dirs = len([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    episode_dir = os.path.join(DATA_DIR, f"episode_{num_dirs:04d}")
    os.makedirs(episode_dir, exist_ok=True)
    
    print(f"Created episode directory: {episode_dir}")
    return episode_dir


# def queue_frame_for_recording(art, ik, camera, simulation_context):
#     """
#     Capture current frame and add to recording queue
#     """
#     try:
#         # Get end effector pose
#         pose = get_franka_end_effector_pose(art, ik)
        
#         # Get camera data
#         data_dict = rgb_and_depth(camera)
#         rgb_resized, depth_resized = resize_images(data_dict["rgb"], data_dict["depth"])
        
#         # Prepare data packet
#         data_packet = {
#             'rgb': rgb_resized.astype(np.uint8),
#             'depth': depth_resized.astype(np.float32),
#             'pose': pose,
#             'timestamp': time.time()
#         }
        
#         # Add to queue
#         recording_queue.put(data_packet)
        
#     except Exception as e:
#         print(f"Error queuing frame: {e}")


def queue_frame_for_recording(art, ik, camera, marker, simulation_context):
    """
    Capture current frame and add to recording queue
    """
    try:
        # Get end effector pose
        ee_pose = get_franka_end_effector_pose(art, ik)
        
        # Get marker in camera pose
        marker_in_camera_pos, marker_in_camera_quat = get_marker_in_camera_pose(camera, marker)
        
        # Get camera data
        data_dict = rgb_and_depth(camera)
        rgb_resized, depth_resized = resize_images(data_dict["rgb"], data_dict["depth"])
        
        # Prepare data packet
        data_packet = {
            'rgb': rgb_resized.astype(np.uint8),
            'depth': depth_resized.astype(np.float32),
            'ee_pose': ee_pose,
            'marker_in_camera_pos': marker_in_camera_pos,
            'marker_in_camera_quat': marker_in_camera_quat,
            'timestamp': time.time()
        }
        
        # Add to queue
        recording_queue.put(data_packet)
        
    except Exception as e:
        print(f"Error queuing frame: {e}")


# def arm_const_speed(art, target_arm, sim, ik, camera, record=False, step_size=0.1, eps=5e-3, is_target=True):
#     """
#     Move arm to target with constant speed, optionally recording
#     """
#     current = art.get_joint_positions().squeeze()

#     if is_target is not True:
#         eps = 1e-2

#     while True:
#         diff = target_arm - current[:7]
#         dist = np.linalg.norm(diff)
#         if dist < eps:
#             break

#         step = diff / (dist + 1e-8) * min(step_size, dist)
#         cmd = current.copy()
#         cmd[:7] += step

#         art.apply_action(ArticulationActions(joint_positions=cmd))
#         sim.step(render=True)
        
#         # Record frame if requested
#         if record:
#             queue_frame_for_recording(art, ik, camera, sim)

#         current = art.get_joint_positions().squeeze()


def arm_const_speed(art, target_arm, sim, ik, camera, marker, record=False, step_size=0.1, eps=5e-3, is_target=True):
    """
    Move arm to target with constant speed, optionally recording
    """
    current = art.get_joint_positions().squeeze()

    if is_target is not True:
        eps = 1e-2

    while True:
        diff = target_arm - current[:7]
        dist = np.linalg.norm(diff)
        if dist < eps:
            break

        step = diff / (dist + 1e-8) * min(step_size, dist)
        cmd = current.copy()
        cmd[:7] += step

        art.apply_action(ArticulationActions(joint_positions=cmd))
        sim.step(render=True)
        
        # Record frame if requested
        if record:
            queue_frame_for_recording(art, ik, camera, marker, sim)

        current = art.get_joint_positions().squeeze()


# def set_gripper(art, width, sim, ik, camera, record=False, steps=50):
#     """
#     Smoothly set gripper width over multiple steps
#     width: target width (e.g. 0.0 ~ 0.08)
#     """
#     joint_pos = art.get_joint_positions().squeeze()

#     start_width = joint_pos[7] + joint_pos[8]
#     target_width = width

#     for i in range(steps):
#         alpha = (i + 1) / steps
#         curr_width = (1 - alpha) * start_width + alpha * target_width

#         cmd = joint_pos.copy()
#         cmd[7] = curr_width / 2
#         cmd[8] = curr_width / 2

#         art.apply_action(ArticulationActions(joint_positions=cmd))
#         sim.step(render=True)

#         if record:
#             queue_frame_for_recording(art, ik, camera, sim)


def set_gripper(art, width, sim, ik, camera, marker, record=False, steps=50):
    """
    Smoothly set gripper width over multiple steps
    """
    joint_pos = art.get_joint_positions().squeeze()

    start_width = joint_pos[7] + joint_pos[8]
    target_width = width

    for i in range(steps):
        alpha = (i + 1) / steps
        curr_width = (1 - alpha) * start_width + alpha * target_width

        cmd = joint_pos.copy()
        cmd[7] = curr_width / 2
        cmd[8] = curr_width / 2

        art.apply_action(ArticulationActions(joint_positions=cmd))
        sim.step(render=True)

        if record:
            queue_frame_for_recording(art, ik, camera, marker, sim)


# def hold_position(art, sim, ik, camera, record=False, duration=2.0):
#     """
#     Hold current position for specified duration
#     """
#     current_cmd = art.get_joint_positions().squeeze()
#     action = ArticulationActions(joint_positions=current_cmd)
    
#     start_time = time.time()
#     frame_count = 0
    
#     while time.time() - start_time < duration:
#         art.apply_action(action)
#         sim.step(render=True)
        
#         # Record at ~10 Hz during hold
#         if record and frame_count % 6 == 0:  # Assuming ~60 Hz simulation
#             queue_frame_for_recording(art, ik, camera, sim)
        
#         frame_count += 1
    
#     print(f"Held position for {duration} seconds")

def hold_position(art, sim, ik, camera, marker, record=False, duration=2.0):
    """
    Hold current position for specified duration
    """
    current_cmd = art.get_joint_positions().squeeze()
    action = ArticulationActions(joint_positions=current_cmd)
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration:
        art.apply_action(action)
        sim.step(render=True)
        
        # Record at ~10 Hz during hold
        if record and frame_count % 6 == 0:
            queue_frame_for_recording(art, ik, camera, marker, sim)
        
        frame_count += 1
    
    print(f"Held position for {duration} seconds")


def get_marker_in_camera_pose(camera, marker):
    """
    Calculate marker pose in camera coordinate system
    Returns: position (3,) and quaternion (4,) - [x, y, z], [qw, qx, qy, qz]
    """
    camera_world_pos, camera_world_quat = camera.get_world_pose()
    marker_world_pos, marker_world_quat = marker.get_world_pose()
    
    # T_world_camera
    R_world_camera = quats_to_rot_matrices(camera_world_quat)
    T_world_camera = np.eye(4)
    T_world_camera[:3, :3] = R_world_camera
    T_world_camera[:3, 3] = camera_world_pos
    
    # T_world_marker
    R_world_marker = quats_to_rot_matrices(marker_world_quat)
    T_world_marker = np.eye(4)
    T_world_marker[:3, :3] = R_world_marker
    T_world_marker[:3, 3] = marker_world_pos
    
    # T_camera_marker
    T_camera_marker = np.linalg.inv(T_world_camera) @ T_world_marker
    
    # Apply camera coordinate transform
    T_tcamera = np.array([
        [0.0,  -1.0,  0.0, 0.0],
        [0.0, 0.0,  -1.0, 0.0],
        [1.0,  0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    T_camera_marker = T_tcamera @ T_camera_marker
    
    marker_in_camera_quat = rot_matrices_to_quats(T_camera_marker[:3, :3])
    marker_in_camera_pos = T_camera_marker[:3, 3]
    
    return marker_in_camera_pos, marker_in_camera_quat


def main():
    # Stage
    print("Opening stage...")
    open_stage(usd_path=asset_path)
    stage = get_current_stage()

    simulation_context = SimulationContext()
    simulation_context.initialize_physics()

    # Robot
    art = Articulation("/fr3")
    art.initialize()
    art_world_pose = art.get_world_poses()
    
    initial_joint_position = np.array([-0.47200201, -0.53468038, 0.41885995, -2.64197119, 0.24759319,
                                       2.1317271, 0.54534657, 0.04, 0.04])
    
    # IK Solver
    ik = LulaKinematicsSolver(
        robot_description_path=robot_description_path,
        urdf_path=urdf_path,
    )
    
    initial_tcp_position, initial_tcp_rotation = ik.compute_forward_kinematics("fr3_hand_tcp",
                                                                                initial_joint_position[:7])
    
    # Initialize robot position
    for i in range(50):
        art.set_joint_positions(initial_joint_position)
        simulation_context.step(render=True)

    # Camera
    camera = initial_camera(camera_prim_path, 60, (1920, 1080))

    # camera_world_pos, camera_world_quat = camera.get_world_pose()
    # print(f"{camera_world_pos=}, {camera_world_quat=}")
    # exit()
    # Object
    marker = XFormPrim(marker_prim_path)
    marker_world_pos, marker_world_quat = marker.get_world_pose()
    print(f"{marker_world_pos=}, {marker_world_quat=}")
    # exit()

    # # Calculate marker related to camera
    # # T_world_camera = pose_to_matrix(camera_world_pos, camera_world_quat)
    # R_world_camera = quats_to_rot_matrices(camera_world_quat)
    # T_world_camera = np.eye(4)
    # T_world_camera[:3, :3] = R_world_camera
    # T_world_camera[:3, 3] = camera_world_pos
    # print(f"{T_world_camera=}")
    # R_world_marker = quats_to_rot_matrices(marker_world_quat)
    # T_world_marker = np.eye(4)
    # T_world_marker[:3, :3] = R_world_marker
    # T_world_marker[:3, 3] = marker_world_pos
    # print(f"{T_world_marker=}")
    # T_camera_marker = np.linalg.inv(T_world_camera) @ T_world_marker
    # print(f"Initial:{T_camera_marker}")
    # T_tcamera = np.array([
    #     [0.0,  -1.0,  0.0, 0.0],
    #     [0.0, 0.0,  -1.0, 0.0],
    #     [1.0,  0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    # T_camera_marker = T_tcamera @ T_camera_marker
    # # print(f"After:{T_camera_marker}")
    # marker_in_camera_quat = rot_matrices_to_quats(T_camera_marker[:3, :3])
    # marker_in_camera_pos = T_camera_marker[:3, 3]

    marker_in_camera_pos, marker_in_camera_quat = get_marker_in_camera_pose(camera, marker)
    print(f"{marker_in_camera_pos=}, {marker_in_camera_quat=}")
    # exit()

    marker_position = marker_world_pos
    marker_quat = marker_world_quat
    # print(f"Marker position: {marker_position}")
    # print(f"Marker quat: {marker_quat}")

    # Target
    target_position = marker_position - art_world_pose[0]
    target_position = target_position.reshape(-1)
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])
    
    end_position = target_position + np.array([0, 0, 0.1])

    # Start simulation
    simulation_context.play()

    # Compute target joint positions
    target_joint_position = ik.compute_inverse_kinematics("fr3_hand_tcp", target_position, target_quat)[0]
    target_joint_position = np.append(target_joint_position, np.array([0.04, 0.04]))

    waypoints_joint_position_1 = np.array([-0.4855259, -0.32121756, 0.48082598, -2.76619067, 0.23539604,
                                           2.46647489, 0.57443971, 0.04, 0.04])
    waypoints_joint_position_2 = np.array([-0.7012182, 0.03962762, 0.64272673, -2.79521795, -0.07836149,
                                           2.82603965, 0.80104943, 0.04, 0.04])

    # Create episode directory
    episode_dir = create_episode_directory()
    
    # Start recording thread
    recording_active.set()
    recording_thread = threading.Thread(target=recording_thread_worker, args=(episode_dir,))
    recording_thread.start()
    
    print("\n" + "="*60)
    print("Starting Grasp Demonstration with Recording")
    print("="*60 + "\n")
    
    print("Phase 1: Moving to initial position...")
    print("Phase 2: Moving to pre-grasp position...")
    arm_const_speed(art, waypoints_joint_position_2[:7], simulation_context, ik, camera, marker,
                   record=True, is_target=False)

    print("Phase 3: Moving to grasp target...")
    arm_const_speed(art, target_joint_position[:7], simulation_context, ik, camera, marker, record=True)
    
    print("Phase 4: Closing gripper...")
    set_gripper(art, width=0.0, sim=simulation_context, ik=ik, camera=camera, marker=marker,
                record=True, steps=50)
    
    print("Phase 5: Returning with object...")
    arm_const_speed(art, waypoints_joint_position_2[:7], simulation_context, ik, camera, marker, record=True)
    hold_position(art, simulation_context, ik, camera, marker, record=True, duration=10.0)
    
    print("\n" + "="*60)
    print("Grasp demonstration completed!")
    print("="*60 + "\n")
    
    # Stop recording thread
    print("Stopping recording thread...")
    recording_active.clear()
    recording_thread.join()
    
    print(f"\nAll data saved to: {episode_dir}")
    
    real_end_joint_position = art.get_joint_positions().squeeze()
    print(f"Final joint position: {real_end_joint_position}")

    simulation_context.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()