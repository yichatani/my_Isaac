import os
import h5py
import time
import queue
import numpy as np
import cv2
from modules.initial_set import rgb_and_depth,save_camera_data
from modules.policy_data import reconstruct_pointcloud, preprocess_point_cloud

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR + "/../../episodes")

command_queue = queue.Queue()

TARGET_HEIGHT = 448
TARGET_WIDTH = 448

def resize_images(rgb, depth):
    """
    Resize both RGB and depth images to a fixed size.
    - RGB: Use bilinear interpolation for smooth resizing.
    - Depth: Use nearest-neighbor interpolation to preserve depth values.
    """
    rgb_resized = cv2.resize(rgb, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(depth, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)  
    return rgb_resized, depth_resized

def create_episode_file(cameras, height, width):
    num_files = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
    episode_path = os.path.join(DATA_DIR, f"episode_{num_files}.h5")
    print(f"Saving to: {episode_path}")
    
    if not os.path.exists(episode_path):
        with h5py.File(episode_path, "w") as f:
            print("Creating episode file...")

            # Global index and trajectory-related datasets
            f.create_dataset("index", shape=(1, 1), maxshape=(None, 1), dtype=np.uint8, compression="lzf")
            f.create_dataset("agent_pos", shape=(1, 7), maxshape=(None, 7), dtype=np.float32, compression="lzf")
            f.create_dataset("action", shape=(1, 7), maxshape=(None, 7), dtype=np.float32, compression="lzf")
            f.create_dataset("label", shape=(1,), dtype=np.uint8, compression="lzf")

            for cam in cameras.keys():
                # RGB and Depth
                f.create_dataset(f"{cam}/rgb", shape=(1, height, width, 3), maxshape=(None, height, width, 3),
                                 dtype=np.uint8, compression="lzf")
                f.create_dataset(f"{cam}/depth", shape=(1, height, width), maxshape=(None, height, width),
                                 dtype=np.float32, compression="lzf")
                # f.create_dataset(f"{cam}/depth", shape=(1, height, width), maxshape=(None, height, width),
                #                  dtype=np.uint16, compression="lzf")

                # Per-frame D_min and D_max
                f.create_dataset(f"{cam}/D_min", shape=(1,), maxshape=(None,), dtype=np.float32, compression="lzf")
                f.create_dataset(f"{cam}/D_max", shape=(1,), maxshape=(None,), dtype=np.float32, compression="lzf")

    print(f"Episode file created: {episode_path}")
    return episode_path



def recording(robot, cameras, episode_path, simulation_context):
    assert robot is not None, "Failed to initialize Articulation"
    
    with h5py.File(episode_path, "a") as f:
        if "index" not in f:
            f.create_dataset("index", shape=(0, 1), maxshape=(None, 1), dtype='i4')
            f.create_dataset("agent_pos", shape=(0, 7), maxshape=(None, 7), dtype='f4')
            f.create_dataset("action", shape=(0, 7), maxshape=(None, 7), dtype='f4')

            for cam in cameras:
                f.create_dataset(f"{cam}/rgb", shape=(0, 128, 128, 3), maxshape=(None, 128, 128, 3), dtype='uint8')
                f.create_dataset(f"{cam}/depth", shape=(0, 128, 128), maxshape=(None, 128, 128), dtype='f4')
                # f.create_dataset(f"{cam}/depth", shape=(0, 128, 128), maxshape=(None, 128, 128), dtype='uint16')
                f.create_dataset(f"{cam}/D_min", shape=(0,), maxshape=(None,), dtype='f4')
                f.create_dataset(f"{cam}/D_max", shape=(0,), maxshape=(None,), dtype='f4')


        index = f["index"].shape[0]
        f["index"].resize((index + 1, 1))
        f["index"][-1] = index

        # Record action
        try:
            action = record_robot_7dofs(robot)
            if action is None or len(action) != 7:
                raise ValueError("Invalid action data received")
        except Exception as e:
            print(f"Error retrieving robot state: {e}")
            action = None

        if action is not None:
            f["action"].resize((index + 1, 7))
            f["action"][-1] = action

        f["agent_pos"].resize((index + 1, 7))
        if index > 0:
            f["agent_pos"][-1] = f["action"][-2]
        else:
            # This should corerespond to the the observing function
            # f["agent_pos"][-1] = f["action"][-1]
            f["agent_pos"][-1] = np.zeros(7)

        for cam in cameras.keys():
            data_dict = rgb_and_depth(cameras[cam], simulation_context)
            data_dict["rgb"], data_dict["depth"] = resize_images(data_dict["rgb"], data_dict["depth"])
            rgb = data_dict["rgb"].astype(np.uint8)
            depth_raw = data_dict["depth"]

            # Clip to valid finite range
            valid_depth = depth_raw[np.isfinite(depth_raw)]
            D_min, D_max = np.percentile(valid_depth, [0, 100])

            # Save D_min and D_max for this frame and camera
            f[f"{cam}/D_min"].resize((index + 1,))
            f[f"{cam}/D_min"][-1] = D_min

            f[f"{cam}/D_max"].resize((index + 1,))
            f[f"{cam}/D_max"][-1] = D_max

            # Encode depth
            # depth_uint16 = ((np.clip(depth_raw, D_min, D_max) - D_min) / (D_max - D_min) * 65535).astype(np.uint16)
            depth_uint16 = depth_raw

            f[f"{cam}/rgb"].resize((index + 1, *rgb.shape))
            f[f"{cam}/rgb"][-1] = rgb

            f[f"{cam}/depth"].resize((index + 1, *depth_uint16.shape))
            f[f"{cam}/depth"][-1] = depth_uint16

        f.flush()
        print(f"Recording frame {index} done.")


def observing(robot, cameras ,simulation_context, data_sample=None):
    assert robot is not None, "Failed to initialize Articulation"
    NUM_PADDING_FRAMES = 6
    data_dict = rgb_and_depth(cameras['front'], simulation_context)
    data_dict["rgb"], data_dict["depth"] = resize_images(data_dict["rgb"], data_dict["depth"])
    rgb = data_dict["rgb"].astype(np.uint8)
    depth_raw = data_dict["depth"]
    pc_raw = reconstruct_pointcloud(rgb, depth_raw)
    # padding
    if data_sample is None:
        pc_list, state_list = [], []
        if pc_raw.shape[0] > 32:
            pc = preprocess_point_cloud(pc_raw, use_cuda=True)
            for _ in range(NUM_PADDING_FRAMES):
                pc_list.append(pc)
                state_list.append(np.zeros(7))
            pc_arr = np.stack(pc_list, axis=0).astype('float32')    
            state_arr = np.stack(state_list, axis=0).astype('float32')
            data_sample = {'obs': 
                {
                    'agent_pos': state_arr.astype(np.float32),
                    'point_cloud': pc_arr.astype(np.float32),
                },
                # 'action': sample['action'].astype(np.float32)
                }
            return data_sample
        else:
            print("Warning: Too few points in point cloud, skipping this frame.")
            return None
    else:
        if pc_raw.shape[0] > 32:
            pc = preprocess_point_cloud(pc_raw, use_cuda=True)
            state = record_robot_7dofs(robot)
            if state.ndim == 1:
                state = state.reshape(1, -1)  # shape: (1, 7)
            pc = pc.reshape(1, *pc.shape)
            if data_sample['obs']['point_cloud'].shape[0] == NUM_PADDING_FRAMES:
                data_sample['obs']['point_cloud'] = np.concatenate((data_sample['obs']['point_cloud'][1:], pc), axis=0)
                data_sample['obs']['agent_pos'] = np.concatenate((data_sample['obs']['agent_pos'][1:], state), axis=0)
            else:
                raise ValueError("Invalid shape for data_sample NUM_PADDING_FRAMES")
            return data_sample
        else:
            print("Warning: Too few points in point cloud, skipping this frame.")
            return data_sample

   

def record_robot_7dofs(robot):
    """
        Record ur10e's 6 DOF and finger_joint positions
    """
    """
        Contain both agent_pos and action
    """
    complete_joint_positions = robot.get_joint_positions()
    robot_7dofs = complete_joint_positions[:7]
    print("robot_7dofs:",robot_7dofs)
    
    return robot_7dofs


############################################################################
############################################################################
############################################################################

def pause_simulation(recording_event, simulation_context):
    while recording_event.is_set():
        try:
            command = command_queue.get(timeout=0.1)
            if command == "pause":
                simulation_context.pause()
            elif command == "play":
                simulation_context.play()
        except queue.Empty:
            pass

def create_point_cloud(data_dict):

    colors = data_dict["rgb"] / 255.0
    depths = data_dict["depth"]

    camera_matrix = [[1281.77, 0.0, 960], [0.0, 1281.77, 540], [0.0, 0.0, 1.0]]
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
    scale = 1.0

    # set workspace to filter output grasps
    xmin, xmax = -0.29, 0.29
    ymin, ymax = -0.29, 0.29
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
    
    # print("points shape:", points.shape, "colors shape:", colors.shape)
    if points.shape[0] == 0:
        print("Warning: Empty point cloud!")

    return points.astype(np.float32), colors.astype(np.float32)


def recording_v0(robot, cameras, episode_path, simulation_context):
    assert robot is not None, "Failed to initialize Articulation"
    
    with h5py.File(episode_path, "a") as f:
        index_dataset = f["index"]
        agent_pos_dataset = f["agent_pos"]
        action_dataset = f["action"]
        D_max_dataset = f["D_max"]
        D_min_dataset = f["D_min"]

        index = index_dataset.shape[0]  # Start from last saved index
        index_dataset.resize((index_dataset.shape[0] + 1, 1))
        index_dataset[-1] = index
        index += 1

        # Record action
        try:
            action = record_robot_7dofs(robot)
            if action is None or len(action) != 7:
                raise ValueError("Invalid action data received")
        except Exception as e:
            print(f"Error retrieving robot state: {e}")
            action = None

        if action is not None:
            action_dataset.resize((action_dataset.shape[0] + 1, 7))
            action_dataset[-1] = action

        agent_pos_dataset.resize((agent_pos_dataset.shape[0] + 1, 7))
        if action_dataset.shape[0] > 1:
            agent_pos_dataset[-1] = action_dataset[-2]

        # ### Create a new group for each frame (avoids shape mismatch)
        # frame_group = f.create_group(f"frame_{index}")

        for cam in cameras.keys():
            data_dict = rgb_and_depth(cameras[cam], simulation_context)

            # Resize images
            data_dict["rgb"], data_dict["depth"] = resize_images(data_dict["rgb"], data_dict["depth"])

            # Get image size dynamically
            height, width = data_dict["rgb"].shape[:2]

            # Handle depth normalization
            depth_raw = data_dict["depth"]
            valid_depth = depth_raw[np.isfinite(depth_raw)]
            if len(valid_depth) > 0:
                D_min, D_max = np.percentile(valid_depth, [5, 95])  # Ignore outliers
                # data_dict["depth"] = (depth_raw - D_min) / (D_max - D_min)
                data_dict["depth"] = ((depth_raw - D_min) / (D_max - D_min) * 65535).astype(np.uint16)
                # Store D_min, D_max only once per episode
                D_min_dataset.resize((D_min_dataset.shape[0] + 1,))
                D_min_dataset[-1] = D_min
                D_max_dataset.resize((D_max_dataset.shape[0] + 1,))
                D_max_dataset[-1] = D_max
                                

            # # Save RGB and depth inside the frame group
            # frame_group.create_dataset(f"{cam}/rgb", data=data_dict["rgb"].astype(np.uint8), compression="lzf")
            # # frame_group.create_dataset(f"{cam}/depth", data=data_dict["depth"].astype(np.float16), compression="lzf")
            # frame_group.create_dataset(f"{cam}/depth", data=data_dict["depth"].astype(np.uint16), compression="gzip", compression_opts=4)
            # Save RGB
            f[f"{cam}/rgb"].resize((f[f"{cam}/rgb"].shape[0] + 1, height, width, 3))
            f[f"{cam}/rgb"][-1] = data_dict["rgb"].astype(np.uint8)

            # Save Depth
            f[f"{cam}/depth"].resize((f[f"{cam}/depth"].shape[0] + 1, height, width))
            f[f"{cam}/depth"][-1] = data_dict["depth"].astype(np.uint16)

        f.flush()
        print(f"Recording frame {index} done.")


