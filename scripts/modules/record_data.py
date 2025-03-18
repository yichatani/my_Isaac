import os
import h5py
import time
import queue
import numpy as np
import cv2
from modules.initial_set import rgb_and_depth,save_camera_data

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

def create_episede_file(cameras, height, width):
    num_files = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
    episode_path = os.path.join(DATA_DIR, f"episode_{num_files}.h5")
    print(f"Saving to: {episode_path}")
    if not os.path.exists(episode_path):
        with h5py.File(episode_path, "w") as f:
            print("Creating episode file...")
            f.create_dataset("index", shape=(1, 1), maxshape=(None, 1), dtype=np.uint8, compression="lzf")
            f.create_dataset("agent_pos", shape=(1, 7), maxshape=(None, 7), dtype=np.float32, compression="lzf")
            f.create_dataset("action", shape=(1, 7), maxshape=(None, 7), dtype=np.float32, compression="lzf")
            f.create_dataset("label", shape=(1,), dtype=np.uint8, compression="lzf")
            f.create_dataset("D_max", shape = (1,), maxshape=(None,), dtype=np.float32, compression="lzf")
            f.create_dataset("D_min", shape = (1,), maxshape=(None,), dtype=np.float32, compression="lzf")
            
            for cam in cameras.keys():

                # Both the rgb and depth should be normalized before training
                f.create_dataset(f"{cam}/rgb", shape=(1, height, width, 3), maxshape=(None, height, width, 3),    
                                    dtype=np.uint8, compression="lzf")
                f.create_dataset(f"{cam}/depth", shape=(1, height, width), maxshape=(None, height, width), # last stop here
                                    dtype=np.uint16, compression="lzf")

    print(f"Episode file created: {episode_path}")

    return episode_path


def recording(robot, cameras, episode_path, simulation_context):
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



def load_episode_data(episode_path):
    """Reads HDF5 data, restoring depth values dynamically per frame."""
    with h5py.File(episode_path, "r") as f:
        index = f["index"][:]
        agent_pos = f["agent_pos"][:]
        action = f["action"][:]

        cameras_data = {}

        for frame in f.keys():
            if frame.startswith("frame_"):  # Ignore index/action datasets
                cameras_data[frame] = {}

                for cam in f[frame].keys():
                    rgb = f[f"{frame}/{cam}/rgb"][:]
                    depth_uint16 = f[f"{frame}/{cam}/depth"][:]

                    # Read D_min and D_max from HDF5
                    D_min = f[f"{frame}/{cam}/D_min"][0]
                    D_max = f[f"{frame}/{cam}/D_max"][0]

                    # Convert depth from uint16 to float32 (meters)
                    depth = (depth_uint16.astype(np.float32) / 65535) * (D_max - D_min) + D_min

                    cameras_data[frame][cam] = {"rgb": rgb, "depth": depth}

    return index, agent_pos, action, cameras_data




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





##########

def recording_thread(robot, cameras, simulation_context, recording_event, stop_event):
    """
        Not use for now.
    """
    assert robot is not None, "Failed to initialize Articulation"

    num_files = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
    episode_path = os.path.join(DATA_DIR, f"episode_{num_files}.h5")
    print(f"Saving to: {episode_path}")

    if not os.path.exists(episode_path):
        with h5py.File(episode_path, "w") as f:
            f.create_dataset("index", shape=(1, 1), maxshape=(None, 1), dtype=np.int32, compression="gzip")
            f.create_dataset("agent_pos", shape=(1, 7), maxshape=(None, 7), dtype=np.float32, compression="gzip")
            f.create_dataset("action", shape=(1, 7), maxshape=(None, 7), dtype=np.float32, compression="gzip")
            f.create_dataset("label",shape=(1, 1), maxshape=(None, 1), dtype=np.int32, compression="gzip")
            
            for cam in cameras.keys():

                # Both the rgb and depth should be normalized before training
                f.create_dataset(f"{cam}/rgb", shape=(1, 448, 448, 3), maxshape=(None, 448, 448, 3),    
                                    dtype=np.float32, compression="gzip")
                f.create_dataset(f"{cam}/depth", shape=(1, 448, 448), maxshape=(None, 448, 448), # last stop here
                                    dtype=np.float32, compression="gzip")
                # f.create_dataset(f"{cam}/point_cloud", shape=(1, 0, 3), maxshape=(None, None, 3),
                #                     dtype=np.float32, compression="gzip")
                # f.create_dataset(f"{cam}/colors", shape=(1, 0, 3), maxshape=(None, None, 3),    # colors = rgb / 255
                #                     dtype=np.float32, compression="gzip")

    with h5py.File(episode_path, "a") as f:
        index_dataset = f["index"]
        agent_pos_dataset = f["agent_pos"]
        action_dataset = f["action"]

        index = index_dataset.shape[0]  # Start from last saved index

        while not stop_event.is_set():

            if not recording_event.wait(timeout=1):  # Wait with a timeout to check stop_event
                continue  # If timeout occurs, check `stop_event` again
            print("Recording triggered by simulation step.")

            # Pause Simulation
            # simulation_context.stop()
            command_queue.put("pause")
            while not simulation_context.is_stopped():
                time.sleep(0.1)
            print("Simulation paused.")


            print(f"Before resize: {index_dataset.shape}")
            index_dataset.resize((index_dataset.shape[0] + 1, 1))
            index_dataset[-1] = index
            print(f"After resize: {index_dataset.shape}")

            index += 1
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
            else:
                print("Skipping action dataset update: Received None or invalid action")

            agent_pos_dataset.resize((agent_pos_dataset.shape[0] + 1, 7))

            if action_dataset.shape[0] > 1:
                agent_pos_dataset[-1] = action_dataset[-2]
            else:
                print("Skipping agent_pos update: Not enough data yet")

            for cam in cameras.keys():

                data_dict = rgb_and_depth(cameras[cam], simulation_context)

                #save_camera_data(data_dict,output_dir=os.path.join(ROOT_DIR + "/../../output_dir"))

                # point_cloud, point_colors = create_point_cloud(data_dict)

                # Save data
                f[f"{cam}/rgb"].resize((f[f"{cam}/rgb"].shape[0] + 1, 448, 448, 3))
                f[f"{cam}/rgb"][-1] = data_dict["rgb"]

                f[f"{cam}/depth"].resize((f[f"{cam}/depth"].shape[0] + 1, 448, 448))
                f[f"{cam}/depth"][-1] = data_dict["depth"]

                # save_camera_data(cam,data_dict,output_dir=os.path.join(ROOT_DIR + "/../../output_dir"))

                # if len(point_cloud) > 0:
                #     num_points = point_cloud.shape[0]   # pointcloud number
                #     f[f"{cam}/point_cloud"].resize((f[f"{cam}/point_cloud"].shape[0] + 1, num_points, 3))
                #     f[f"{cam}/point_cloud"][-1] = point_cloud

                #     f[f"{cam}/colors"].resize((f[f"{cam}/colors"].shape[0] + 1, num_points, 3))
                #     f[f"{cam}/colors"][-1] = point_colors  # Ensure same size as point cloud

                ##
            

            f.flush()  # Ensure data is saved
            print("Recording done. ")

            # Restart simulation
            command_queue.put("play")
            while simulation_context.is_stopped():
                time.sleep(0.1)
            print("Simulation play.")

            recording_event.clear()
            # time.sleep(0.01)


def recording_deprecated(robot, cameras, episode_path, simulation_context):

    assert robot is not None, "Failed to initialize Articulation"
    with h5py.File(episode_path, "a") as f:
        index_dataset = f["index"]
        agent_pos_dataset = f["agent_pos"]
        action_dataset = f["action"]

        index = index_dataset.shape[0]  # Start from last saved index

        print("Recording triggered by simulation step.")

        print(f"Before resize: {index_dataset.shape}")
        index_dataset.resize((index_dataset.shape[0] + 1, 1))
        index_dataset[-1] = index
        print(f"After resize: {index_dataset.shape}")

        index += 1
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
        else:
            print("Skipping action dataset update: Received None or invalid action")

        agent_pos_dataset.resize((agent_pos_dataset.shape[0] + 1, 7))

        if action_dataset.shape[0] > 1:
            agent_pos_dataset[-1] = action_dataset[-2]
        else:
            print("Skipping agent_pos update: Not enough data yet")

        for cam in cameras.keys():

            data_dict = rgb_and_depth(cameras[cam], simulation_context)

            depth_raw = data_dict["depth"]
            
            # Remove invalid values (NaN, Inf)
            valid_depth = depth_raw[np.isfinite(depth_raw)]

            if len(valid_depth) > 0:
                D_min, D_max = np.percentile(valid_depth, [5, 95])  # Ignore top/bottom 5%

                # Normalize depth (to range [0,1])
                data_dict["depth"] = (depth_raw - D_min) / (D_max - D_min)

            # Get RGB shape dynamically
            height, width = data_dict["rgb"].shape[:2]

            # # Save data
            # f[f"{cam}/rgb"].resize((f[f"{cam}/rgb"].shape[0] + 1, 448, 448, 3)) ## Here to change the recording size of the image.
            # f[f"{cam}/rgb"][-1] = data_dict["rgb"].astype(np.uint8)

            # f[f"{cam}/depth"].resize((f[f"{cam}/depth"].shape[0] + 1, 448, 448))
            # f[f"{cam}/depth"][-1] = data_dict["depth"].astype(np.float16)

            # Save RGB
            f[f"{cam}/rgb"].resize((f[f"{cam}/rgb"].shape[0] + 1, height, width, 3))
            f[f"{cam}/rgb"][-1] = data_dict["rgb"].astype(np.uint8)

            # Save Depth
            f[f"{cam}/depth"].resize((f[f"{cam}/depth"].shape[0] + 1, height, width))
            f[f"{cam}/depth"][-1] = data_dict["depth"].astype(np.float16)
        

        f.flush()  # Ensure data is saved
        print("Recording done. ")


def load_episode_data_deprecated(episode_path):
    with h5py.File(episode_path, "r") as f:
        index = f["index"][:]
        agent_pos = f["agent_pos"][:]
        action = f["action"][:]

        cameras_data = {}

        for cam in f.keys():
            if cam == "index" or cam == "agent_pos" or cam == "action":
                continue

            rgb = f[f"{cam}/rgb"][:]
            depth_normalized = f[f"{cam}/depth"][:]

            # Compute D_min and D_max from stored depth values
            valid_depth = depth_normalized[np.isfinite(depth_normalized)]
            if len(valid_depth) > 0:
                D_min, D_max = np.percentile(valid_depth, [5, 95])  # Restore original range

                # Denormalize depth
                depth = depth_normalized * (D_max - D_min) + D_min
            else:
                depth = depth_normalized  # No valid depth values, return as is

            cameras_data[cam] = {"rgb": rgb, "depth": depth}

    return index, agent_pos, action, cameras_data