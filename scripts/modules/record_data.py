import os
import h5py
import time
import threading
import numpy as np
from PIL import Image
from omni.isaac.core.articulations import Articulation # type: ignore
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR + "/../../episodes")

robot_path = "/ur10e"

recording_event = threading.Event()

def recording(robot, simulation_context):
    assert robot is not None, "Failed to initialize Articulation"

    num_files = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
    episode_path = os.path.join(DATA_DIR, f"episode_{num_files}.h5")
    print(f"Saving to: {episode_path}")

    if not os.path.exists(episode_path):
        with h5py.File(episode_path, "w") as f:
            f.create_dataset("index", shape=(1, 1), maxshape=(None, 1), dtype=np.int32, compression="gzip")
            f.create_dataset("agent_pos", shape=(1, 7), maxshape=(None, 7), dtype=np.float32, compression="gzip")
            f.create_dataset("action", shape=(1, 7), maxshape=(None, 7), dtype=np.float32, compression="gzip")

    with h5py.File(episode_path, "a") as f:
        index_dataset = f["index"]
        agent_pos_dataset = f["agent_pos"]
        action_dataset = f["action"]

        index = index_dataset.shape[0]  # Start from last saved index

        while True:

            recording_event.wait() # wait signal from main thread
            recording_event.clear() # prevent from repeating

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

            f.flush()  # Ensure data is saved
            time.sleep(0.01)

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

def record_rgb_and_depth(data_dict, output_dir):
    """
        This will used for 3 cameras in this project,
        in_hand, up and front

        To use with the camera initial function in the initial set.
    """

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
    
    """
    Change to numpy array to save.
    """

def recreate_point_cloud():
    pass

