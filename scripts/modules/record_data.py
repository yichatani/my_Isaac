import os
import h5py
import time
import numpy as np
from PIL import Image
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR + "/../..")


def recording(robot):

    with h5py.File(os.path.join(DATA_DIR + "/episode1.h5"), "w") as f:
        f.create_dataset("agent_pos", shape=(0, 7), maxshape=(None, 7), dtype=np.float32, compression="gzip")
        f.create_dataset("action", shape=(0, 7), maxshape=(None, 7), dtype=np.float32, compression="gzip")

    timestamp = 0
    with h5py.File(os.path.join(DATA_DIR + "/episode1.h5"), "a") as f:
        timestamp_dataset = f["time_stamp"]
        agent_pos_dataset = f["agent_pos"]
        action_dataset = f["action"]
    
    while True:

        action = record_robot_7dofs(robot)
        action_dataset.resize((action_dataset.shape[0] + 1, 7))
        action_dataset[-1] = action

        agent_pos_dataset.resize((agent_pos_dataset.shape[0] + 1, 7))
        agent_pos_dataset[-1] = action_dataset[-2]

        timestamp += 1
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

