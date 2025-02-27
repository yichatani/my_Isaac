import os
import h5py
import time
import open3d as o3d
import numpy as np
from PIL import Image
from omni.isaac.sensor import Camera # type: ignore
from modules.initial_set import initial_camera,rgb_and_depth,save_camera_data

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR + "/../../episodes")

robot_path = "/ur10e"
camera_path = "/ur10e/tool0/Camera"
camera_1_path = "/ur10e/tool0/Camera1"
camera_2_path = "/ur10e/tool0/Camera2"

camera_paths = {
    "in_hand": "/ur10e/tool0/Camera",
    "up": "/ur10e/tool0/Camera1",
    "front": "/ur10e/tool0/Camera2"
}
cam = "in_hand"
# initial_camera(camera_paths["in_hand"])

def recording(robot, cameras, simulation_context, recording_event, stop_event):
    assert robot is not None, "Failed to initialize Articulation"

    num_files = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
    episode_path = os.path.join(DATA_DIR, f"episode_{num_files}.h5")
    print(f"Saving to: {episode_path}")

    if not os.path.exists(episode_path):
        with h5py.File(episode_path, "w") as f:
            f.create_dataset("index", shape=(1, 1), maxshape=(None, 1), dtype=np.int32, compression="gzip")
            f.create_dataset("agent_pos", shape=(1, 7), maxshape=(None, 7), dtype=np.float32, compression="gzip")
            f.create_dataset("action", shape=(1, 7), maxshape=(None, 7), dtype=np.float32, compression="gzip")
            
            for cam in cameras.keys():

                # Both the rgb and depth should be normalized before training
                f.create_dataset(f"{cam}/rgb", shape=(1, 480, 640, 3), maxshape=(None, 480, 640, 3),    
                                    dtype=np.float32, compression="gzip")
                f.create_dataset(f"{cam}/depth", shape=(1, 480, 640), maxshape=(None, 480, 640), # last stop here
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
                f[f"{cam}/rgb"].resize((f[f"{cam}/rgb"].shape[0] + 1, 480, 640, 3))
                f[f"{cam}/rgb"][-1] = data_dict["rgb"]

                f[f"{cam}/depth"].resize((f[f"{cam}/depth"].shape[0] + 1, 480, 640))
                f[f"{cam}/depth"][-1] = data_dict["depth"]

                # if len(point_cloud) > 0:
                #     num_points = point_cloud.shape[0]   # pointcloud number
                #     f[f"{cam}/point_cloud"].resize((f[f"{cam}/point_cloud"].shape[0] + 1, num_points, 3))
                #     f[f"{cam}/point_cloud"][-1] = point_cloud

                #     f[f"{cam}/colors"].resize((f[f"{cam}/colors"].shape[0] + 1, num_points, 3))
                #     f[f"{cam}/colors"][-1] = point_colors  # Ensure same size as point cloud

                ##

            f.flush()  # Ensure data is saved
            print("Recording done. ")
            recording_event.clear()
            # time.sleep(0.01)
            

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


# def record_rgb_and_depth(camera_path,simulation_context):
#     """
#         This will used for 3 cameras in this project,
#         in_hand, up and front

#         To use with the camera initial function in the initial set.
#     """

#     data_dict = initial_camera(camera_path,simulation_context)

#     # colors = data_dict["rgb"] / 255.0

#     # # Normalized grayscale image
#     # depth_data = data_dict["depth"]
#     # depth_normalized = ((depth_data - np.min(depth_data)) / np.ptp(depth_data) * 255).astype(np.uint8)
    
#     # data_dict["rgb"] = colors
#     # data_dict["depth"] = depth_normalized
    
#     return data_dict