import os
from PIL import Image
import cv2
import random
import numpy as np
import argparse
import open3d as o3d
from gsnet import AnyGrasp # type: ignore
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from modules.transform import create_rotation_matrix


############ AnyGrasp Parts
parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path', default=os.path.join(ROOT_DIR, "../log/checkpoint_detection.tar"), help='Model checkpoint path')
#parser.add_argument('--checkpoint_path', default=os.path.join(ROOT_DIR, "../log/mega.tar"), help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.14, help='Maximum gripper width (<=0.1m)')
# parser.add_argument('--gripper_height', type=float, default=0.035, help='Gripper height')
# parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
# parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.checkpoint_path = os.path.join(ROOT_DIR, "../log/checkpoint_detection.tar")
cfgs.gripper_height = 0.035
cfgs.max_gripper_width = max(0, min(0.14, cfgs.max_gripper_width))
cfgs.top_down_grasp = True
cfgs.debug = True
anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()
############

# with open(os.path.join(ROOT_DIR + '/model_config.yaml'), 'r') as f:
#     model_config = yaml.load(f, Loader=yaml.FullLoader)


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



def vis_grasps(gg,cloud):
    """Visualize the grasp"""
    trans_mat = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    cloud.transform(trans_mat)
    grippers = gg.to_open3d_geometry_list()
    for gripper in grippers:
        gripper.transform(trans_mat)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, 
        origin=[0, 0, 0]
    )
    # o3d.visualization.draw_geometries([*grippers, cloud, coord_frame])
    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    # sphere.translate(gg[0].translation)
    # sphere.paint_uniform_color([1,0,0])
    # o3d.visualization.draw_geometries([grippers[0], cloud, sphere])
    # o3d.visualization.draw_geometries([grippers[0], cloud])
    o3d.visualization.draw_geometries([*grippers, cloud])
    

def remove_distortion(camera_matrix,dist_coeffs,colors,depths):
    """Undistort the colors and depths"""
    # Undistort the RGB image
    h, w = colors.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_colors = cv2.undistort((colors * 255).astype(np.uint8), camera_matrix, dist_coeffs, None, new_camera_matrix)
    undistorted_colors = undistorted_colors / 255.0

    # Undistort the depth image
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)
    undistorted_depths = cv2.remap(depths, map1, map2, interpolation=cv2.INTER_NEAREST)
    return undistorted_colors, undistorted_depths

def any_grasp(data_dict):
    
    colors = data_dict["rgb"] / 255.0
    depths = data_dict["depth"]

    # camera_matrix = [[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]   970.94244   600.37482
    camera_matrix = [[1281.77, 0.0, 960], [0.0, 1281.77, 540], [0.0, 0.0, 1.0]]
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
    scale = 1.0

    # set workspace to filter output grasps
    # xmin, xmax = -0.29, 0.29
    # ymin, ymax = -0.29, 0.29
    # zmin, zmax = 0.0, 1.0

    xmin, xmax = -0.5, 0.5
    ymin, ymax = -0.5, 0.5
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
    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=False, dense_grasp=False, collision_detection=False)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')
        return False

    gg = gg.nms().sort_by_score()
    gg = gg[0:20]
    # vis_grasps(gg,cloud)
    # exit()

    # target_grasp_pose_to_cam = define_grasp_pose(gg[random.randint(0, 2)])
    target_grasp_pose_to_cam = define_grasp_pose(gg[0])

    data_dict = {
        "width":gg[0].width,
        "depth":gg[0].depth,
        "T":target_grasp_pose_to_cam,
        "score":gg[0].score
    }

    return data_dict