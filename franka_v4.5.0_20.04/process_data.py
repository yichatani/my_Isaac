import os
import numpy as np
import open3d as o3d
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def reconstruct_pointcloud(episode_dir, frame_idx=0, visualize=False):
    """
    Reconstruct point cloud from RGB + depth.

    Args:
        episode_dir: path to episode directory (e.g., episodes/episode_0000)
        frame_idx: frame index to reconstruct
        visualize: whether to visualize the point cloud

    Returns:
        point_cloud: (Np, 6) numpy array, columns: [x, y, z, r, g, b]
    """
    rgb_path = os.path.join(episode_dir, 'rgb', f'{frame_idx:06d}.png')
    depth_path = os.path.join(episode_dir, 'depth', f'{frame_idx:06d}.png')
    
    colors = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
    depths = np.array(Image.open(depth_path)) / 1000.0
    
    camera_matrix = [[299.08, 0.0, 224], [0.0, 531.70, 224], [0.0, 0.0, 1.0]]
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
    scale = 1.0

    # Construct pixel grid
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0) & (points_z < 3)
    points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)

    if points.shape[0] == 0:
        print("Warning: Empty point cloud!")
        return np.zeros((0, 6), dtype=np.float32)

    if visualize:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([cloud])

    # Combine [x, y, z, r, g, b]
    point_cloud = np.concatenate([points, colors], axis=1)  # shape: (Np, 6)
    return point_cloud


class EpisodeReader:
    def __init__(self, episode_dir):
        """
        Args:
            episode_dir: path to episode directory (e.g., episodes/episode_0000)
        """
        self.episode_dir = episode_dir
        self.rgb_dir = os.path.join(episode_dir, "rgb")
        self.depth_dir = os.path.join(episode_dir, "depth")
        self.ee_pose_dir = os.path.join(episode_dir, "ee_pose")
        self.marker_pose_dir = os.path.join(episode_dir, "marker_pose")

        # ---- load end effector poses ----
        ee_poses_path = os.path.join(self.ee_pose_dir, "end_poses.npz")
        assert os.path.exists(ee_poses_path), f"end_poses.npz not found at {ee_poses_path}"

        ee_data = np.load(ee_poses_path)
        self.ee_poses = ee_data["poses"]          # (T, 8)
        self.ee_timestamps = ee_data["timestamps"]
        self.ee_indices = ee_data["indices"]

        # ---- load marker poses ----
        marker_poses_path = os.path.join(self.marker_pose_dir, "marker_poses.npz")
        assert os.path.exists(marker_poses_path), f"marker_poses.npz not found at {marker_poses_path}"

        marker_data = np.load(marker_poses_path)
        self.marker_poses = marker_data["poses"]  # (T, 7)
        self.marker_timestamps = marker_data["timestamps"]
        self.marker_indices = marker_data["indices"]

        self.length = self.ee_poses.shape[0]
        assert self.length == self.marker_poses.shape[0], "EE and marker pose counts mismatch"

    def get_frame(self, idx):
        """
        Returns:
            rgb:         (H, W, 3), uint8
            depth:       (H, W), float32 (meters)
            ee_pose:     (8,) - [x, y, z, qw, qx, qy, qz, gripper_width]
            marker_pose: (7,) - [x, y, z, qw, qx, qy, qz]
        """
        assert 0 <= idx < self.length, f"Index {idx} out of range [0, {self.length})"

        rgb_path = os.path.join(self.rgb_dir, f"{idx:06d}.png")
        depth_path = os.path.join(self.depth_dir, f"{idx:06d}.png")

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth = depth_mm / 1000.0  # mm -> meters

        ee_pose = self.ee_poses[idx]
        marker_pose = self.marker_poses[idx]

        return rgb, depth, ee_pose, marker_pose

    def __len__(self):
        return self.length


def subsample_episode(src_episode_dir, dst_root_dir, step=2):
    """
    每隔 step 帧抽一帧，保存到新目录：
        dst_root_dir/episode_xxxx/{rgb, depth, ee_pose, marker_pose}

    - RGB / Depth: 直接拷贝 PNG（不重新编码，避免误差）
    - Pose: 重新写 npz 文件，只包含抽出来的帧
    """
    reader = EpisodeReader(src_episode_dir)

    episode_name = os.path.basename(src_episode_dir.rstrip("/"))
    out_episode_dir = os.path.join(dst_root_dir, episode_name)

    rgb_out_dir = os.path.join(out_episode_dir, "rgb")
    depth_out_dir = os.path.join(out_episode_dir, "depth")
    ee_pose_out_dir = os.path.join(out_episode_dir, "ee_pose")
    marker_pose_out_dir = os.path.join(out_episode_dir, "marker_pose")

    os.makedirs(rgb_out_dir, exist_ok=True)
    os.makedirs(depth_out_dir, exist_ok=True)
    os.makedirs(ee_pose_out_dir, exist_ok=True)
    os.makedirs(marker_pose_out_dir, exist_ok=True)

    selected_indices = np.arange(0, len(reader), step)

    # Subsample poses
    sub_ee_poses = reader.ee_poses[selected_indices]
    sub_ee_timestamps = reader.ee_timestamps[selected_indices]
    sub_ee_indices = reader.ee_indices[selected_indices]

    sub_marker_poses = reader.marker_poses[selected_indices]
    sub_marker_timestamps = reader.marker_timestamps[selected_indices]
    sub_marker_indices = reader.marker_indices[selected_indices]

    print(f"Subsampling {src_episode_dir}: {len(reader)} -> {len(selected_indices)} frames")

    for new_idx, orig_idx in enumerate(selected_indices):
        src_rgb = os.path.join(reader.rgb_dir, f"{orig_idx:06d}.png")
        src_depth = os.path.join(reader.depth_dir, f"{orig_idx:06d}.png")

        dst_rgb = os.path.join(rgb_out_dir, f"{new_idx:06d}.png")
        dst_depth = os.path.join(depth_out_dir, f"{new_idx:06d}.png")

        if not os.path.exists(src_rgb) or not os.path.exists(src_depth):
            print(f"Warning: missing files for frame {orig_idx:06d}, skip")
            continue

        shutil.copy2(src_rgb, dst_rgb)
        shutil.copy2(src_depth, dst_depth)

    # Save end effector poses
    ee_poses_out_path = os.path.join(ee_pose_out_dir, "end_poses.npz")
    np.savez(
        ee_poses_out_path,
        poses=sub_ee_poses,
        timestamps=sub_ee_timestamps,
        indices=sub_ee_indices,
    )
    print(f"Saved subsampled EE poses to {ee_poses_out_path}")

    # Save marker poses
    marker_poses_out_path = os.path.join(marker_pose_out_dir, "marker_poses.npz")
    np.savez(
        marker_poses_out_path,
        poses=sub_marker_poses,
        timestamps=sub_marker_timestamps,
        indices=sub_marker_indices,
    )
    print(f"Saved subsampled marker poses to {marker_poses_out_path}")


if __name__ == '__main__':

    # Example 1: Subsample episode
    src_episode_dir = os.path.join(ROOT_DIR, "episodes", "episode_0001")
    dst_root_dir = os.path.join(ROOT_DIR, "episodes_subsampled")
    subsample_episode(src_episode_dir, dst_root_dir, step=6)
    print("Done subsampling.")

    # # Example 2: Reconstruct point cloud
    # episode_dir = os.path.join(ROOT_DIR, "episodes", "episode_0000")
    # reconstruct_pointcloud(episode_dir, frame_idx=0, visualize=True)

    # # Example 3: Read frames
    # reader = EpisodeReader(os.path.join(ROOT_DIR, "episodes", "episode_0000"))
    # print(f"Episode length: {reader.length}")
    # rgb, depth, ee_pose, marker_pose = reader.get_frame(10)
    # print(f"EE Pose (8D): {ee_pose}")
    # print(f"Marker Pose (7D): {marker_pose}")

    # # Example 4: 
    # episode_dir = os.path.join(ROOT_DIR, "episodes", "episode_0000")
    # reader = EpisodeReader(episode_dir)
    
    # print(f"Total frames in episode: {len(reader)}\n")
    
    # frames_to_print = [0, 10, 20, 30]
    
    # for idx in frames_to_print:
    #     if idx >= len(reader):
    #         print(f"Frame {idx} out of range, skipping...")
    #         continue
            
    #     print("="*80)
    #     print(f"Frame {idx}")
    #     print("="*80)
        
    #     rgb, depth, ee_pose, marker_pose = reader.get_frame(idx)
        
    #     # RGB信息
    #     print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    #     print(f"RGB range: [{rgb.min()}, {rgb.max()}]")
        
    #     # Depth信息
    #     print(f"\nDepth shape: {depth.shape}, dtype: {depth.dtype}")
    #     print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}] meters")
    #     print(f"Depth mean: {depth.mean():.4f} meters")
        
    #     # End Effector Pose (8D: position, quaternion, gripper)
    #     print(f"\nEnd Effector Pose (8D):")
    #     print(f"  Position [x, y, z]: {ee_pose[:3]}")
    #     print(f"  Quaternion [qw, qx, qy, qz]: {ee_pose[3:7]}")
    #     print(f"  Gripper width: {ee_pose[7]:.4f}")
        
    #     # Marker Pose (7D: position, quaternion)
    #     print(f"\nMarker Pose in Camera Frame (7D):")
    #     print(f"  Position [x, y, z]: {marker_pose[:3]}")
    #     print(f"  Quaternion [qw, qx, qy, qz]: {marker_pose[3:7]}")
        
    #     # Timestamps
    #     print(f"\nTimestamps:")
    #     print(f"  EE timestamp: {reader.ee_timestamps[idx]:.6f}")
    #     print(f"  Marker timestamp: {reader.marker_timestamps[idx]:.6f}")
        
    #     print()

# if __name__ == '__main__':

#     # src_episode_dir = os.path.join(ROOT_DIR, "episodes", "episode_0000")

#     # dst_root_dir = os.path.join(ROOT_DIR, "episodes_subsampled_2")

#     # subsample_episode(src_episode_dir, dst_root_dir, step=6)
#     # print("Done.")


#     reconstruct_pointcloud(ROOT_DIR + "/episodes/episode_0000", visualize=True)
#     # reader = EpisodeReader(ROOT_DIR + "/episodes_subsampled_2/episode_0000/pose")
#     # print(f"{reader.length}")
#     # rgb, depth, pose = reader.get_frame(10)
#     # print("Pose:", pose)