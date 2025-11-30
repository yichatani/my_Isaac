import os
import numpy as np
import open3d as o3d
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# print(f"{ROOT_DIR=}")
# exit()

def reconstruct_pointcloud(data_dir, visualize=False):
    """
    Reconstruct point cloud from RGB + depth.

    Returns:
        point_cloud: (Np, 6) numpy array, columns: [x, y, z, r, g, b]
    """
    # Normalize RGB to [0,1]
    # colors = rgb[..., :3] / 255.0
    # depths = depth
    colors = np.array(Image.open(os.path.join(data_dir, 'rgb_000000.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(data_dir, 'depth_000000.png'))) / 1000.0
    camera_matrix = [[299.08, 0.0, 224], [0.0, 531.70, 224], [0.0, 0.0, 1.0]]
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
    scale = 1.0  # if your depth is in mm, scale it to meters

    # Construct pixel grid
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # mask = (points_z > 0) & (points_z < 2)  # optional: crop invalid range
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
        self.episode_dir = episode_dir

        # ---- load poses ----
        poses_path = os.path.join(episode_dir, "poses.npz")
        assert os.path.exists(poses_path), "poses.npz not found"

        data = np.load(poses_path)
        self.poses = data["poses"]          # (T, 8)
        self.timestamps = data["timestamps"]
        self.indices = data["indices"]

        self.length = self.poses.shape[0]

    def get_frame(self, idx):
        """
        Returns:
            rgb:   (H, W, 3), uint8
            depth: (H, W), float32 (meters)
            pose:  (8,)
        """
        assert 0 <= idx < self.length

        rgb_path = os.path.join(self.episode_dir, f"rgb_{idx:06d}.png")
        depth_path = os.path.join(self.episode_dir, f"depth_{idx:06d}.png")

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth = depth_mm / 1000.0  # mm -> meters

        pose = self.poses[idx]

        return rgb, depth, pose

    def __len__(self):
        return self.length


def subsample_episode(src_episode_dir, dst_root_dir, step=2):
    """
    每隔 step 帧抽一帧，保存到新目录：
        dst_root_dir/episode_xxxx/{rgb, depth, pose}

    - RGB / Depth: 直接拷贝 PNG（不重新编码，避免误差）
    - Pose: 重新写一个 poses.npz，只包含抽出来的帧
    """
    reader = EpisodeReader(src_episode_dir)

    episode_name = os.path.basename(src_episode_dir.rstrip("/"))
    out_episode_dir = os.path.join(dst_root_dir, episode_name)

    rgb_out_dir = os.path.join(out_episode_dir, "rgb")
    depth_out_dir = os.path.join(out_episode_dir, "depth")
    pose_out_dir = os.path.join(out_episode_dir, "pose")

    os.makedirs(rgb_out_dir, exist_ok=True)
    os.makedirs(depth_out_dir, exist_ok=True)
    os.makedirs(pose_out_dir, exist_ok=True)

    selected_indices = np.arange(0, len(reader), step)

    sub_poses = reader.poses[selected_indices]
    sub_timestamps = reader.timestamps[selected_indices]
    sub_indices = reader.indices[selected_indices]

    print(f"Subsampling {src_episode_dir}: {len(reader)} -> {len(selected_indices)} frames")

    for new_idx, orig_idx in enumerate(selected_indices):
        src_rgb = os.path.join(src_episode_dir, f"rgb_{orig_idx:06d}.png")
        src_depth = os.path.join(src_episode_dir, f"depth_{orig_idx:06d}.png")

        dst_rgb = os.path.join(rgb_out_dir, f"rgb_{new_idx:06d}.png")
        dst_depth = os.path.join(depth_out_dir, f"depth_{new_idx:06d}.png")

        if not os.path.exists(src_rgb) or not os.path.exists(src_depth):
            print(f"Warning: missing files for frame {orig_idx:06d}, skip")
            continue

        shutil.copy2(src_rgb, dst_rgb)
        shutil.copy2(src_depth, dst_depth)

    poses_out_path = os.path.join(pose_out_dir, "poses.npz")
    np.savez(
        poses_out_path,
        poses=sub_poses,
        timestamps=sub_timestamps,
        indices=sub_indices,
    )
    print(f"Saved subsampled poses to {poses_out_path}")


if __name__ == '__main__':

    # src_episode_dir = os.path.join(ROOT_DIR, "episodes", "episode_0001")

    # dst_root_dir = os.path.join(ROOT_DIR, "episodes_subsampled")

    # subsample_episode(src_episode_dir, dst_root_dir, step=3)
    # print("Done.")


    # reconstruct_pointcloud(ROOT_DIR + "/episodes/episode_0000", visualize=True)
    reader = EpisodeReader(ROOT_DIR + "/episodes_subsampled/episode_0001/pose")
    print(f"{reader.length}")
    # rgb, depth, pose = reader.get_frame(10)
    # print("Pose:", pose)