import numpy as np
import torch
import open3d as o3d
from pytorch3d.ops import sample_farthest_points

def reconstruct_pointcloud(rgb, depth, visualize=False):
    """
    Reconstruct point cloud from RGB + depth.

    Returns:
        point_cloud: (Np, 6) numpy array, columns: [x, y, z, r, g, b]
    """
    # Normalize RGB to [0,1]
    colors = rgb[..., :3] / 255.0
    depths = depth
    camera_matrix = [[531.29, 0.0, 224], [0.0, 531.29, 224], [0.0, 0.0, 1.0]]
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

def preprocess_point_cloud(points, num_points=1024, use_cuda=True):
    extrinsics_matrix = np.array([
        [-0.61193014,  0.2056703,  -0.76370232,  2.22381139],
        [ 0.78640693,  0.05530829, -0.61522771,  1.06986129],
        [-0.084295,   -0.97705717, -0.19558536,  0.90482569],
        [ 0.,          0.,          0.,          1.        ],
    ])
    WORK_SPACE = [[-0.12, 1.12], [-1.00, 1.00], [0.128, 1.5]]
    # WORK_SPACE = [[-0.12, 1.12], [-0.40, 0.80], [0.128, 1.5]]
    # point_xyz = points[..., :3] * 0.00025
    point_xyz = points[..., :3]
    point_hom = np.concatenate([point_xyz, np.ones((point_xyz.shape[0], 1))], axis=1)
    point_xyz = point_hom @ extrinsics_matrix.T
    points[..., :3] = point_xyz[..., :3]

    mask = (
        (points[:, 0] > WORK_SPACE[0][0]) & (points[:, 0] < WORK_SPACE[0][1]) &
        (points[:, 1] > WORK_SPACE[1][0]) & (points[:, 1] < WORK_SPACE[1][1]) &
        (points[:, 2] > WORK_SPACE[2][0]) & (points[:, 2] < WORK_SPACE[2][1])
    )
    points = points[mask]
    if points.shape[0] == 0:
        raise ValueError("All points filtered out by WORK_SPACE constraints.")

    if use_cuda:
        pts_tensor = torch.from_numpy(points[:, :3]).unsqueeze(0).cuda()
    else:
        pts_tensor = torch.from_numpy(points[:, :3]).unsqueeze(0)
    sampled_pts, indices = sample_farthest_points(pts_tensor, K=num_points)
    sampled_pts = sampled_pts.squeeze(0).cpu().numpy()
    indices = indices.cpu().squeeze(0)
    rgb = points[indices.numpy(), 3:]
    return np.hstack((sampled_pts, rgb))