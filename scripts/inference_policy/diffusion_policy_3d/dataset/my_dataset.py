import os
import h5py
import numpy as np
import torch
import copy
import open3d as o3d
from typing import Dict
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer


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

    mask = (points_z > 0) & (points_z < 2)  # optional: crop invalid range
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


class IsaacZarrDataset(BaseDataset):
    def __init__(self,
                 zarr_path,
                 horizon=16,
                 pad_before=5,
                 pad_after=15,
                 seed=42,
                 val_ratio=0.1,
                 max_train_episodes=None,
                 task_name=None):
        super().__init__()
        self.task_name = task_name
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        if not os.path.isabs(zarr_path):
            zarr_path = os.path.abspath(os.path.join(os.getcwd(), zarr_path))
        print("[DEBUG] Using zarr path:", zarr_path)

        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path)

        val_mask = get_val_mask(self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = downsample_mask(~val_mask, max_n=max_train_episodes, seed=seed)
        self.train_mask = train_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)

    def _sample_to_data(self, sample):
        return {
            'obs': {
                'agent_pos': sample['state'].astype(np.float32),
                'point_cloud': sample['point_cloud'].astype(np.float32),
            },
            'action': sample['action'].astype(np.float32)
        }

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'agent_pos': self.replay_buffer['state'],
            'action': self.replay_buffer['action'],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer



# class IsaacHDF5Dataset(BaseDataset):
#     def __init__(self,
#                  hdf5_dir,
#                  horizon=1,
#                  pad_before=0,
#                  pad_after=0,
#                  seed=42,
#                  val_ratio=0.1,
#                  max_train_episodes=None,
#                  task_name=None):
#         super().__init__()
#         self.task_name = task_name
#         self.horizon = horizon
#         self.pad_before = pad_before
#         self.pad_after = pad_after

#         # Step 1: 加载所有 episodes
#         self.replay_buffer = ReplayBuffer.create_empty()
#         episode_files = sorted([os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith(".h5")])
#         for path in episode_files:
#             data = self.load_hdf5_episode(path)
#             self.replay_buffer.add_episode(data)

#         # Step 2: 创建训练/验证 mask
#         val_mask = get_val_mask(self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
#         train_mask = downsample_mask(~val_mask, max_n=max_train_episodes, seed=seed)
#         self.train_mask = train_mask

#         # Step 3: 创建序列采样器
#         self.sampler = SequenceSampler(
#             replay_buffer=self.replay_buffer,
#             sequence_length=horizon,
#             pad_before=pad_before,
#             pad_after=pad_after,
#             episode_mask=train_mask
#         )

#     def load_hdf5_episode(self, path):
#         with h5py.File(path, "r") as f:
#             index = f["index"][:]
#             agent_pos = f["agent_pos"][:]
#             action = f["action"][:]

#             # 按需选择相机
#             cam = "front"
#             rgb = f[f"{cam}/rgb"][:]
#             depth = f[f"{cam}/depth"][:]

#         # 构造模拟 point cloud 输入，这里直接拼接 rgb + depth 做 placeholder
#         # 可以改为 reconstruct_pointcloud(rgb, depth)
#         # num_frames = len(index)
#         # rgb = rgb.reshape((num_frames, -1))  # flatten image
#         # depth = depth.reshape((num_frames, -1))
#         # point_cloud = np.concatenate([rgb, depth[..., None]], axis=-1)
#         point_cloud = reconstruct_pointcloud(rgb, depth)

#         return {
#             "state": agent_pos.astype(np.float32),       # (T, D_state)
#             "action": action.astype(np.float32),         # (T, D_action)
#             "point_cloud": point_cloud.astype(np.float32),  # (T, D_pc)
#         }

#     def get_validation_dataset(self):
#         val_set = copy.copy(self)
#         val_set.sampler = SequenceSampler(
#             replay_buffer=self.replay_buffer,
#             sequence_length=self.horizon,
#             pad_before=self.pad_before,
#             pad_after=self.pad_after,
#             episode_mask=~self.train_mask
#         )
#         val_set.train_mask = ~self.train_mask
#         return val_set

#     def get_normalizer(self, mode='limits', **kwargs):
#         data = {
#             'action': self.replay_buffer['action'],
#             'agent_pos': self.replay_buffer['state'],
#             'point_cloud': self.replay_buffer['point_cloud']
#         }
#         normalizer = LinearNormalizer()
#         normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
#         return normalizer

#     def __len__(self):
#         return len(self.sampler)

#     def _sample_to_data(self, sample):
#         return {
#             'obs': {
#                 'agent_pos': sample['state'].astype(np.float32),
#                 'point_cloud': sample['point_cloud'].astype(np.float32),
#             },
#             'action': sample['action'].astype(np.float32)
#         }

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         sample = self.sampler.sample_sequence(idx)
#         data = self._sample_to_data(sample)
#         torch_data = dict_apply(data, torch.from_numpy)
#         return torch_data
