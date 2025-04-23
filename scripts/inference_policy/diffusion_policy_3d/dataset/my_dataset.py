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
                 n_obs_steps=2,
                 n_action_steps=5,
                 pad_before=None,
                 pad_after=None,
                 seed=42,
                 val_ratio=0.1,
                 max_train_episodes=None,
                 task_name=None):
        super().__init__()
        self.task_name = task_name

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = n_obs_steps + n_action_steps - 1

        # padding ：pad_before = n_obs_steps - 1, pad_after = n_action_steps - 1
        self.pad_before = pad_before if pad_before is not None else n_obs_steps - 1
        self.pad_after = pad_after if pad_after is not None else n_action_steps - 1

        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path)
        assert self.replay_buffer.n_episodes > 0, f"Replay buffer is empty. Please check the path: {zarr_path}"

        val_mask = get_val_mask(self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = downsample_mask(~val_mask, max_n=max_train_episodes, seed=seed)
        self.train_mask = train_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.train_mask
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        if idx == 0:
            print(f"[Dataset] agent_pos shape: {data['obs']['agent_pos'].shape}")
            print(f"[Dataset] point_cloud shape: {data['obs']['point_cloud'].shape}")
            print(f"[Dataset] action shape: {data['action'].shape}")
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
