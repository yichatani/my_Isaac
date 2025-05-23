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

class IsaacZarrDataset(BaseDataset):
    def __init__(self,
                 zarr_path,
                 n_obs_steps=None,
                 n_action_steps=None,
                 horizon = None,
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
        self.horizon = horizon

        self.pad_before = pad_before if pad_before is not None else n_obs_steps - 1
        self.pad_after = pad_after if pad_after is not None else n_action_steps - 1

        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path)

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