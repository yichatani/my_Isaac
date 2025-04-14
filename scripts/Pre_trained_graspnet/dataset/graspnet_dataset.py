""" GraspNet dataset processing.
    To generate the graspNet dataset, you need to first run generate_graspness.py if you want to use graspness as one of the training features.
    Then you need to run simplify_dataset.py to simplify the data labels. Then everything is ready to use.
    Author: chenxi-wang
"""

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import yaml
import math
import random
import numpy as np
import scipy.io as scio
from PIL import Image
from tqdm import tqdm
import json
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask
import MinkowskiEngine as ME
import collections.abc as container_abcs

# load model config
model_config_path = os.path.join(ROOT_DIR, 'models/model_config.yaml')
with open(model_config_path, 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)


class GraspNetDataset(Dataset):
    def __init__(self, root, grasp_labels, suction_labels_root, camera='realsense', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=True, augment=False, load_label=True):
        # assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}
        self.adapt_radius = True
        self.suction_labels_root = suction_labels_root

        if split == 'train':
            self.sceneIds = list(range(0,151))
        elif split == "test_one":
            self.sceneIds = list(range(1))
        elif split == 'test':
            self.sceneIds = list(range(169, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(59, 60))
            #self.sceneIds = [14]
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        elif split == 'validation':
            self.sceneIds = list(range(151, 169))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        #self.sceneIds = ['{}'.format(str(x).zfill(6)) for x in self.sceneIds]

        self.depthpath = []
        self.colorpath = []
        self.normalpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        self.flatnesspath = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(256):
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.graspnesspath.append(os.path.join(root, 'Affordance', 'graspness_avg', x, camera, str(img_num).zfill(4) + '.npy'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]
        # self.sceneIds = ['{}'.format(str(x).zfill(6)) for x in self.sceneIds]
        # for x in tqdm(self.sceneIds, desc = 'Loading data path ...'):
        #     for img_num in range(39,40):
        #         self.depthpath.append(os.path.join(root, str(x).zfill(6), 'depth', str(img_num).zfill(6) + '.png'))
        #         self.colorpath.append(os.path.join(root, str(x).zfill(6), 'rgb', str(img_num).zfill(6) + '.jpg'))
        #         self.scenename.append(x)
        #         self.frameid.append(img_num)

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
        else:
            flip_mat = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list, flip_mat, rot_mat

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        # print('HELLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
        depth = np.array(Image.open(self.depthpath[index]))
        color = np.array(Image.open(self.colorpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)
        
        # with open('/home/zhy/yc_dir/realsense_yc/results.json') as f:
        #     fil = json.load(f)
        #     fx, fy, cx, cy, width, height = fil['fx'], fil['fy'], fil['cx'], fil['cy'], fil['width'],  fil['height']
        # factor_depth = 1000
        # #width, height = 640, 480
        # camera = CameraInfo(width, height, fx, fy, cx, cy, factor_depth)

        # generate cloud, color, normal
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        color = color / 255


        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        # normal_masked = normal[mask]

        if return_raw_cloud:
            return cloud_masked
        # sample points random
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        # normal_sampled = normal_masked[idxs]

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),

                    'color': color_sampled.astype(np.float32),
                    # 'normal': normal_sampled.astype(np.float32),
                    # 'camera_poses': camera_poses[self.frameid[index]],
                    }
        return ret_dict

    def get_data_label(self, index):
        depth = np.array(Image.open(self.depthpath[index]))
        color = np.array(Image.open(self.colorpath[index]))
        # normal = np.array(Image.open(self.normalpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])

        # graspness
        graspness = np.load(self.graspnesspath[index])
        scene_idx, anno_idx = int(index/256), int(index%256)

        wrenchness = np.zeros_like(depth, dtype=np.float32) # load wrench score, [720, 1280]


        # camera info
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        color = color / 255
        # normal = normal / 255

        # get valid points
        depth_mask = (depth > 0)

        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        # normal_masked = normal[mask]
        seg_masked = seg[mask]
        # sealness_masked = sealness[mask]
        wrenchness_masked = wrenchness[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        seg_sampled = seg_masked[idxs]
        graspness_sampled = graspness[idxs]
        
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1
        wrenchness_sampled = wrenchness_masked[idxs]

        # pose labels
        object_poses_list = []
        grasp_points_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        for i, obj_idx in enumerate(obj_idxs):
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, widths, scores = self.grasp_labels[obj_idx]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)

            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_widths_list.append(widths[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)

        # point cloud augment
        if self.augment:
            cloud_sampled, object_poses_list, flip_mat, rot_mat = self.augment_data(cloud_sampled, object_poses_list)

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'color': color_sampled.astype(np.float32),
                    # 'normal': normal_sampled.astype(np.float32),
                    'camera_poses': camera_poses[self.frameid[index]].astype(np.float32),

                    # affordance label
                    'objectness_label': objectness_label.astype(np.int64),
                    'graspness_label': graspness_sampled.astype(np.float32),
                    # 'flatness_label': flatness_sampled.astype(np.float32),
                    # 'sealness_label': sealness_sampled.astype(np.float32),
                    'wrenchness_label': wrenchness_sampled.astype(np.float32),
                    'seg_label': seg_sampled.astype(np.int64),

                    # pose label
                    'object_poses_list': object_poses_list,
                    'grasp_points_list': grasp_points_list,
                    'grasp_widths_list': grasp_widths_list,
                    'grasp_scores_list': grasp_scores_list}

        # MinkowskiEngine input for voxelization
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        if model_config['Backbone']['in_channels'] == 3:
            ret_dict['feats'] = np.ones_like(cloud_sampled).astype(np.float32)
        elif model_config['Backbone']['in_channels'] == 6:
            ret_dict['feats'] = np.concatenate((cloud_sampled, color_sampled), axis=1).astype(np.float32) # RGBD input

        return ret_dict


def load_grasp_labels(root):
    
    dir_path = os.path.join(root, 'grasp_label_simplified')
    count = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
    print('File Count:', count)
    obj_names = list(range(1, count))

    # obj_names = [15, 1, 6, 16, 21, 49, 67, 71, 47]
    grasp_labels = {}
    for obj_name in tqdm(obj_names, desc='Loading grasping labels...'):
        label = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_name - 1).zfill(3))))
        grasp_labels[obj_name] = (label['points'].astype(np.float32), label['width'].astype(np.float32),
                                  label['scores'].astype(np.float32))

    return grasp_labels



def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }
    def pad_tensor(tensor, target_size):
        current_size = tensor.size(0)
        if current_size < target_size:
            padding_size = target_size - current_size
            padded_tensor = F.pad(tensor, (0, 0, 0, 0, 0, padding_size))
            return padded_tensor
        else:
            return tensor


    def batch_variable_size_tensors(batch):
        """
        Batches a list of variable-sized numpy arrays into a single PyTorch tensor.
        
        Args:
        batch: List of numpy arrays with potentially different sizes.
        
        Returns:
        A tuple containing:
        - batched_tensor: PyTorch tensor with padded data
        - original_sizes: List of original sizes of each tensor
        """
        # Convert numpy arrays to PyTorch tensors and get their shapes
        tensors = [torch.from_numpy(b) for b in batch]
        shapes = [t.shape for t in tensors]
        
        # Find the maximum size for each dimension
        max_shape = [max(s[i] for s in shapes) for i in range(max(len(shape) for shape in shapes))]
        
        # Create a list to store padded tensors
        padded_tensors = []
        
        for tensor in tensors:
            # Calculate padding sizes
            pad_sizes = []
            for i, dim in enumerate(tensor.shape):
                pad_size = max_shape[i] - dim
                pad_sizes.extend([0, pad_size])  # Padding at the end of each dimension
            
            # Reverse pad_sizes because F.pad expects them in reverse order
            padded_tensor = torch.nn.functional.pad(tensor, reverse(pad_sizes))
            padded_tensors.append(padded_tensor)
        
        # Stack the padded tensors
        batched_tensor = torch.stack(padded_tensors, dim=0)
        
        return batched_tensor, shapes

    def reverse(lst):
        return lst[::-1]

    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            # max_size = max([b.shape[0] for b in batch])
            # batch = [pad_tensor(torch.from_numpy(b), max_size) for b in batch]
            # return torch.stack(batch, 0)
            # return torch.cat([torch.from_numpy(b) for b in batch])
            return torch.stack([torch.from_numpy(b) for b in batch])
            # return [torch.from_numpy(b) for b in batch]
            # return torch.cat(([torch.from_numpy(b).unsqueeze(0) for b in batch]), dim=0) 
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    res = collate_fn_(list_data)

    return res