# """
#     Input: epoch_{epoch}.tar
#     Inference:
#         log_dir/dump_{epoch}_{split}
#     Evaluate:
#         log_dir/dump_{epoch}_{split}/ap_{epoch}_{split}.npy
# """
# import os
# import sys
# import yaml
# import time
# import torch
# import json
# import argparse
# from PIL import Image
# import numpy as np
# import open3d as o3d
# from tqdm import tqdm
# import cv2
# from torch.utils.data import DataLoader
# # from suctionnetAPI import SuctionNetEval
# from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
# sys.path.insert(0, ROOT_DIR)
# from Pre_trained_graspnet.utils.data_utils import CameraInfo,create_point_cloud_from_depth_image
# from Pre_trained_graspnet.models.graspnet import GraspNet, pred_grasp_decode
# from Pre_trained_graspnet.dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn, load_grasp_labels
# from Pre_trained_graspnet.utils.collision_detector import ModelFreeCollisionDetector
# from modules.grasp_generator import define_grasp_pose

# parser = argparse.ArgumentParser()
# # variable in shell
# parser.add_argument('--infer', action='store_true', default=False)
# parser.add_argument('--eval', action='store_true', default=False)
# parser.add_argument('--log_dir', default=os.path.join(ROOT_DIR + '/logs'), required=False)
# # parser.add_argument('--chosen_model', default='1billion.tar', required=False)
# parser.add_argument('--collision_thresh', type=float, default=0.01,
#                     help='Collision Threshold in collision detection [default: 0.01]')
# parser.add_argument('--epoch_range', type=list, default=[79], help='epochs to infer&eval')

# # default in shell
# parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
# parser.add_argument('--seed_feat_dim', default=256, type=int, help='Point wise feature dim')
# parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
# parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
# parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
# parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
# cfgs = parser.parse_args()
# DATA_PATH = os.path.join(ROOT_DIR, "example_data")

# # load model config
# with open(os.path.join(ROOT_DIR + '/models/model_config.yaml'), 'r') as f:
#     model_config = yaml.load(f, Loader=yaml.FullLoader)

# def process_data(data_dict = None, return_raw_cloud = False):
    
#     if data_dict is None:
#         raise ValueError("data_dict is None")
#         depth = np.array(Image.open(os.path.join(DATA_PATH, 'depth.png')))
#         color = np.array(Image.open(os.path.join(DATA_PATH, 'color.png')),dtype=np.float32) / 255.0
#     else:
#         color = data_dict["rgb"] / 255.0
#         depth = data_dict["depth"]

#     fx, fy = 927.17, 927.37
#     cx, cy = 651.32, 349.62
#     width, height = 1280, 720
#     factor_depth = 1000
#     camera = CameraInfo(width, height, fx, fy, cx, cy, factor_depth)

#     cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
#     # print("cloud shape:", cloud.shape)
    
#     cloud_masked = cloud.reshape(-1,3)
#     mask = (cloud_masked[:, 2] >= 0) & (cloud_masked[:, 2] <= 1) # filter out points outside the table
#     # print("cloud_masked shape:", cloud_masked.shape)
#     cloud_masked = cloud_masked[mask]

#     color_masked = color.reshape(-1, color.shape[-1])
#     color_masked = color_masked[mask]
    
#     if return_raw_cloud:
#         return cloud_masked

#     if len(cloud_masked) >= cfgs.num_point:
#         idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
#     else:
#         idxs1 = np.arange(len(cloud_masked))
#         idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
#         idxs = np.concatenate([idxs1, idxs2], axis=0)
#     cloud_sampled = cloud_masked[idxs]
#     color_sampled = color_masked[idxs]  

#     ret_dict = {
#                 'raw_point_clouds': cloud_masked.astype(np.float32),
#                 'raw_color': color_masked.astype(np.float32),
#                 'point_clouds': cloud_sampled.astype(np.float32),
#                 'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
#                 'feats': np.ones_like(cloud_sampled).astype(np.float32),
#                 'color': color_sampled.astype(np.float32),
#                 }
#     return ret_dict


# def pretrained_graspnet(data_dict = None, chosen_model = '1billion.tar'):
#     sample_data = process_data(data_dict)
#     raw_pointclouds = sample_data['raw_point_clouds']
#     raw_color = sample_data['raw_color']
#     net = GraspNet(model_config, is_training=False)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net.to(device)
    
#     #checkpoint_path = os.path.join(cfgs.log_dir, 'epoch_{}.tar'.format(epoch))
#     checkpoint_path = os.path.join(cfgs.log_dir, chosen_model)
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['model_state_dict'])
#     sample_data = minkowski_collate_fn([sample_data])
#     for key in sample_data:
#         if 'list' in key:
#             for i in range(len(sample_data[key])):
#                 for j in range(len(sample_data[key][i])):
#                     sample_data[key][i][j] = sample_data[key][i][j].to(device)
#         else:
#             sample_data[key] = sample_data[key].to(device)

#     net.eval()
#     with torch.no_grad():
#         source_end_points = net(sample_data)
#         grasp_preds = pred_grasp_decode(source_end_points)  

#     preds = grasp_preds[0].detach().cpu().numpy()
#     gg = GraspGroup(preds)

#     # print("====")
#     # print(len(gg))
#     # print("====")
#     if cfgs.collision_thresh >0:
#         cloud = process_data(data_dict,return_raw_cloud=True)  
#         mfcdetector = ModelFreeCollisionDetector(cloud,voxel_size=cfgs.voxel_size_cd)
#         collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
#         gg = gg[~collision_mask]

#     gg = gg.sort_by_score()


#     indices_to_remove = []
#     # for i, grasp in enumerate(gg):
#     #     if (grasp.translation[0] <= -0.2 or grasp.translation[0] >= 0.35) or \
#     #         (grasp.translation[1] <= -0.2 or grasp.translation[1] >= 0.1):
#     #         indices_to_remove.append(i)


#     for i, grasp in enumerate(gg):
#         if grasp.score < 0.15:
#             indices_to_remove.append(i)
#     gg = gg.remove(indices_to_remove)


#     # print(gg[0].score)    

#     grippers = gg[0:20].to_open3d_geometry_list()
#     cloud = o3d.geometry.PointCloud()
#     cloud.points = o3d.utility.Vector3dVector(raw_pointclouds.astype(np.float32))
#     cloud.colors = o3d.utility.Vector3dVector(np.asarray(raw_color, dtype=np.float32))
#     o3d.visualization.draw_geometries([cloud, *grippers])

#     target_grasp_pose_to_cam = define_grasp_pose(gg[0])

#     data_dict = {
#         "width":gg[0].width,
#         "depth":gg[0].depth,
#         "T":target_grasp_pose_to_cam,
#         "score":gg[0].score
#     }
#     return data_dict

   

# if __name__ == '__main__':
        
#     data_dict = pretrained_graspnet(chosen_model = '1billion.tar')

#     # data_dict = pretrained_graspnet(chosen_model='mega.tar')


import os
import sys
import yaml
import torch
from PIL import Image
import numpy as np
import open3d as o3d
from types import SimpleNamespace
from graspnetAPI.graspnet_eval import GraspGroup
from Pre_trained_graspnet.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from Pre_trained_graspnet.models.graspnet import GraspNet, pred_grasp_decode
from Pre_trained_graspnet.dataset.graspnet_dataset import minkowski_collate_fn
from Pre_trained_graspnet.utils.collision_detector import ModelFreeCollisionDetector
from modules.grasp_generator import define_grasp_pose

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, ROOT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "example_data")

# load model config
with open(os.path.join(ROOT_DIR, 'models/model_config.yaml'), 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)


def default_cfgs():
    return SimpleNamespace(
        log_dir=os.path.join(ROOT_DIR, 'logs'),
        collision_thresh=0.01,
        num_point=15000,
        voxel_size=0.005,
        voxel_size_cd=0.01,
    )


def process_data(data_dict=None, return_raw_cloud=False, cfgs=None):
    if data_dict is None:
        raise ValueError("data_dict is None")

    color = data_dict["rgb"] / 255.0
    depth = data_dict["depth"]

    camera_matrix = [[1281.77, 0.0, 960], [0.0, 1281.77, 540], [0.0, 0.0, 1.0]]
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
    width, height = 1920, 1080
    factor_depth = 1.0
    camera = CameraInfo(width, height, fx, fy, cx, cy, factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud_masked = cloud.reshape(-1, 3)
    mask = (cloud_masked[:, 2] >= 0) & (cloud_masked[:, 2] <= 1)
    cloud_masked = cloud_masked[mask]
    color_masked = color.reshape(-1, color.shape[-1])[mask]

    if return_raw_cloud:
        return cloud_masked

    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    ret_dict = {
        'raw_point_clouds': cloud_masked.astype(np.float32),
        'raw_color': color_masked.astype(np.float32),
        'point_clouds': cloud_sampled.astype(np.float32),
        'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
        'feats': np.ones_like(cloud_sampled).astype(np.float32),
        'color': color_sampled.astype(np.float32),
    }
    return ret_dict


def pretrained_graspnet(data_dict=None, chosen_model='1billion.tar', cfgs=None):
    if cfgs is None:
        cfgs = default_cfgs()

    sample_data = process_data(data_dict, cfgs=cfgs)
    raw_pointclouds = sample_data['raw_point_clouds']
    raw_color = sample_data['raw_color']

    net = GraspNet(model_config, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(">>>>>>Loading model>>>>>>")
    checkpoint_path = os.path.join(cfgs.log_dir, chosen_model)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    sample_data = minkowski_collate_fn([sample_data])
    for key in sample_data:
        if 'list' in key:
            for i in range(len(sample_data[key])):
                for j in range(len(sample_data[key][i])):
                    sample_data[key][i][j] = sample_data[key][i][j].to(device)
        else:
            sample_data[key] = sample_data[key].to(device)

    net.eval()
    with torch.no_grad():
        source_end_points = net(sample_data)
        grasp_preds = pred_grasp_decode(source_end_points)

    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)

    if cfgs.collision_thresh > 0:
        cloud = process_data(data_dict, return_raw_cloud=True, cfgs=cfgs)
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]

    gg = gg.sort_by_score()

    indices_to_remove = [i for i, grasp in enumerate(gg) if grasp.score < 0.15]
    gg = gg.remove(indices_to_remove)

    # grippers = gg[0:20].to_open3d_geometry_list()
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(raw_pointclouds.astype(np.float32))
    # cloud.colors = o3d.utility.Vector3dVector(np.asarray(raw_color, dtype=np.float32))
    # o3d.visualization.draw_geometries([cloud, *grippers])

    target_grasp_pose_to_cam = define_grasp_pose(gg[0])

    result_dict = {
        "width": gg[0].width,
        "depth": gg[0].depth,
        "T": target_grasp_pose_to_cam,
        "score": gg[0].score
    }
    return result_dict


# Optional CLI entry
if __name__ == '__main__':
    dummy_data = {
        "rgb": np.array(Image.open(os.path.join(DATA_PATH, 'color.png')), dtype=np.float32),
        "depth": np.array(Image.open(os.path.join(DATA_PATH, 'depth.png')))
    }
    result = pretrained_graspnet(data_dict=dummy_data, chosen_model='1billion.tar')
    print(result)

