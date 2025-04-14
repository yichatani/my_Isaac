import yaml
import os
import sys 

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
import numpy as np

from PIL import Image
import json
from tqdm import tqdm

from torch.utils.data import Dataset
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image



model_config_path = os.path.join(ROOT_DIR, 'models/model_config.yaml')
with open(model_config_path, 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

class MetaGraspnetDataset(Dataset):
  """
    This class is written based on graspnet_dataset.py, so it is very similar to GraspNetDataset
  """
  def __init__(self, root,split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=True, augment=False, load_label=True):
        # assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.augment = augment
        self.load_label = load_label
        self.adapt_radius = True
        # self.grasp_labels = grasp_labels

        if split == 'train':
            self.sceneIds = list(range(2132,2210))
        elif split == "test_one":
            self.sceneIds = list(range(1))
        elif split == 'test':
            self.sceneIds = list(range(2102, 2200))
        elif split == 'test_seen':
            self.sceneIds = list(range(59, 60))
            #self.sceneIds = [14]
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        elif split == 'validation':
            self.sceneIds = list(range(151, 169))
        self.sceneIds = ['scene{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        #self.sceneIds = ['{}'.format(str(x).zfill(6)) for x in self.sceneIds]

        self.depthpath = []
        self.colorpath = []
        self.normalpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        self.flatnesspath = []
        self.lablepath = []

        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(10):
                self.depthpath.append(os.path.join(root,  x,  str(img_num).zfill(1) + '.npz'))
                self.colorpath.append(os.path.join(root,  x,   str(img_num).zfill(1) + '_rgb.png'))
                self.metapath.append(os.path.join(root,  x,  str(img_num).zfill(1) + '_camera_params.json'))
                self.lablepath.append(os.path.join(root,  x,  str(img_num).zfill(3) + '_labels.npz'))
                # self.lablepath.append(os.path.join(root,  'grasp_label',  x + '_' + str(img_num).zfill(6) + '_labels.npz'))
                # self.graspnesspath.append(os.path.join(root, 'Affordance', 'graspness_avg', x, str(img_num).zfill(4) + '.npy'))
                
                self.scenename.append(x.strip())
                self.frameid.append(img_num)

  def scene_list(self):
    return self.scenename

  def __len__(self):
    return len(self.depthpath)
  

  def __getitem__(self, index):
    if self.load_label:
        return self.get_data_label(index)
    else:
        return self.get_data(index)

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



  def get_data(self, index, return_raw_cloud=False):
    # print('HELLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
    # depth = np.array(Image.open(self.depthpath[index]))
    with np.load(self.depthpath[index]) as data:
        depth_np = data['depth']
        seg_np = data['instances_objetcs']

    for i in range(depth_np.shape[0]):
        for j in range(depth_np.shape[1]):
            if np.isnan(depth_np[i,j]):
                depth_np[i,j] = 0.0

    for i in range(seg_np.shape[0]):
        for j in range(seg_np.shape[1]):
            if np.isnan(seg_np[i,j]):
                seg_np[i,j] = 0.0

    depth = depth_np
    seg = seg_np
    color = np.array(Image.open(self.colorpath[index]))
    with open(self.metapath[index], 'r') as file:
        meta = json.load(file)
    scene = self.scenename[index]
    try:
        fx, fy = meta['fx'], meta['fy']
        width, height = meta['resolution']['width'], meta['resolution']['height']
    except Exception as e:
        print(repr(e))
        print(scene)

    factor_depth = 100
    camera_info = {"fx": 1784.49072265625, "fy": 1786.48681640625, "cx": 975.0308837890625, "cy": 598.6246337890625, "width": 1944, "height": 1200}
    camera = CameraInfo(width, height, fx, fy, width/2, height/2, factor_depth)
    

    # generate cloud, color, normal
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    color = color / 255


    # get valid points
    depth_mask = (depth > 0)
    # if self.remove_outlier:
    #     camera_poses = np.load(os.path.join(self.root, 'scenes', scene,  'camera_poses.npy'))
    #     align_mat = np.load(os.path.join(self.root, 'scenes', scene,  'cam0_wrt_table.npy'))
    #     trans = np.dot(align_mat, camera_poses[self.frameid[index]])
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

                }
    return ret_dict
  

  def get_data_label(self, index):
    with np.load(self.depthpath[index]) as data:
        depth_np = data['depth']
        seg_np = data['instances_objects']

    for i in range(depth_np.shape[0]):
        for j in range(depth_np.shape[1]):
            if np.isnan(depth_np[i,j]):
                depth_np[i,j] = 0.0

    for i in range(seg_np.shape[0]):
        for j in range(seg_np.shape[1]):
            if np.isnan(seg_np[i,j]):
                seg_np[i,j] = 0.0

    depth = depth_np
    seg = seg_np
    color = np.array(Image.open(self.colorpath[index]))
    # normal = np.array(Image.open(self.normalpath[index]))
    with open(self.metapath[index], 'r') as file:
        meta = json.load(file)

    # graspness
    # graspness = np.load(self.graspnesspath[index])
    scene_idx, anno_idx = int(index/256), int(index%256)

    wrenchness = np.zeros_like(depth, dtype=np.float32) # load wrench score, [720, 1280]


    # camera info
    scene = self.scenename[index]
    try:
        # poses = meta['poses']
        fx, fy = meta['fx'], meta['fy']
        width, height = meta['resolution']['width'], meta['resolution']['height']
    except Exception as e:
        print(repr(e))
        print(scene)

    factor_depth = 100
    camera_info = {"fx": 1784.49072265625, "fy": 1786.48681640625, "cx": 975.0308837890625, "cy": 598.6246337890625, "width": 1944, "height": 1200}
    camera = CameraInfo(width, height, fx, fy, width/2, height/2, factor_depth)

    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    color = color / 255
    # normal = normal / 255

    # get valid points
    depth_mask = (depth > 0)

    # if self.remove_outlier:
    #     # camera_poses = np.load(os.path.join(self.root, 'scenes', scene, 'camera_poses.npy'))
    #     # align_mat = np.load(os.path.join(self.root, 'scenes', scene,  'cam0_wrt_table.npy'))
        
    #     trans = np.dot(align_mat, camera_poses[self.frameid[index]])


    mask = depth_mask
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    seg_masked = seg[mask]
    
    
    # normal_masked = normal[mask]

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
    objectness_label = seg_sampled.copy()
    objectness_label[objectness_label > 1] = 1


    # graspness_sampled = graspness[idxs]
    wrenchness_sampled = wrenchness_masked[idxs]

    # pose labels


    # for i, obj_idx in enumerate(obj_idxs):
    #     if (0 == obj_idx).sum() < 50:
    #         continue
    label_path = self.lablepath[index]
    label = np.load(label_path)
    
    # grasp_points_list, grasp_widths_list, grasp_scores_list, grasp_rots_list =label['points'].astype(np.float32), label['width'].astype(np.float32), label['scores'].astype(np.float32), label['rot'].astype(np.float32)
    grasp_points_list, grasp_widths_list, grasp_scores_list =label['points'].astype(np.float32), label['width'].astype(np.float32), label['scores'].astype(np.float32)
    # print('grasp_points_list', grasp_points_list)

    # idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)


    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'color': color_sampled.astype(np.float32),
                # 'normal': normal_sampled.astype(np.float32),
                # 'camera_poses': camera_poses[self.frameid[index]].astype(np.float32),

                # affordance label
                'objectness_label': objectness_label.astype(np.int64),
                # 'graspness_label': graspness_sampled.astype(np.float32),
                # 'flatness_label': flatness_sampled.astype(np.float32),
                # 'sealness_label': sealness_sampled.astype(np.float32),
                'wrenchness_label': wrenchness_sampled.astype(np.float32),

                # pose label
                # 'grasp_rots_list':grasp_rots_list,
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
  
