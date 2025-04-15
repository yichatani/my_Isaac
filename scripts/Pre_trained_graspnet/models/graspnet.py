""" GraspNet baseline model definition.follows
    Author: chenxi-wang
"""

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import yaml
import torch
import torch.nn as nn

import numpy as np

from models.Pointnet2_backbone.pointnet2_backbone import Pointnet2Backbone
from models.modules import GraspableNet, SuctionableNet, ApproachNet, CloudCrop, SWADNet


from Pre_trained_graspnet.utils.suction_utils import suction_normal
from Pre_trained_graspnet.utils.label_generation import process_grasp_labels, process_meta_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn
try:
    import pointnet2._ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

class GraspNet(nn.Module):
    def __init__(self, model_config, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.M_points = model_config['Global']['M_POINT']
        self.feat_dim = model_config['Global']['feat_dim']
        self.backbone_name = model_config['Backbone']['name']
        self.backbone_in = model_config['Backbone']['in_channels']
        self.training_branch = model_config['Global']['training_branch']
        self.GRASPNESS_THRESHOLD = model_config['Global']['GRASPNESS_THRESHOLD']
        self.SUCTIONESS_THRESHOLD = model_config['Global']['SUCTIONESS_THRESHOLD']

        if self.backbone_name == 'Pointnet2':
            self.backbone = Pointnet2Backbone()

        self.graspable = GraspableNet(feat_dim=self.feat_dim)
        self.suctionable = SuctionableNet(feat_dim=self.feat_dim)
        self.approach = ApproachNet(model_config, is_training=self.is_training)
        self.crop = CloudCrop(model_config)
        self.swad = SWADNet(model_config)

    def backbone_forward(self, end_points):

        if self.backbone_name == 'Pointnet2':
            end_points['features'] = self.backbone(end_points)  # (B, C, N)
        return end_points


    def graspable_fps(self, end_points):
        """
            input: 
                points & features: [B, N, 3], [B, C, N]
            output:
                graspable points & features: [B, M, 3], [B, C, M]
        """
        B, _, _ = end_points['features'].shape
        seed_xyz, seed_features = end_points['point_clouds'], end_points['features']
        seed_features_flipped = seed_features.transpose(1, 2)
        end_points = self.graspable(seed_features, end_points)
        # graspable mask
        objectness_score = end_points['objectness_score']
        graspness_score = end_points['graspness_score']
        if end_points['graspness_score'].shape[1] == 1:
            graspness_score = end_points['graspness_score'].squeeze(1)

        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        #check which graspness are valid
        graspness_mask = graspness_score > self.GRASPNESS_THRESHOLD
        graspable_mask = objectness_mask & graspness_mask

        seed_features_graspable = []
        seed_xyz_graspable = []
        seed_graspness_graspable = []
        graspable_num_batch = 0.
        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_seed_xyz = seed_xyz[i][cur_mask] # (Ns, 3)
            cur_seed_feat = seed_features_flipped[i][cur_mask] # (Ns, C)
            graspness_score = graspness_score.unsqueeze(2) # (B, N, 1)
            cur_seed_graspness = graspness_score[i][cur_mask].reshape(-1, 1) # (Ns, 1)
            
            #check if there is no valid grasps
            if cur_seed_feat.shape[0] == 0: # Handle 0 Exception
                print('Exception: no graspable points!')
                graspness_score = graspness_score.reshape(B, -1) # (B, N)
                topk_values, topk_idx = torch.topk(graspness_score[i], self.M_points)
                seed_xyz_graspable.append(seed_xyz[i][topk_idx])
                seed_features_graspable.append(seed_features_flipped[i][topk_idx].transpose(0, 1))
                seed_graspness_graspable.append(graspness_score[i][topk_idx].reshape(-1, 1))
                continue

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # (1, Ns, 3)
            #using farthest point
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points) # (1, M)
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous() # (1, 3, Ns)
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0) # M * 3
            # cur_seed_xyz =_ext.gather_points(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2)
            # cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2)
            #if cur_seed_xyz.shape[0] == 1:
            cur_seed_xyz = cur_seed_xyz.squeeze(0)
            cur_feat_flipped = cur_seed_feat.unsqueeze(0).transpose(1, 2).contiguous() # (1, C, Ns)

            cur_seed_feat = gather_operation(cur_feat_flipped, fps_idxs)# (C, M)
            # cur_seed_feat = _ext.gather_points(cur_feat_flipped, fps_idxs)# (C, M)
            # if cur_seed_feat.shape[0] == 1:
            cur_seed_feat = cur_seed_feat.squeeze(0)
            cur_seed_feat = cur_seed_feat.contiguous()
            cur_graspness_flipped = cur_seed_graspness.unsqueeze(0).transpose(1, 2).contiguous() # (1, 1, Ns)

            cur_seed_graspness = gather_operation(cur_graspness_flipped, fps_idxs).transpose(1, 2)# (M, 1)
            #if cur_seed_graspness.shape[0] == 1:
            # cur_seed_graspness = _ext.gather_points(cur_graspness_flipped, fps_idxs).transpose(1, 2)# (M, 1)
            cur_seed_graspness = cur_seed_graspness.squeeze(0)
            cur_seed_graspness = cur_seed_graspness.contiguous()

            seed_xyz_graspable.append(cur_seed_xyz)
            seed_features_graspable.append(cur_seed_feat)
            seed_graspness_graspable.append(cur_seed_graspness)
                    
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0) # (B, M, 3)
        seed_features_graspable = torch.stack(seed_features_graspable) # (B, C, M)
        seed_graspness_graspable = torch.stack(seed_graspness_graspable) # (B, M, 1)
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['features_graspable'] = seed_features_graspable
        end_points['scores_graspable'] = seed_graspness_graspable
        end_points['graspable_count_stage1'] = graspable_num_batch / B

        return end_points


    def suctionable_fps(self, end_points):
        """
            input: 
                points & features: [B, N, 3], [B, C, N]
            output:
                suctionable points & features: [B, M, 3], [B, C, M]
        """
        B, _, _ = end_points['features'].shape
        seed_xyz, seed_features = end_points['point_clouds'], end_points['features']

        # suctionable mask
        end_points = self.suctionable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim
        objectness_score = end_points['objectness_score']
        sealness_score = end_points['sealness_score']
        wrenchness_score = end_points['wrenchness_score']
        flatness_score = end_points['flatness_score']

        suctioness_score = sealness_score.clamp(0, 1) * wrenchness_score
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        suctioness_mask = suctioness_score > self.SUCTIONESS_THRESHOLD
        suctionable_mask = objectness_mask & suctioness_mask

        # suctionable points (xyz, feature)
        seed_features_suctionable = []
        seed_xyz_suctionable = []
        seed_scores_suctionable = []
        suctionable_num_batch = 0.
        for i in range(B):
            cur_mask = suctionable_mask[i]
            suctionable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns * feat_dim, maybe 0 * feat_dim
            cur_seed_xyz = seed_xyz[i][cur_mask] # Ns * 3, maybe 0 * 3
            cur_seed_score = suctioness_score[i][cur_mask].reshape(-1, 1) # Ns * 1, maybe 0 * 1

            if cur_feat.shape[0] == 0: # Handle 0 Exception: no suctionable points
                topk_values, topk_idx = torch.topk(suctioness_score[i], self.M_points)
                seed_xyz_suctionable.append(seed_xyz[i][topk_idx])
                seed_features_suctionable.append(seed_features_flipped[i][topk_idx].transpose(0, 1))
                seed_scores_suctionable.append(suctioness_score[i][topk_idx].reshape(-1, 1))
                continue

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1 * Ns * 3
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous() # 1 * 3 * Ns
            # cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0) # M * 3
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2)
            if cur_seed_xyz.shape[0] == 1:
                cur_seed_xyz = cur_seed_xyz.squeeze(0)
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1 * feat_dim * Ns
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs) # feat_dim * M
            if cur_feat.shape[0] == 1:
                cur_feat = cur_feat.squeeze(0)
            cur_feat = cur_feat.contiguous()
            cur_seed_score_flipped = cur_seed_score.unsqueeze(0).transpose(1, 2).contiguous()  # 1 * 1 * Ns
            cur_seed_score = gather_operation(cur_seed_score_flipped, fps_idxs).transpose(1, 2) # M * 1
            if cur_seed_score.shape[0] == 1:
                cur_seed_score = cur_seed_score.squeeze(0)
            cur_seed_score = cur_seed_score.contiguous()
            seed_features_suctionable.append(cur_feat)
            seed_xyz_suctionable.append(cur_seed_xyz)
            seed_scores_suctionable.append(cur_seed_score)

        seed_xyz_suctionable = torch.stack(seed_xyz_suctionable, 0) # B * M * 3
        seed_features_suctionable = torch.stack(seed_features_suctionable)  # B * feat_dim * M
        seed_scores_suctionable = torch.stack(seed_scores_suctionable)  # B * M * 1

        end_points['xyz_suctionable'] = seed_xyz_suctionable
        end_points['features_suctionable'] = seed_features_suctionable
        end_points['scores_suctionable'] = seed_scores_suctionable
        end_points['suctionable_count_stage1'] = suctionable_num_batch / B

        return end_points


    def forward(self, end_points, dataset_type ='graspnet'):
        # backbone forward
        end_points = self.backbone_forward(end_points)
        # print('backbone outputs',end_points )

        if self.training_branch == 'grasp':
            # graspable mask & FPS
            end_points = self.graspable_fps(end_points)

        # View prediction
        seed_xyz_graspable = end_points['xyz_graspable']
        seed_features_graspable = end_points['features_graspable']
        end_points, res_feat= self.approach(end_points)
        # end_points, res_feat = end_points['res_features']
        seed_features_graspable = seed_features_graspable + res_feat

        # select best view
        if self.is_training:
            if dataset_type == 'graspnet':
                end_points = process_grasp_labels(end_points)
            elif dataset_type == 'meta':
                end_points = process_meta_grasp_labels(end_points)
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']

        group_features = self.crop(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        end_points['vp_features'] = group_features
        # end_points = self.swad(group_features, end_points)
        end_points = self.swad(end_points)

        # print('end_points', end_points)

        return end_points
    def export_to_onnx(self, outpath, device):
        point_clouds = torch.randn(100000, 3).numpy()
        voxel_size=0.005
        point_clouds = point_clouds.astype(np.float32)
        coors = point_clouds.astype(np.float32) / voxel_size
        feats = np.ones_like(point_clouds).astype(np.float32)
        color = point_clouds.astype(np.float32)

        dummy_inputs = {
            'coors': coors,
            'feats': feats,
            'point_clouds': point_clouds,
            'color': color,
        }
        dummy_inputs = minkowski_collate_fn([dummy_inputs])

        for key in dummy_inputs:
            if 'list' in key:
                for i in range(len(dummy_inputs[key])):
                    for j in range(len(dummy_inputs[key][i])):
                        dummy_inputs[key][i][j] = dummy_inputs[key][i][j].to(device)
            else:
                dummy_inputs[key] = dummy_inputs[key].to(device)


        end_points = self.backbone_forward(dummy_inputs)
        end_points = self.graspable_fps(end_points)

        seed_xyz_graspable = end_points['xyz_graspable']
        seed_features_graspable = end_points['features_graspable']

        # export approach net
        approach_inputs = end_points
        approach_input_names = ['coors', 'feats', 'quantize2original', 'point_clouds', 'color', 'features', 'objectness_score', 
                                'graspness_score', 'xyz_graspable', 'features_graspable', 'scores_graspable', 'graspable_count_stage1']
        approach_output_names = ['coors', 'feats', 'quantize2original', 'point_clouds', 'color', 'features', 'objectness_score', 
                                'graspness_score', 'xyz_graspable', 'features_graspable', 'scores_graspable', 'graspable_count_stage1', 
                                'view_score', 'grasp_top_view_xyz', 'grasp_top_view_rot', 'grasp_top_view_inds', 'vp_features']
        torch.onnx.export(self.approach, approach_inputs, f'{outpath}onnx_approach.onnx', input_names = approach_input_names, output_names = approach_output_names,
                        dynamic_axes = {'coors': {0: 'batch_size', 1: 'sequence_length'}, 
                                        'feats':{0: 'batch_size', 1: 'sequence_length'},
                                        'quantize2original':{0: 'batch_size', 1: 'sequence_length'},
                                        'point_clouds':{0: 'batch_size', 1: 'sequence_length'},
                                        'color':{0: 'batch_size', 1: 'sequence_length'},
                                        'features':{0: 'batch_size', 1: 'sequence_length', 2: 'width'},
                                        'objectness_score':{0: 'batch_size', 1: 'sequence_length', 2: 'width'},
                                        'graspness_score':{0: 'batch_size', 1: 'sequence_length'},
                                        'xyz_graspable':{0: 'batch_size', 1: 'sequence_length'}, 
                                        'features_graspable':{0: 'batch_size', 1: 'sequence_length'}, 
                                        'scores_graspable':{0: 'batch_size', 1: 'sequence_length'}, 
                                        'graspable_count_stage1':{0: 'batch_size', 1: 'sequence_length'},
                                        'view_score':{0: 'batch_size', 1: 'sequence_length'},
                                        'grasp_top_view_xyz': {0: 'batch_size', 1: 'sequence_length'},
                                        'grasp_top_view_rot':{0: 'batch_size', 1: 'sequence_length'},
                                        'grasp_top_view_inds':{0: 'batch_size', 1: 'sequence_length'},
                                        'vp_features':{0: 'batch_size', 1: 'sequence_length'}}, opset_version=12)

        end_points, res_features = self.approach(end_points)
        seed_features_graspable = seed_features_graspable + res_features

        crop_input_names = [ 'features_graspable',   'seed_features_graspable', 'grasp_top_view_rot',]

        # crop model: not converted to onnx
        grasp_top_views_rot = end_points['grasp_top_view_rot']
        crop_input = (seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)

        torch.onnx.export(self.crop, crop_input, f'{outpath}onnx_crop.onnx', input_names = crop_input_names, 
                        dynamic_axes = {'features_graspable':{0: 'batch_size', 1: 'sequence_length'}, 
                                        'grasp_top_view_rot':{0: 'batch_size', 1: 'sequence_length'},
                                        'seed_features_graspable':{0: 'batch_size', 1: 'sequence_length'} }, opset_version=12)

        group_features = self.crop(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)



        # convert the swad model to onnx

        end_points['vp_features'] = group_features
        swad_input_names = ['coors', 'feats', 'quantize2original', 'point_clouds', 'color', 'features', 'objectness_score', 'graspness_score', 
                            'xyz_graspable', 'features_graspable', 'scores_graspable', 'graspable_count_stage1', 'view_score', 'grasp_top_view_xyz', 'grasp_top_view_rot', 
                            'grasp_top_view_inds','vp_features']

        swad_output_names = ['coors', 'feats', 'quantize2original', 'point_clouds', 'color', 'features', 'objectness_score', 'graspness_score', 'xyz_graspable', 
                            'features_graspable', 'scores_graspable', 'graspable_count_stage1', 'view_score', 'grasp_top_view_xyz', 'grasp_top_view_rot', 
                            'grasp_top_view_inds', 'vp_features', 'grasp_score_pred', 'grasp_width_pred']

        torch.onnx.export(self.swad, end_points, f'{outpath}onnx_swad.onnx', input_names = swad_input_names, output_names = swad_output_names,
                        dynamic_axes = {
                                        'coors': {0: 'batch_size', 1: 'sequence_length'}, 
                                        'feats':{0: 'batch_size', 1: 'sequence_length'},
                                        'quantize2original':{0: 'batch_size', 1: 'sequence_length'},
                                        'point_clouds':{0: 'batch_size', 1: 'sequence_length'},
                                        'color':{0: 'batch_size', 1: 'sequence_length'},
                                        'features':{0: 'batch_size', 1: 'sequence_length', 2: 'width'},
                                        'objectness_score':{0: 'batch_size', 1: 'sequence_length', 2: 'width'},
                                        'graspness_score':{0: 'batch_size', 1: 'sequence_length'},
                                        'xyz_graspable':{0: 'batch_size', 1: 'sequence_length'}, 
                                        'features_graspable':{0: 'batch_size', 1: 'sequence_length'}, 
                                        'scores_graspable':{0: 'batch_size', 1: 'sequence_length'}, 
                                        'graspable_count_stage1':{0: 'batch_size', 1: 'sequence_length'},
                                        'view_score':{0: 'batch_size', 1: 'sequence_length'},
                                        'grasp_top_view_xyz': {0: 'batch_size', 1: 'sequence_length'},
                                        'grasp_top_view_rot':{0: 'batch_size', 1: 'sequence_length'},
                                        'grasp_top_view_inds':{0: 'batch_size', 1: 'sequence_length'},
                                        'vp_features':{0: 'batch_size', 1: 'sequence_length', 2:'wdith'},
                                        'grasp_score_pred': {0: 'batch_size', 1: 'sequence_length', 2:'wdith'},
                                        'grasp_width_pred': {0: 'batch_size', 1: 'sequence_length', 2:'wdith'}}, opset_version=12)
    



def pred_grasp_decode(end_points):
    GRASP_MAX_WIDTH, NUM_ANGLE, NUM_DEPTH, M_POINT = 0.1, 12, 4, 1024
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()

        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        grasp_angle = (grasp_score_inds // NUM_DEPTH) * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        scale = 10.
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] 
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M_POINT, 9)

        print('grasp angles CHECKING NOW', grasp_angle)
        print('grasp width CHECKING NOW', grasp_width)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds


def pred_suction_decode(end_points):
    point_cloud = end_points['point_clouds']
    point_cloud = point_cloud.squeeze(0).detach().cpu().numpy().reshape(-1, 3) # [N, 3]
    point_color = end_points['color']
    point_color = point_color.squeeze(0).detach().cpu().numpy().reshape(-1, 3) # [N, 3]

    suction_points = end_points['xyz_suctionable']
    suction_points = suction_points.squeeze(0).detach().cpu().numpy() # [M, 3]
    suction_normals = suction_normal(suction_points, point_cloud) # [M, 3]
    suction_scores = end_points['scores_suctionable']
    suction_scores = suction_scores.squeeze(0).detach().cpu().numpy() # [M, 1]

    # concatnate scores, points, normals
    suction_preds = np.concatenate((suction_scores, suction_normals, suction_points), axis=1) # [M, 7]
    return suction_preds


