import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

import pointnet2.pytorch_utils as pt_utils
# from pointnet2_.pointnet2_utils import CylinderQueryAndGroup, BallQuery, furthest_point_sample, gather_operation
from pointnet2_.pointnet2_utils import CylinderQueryAndGroup
# from utils.model_utils import sample_and_group, query_ball_point, index_points
from Pre_trained_graspnet.utils.loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix
# from knn.knn_modules import knn


class GraspableNet(nn.Module):
    # predict Graspness
    def __init__(self, feat_dim):
        super().__init__()
        self.in_dim = feat_dim
        self.conv_graspable = nn.Sequential(nn.Conv1d(self.in_dim, 3, 1))
        
    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, M)
        end_points['objectness_score'] = graspable_score[:, :2]
        end_points['graspness_score'] = graspable_score[:, 2]
        return end_points


class SuctionableNet(nn.Module):
    # predict Sealness, Wrenchness, Flatness
    def __init__(self, feat_dim):
        super().__init__()
        self.in_dim = feat_dim
        self.conv_suctionable = nn.Sequential(nn.Conv1d(self.in_dim, 3, 1))

    def forward(self, seed_features, end_points):
        suctionable_score = self.conv_suctionable(seed_features)  # (B, 3, M)
        end_points['sealness_score'] = suctionable_score[:, 0]
        end_points['wrenchness_score'] = suctionable_score[:, 1]
        end_points['flatness_score'] = suctionable_score[:, 2]
        return end_points


class ApproachNet(nn.Module):
    def __init__(self, config, is_training=True):
        super().__init__()
        self.num_view = config['Global']['NUM_VIEW']
        self.in_dim = config['Global']['feat_dim']
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)


    def forward(self, end_points):
        seed_features = end_points['features_graspable']
        B, _, num_seed = seed_features.size()
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        view_features = self.conv2(res_features)
        view_score = view_features.transpose(1, 2).contiguous() # (B, num_seed, num_view)
        end_points['view_score'] = view_score

        if self.is_training:
            # normalize view graspness score to 0~1
            view_score_ = view_score.clone().detach()  
            view_score_max, _ = torch.max(view_score_, dim=2)
            view_score_min, _ = torch.min(view_score_, dim=2)
            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)

            top_view_inds = []
            for i in range(B):
                top_view_inds_batch = torch.multinomial(view_score_[i], 1, replacement=False)
                top_view_inds.append(top_view_inds_batch)
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)  # B, num_seed
        else:
            _, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed)

            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            template_views = generate_grasp_views(self.num_view).to(seed_features.device)  # (num_view, 3)
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        # end_points['res_features'] = res_features
        return end_points, res_features


class CloudCrop(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_dim = config['Global']['feat_dim']
        self.nsample = config['CloudCrop']['nsample']
        self.hmin = config['CloudCrop']['hmin']
        self.hmax = config['CloudCrop']['hmax']
        self.cylinder_radius = config['CloudCrop']['cylinder_radius']
        self.out_dim = config['CloudCrop']['out_dim']

        self.grouper = CylinderQueryAndGroup(radius=self.cylinder_radius, hmin=self.hmin, hmax=self.hmax, nsample=self.nsample,
                                             use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP([3 + self.in_dim, self.out_dim, self.out_dim], bn=True) # use xyz, so plus 3

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable) # B * (3 + feat_dim) * M * K
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
        kernel_size = new_features.size(3)
        if torch.is_tensor(kernel_size):
            kernel_size = kernel_size.item()
        new_features = F.max_pool2d(new_features, kernel_size=[1, kernel_size])  # (batch_size, mlps[-1], M, 1)
        # if new_features.shape[-1] == 1:
        #     new_features = new_features.squeeze(-1)
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features


class Local_attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = config['CloudCrop']['out_dim']
        self.q_conv = nn.Conv1d(self.channels, self.channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(self.channels, self.channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(self.channels, self.channels, 1)
        self.trans_conv = nn.Conv1d(self.channels, self.channels, 1)
        self.after_norm = nn.BatchNorm1d(self.channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        #  x.shape   b*n, c, k
        x_q = self.q_conv(x).permute(0, 2, 1)  # b*n, k, c
        x_k = self.k_conv(x)  # b*n, c, k
        x_v = self.v_conv(x)  # b*n, c, k
        energy = x_q @ x_k  # b*n, k, k
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=2, keepdims=True))
        x_r = x_v @ attention  # b*n, c, k
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class SWADNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_angle = config['Global']['NUM_ANGLE']
        self.num_depth = config['Global']['NUM_DEPTH']
        self.in_dim = config['CloudCrop']['out_dim']
        self.in_dim_div2 = int(self.in_dim / 2)

        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(self.in_dim, 2*self.num_angle*self.num_depth, 1)

    def forward(self,  end_points):
        vp_features = end_points['vp_features']

        B, _, num_seed = vp_features.size()
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]  # B * num_seed * num_angle * num_depth
        end_points['grasp_width_pred'] = vp_features[:, 1]
        return end_points



# class Graspable_GroupRegion(nn.Module):
#     '''
#         Group neighbor features of each graspable point
#     '''
#     def __init__(self, config):
#         super(Graspable_GroupRegion, self).__init__()
#         self.in_channel = config['Global']['feat_dim']
#         self.nsample = config['GroupRegion']['nsample']
#         self.knn = config['GroupRegion']['knn']
#         self.mlp = config['GroupRegion']['mlp']
#         self.radius = config['GroupRegion']['radius'] # unused if knn is True

#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()

#         last_channel = self.in_channel
#         for out_channel in self.mlp:
#             self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm2d(out_channel))
#             last_channel = out_channel

#     def forward(self, end_points):
#         """
#             Input:
#                 xyz: input points position data, [B, M, C_3]
#                 points: input points features, [B, M, C], C=256
#             Return:
#                 new_xyz: fused points position data, [B, M, C_3]
#                 new_points_concat: fused points features, [B, M, D], D is the last channel of MLP_List
#         """
#         xyz, points = end_points['point_clouds'], end_points['features'].permute(0, 2, 1)
#         seed_xyz, seed_points = end_points['xyz_graspable'], end_points['features_graspable'].permute(0, 2, 1)

#         # [B, M, nsample, C_3] & [B, M, nsample, C=256]
#         new_xyz, new_points = sample_and_group(self.radius, self.nsample, xyz, points, seed_xyz, seed_points, knn=self.knn)

#         new_points = new_points.permute(0, 3, 2, 1) # [B, C, nsample, M]
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points =  F.relu(bn(conv(new_points)))

#         new_xyz = torch.mean(new_xyz, dim=2)
#         new_points = torch.max(new_points, 2)[0].transpose(1, 2)
#         end_points['xyz_graspable'] = new_xyz
#         end_points['features_graspable'] = new_points
#         return end_points