import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .pw_attention import PA
from .cw_attention import CAA_Module
from .pointnet_util import *


# # original version
# class TransitionDown(nn.Module):
#     def __init__(self, k, nneighbor, channels):
#         super().__init__()
#         self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
#     def forward(self, xyz, points):
#         return self.sa(xyz, points)


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.npoint = k
        self.radius = 0
        self.nsample = nneighbor
        self.knn = True
        self.input_dim = channels[0] - 3
        self.output_dim = channels[1]
        
        self.w_qs = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.w_ks = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.w_vs = nn.Linear(self.input_dim, self.output_dim, bias=False)

        self.fc_delta = nn.Sequential(
            nn.Linear(3, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        self.fc_last = nn.Linear(self.output_dim, self.output_dim)
        self.bn_last = nn.BatchNorm1d(self.output_dim)
        
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, C]
        Return:
            sampled_xyz: sampled points position data, [B, S, 3]
            sampled_points_concat: sample points feature data, [B, S, C']
        Default params: S = N / 4, C' = C * 2
        """

        # FPS & KNN, sampled_xyz: (B, S, 3), neighbors: (B, S, K, 3+C)
        sampled_xyz, neighbors, _, fps_idx = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, 
                                                                    knn=self.knn, returnfps=True)
        sampled_points = index_points(points, fps_idx) # (B, S, C)
        knn_xyz_norm, knn_points = neighbors[..., :3], neighbors[..., 3:] # (B, S, K, 3), (B, S, K, C)
        
        # vector attention
        q, k, v = self.w_qs(sampled_points), self.w_ks(knn_points), self.w_vs(knn_points) # (B, S, C'), (B, S, K, C'), (B, S, K, C')
        pos_enc = self.fc_delta(knn_xyz_norm) # (B, S, K, C')
        att_map = self.fc_gamma(q[:, :, None] - k + pos_enc) # (B, S, K, C')
        att_map = F.softmax(att_map / np.sqrt(self.output_dim), dim=-2) # (B, S, K, C')
        
        res = torch.einsum('bmnf, bmnf -> bmf', att_map, v + pos_enc)
        res = F.relu(self.bn_last(self.fc_last(res).transpose(1, 2)).transpose(1, 2))
        # res = self.fc_last(res) # no bn version, need to comment 'self.bn_last'

        return sampled_xyz, res # (B, S, 3), (B, S, C')


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2



# # original version
# class TransformerBlock(nn.Module):
#     def __init__(self, d_points, d_model, k) -> None:
#         super().__init__()
#         self.fc1 = nn.Linear(d_points, d_model)
#         self.fc2 = nn.Linear(d_model, d_points)
#         self.fc_delta = nn.Sequential(
#             nn.Linear(3, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, d_model)
#         )
#         self.fc_gamma = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, d_model)
#         )
#         self.w_qs = nn.Linear(d_model, d_model, bias=False)
#         self.w_ks = nn.Linear(d_model, d_model, bias=False)
#         self.w_vs = nn.Linear(d_model, d_model, bias=False)
#         self.k = k
        
#     # xyz: b x n x 3, features: b x n x f
#     def forward(self, xyz, features):
#         dists = square_distance(xyz, xyz)
#         knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
#         knn_xyz = index_points(xyz, knn_idx)
        
#         pre = features
#         x = self.fc1(features)
#         q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

#         pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        
#         attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
#         # print('attn: ', attn.shape)
#         attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
#         res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
#         res = self.fc2(res) + pre
#         return res, attn



# Dual branch: point-wise + channel-wise attention
class TransformerBlock(nn.Module):
    '''
    Input: 
        point features (B, N, C)
    Output:
        aggregated point features (B, N, C)
    '''
    def __init__(self, channels, d_model, k) -> None:
        super().__init__()
        self.input_dim = channels
        self.point_wise = PA(self.input_dim)
        self.channel_wise = CAA_Module(self.input_dim)

        self.bn1 = nn.BatchNorm1d(self.input_dim)
        self.bn2 = nn.BatchNorm1d(self.input_dim)
        self.fc_2 = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim)
        )

    def forward(self, xyz, x):
        # global attention, so we do not need xyz to group knn here.

        # point & channel wise attention
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x_p, _ = self.point_wise(x)
        x_c, _ = self.channel_wise(x)
        x = x_p + x_c + 2 * x

        # batch norm 2 & FFN
        x_norm = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x_norm_ffn = self.fc_2(x_norm)
        x = x + x_norm_ffn

        # # *****************************
        # x_p, att_p = self.point_wise(x)
        # x_c, att_c = self.channel_wise(x)
        # x = x_p + x_c


        return x