""" PointNet2 backbone for feature learning.
    Author: Charles R. Qi
"""
import os
import sys
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule


class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0):
        super().__init__()
        """
            Sampling layer: choose a subset of points using iterative farthest point sampling.
            Grouping layer: groups the sets using KNN or certain Manhattan distance
            PointNet layer: abstracted the centroid of each grouping set
            
            npoints: number of points to samples
            radius: radii for multi-scale grouping
            nsample:  number of samples in each group
            mlp: MLP configurations for each scale
            use_xyz: Whether to use xyz coordinates as features

            **********************
            So it has 4 set abstractions
            **********************
        
        """

        self.sa1 = PointnetSAModuleVotes(
            npoint=2048,
            radius=0.04,
            nsample=64,
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.1,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.2,
            nsample=16,
            mlp=[256, 256, 256, 512],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=256,
            radius=0.3,
            nsample=16,
            mlp=[512, 256, 256, 512],
            use_xyz=True,
            normalize_xyz=True
        )

        self.fp1 = PointnetFPModule(mlp=[512 + 512, 512, 512])
        self.fp2 = PointnetFPModule(mlp=[512 + 256, 512, 512])
        self.fp3 = PointnetFPModule(mlp=[256 + 128, 512, 512])
        self.fp4 = PointnetFPModule(mlp=[0 + 128, 512, 512])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,D,K)
                XXX_inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        pointcloud = pointcloud['point_clouds']

        batch_size = pointcloud.shape[0]


        xyz, features = self._break_up_pc(pointcloud)
        end_points['input_xyz'] = xyz
        end_points['input_features'] = features

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features)  # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # # --------- 2 FEATURE UPSAMPLING LAYERS --------
        # features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'],
        #                     end_points['sa4_features'])
        # features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        # end_points['fp2_features'] = features
        # end_points['fp2_xyz'] = end_points['sa2_xyz']
        # num_seed = end_points['fp2_xyz'].shape[1]
        # end_points['fp2_inds'] = end_points['sa1_inds'][:, 0:num_seed]  # indices among the entire input point clouds
        #
        # return features, end_points['fp2_xyz'], end_points

        # --------- 4 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], end_points['sa3_features'])
        features = self.fp3(end_points['sa1_xyz'], end_points['sa2_xyz'], end_points['sa1_features'], end_points['sa2_features'])
        features = self.fp4(end_points['input_xyz'], end_points['sa1_xyz'], end_points['input_features'], end_points['sa1_features'])
        return features


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_input = torch.rand(2, 20000, 3).to(device)
    backbone = Pointnet2Backbone().to(device)
    input_feat, input_xyz, _ = backbone(test_input)

    # print('input_xyz: ', input_xyz.shape)
    # print('input_feat: ', input_feat.shape)
