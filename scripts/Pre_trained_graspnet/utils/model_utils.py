import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def square_distance(src, dst):
    """
        Calculate Euclid distance between each two points.
        Input:
            src: source points, [B, M, C]
            dst: target points, [B, N, C]
        Output:
            dist: per-point square distance, [B, M, N]
    """

    # B, _, N = src.shape
    # _, _, M = dst.shape
    # dist = -2 * torch.matmul(src.permute(0, 2, 1), dst)
    # dist += torch.sum(src ** 2, 1).view(B, N, 1)
    # dist += torch.sum(dst ** 2, 1).view(B, 1, M)
    # return dist

    B, N, _ = src.shape
    _, M, _ = dst.shape
    # print('src: ', src.shape)
    # print('dst: ', dst.permute(0, 2, 1).shape)
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, 2).view(B, N, 1)
    dist += torch.sum(dst ** 2, 2).view(B, 1, M)

    return dist


def index_points(points, idx):
    """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S, [K]]
        Return:
            new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
        Input:
            radius: local region radius
            nsample: max sample number in local region
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, S, C]
        Return:
            group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)
    if radius is not None:
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
    else:
        group_idx = torch.sort(sqrdists, dim=-1)[1][:,:,:nsample]
    return group_idx


def sample_and_group(radius, nsample, xyz, points, seed_xyz, seed_points, knn=True):
    """
        Input:
            radius: the radius of ball query if knn is false
            nsample: neighbor nums
            xyz: input points position, [B, N, C_3]
            points: input points features, [B, N, C]
            seed_xyz: seed points position, [B, M, C_3]
            seed_points: seed points features, [B, M, C]
        Return:
            new_xyz: sampled points position, [B, M, nsample, C_3]
            new_points: sampled points features, [B, M, nsample, C]
    """
    B, C_3, M = seed_xyz.shape

    if knn:
        dists = square_distance(seed_xyz, xyz)  # B x M x N
        idx = dists.argsort()[:, :, :nsample]  # B x M x K
    else:
        idx = query_ball_point(radius, nsample, xyz, seed_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, M, nsample, C_3]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - seed_xyz.unsqueeze(2) # [B, M, nsample, C_3]
    torch.cuda.empty_cache()

    # -> new_xyz & new_points
    new_xyz = grouped_xyz
    new_points = index_points(points, idx) # [B, M, nsample, C]

    return new_xyz, new_points


