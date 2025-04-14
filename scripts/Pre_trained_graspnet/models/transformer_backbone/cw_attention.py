'''
    Various channel wise attention
'''

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# GBNet channel-wise attention
class CAA_Module(nn.Module):
    '''
    Input: 
        point features (B, C, N)
    Output:
        aggregated point features (B, C, N)
    '''
    def __init__(self, in_dim):
        super(CAA_Module, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_dim//4)
        self.bn2 = nn.BatchNorm1d(in_dim//4)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1, bias=False),
                                        self.bn2,
                                        nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        # Compact Channel-wise Comparator block
        x_hat = x.permute(0, 2, 1) # (B, C, N)
        proj_query = self.query_conv(x_hat) 
        proj_key = self.key_conv(x_hat).permute(0, 2, 1) 
        similarity_mat = torch.bmm(proj_key, proj_query)

        # Channel Affinity Estimator block
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat)-similarity_mat
        affinity_mat = self.softmax(affinity_mat) # (B, N, N)
        
        proj_value = self.value_conv(x_hat) # (B, C, N)
        out = torch.bmm(affinity_mat, proj_value.permute(0, 2, 1))
        # residual connection with a learnable weight
        out = self.alpha*out + x

        return out, affinity_mat