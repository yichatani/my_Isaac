'''
    Attention modules for ResUnet 3D shotcut connection
'''

import os
import yaml
models_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(models_path, 'model_config.yaml'), 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME


class ResUnet_Attention_Module(nn.Module):
    '''
    Input: 
        Sparse Tensor features (N, C)
    Output:
        Attention Sparse Tensor features (N, C)
    '''
    def __init__(self, channels):
        super(ResUnet_Attention_Module, self).__init__()
        self.backbone_name = model_config['Backbone']['name']
        if self.backbone_name == 'ResUnet_PA':
            self.position_attention = ResUnet_Position_Attention(channels)
        elif self.backbone_name == 'ResUnet_CA':
            self.channel_attention = ResUnet_Channel_Attention(channels)
        else:
            self.position_attention = ResUnet_Position_Attention(channels)
            self.channel_attention = ResUnet_Channel_Attention(channels)

    def forward(self, x):
        if self.backbone_name == 'ResUnet_PA':
            x = self.position_attention(x)
        elif self.backbone_name == 'ResUnet_CA':
            x = self.channel_attention(x)
        else:
            x = self.position_attention(x) + self.channel_attention(x)
        return x


class ResUnet_Position_Attention(nn.Module):
    '''
    Input: 
        Sparse Tensor features (N, C)
    Output:
        Attention Sparse Tensor features (N, C)
    '''
    def __init__(self, channels):
        super(ResUnet_Position_Attention, self).__init__()
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        # self.q.weight = self.k.weight 
        self.v = nn.Linear(channels, channels)
        self.softmax = nn.Softmax(dim=-1)
        # LBR
        self.LBR = nn.Sequential(nn.Linear(channels, channels),
                                 nn.BatchNorm1d(channels),
                                 nn.ReLU())

    def forward(self, x):
        x_q = self.q(x.F) # (N, Cq)
        x_k = self.k(x.F) # (N, Ck)
        x_v = self.v(x.F) # (N, C)
        
        energy = torch.matmul(x_q, x_k.T) # (N, N)
        energy /= np.sqrt(x_q.size(-1)) # (N, N)
        attn_map = self.softmax(energy) # (N, N)
        # print('attn_map: ', attn_map.shape)

        x_r = torch.matmul(attn_map, x_v) # (N, C)
        x_r = x_r - x_v # (N, C)
        x_r = self.LBR(x_r) # (N, C)

        P = ME.SparseTensor(
            # coordinates=coords, not required
            features=x_r,
            tensor_stride=x.tensor_stride,
            quantization_mode=x.quantization_mode,
            coordinate_manager=x.coordinate_manager,  # must share the same coordinate manager
            coordinate_map_key=x.coordinate_map_key,  # For inplace, must share the same coords key
            device=x.device
        )
        P = P + x
        return P


class ResUnet_Channel_Attention(nn.Module):
    '''
    Input: 
        Sparse Tensor features (N, C)
    Output:
        Attention Sparse Tensor features (N, C)
    '''
    def __init__(self, channels):
        super(ResUnet_Channel_Attention, self).__init__()
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        # self.q.weight = self.k.weight 
        self.v = nn.Linear(channels, channels)
        self.softmax = nn.Softmax(dim=-1)
        # LBR
        self.LBR = nn.Sequential(nn.Linear(channels, channels),
                                 nn.BatchNorm1d(channels),
                                 nn.ReLU())

    def forward(self, x):
        x_q = self.q(x.F) # (N, Cq)
        x_k = self.k(x.F) # (N, Ck)
        x_v = self.v(x.F) # (N, C)
        
        energy = torch.matmul(x_q.T, x_k) # (Cq, Ck)
        energy /= np.sqrt(x_q.size(-1)) # (Cq, Ck), / Cq
        attn_map = self.softmax(energy) # (Cq, Ck)

        x_r = torch.matmul(x_v, attn_map) # (N, C)
        x_r = x_r - x_v # (N, C)
        x_r = self.LBR(x_r) # (N, C)

        P = ME.SparseTensor(
            # coordinates=coords, not required
            features=x_r,
            tensor_stride=x.tensor_stride,
            quantization_mode=x.quantization_mode,
            coordinate_manager=x.coordinate_manager,  # must share the same coordinate manager
            coordinate_map_key=x.coordinate_map_key,  # For inplace, must share the same coords key
            device=x.device
        )
        P = P + x
        return P



