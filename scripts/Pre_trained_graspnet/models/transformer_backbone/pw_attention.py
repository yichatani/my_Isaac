'''
    Various point wise attention
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# PA "Point attention network for semantic segmentation of 3D point clouds"
class PA(nn.Module):
    '''
    Input: 
        point features (B, N, C)
    Output:
        aggregated point features (B, N, C)
    '''
    def __init__(self, channels):
        super(PA, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        # self.q_conv.conv.weight = self.k_conv.conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, N, C) -> (B, C, N)
        x_q = self.q_conv(x).permute(0, 2, 1) # (B, N, Cq)
        x_k = self.k_conv(x) # (B, Ck, N)
        x_v = self.v_conv(x) # (B, Cv, N)
        
        energy = torch.bmm(x_q, x_k) # (B, N, N)
        energy /= np.sqrt(x_q.size(-1)) # (B, N, N)
        attention = self.softmax(energy) # (B, N, N)

        x_r = torch.bmm(x_v, attention) # (B, Cv, N)
        x = x + x_r # (B, Cv, N)
        return x.permute(0, 2, 1), attention # (B, N, C)


# A_SCN "Attentional shapecontextnet for point cloud recognition"
class A_SCN(nn.Module):
    '''
    Input: 
        point features (B, N, C)
    Output:
        aggregated point features (B, N, C)
    '''
    def __init__(self, channels):
        super(A_SCN, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        # self.q_conv.conv.weight = self.k_conv.conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, N, C) -> (B, C, N)
        x_q = self.q_conv(x).permute(0, 2, 1) # (B, N, Cq)
        x_k = self.k_conv(x) # (B, Ck, N)
        x_v = self.v_conv(x) # (B, Cv, N)
        energy = torch.bmm(x_q, x_k) # (B, N, N)

        # SS(scale + softmax) or SL(softmax + L1 norm)
        energy /= np.sqrt(x_q.size(-1)) # (B, N, N)
        attention = self.softmax(energy) # (B, N, N)
        # attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True)) # (B, N, N)

        x_r = torch.bmm(x_v, attention) # (B, Cv, N)
        x = x_v + x_r # (B, Cv, N)
        return x.permute(0, 2, 1), attention # (B, N, C)


# PCT "Point cloud transformer"
class PCT(nn.Module):
    '''
    Input: 
        point features (B, N, C)
    Output:
        aggregated point features (B, N, C)
    '''
    def __init__(self, channels):
        super(PCT, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        # self.q_conv.conv.weight = self.k_conv.conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, N, C) -> (B, C, N)
        x_q = self.q_conv(x).permute(0, 2, 1) # (B, N, Cq)
        x_k = self.k_conv(x) # (B, Ck, N)    
        x_v = self.v_conv(x) # (B, Cv, N)
        energy = torch.bmm(x_q, x_k) # (B, N, N)
        attention = self.softmax(energy) # (B, N, N)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True)) # (B, N, N)
        x_r = torch.bmm(x_v, attention) # (B, Cv, N)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r))) # (B, Cv, N)
        x = x + x_r # (B, Cv=C, N)
        return x.permute(0, 2, 1), attention # (B, N, C)


# PT "Point Transformer" Local !!!
class PT_PWA(nn.Module):
    '''
    Input: 
        point xyz (B, N, 3)
        point features (B, N, C)
    Output:
        attention map (B, N, K, C)
        aggregated features (B, N, C)
    '''
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    def forward(self, xyz, features):
        from .pointnet_util import index_points, square_distance
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k] # (B, N, K)
        knn_xyz = index_points(xyz, knn_idx)

        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx) # (B, N, C), (B, N, K, C), (B, N, K, C)
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz) # (B, N, K, C)
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc) # (B, N, K, C)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2) # (B, N, K, C)
        
        new_features = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        new_features = self.fc2(new_features) + features
        return new_features, attn


# "LighTN: Light-weight Transformer Network for Performance-overhead Tradeoff in Point Cloud Downsampling"
class LighTN(nn.Module):
    '''
    Input: 
        point features (B, N, C)
    Output:
        aggregated point features (B, N, C)
    '''
    def __init__(self, channels):
        super(LighTN, self).__init__()
        self.conv_1d = nn.Conv1d(2 * channels, channels, 1)
        self.FFN = nn.Conv1d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, C, N)
        energy = torch.bmm(x.permute(0, 2, 1), x) # (B, N, N)
        energy /= np.sqrt(x.size(-2)) # (B, N, N)
        attention = self.softmax(energy) # (B, N, N)
        x_r = torch.bmm(x, attention) # (B, C, N)
        x_r = torch.cat((x, x_r), dim=1) # (B, 2C, N)
        x_r = self.conv_1d(x_r) # (B, C, N)
        x = self.FFN(x + x_r) # (B, C, N)
        return x.permute(0, 2, 1), attention # (B, N, C)



if __name__ == '__main__':

    B, N, C = 2, 1000, 512
    x = torch.randn((B, N, C))

    # test bug
    # A_SCN = A_SCN(channels=C)
    # x = A_SCN(x)
    # print('x shape: ', x.shape)

    # PA = PA(channels=C)
    # x = PA(x)
    # print('x shape: ', x.shape)

    # PCT = PCT(channels=C)
    # x = PCT(x)
    # print('x shape: ', x.shape) LighTN

    LighTN = LighTN(channels=C)
    x = LighTN(x)
    print('x shape: ', x.shape)