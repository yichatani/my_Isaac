# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Dict, Tuple, Union, List, Type
# from termcolor import cprint

# def create_mlp(
#         input_dim: int,
#         output_dim: int,
#         net_arch: List[int],
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         squash_output: bool = False,
# ) -> List[nn.Module]:
#     if len(net_arch) > 0:
#         modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
#     else:
#         modules = []

#     for idx in range(len(net_arch) - 1):
#         modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
#         modules.append(activation_fn())

#     if output_dim > 0:
#         last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
#         modules.append(nn.Linear(last_layer_dim, output_dim))
#     if squash_output:
#         modules.append(nn.Tanh())
#     return modules


# class AdvancedPointNetEncoderWithStateAttention(nn.Module):
#     """
#     一个PointNet风格的编码器，它使用状态向量作为Query来关注点云的逐点特征。
#     """
#     def __init__(self,
#                  point_input_dim: int = 3,          # 点的输入维度 (例如，3 for XYZ)
#                  state_input_dim: int = 64,         # state_feat 的维度
#                  per_point_feat_dim: int = 128,     # 点的内部特征维度，也是注意力机制的嵌入维度
#                  final_out_dim: int = 256,          # 此编码器最终输出的维度
#                  num_attn_heads: int = 4,           # 注意力头数
#                  use_layernorm_in_point_mlp: bool = False, # 是否在初始point_mlp中使用LayerNorm
#                  point_mlp_hidden_dims: List[int] = [64, 128] # 初始point_mlp在达到per_point_feat_dim之前的隐藏层维度
#                 ):
#         super().__init__()
        
#         # 1. MLP 用于提取每个点的初始特征
#         initial_mlp_layers = []
#         current_dim = point_input_dim
#         for h_dim in point_mlp_hidden_dims:
#             initial_mlp_layers.append(nn.Linear(current_dim, h_dim))
#             if use_layernorm_in_point_mlp:
#                 initial_mlp_layers.append(nn.LayerNorm(h_dim)) # LayerNorm在Linear之后，激活之前
#             initial_mlp_layers.append(nn.ReLU())
#             current_dim = h_dim
#         initial_mlp_layers.append(nn.Linear(current_dim, per_point_feat_dim))
#         # 最后一层到 per_point_feat_dim 通常不加激活或归一化，直接送入注意力
#         self.point_mlp_initial = nn.Sequential(*initial_mlp_layers)

#         # 2. 将 state_feat 投影为 Query 向量
#         self.state_query_proj = nn.Linear(state_input_dim, per_point_feat_dim)

#         # 3. 多头注意力层
#         self.attention = nn.MultiheadAttention(
#             embed_dim=per_point_feat_dim,
#             num_heads=num_attn_heads,
#             batch_first=True  # 输入输出格式: (Batch, Sequence, Feature)
#         )
        
#         # 可选: 在注意力之后应用 LayerNorm (常见的做法)
#         self.attention_layernorm = nn.LayerNorm(per_point_feat_dim)

#         # 4. 最终的 MLP，将经过注意力处理的特征投影到 final_out_dim
#         # 你可以选择一个简单的线性层或者一个更深的网络
#         self.final_projection_mlp = nn.Sequential(
#             nn.Linear(per_point_feat_dim, per_point_feat_dim * 2), # 示例：先放大
#             nn.ReLU(),
#             nn.Linear(per_point_feat_dim * 2, final_out_dim)
#         )
#         # 或者仅一个线性层:
#         # self.final_projection_mlp = nn.Linear(per_point_feat_dim, final_out_dim)

#     def forward(self, points: torch.Tensor, state_feat: torch.Tensor) -> torch.Tensor:
#         # points: (B, N, point_input_dim) e.g., (BatchSize, NumPoints, 3)
#         # state_feat: (B, state_input_dim) e.g., (BatchSize, 64)

#         # 1. 获取每个点的特征
#         # 输出形状: (B, N, per_point_feat_dim)
#         per_point_features = self.point_mlp_initial(points)

#         # 2. 将 state_feat 投影为 Query
#         # 输出形状: (B, per_point_feat_dim)
#         state_query_unexpanded = self.state_query_proj(state_feat)
#         # 为注意力机制增加序列长度维度: (B, 1, per_point_feat_dim)
#         query = state_query_unexpanded.unsqueeze(1)

#         # 3. 应用Cross-Attention
#         # query: (B, L_query=1, E_attn)
#         # key:   (B, S_key=N, E_attn) -> per_point_features
#         # value: (B, S_value=N, E_attn) -> per_point_features
#         # attended_features 输出形状: (B, L_query=1, E_attn)
#         # attn_output_weights 输出形状: (B, L_query=1, S_key=N) - 可用于可视化分析
#         attended_features, attn_output_weights = self.attention(
#             query=query,
#             key=per_point_features,
#             value=per_point_features
#         )

#         # attended_features 代表了根据state关注后的场景的“摘要”
#         # 移除序列长度维度: (B, per_point_feat_dim)
#         global_context_vector = attended_features.squeeze(1)
        
#         # 可选: 应用 LayerNorm
#         global_context_vector = self.attention_layernorm(global_context_vector)

#         # 4. 最终投影
#         # 输出形状: (B, final_out_dim)
#         final_output = self.final_projection_mlp(global_context_vector)

#         return final_output # 如果需要，可以返回 attn_output_weights 用于调试
#         # return final_output, attn_output_weights


# class DP3Encoder(nn.Module):
#     def __init__(self,
#                  observation_space: Dict,
#                  img_crop_shape=None, # 在当前实现中未使用，但保留以兼容API
#                  # ---- 新的编码器配置 ----
#                  point_cloud_key: str = 'point_cloud',
#                  state_key: str = 'agent_pos',
#                  imagination_key: str = 'imagin_robot',
#                  point_input_dim: int = 3, # XYZ
#                  use_pc_color: bool = False, # 如果为True, point_input_dim 将变为 6
#                  # state_mlp 用于预处理原始 state_feat
#                  state_mlp_size: Tuple[int, ...] = (64, 64), 
#                  state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
#                  # AdvancedPointNetEncoderWithStateAttention 的参数
#                  adv_pn_per_point_feat_dim: int = 128,
#                  adv_pn_final_out_dim: int = 256, # AdvancedPointNetEncoder的输出维度
#                  adv_pn_num_attn_heads: int = 4,
#                  adv_pn_use_layernorm_in_point_mlp: bool = False,
#                  adv_pn_point_mlp_hidden_dims: List[int] = [64, 128],
#                  # ---- 是否将 state_aware_pc_feat 与 state_feat 再次拼接 ----
#                  concatenate_final_state: bool = False, # 如果为True, 会把 adv_pn_final_out_dim 和 processed_state_dim 拼起来
#                  final_fusion_mlp_arch: Optional[List[int]] = None, # 如果 concatenate_final_state 为 True, 可以再接一个MLP
#                  **kwargs # 用于吸收其他未使用的旧配置参数，如 pointnet_type, pointcloud_encoder_cfg 等
#                 ):
#         super().__init__()
#         self.imagination_key = imagination_key
#         self.state_key = state_key
#         self.point_cloud_key = point_cloud_key

#         self.use_imagined_robot = self.imagination_key in observation_space.keys()
#         if self.state_key not in observation_space:
#             raise ValueError(f"'{self.state_key}' not found in observation_space.")
#         self.state_shape = observation_space[self.state_key]
        
#         actual_point_input_dim = point_input_dim
#         if use_pc_color:
#             actual_point_input_dim = 6 # XYZ + RGB

#         # 1. 预处理原始 state_feat 的 MLP (与原版DP3Encoder类似)
#         # 这个MLP的输出将作为 state_input_dim 输入到 AdvancedPointNetEncoderWithStateAttention
#         if not state_mlp_size or len(state_mlp_size) == 0:
#             raise ValueError("state_mlp_size must be defined and non-empty.")
#         state_mlp_net_arch_list = list(state_mlp_size[:-1]) # Mypy happy
#         processed_state_dim = state_mlp_size[-1]

#         self.state_mlp = nn.Sequential(
#             *create_mlp(self.state_shape[0], processed_state_dim, state_mlp_net_arch_list, state_mlp_activation_fn)
#         )

#         # 2. 实例化新的包含状态注意力的点云编码器
#         self.point_cloud_extractor = AdvancedPointNetEncoderWithStateAttention(
#             point_input_dim=actual_point_input_dim,
#             state_input_dim=processed_state_dim, # 使用 self.state_mlp 的输出维度
#             per_point_feat_dim=adv_pn_per_point_feat_dim,
#             final_out_dim=adv_pn_final_out_dim,
#             num_attn_heads=adv_pn_num_attn_heads,
#             use_layernorm_in_point_mlp=adv_pn_use_layernorm_in_point_mlp,
#             point_mlp_hidden_dims=adv_pn_point_mlp_hidden_dims
#         )

#         # 3. 确定最终输出特征的维度
#         self.concatenate_final_state = concatenate_final_state
#         if self.concatenate_final_state:
#             self._current_feature_dim = adv_pn_final_out_dim + processed_state_dim
#             if final_fusion_mlp_arch and len(final_fusion_mlp_arch) > 0:
#                 self.final_fusion_mlp = nn.Sequential(
#                     *create_mlp(self._current_feature_dim, final_fusion_mlp_arch[-1], final_fusion_mlp_arch[:-1])
#                 )
#                 self._final_feature_dim = final_fusion_mlp_arch[-1]
#             else:
#                 self.final_fusion_mlp = nn.Identity()
#                 self._final_feature_dim = self._current_feature_dim
#         else:
#             # 如果不拼接，最终特征就是点云编码器的输出 (它已经是 state-aware 的了)
#             self._final_feature_dim = adv_pn_final_out_dim
#             self.final_fusion_mlp = nn.Identity()


#         cprint(f"--- [DP3Encoder with State Attention Init] ---", 'green')
#         cprint(f"Point cloud input source dim: {actual_point_input_dim} (color: {use_pc_color})", "yellow")
#         cprint(f"Raw state input dim from obs: {self.state_shape[0]}", "yellow")
#         cprint(f"Processed state dim (output of state_mlp, input to AdvPointNet): {processed_state_dim}", "yellow")
#         cprint(f"AdvancedPointNetEncoder output dim (state-aware pc_feat): {adv_pn_final_out_dim}", "yellow")
#         cprint(f"Concatenate final state with pc_feat: {self.concatenate_final_state}", "yellow")
#         if self.concatenate_final_state and hasattr(self.final_fusion_mlp, 'in_features'): # Check if it's a real MLP
#              cprint(f"Final fusion MLP input dim: {self._current_feature_dim}", "yellow")
#         cprint(f"FINAL DP3Encoder output dim: {self._final_feature_dim}", "red")
#         cprint(f"--- [End DP3Encoder Init] ---", 'green')


#     def forward(self, observations: Dict) -> torch.Tensor:
#         points = observations[self.point_cloud_key]
#         # 预期的点云形状: (Batch, NumPoints, PointDim)
#         assert len(points.shape) == 3, f"Point cloud shape: {points.shape}, should be (B, N, D_point_in)"

#         if self.use_imagined_robot:
#             # 确保 imagination_key 存在并且维度可以对齐
#             if self.imagination_key in observations:
#                 img_points = observations[self.imagination_key][..., :points.shape[-1]] # 对齐最后一个维度
#                 points = torch.cat([points, img_points], dim=1)
#             else:
#                 cprint(f"Warning: use_imagined_robot is True but '{self.imagination_key}' not in observations.", "magenta")


#         raw_state = observations[self.state_key]
#         # 预处理 state: (B, processed_state_dim)
#         processed_state = self.state_mlp(raw_state)

#         # 获取 state-aware 的点云特征: (B, adv_pn_final_out_dim)
#         state_aware_pc_feat = self.point_cloud_extractor(points, processed_state)

#         if self.concatenate_final_state:
#             # 选项：将 state-aware 点云特征与 (已处理的) state 特征再次拼接
#             final_feat_before_fusion_mlp = torch.cat([state_aware_pc_feat, processed_state], dim=-1)
#             final_feat = self.final_fusion_mlp(final_feat_before_fusion_mlp)
#         else:
#             # 选项：直接使用 state-aware 点云特征作为最终特征
#             final_feat = state_aware_pc_feat
            
#         return final_feat

#     def output_shape(self) -> int:
#         return self._final_feature_dim
    




###############################################################################################################################
###############################################################################################################################
###############################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels