# from graspnet import GraspNet
import os
import sys
import yaml
import argparse
import numpy as np
# from graspnet import GraspNet


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from dataset.graspnet_dataset import minkowski_collate_fn
from tqdm import tqdm

print(ROOT_DIR)
import torch


from graspnet import GraspNet



parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='logs/log', required=False)
 # corresp to evaluate func
cfgs = parser.parse_args()
with open('model_config.yaml', 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
M_points = model_config['Global']['M_POINT']
M_points = model_config['Global']['M_POINT']
GRASPNESS_THRESHOLD = model_config['Global']['GRASPNESS_THRESHOLD']


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

net = GraspNet(model_config, is_training=False)
checkpoint_path = os.path.join('..', cfgs.log_dir, 'epoch_59.tar')
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']



outpath = '/home/zhy/Grasp_pointcloud/new_structure/models/onnx_models/'



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
net.eval()

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

# convert the model in complete mode
# torch.onnx.export(net, dummy_inputs, f'{outpath}onnx_graspnet.onnx',   input_names = ['coors', 'feats', 'quantize2original','point_clouds', 'color'],
#                   output_names = ['coors','feats', 'quantize2original', 'point_clouds', 'color', 'features', 'objectness_score',  'graspness_score',
#                                   'xyz_graspable', 'features_graspable', 'scores_graspable', 'graspable_count_stage1', 'view_score', 'grasp_top_view_xyz',
#                                   'grasp_top_view_rot', 'vp_features', 'grasp_top_view_inds', 'grasp_score_pred', 'grasp_width_pred'],
#                                   dynamic_axes = {'coors': {0: 'sequence_length'}, 'feats':{0: 'sequence_length'},
#                                                   'quantize2original':{0: 'sequence_length'}, 
#                                                   'point_clouds': {0: 'batch_size', 1: 'sequence_length'}, 
#                                                   'color': {0: 'batch_size', 1: 'sequence_length'},
#                                                   'features':{0: 'batch_size', 1: 'sequence_length', 2:'width'},
#                                                   'objectness_score':{0: 'batch_size', 1: 'sequence_length', 2:'width'},
#                                                   'graspness_score':{0: 'batch_size', 1: 'sequence_length', 2:'width'},
#                                                   'xyz_graspable':{0: 'batch_size', 1: 'sequence_length', 2:'width'},
#                                                   'features_graspable':{0: 'batch_size', 1: 'sequence_length', 2:'width'},
#                                                   'scores_graspable':{0: 'batch_size', 1: 'sequence_length', 2:'width'},
#                                                   'graspable_count_stage1':{0: 'sequence_length'},
#                                                   'view_score':{0: 'batch_size', 1: 'sequence_length', 2:'width'},
#                                                   'grasp_top_view_xyz':{0: 'batch_size', 1: 'sequence_length', 2:'width'},
#                                                   'grasp_top_view_rot':{0: 'batch_size', 1: 'sequence_length', 2:'width'},
#                                                   'vp_features':{0: 'batch_size', 1: 'sequence_length'},
#                                                   'grasp_top_view_inds':{0: 'batch_size', 1: 'sequence_length', 2:'width'},
#                                                   'grasp_score_pred':{0: 'batch_size', 1: 'sequence_length', 2:'width'},
#                                                   'grasp_width_pred':{0: 'batch_size', 1: 'sequence_length', 2:'width'}}, 
#                                                   opset_version=12, verbose = False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX, do_constant_folding=True)

# convert the model in split mode:

net.export_to_onnx(outpath, device)





