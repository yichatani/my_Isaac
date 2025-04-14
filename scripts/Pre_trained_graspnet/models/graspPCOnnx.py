import onnxruntime as _ort
import numpy as np
import os
import sys
import torch
import argparse
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
print(ROOT_DIR)
import yaml

from visual.vis_poses import Load_Real_Data_SCENE, view_grasps, Load_Real_Data_meta_dataset

from dataset.graspnet_dataset import minkowski_collate_fn
from models.graspnet import GraspNet, pred_grasp_decode


parser = argparse.ArgumentParser()
# variable in shell
parser.add_argument('--scene_id', type=int, default=0, required=False, help='scene id')
parser.add_argument('--anno_id', type=int, default=0, required=False, help='annoid')
parser.add_argument('--log_dir', default='logs/log')

cfgs = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open('/home/zhy/Grasp_pointcloud/new_structure/models/model_config.yaml', 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

class GraspPcOnnx():
  def __init__(self, onnx_dir):
    providers = ['CPUExecutionProvider']
    options = _ort.SessionOptions()
    options.enable_mem_reuse = False
    self.feat_dim = model_config['Global']['feat_dim']
    self.GRASPNESS_THRESHOLD = model_config['Global']['GRASPNESS_THRESHOLD']
    self.SUCTIONESS_THRESHOLD = model_config['Global']['SUCTIONESS_THRESHOLD']
    self.M_points = model_config['Global']['M_POINT']
    # self.onnx_graspnet = _ort.InferenceSession(f'{onnx_dir}/onnx_graspnet.onnx', providers=providers, sess_options=options)
    self.backbone_onnx = _ort.InferenceSession(f'{onnx_dir}/onnx_backbone.onnx', providers=providers, sess_options=options)
    self.onnx_approach = _ort.InferenceSession(f'{onnx_dir}/onnx_approach.onnx', providers=providers, sess_options=options)
    self.onnx_crop = _ort.InferenceSession(f'{onnx_dir}/onnx_approach.onnx', providers=providers, sess_options=options)
    self.onnx_swad = _ort.InferenceSession(f'{onnx_dir}/onnx_swad.onnx', providers=providers, sess_options=options)
  

  def inference(self, root_path, scene_id, dataset_type):
    # load model
    net = GraspNet(model_config, is_training=False)
    net.to(device)
    net.eval()
    checkpoint_path = os.path.join('..', cfgs.log_dir, 'epoch_59.tar')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    if dataset_type == 'graspnet':
      # load dataset
      dict_data = Load_Real_Data_SCENE(root_path, '0000')

      batch_data = minkowski_collate_fn([dict_data])
      for key in batch_data:
        batch_data[key] = batch_data[key].to(device)
    elif dataset_type == 'meta':
      sample = Load_Real_Data_meta_dataset(scene_id,0)
      batch_data = minkowski_collate_fn([sample])
      for key in batch_data:
          if 'list' in key:
              for i in range(len(batch_data[key])):
                  for j in range(len(batch_data[key][i])):
                      batch_data[key][i][j] = batch_data[key][i][j].to(device)
          else:
              batch_data[key] = batch_data[key].to(device)
    end_points = net.backbone_forward(batch_data)
    end_points = net.graspable_fps(end_points)

    
    
    for key in end_points:
        if 'list' in key:
            for i in range(len(end_points[key])):
                for j in range(len(end_points[key][i])):
                    end_points[key][i][j] = end_points[key][i][j].to(device)
        else:
            end_points[key] = end_points[key].detach().cpu().numpy()
    seed_xyz_graspable = end_points['xyz_graspable']
    seed_features_graspable = end_points['features_graspable']

    

    # check approach net
    appraoch_output_names = ['coors', 'feats', 'quantize2original', 'point_clouds', 'color', 'features', 'objectness_score', 
                             'graspness_score', 'xyz_graspable', 'features_graspable', 'scores_graspable', 'graspable_count_stage1', 'view_score', 
                             'grasp_top_view_xyz', 'grasp_top_view_rot', 'grasp_top_view_inds', 'vp_features']
    end_points = self.onnx_approach.run(appraoch_output_names, end_points)
    approach_end_points_dict = {name: array for name, array in zip(appraoch_output_names, end_points)}


    # check swad model
    #crop 
    seed_features_graspable = torch.tensor(seed_features_graspable + approach_end_points_dict['vp_features'])

    grasp_top_views_rot = approach_end_points_dict['grasp_top_view_rot']
    crop_outputs = net.crop(torch.tensor(seed_xyz_graspable).contiguous().to(device), seed_features_graspable.contiguous().to(device), torch.tensor(grasp_top_views_rot).to(device))
    approach_end_points_dict['vp_features'] = crop_outputs.detach().cpu().numpy()
    swad_output_names = ['coors', 'feats', 'quantize2original', 'point_clouds', 'color', 'features', 'objectness_score', 'graspness_score', 'xyz_graspable', 
                     'features_graspable', 'scores_graspable', 'graspable_count_stage1', 'view_score', 'grasp_top_view_xyz', 'grasp_top_view_rot', 
                     'grasp_top_view_inds', 'vp_features', 'grasp_score_pred', 'grasp_width_pred']
    swad_outputs = self.onnx_swad.run(swad_output_names, approach_end_points_dict)

    
    swad_end_points_dict = {name: torch.from_numpy(array) for name, array in zip(swad_output_names, swad_outputs)}
    grasp_preds = pred_grasp_decode(swad_end_points_dict)
    

    outputs = {
       'end_points':swad_end_points_dict,
       'grasps':grasp_preds,
       'approach_outputs':approach_end_points_dict
    }
    return outputs


    
  def compare_original_and_onnx(self, end_points_dict, end_points_copy, res_copy):
    for key in end_points_dict.keys() :
      output_a = end_points_dict[key].numpy()
      if key != 'res_features':
        
        output_b = end_points_copy[key].detach().cpu().numpy()
        self.check_array_equality(output_a, output_b)
      else:
        output_b = res_copy.detach().cpu().numpy()
        self.check_array_equality(output_a, output_b)




  
  def view_grasps(self,  outputs):
    grasp_preds = outputs['grasps'][0]
    end_points = outputs['end_points']
    for key in end_points:
        end_points[key] = end_points[key].numpy()
    preds = grasp_preds.detach().cpu().numpy()
    view_grasps(grasps=preds, end_points=end_points)

  def check_array_equality(self, output_a, output_b):
    if not np.allclose(output_a, output_b, rtol=1e-03, atol=1e-06):
      print("The arrays are not the same!")
      
      # Calculate the differences
      difference = output_a - output_b

      # Print the difference
      print("Difference between arrays:")
      print(difference)

      # Print indices where arrays differ significantly
      indices = np.where(np.abs(difference) > 1e-05)
      print("Indices with significant differences:", indices)

      # Print values at those indices
      for idx in zip(*indices):
          print(f"onnx_outputs{idx} = {output_a[idx]}")
          print(f"original_outputs{idx} = {output_b[idx]}")
          print(f"difference = {difference[idx]}")
      return False
    
    return True
    
    


if __name__ == '__main__':

  scene_id = cfgs.scene_id
  file_id = '0000'
  if scene_id / 10 >= 1 and scene_id / 10 < 10:
    file_id = '00' + str(scene_id)
  elif scene_id/10 >= 10 and scene_id / 10 <100:
    file_id = '0' + str(scene_id)
  else:
    file_id = '000' + str(scene_id)

  onnx_dir = '/home/zhy/Grasp_pointcloud/new_structure/models/onnx_models'
  data_path = '/media/zhy/Data2TB1/GraspNet1B/scenes/scene_'+ file_id + '/realsense/'
  grasp_onnx = GraspPcOnnx(onnx_dir)
  outputs = grasp_onnx.inference(data_path, '0000')
  outputs = grasp_onnx.inference_meta(cfgs.scene_id, cfgs.anno_id)
  grasp_onnx.view_grasps(outputs=outputs)
  

  