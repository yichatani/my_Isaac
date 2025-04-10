import os
# import sys
# sys.path = sorted(sys.path, key=lambda p: "isaac-sim" in p)
# for p in sys.path: print(p)
import torch
import torch
print(torch.__version__)
print(dir(torch.library))

import dill
print("AAAAA")
import hydra
print("BBBBB")
import pathlib
import numpy as np
# from omegaconf import OmegaConf
from diffusion_policy_3d.policy.dp3 import DP3
print("CCCCC")
from diffusion_policy_3d.dataset.my_dataset import IsaacZarrDataset
print("DDDDD")
from diffusion_policy_3d.common.pytorch_util import dict_apply
print("EEEEE")
ROOT_PATH = os.path.dirname(__file__)

def load_model_from_ckpt(ckpt_path):
    print(f"Loading model from checkpoint: {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']

    # overwrite zarr_path if empty or wrong
    zarr_path = cfg.task.dataset.get("zarr_path", "")
    if not zarr_path or not os.path.exists(zarr_path):
        fixed_path = os.path.join(ROOT_PATH, "data/positive_1.zarr")
        print(f"[INFO] Overriding zarr_path to: {fixed_path}")
        cfg.task.dataset.zarr_path = fixed_path

    model: DP3 = hydra.utils.instantiate(cfg.policy)
    state_dict = payload['state_dicts'].get('ema_model', payload['state_dicts']['model'])
    model.load_state_dict(state_dict)

    # load normalizer
    dataset: IsaacZarrDataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    model.set_normalizer(normalizer)

    model.eval().cuda()
    return model, cfg, dataset


def inferernce(data_sample):
    """
    Perform inference using the loaded model and configuration.
    Args:
        data_sample (dict): A dictionary containing the input data for inference.
    Returns:
        np.ndarray: The predicted action as a numpy array.
    """
    ckpt_path = pathlib.Path(ROOT_PATH + "/checkpoints/latest.ckpt")
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    model, cfg, _ = load_model_from_ckpt(ckpt_path)
    if data_sample == None:
        raise ValueError("data_sample is None")
        # obs_dict = {k: v.unsqueeze(0) for k, v in _[0]['obs'].items()}
    else:
        obs_dict = {k: v.unsqueeze(0) for k, v in data_sample['obs'].items()}
    obs_dict = dict_apply(obs_dict, lambda x: x.cuda())

    with torch.no_grad():
        result = model.predict_action(obs_dict)
        action = result['action_pred'].squeeze(0).cpu().numpy()

    print("Predicted action:", action[0])
    return np.array(action[0], dtype=np.float32)


if __name__ == "__main__":
    inferernce(None)
