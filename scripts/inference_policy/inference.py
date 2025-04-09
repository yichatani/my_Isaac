import os
import sys
# sys.path = sorted(sys.path, key=lambda p: "isaac-sim" in p)
# for p in sys.path: print(p)
import torch
import dill
import hydra
import argparse
import pathlib
import numpy as np
from omegaconf import OmegaConf
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.my_dataset import IsaacZarrDataset
from diffusion_policy_3d.common.pytorch_util import dict_apply
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


def prepare_obs_dict(dataset, index=0):
    item = dataset[index]

    # print(item)
    # exit()
    
    obs_dict = item['obs']
    obs_dict = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
    return obs_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=ROOT_PATH + "/checkpoints/latest.ckpt", help='Path to checkpoint file (e.g. checkpoints/latest.ckpt)')
    parser.add_argument('--index', type=int, default=0, help='Index of sample from dataset to use')
    parser.add_argument('--save_action', type=str, default=None, help='Optional path to save predicted action as .npy')
    args = parser.parse_args()

    ckpt_path = pathlib.Path(args.ckpt)
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    model, cfg, dataset = load_model_from_ckpt(ckpt_path)
    obs_dict = prepare_obs_dict(dataset, index=args.index)
    obs_dict = dict_apply(obs_dict, lambda x: x.cuda())

    with torch.no_grad():
        result = model.predict_action(obs_dict)
        action = result['action_pred'].squeeze(0).cpu().numpy()

    print("Predicted action:", action)
    if args.save_action:
        np.save(args.save_action, action)
        print(f"Saved predicted action to: {args.save_action}")


if __name__ == "__main__":
    main()
