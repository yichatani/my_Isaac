try:
    import os
    import torch
    import dill
    import hydra
    import pathlib
    import numpy as np
    from diffusion_policy_3d.policy.dp3 import DP3
    from diffusion_policy_3d.dataset.my_dataset import IsaacZarrDataset
    from diffusion_policy_3d.common.pytorch_util import dict_apply
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval)
except:
    raise ImportError("inference.py import error, please check your environment.")
ROOT_PATH = os.path.dirname(__file__)
ROOT_DIR = str(pathlib.Path(__file__).parent.resolve())


def load_model_from_ckpt(ckpt_path):
    print(f"Loading model from checkpoint: {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    # print(payload.keys())
    cfg = payload['cfg']

    # print("########")
    print(cfg.task.dataset)
    # exit()
    model: DP3 = hydra.utils.instantiate(cfg.policy)
    state_dict = payload['state_dicts'].get('ema_model', payload['state_dicts']['model'])
    model.load_state_dict(state_dict)

    cfg.task.dataset.zarr_path = os.path.join(ROOT_DIR, cfg.task.dataset.zarr_path)
    # print(cfg.task.dataset)
    dataset: IsaacZarrDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    model.set_normalizer(normalizer)

    model.eval().cuda()
    print(f"Model loaded successfully from {ckpt_path}")
    return model

def prepare_inference_data(data_sample: dict):
    """
        Organize data
    """
    agent_pos = np.array(data_sample['obs']['agent_pos'], dtype=np.float32)  # (T, 7)
    point_cloud = np.array(data_sample['obs']['point_cloud'], dtype=np.float32)  # (T, N, 6)
    # delta = np.array(data_sample['obs']['delta'], dtype=np.float32)  # (T, 7)

    return {
        'obs': {
            'agent_pos': agent_pos,
            'point_cloud': point_cloud,
            # 'delta': delta,
        }
    }


def inference_policy(data_sample,action_steps=6):
    """
    Perform inference using the loaded model and configuration.
    Args:
        data_sample (dict): A dictionary containing the input data for inference.
    Returns:
        np.ndarray: The predicted action as a numpy array.
    """
    ckpt_path = pathlib.Path(ROOT_PATH + "/checkpoints/cup.ckpt")
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    model = load_model_from_ckpt(ckpt_path)
    data_dict = prepare_inference_data(data_sample)
    obs_dict = dict_apply(data_dict['obs'], lambda x: torch.tensor(x).unsqueeze(0).cuda())


    with torch.no_grad():
        result = model.predict_action(obs_dict)
        action = result['action_pred'].squeeze(0).cpu().numpy()

    print("Action shape:", action.shape)
    print("Predicted action:", action[-action_steps:])
    return np.array(action[-action_steps:], dtype=np.float32)


if __name__ == "__main__":
    inference_policy(None)
