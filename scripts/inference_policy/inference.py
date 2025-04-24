try:
    import os
    import torch
    import dill
    import hydra
    import pathlib
    import numpy as np
    from diffusion_policy_3d.policy.dp3 import DP3
    from diffusion_policy_3d.common.pytorch_util import dict_apply
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval)
except:
    raise ImportError("inference.py import error, please check your environment.")
ROOT_PATH = os.path.dirname(__file__)


def check_pickles_keys():
    payload = torch.load(ROOT_PATH + "/checkpoints/latest.ckpt", pickle_module=dill, map_location="cpu")
    print("pickles keys:", payload.get("pickles", {}).keys())


def load_model_from_ckpt(ckpt_path):
    print(f"Loading model from checkpoint: {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    # print(payload.keys())
    cfg = payload['cfg']

    model: DP3 = hydra.utils.instantiate(cfg.policy)
    state_dict = payload['state_dicts'].get('ema_model', payload['state_dicts']['model'])
    model.load_state_dict(state_dict)

    with open(ROOT_PATH + "/checkpoints/dp3_cube_normalizer.pkl", "rb") as f:
        normalizer = dill.load(f)
    model.set_normalizer(normalizer)
    # if 'pickles' in payload and 'normalizer' in payload['pickles']:
    #     normalizer = dill.loads(payload['pickles']['normalizer'])
    #     model.set_normalizer(normalizer)
    #     print("Loaded normalizer from checkpoint.")
    # else:
    #     raise ValueError("Normalizer not found in checkpoint.")

    model.eval().cuda()
    print(f"Model loaded successfully from {ckpt_path}")
    return model


def inference_policy(data_sample,obs_steps=3,action_steps=6):
    """
    Perform inference using the loaded model and configuration.
    Args:
        data_sample (dict): A dictionary containing the input data for inference.
    Returns:
        np.ndarray: The predicted action as a numpy array.
    """
    ckpt_path = pathlib.Path(ROOT_PATH + "/checkpoints/dp3_cube.ckpt")
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    model = load_model_from_ckpt(ckpt_path)
    if data_sample == None:
        raise ValueError("data_sample is None")
    else:
        assert len(data_sample['obs']['agent_pos']) == obs_steps, "agent_pos should be of length obs_steps"
        data_tensor = data_sample
        data_tensor['obs'] = {
            'agent_pos': torch.tensor(data_tensor['obs']['agent_pos']),        # -> [1, obs_steps, 7]
            'point_cloud': torch.tensor(data_tensor['obs']['point_cloud'])     # -> [1, obs_steps, N, 6]
        }
        obs_dict = {k: v.unsqueeze(0) for k, v in data_tensor['obs'].items()}
    obs_dict = dict_apply(obs_dict, lambda x: x.cuda())

    with torch.no_grad():
        result = model.predict_action(obs_dict)
        action = result['action_pred'].squeeze(0).cpu().numpy()

    print("Action shape:", action.shape)
    print("Predicted action:", action[1:action_steps])
    return np.array(action[1:action_steps], dtype=np.float32)


if __name__ == "__main__":
    inference_policy(None)
