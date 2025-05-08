try:
    import os
    import torch
    import dill
    import hydra
    import pathlib
    import numpy as np
    from diffusion_policy_3d.policy.ani_dp3 import DP3
    from diffusion_policy_3d.dataset.my_dataset import IsaacZarrDataset
    from diffusion_policy_3d.model.phase_encoder import PhaseEncoder
    from diffusion_policy_3d.common.pytorch_util import dict_apply
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval)
except:
    raise ImportError("inference.py import error, please check your environment.")
ROOT_PATH = os.path.dirname(__file__)
ROOT_DIR = str(pathlib.Path(__file__).parent.resolve())


# def check_pickles_keys():
#     payload = torch.load(ROOT_PATH + "/checkpoints/phase_latent_p.ckpt", pickle_module=dill, map_location="cpu")
#     # print("pickles keys:", payload.get("pickles", {}).keys())
#     print(payload['cfg'].get("policy",{}).keys())
# check_pickles_keys()
# exit()

def preprocess_observation(obs_dict):
    if 'agent_pos' not in obs_dict:
        raise KeyError(f"'agent_pos' not found in observation dictionary.")
    obs_tensor = obs_dict['agent_pos']  # (B, T, 7)
    assert obs_tensor.dim() == 3, f"Expected 3D tensor for agent_pos, got {obs_tensor.dim()}D."
    return obs_tensor

def load_model_from_ckpt(ckpt_path):
    print(f"Loading model from checkpoint: {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    # print(payload.keys())
    cfg = payload['cfg']

    phase_encoder = PhaseEncoder(
            input_dim=cfg.task.phase_encoder.input_dim,
            latent_dim=cfg.task.phase_encoder.latent_dim 
        ).eval().cuda()

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
    return model, phase_encoder


def inference_policy(data_sample,obs_steps=3,action_steps=6):
    """
    Perform inference using the loaded model and configuration.
    Args:
        data_sample (dict): A dictionary containing the input data for inference.
    Returns:
        np.ndarray: The predicted action as a numpy array.
    """
    ckpt_path = pathlib.Path(ROOT_PATH + "/checkpoints/phase_latent_p.ckpt")
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    model,phase_encoder = load_model_from_ckpt(ckpt_path)
    if data_sample == None:
        raise ValueError("data_sample is None")
    else:
        assert len(data_sample['obs']['agent_pos']) == obs_steps, "agent_pos should be of length obs_steps"
        data_tensor = data_sample
        data_tensor['obs'] = {   
            'agent_pos': torch.tensor(data_tensor['obs']['agent_pos'], dtype=torch.float32),    # -> [1, obs_steps, 7]
            'point_cloud': torch.tensor(data_tensor['obs']['point_cloud'], dtype=torch.float32) # -> [1, obs_steps, N, 6]
        }
        obs_dict = {k: v.unsqueeze(0) for k, v in data_tensor['obs'].items()}
    obs_dict = dict_apply(obs_dict, lambda x: x.cuda())

    with torch.no_grad():
        obs_tensor = preprocess_observation(obs_dict)
        phase_latent = phase_encoder(obs_tensor)
        result = model.predict_action(obs_dict,phase_latent=phase_latent)
        action = result['action_pred'].squeeze(0).cpu().numpy()

    print("Action shape:", action.shape)
    print("Predicted action:", action[-action_steps:])
    return np.array(action[-action_steps:], dtype=np.float32)


if __name__ == "__main__":
    inference_policy(None)
