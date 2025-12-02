# ==========================================================
# Policy Server (DIRECT → ISAAC)
#   - REP: receive obs from SegServer
#   - Inference
#   - REQ: send action directly to Isaac
# ==========================================================

import zmq
import numpy as np
import torch
import torch.nn as nn
import hydra
import logging

log = logging.getLogger(__name__)

# ==========================================================
# Network config (★★重要：IP 在这里统一管理★★)
# ==========================================================
SEG_BIND_ADDR   = "tcp://192.168.56.56:5555"              # Seg → Policy
ISAAC_ACTION_ADDR = "tcp://192.168.56.56:5557"  # Policy → Isaac

# ==========================================================
# Normalizer
# ==========================================================
class LinearNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stats = nn.ParameterDict()

    def _normalize(self, x, key):
        s = self.stats[key]
        x = (x - s["min"]) / (s["max"] - s["min"] + 1e-6)
        return 2.0 * x - 1.0

    def _denormalize(self, x, key):
        s = self.stats[key]
        x = (x + 1.0) / 2.0
        return x * (s["max"] - s["min"]) + s["min"]

    def forward(self, x, key, forward=True):
        return self._normalize(x, key) if forward else self._denormalize(x, key)

    def _turn_off_gradients(self):
        for k in self.stats:
            for sk in self.stats[k]:
                self.stats[k][sk].requires_grad = False


# ==========================================================
# Policy Server
# ==========================================================
class PolicyServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        # ----------------------------
        # Model
        # ----------------------------
        self.model = hydra.utils.instantiate(cfg.model).to(self.device)
        self.model.eval()

        # ----------------------------
        # Normalizer
        # ----------------------------
        self.normalizer = LinearNormalizer().to(self.device)
        self._load_normalizer(cfg.normalization_path)

        # ----------------------------
        # Weights
        # ----------------------------
        self._load_checkpoint(cfg.base_policy_path)

        # ----------------------------
        # ZMQ
        # ----------------------------
        self.ctx = zmq.Context.instance()

        # ← Seg server → Policy
        self.seg_sock = self.ctx.socket(zmq.REP)
        self.seg_sock.bind(SEG_BIND_ADDR)

        # → Isaac (direct)
        self.isaac_sock = self.ctx.socket(zmq.REQ)
        self.isaac_sock.connect(ISAAC_ACTION_ADDR)

        print(f"[Policy] REP bind on {SEG_BIND_ADDR}")
        print(f"[Policy] REQ connect to Isaac at {ISAAC_ACTION_ADDR}")

    # ==================================================
    # Main loop
    # ==================================================
    def serve(self):
        print("[Policy] Policy Server started")

        while True:
            # ------------------------------------------------
            # 1. recv obs from Seg
            # ------------------------------------------------
            obs = self.seg_sock.recv_pyobj()

            try:
                action = self.infer_from_obs(obs)
            except Exception as e:
                print("[Policy] Inference error:", e)
                action = np.zeros(
                    (self.cfg.act_steps, self.cfg.action_dim),
                    dtype=np.float32
                )

            # ------------------------------------------------
            # 2. ACK Seg immediately（不走 action）
            # ------------------------------------------------
            self.seg_sock.send(b"ok")

            # ------------------------------------------------
            # 3. send action directly → Isaac
            # ------------------------------------------------
            self.isaac_sock.send_pyobj(action)
            self.isaac_sock.recv()  # wait for Isaac ACK

    # ==================================================
    # Inference
    # ==================================================
    def infer_from_obs(self, obs):
        with torch.no_grad():
            # # ---- state ----
            # state = torch.from_numpy(obs["state"]).float().to(self.device)
            # state = self.normalizer(state, "states", forward=True)
            # state = state.unsqueeze(0)  # (1,T,D)

            # ---- image ----
            print("AAAAAA")
            img = torch.from_numpy(obs).float()
            print("BBBBBB")
            img = img.permute(0, 3, 1, 2)  # T,C,H,W
            print("CCCCCC")
            img = img.unsqueeze(0).to(self.device)
            print("DDDDDD")
            cond = {"rgb": img}

            print("EEEEEE")
            # ---- model ----
            samples = self.model.sample(cond, self.cfg.denoising_steps)

            print("FFFFFF")
            action = samples.trajectories[0]
            print("GGGGGG")
            action = self.normalizer(action, "actions", forward=False)
            print("HHHHHH")
        print(f"{action.cpu().numpy()=}")
        return action.cpu().numpy()

    # ==================================================
    # Utils
    # ==================================================
    def _load_checkpoint(self, ckpt):
        data = torch.load(ckpt, map_location=self.device, weights_only=True)
        if "ema" in data:
            self.model.load_state_dict(data["ema"], strict=False)
            print("[Policy] EMA loaded")
        elif "policy" in data:
            self.model.load_state_dict(data["policy"], strict=False)
            print("[Policy] Policy loaded")
        else:
            raise RuntimeError("Checkpoint format not supported")

    def _load_normalizer(self, path):
        norm = np.load(path)
        self.normalizer.stats = nn.ParameterDict({
            "states": nn.ParameterDict({
                "min": nn.Parameter(torch.tensor(norm["obs_min"], device=self.device), requires_grad=False),
                "max": nn.Parameter(torch.tensor(norm["obs_max"], device=self.device), requires_grad=False),
            }),
            "actions": nn.ParameterDict({
                "min": nn.Parameter(torch.tensor(norm["action_min"], device=self.device), requires_grad=False),
                "max": nn.Parameter(torch.tensor(norm["action_max"], device=self.device), requires_grad=False),
            }),
        })
        self.normalizer._turn_off_gradients()

    def run(self):
        """
        Hydra / launcher compatibility entrypoint.
        """
        self.serve()


# ==========================================================
# main
# ==========================================================
@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg):
    server = PolicyServer(cfg)
    server.serve()


if __name__ == "__main__":
    main()
