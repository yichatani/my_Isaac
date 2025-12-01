# policy_client.py
import zmq
import time
import numpy as np

class PolicyClient:
    def __init__(self, addr="tcp://127.0.0.1:5556"):
        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.REQ)
        self.socket.connect(addr)

    def infer(self, obs):
        """
        obs: dict / np.ndarray
        return: np.ndarray
        """
        self.socket.send_pyobj(obs)
        action = self.socket.recv_pyobj()
        return action

if __name__ == '__main__':
    # -------------------------
    # 1. 创建 policy client
    # -------------------------
    policy = PolicyClient("tcp://127.0.0.1:5556")

    # -------------------------
    # 2. 假装仿真参数
    # -------------------------
    CONTROL_HZ = 20
    DT = 1.0 / CONTROL_HZ
    T = 0

    print("[FakeSim] Start fake simulation loop")

    # -------------------------
    # 3. fake sim loop
    # -------------------------
    while True:
        # =====================
        # fake observation
        # =====================
        joint_pos = np.sin(np.linspace(0, 1, 7) + T).astype(np.float32)
        joint_vel = np.cos(np.linspace(0, 1, 7) + T).astype(np.float32)
        pointcloud = np.random.randn(1024, 3).astype(np.float32)

        obs = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "pointcloud": pointcloud,
        }

        # =====================
        # policy inference
        # =====================
        t0 = time.time()
        action = policy.infer(obs)
        infer_dt = (time.time() - t0) * 1000

        # =====================
        # fake apply action
        # =====================
        print(f"[FakeSim] t={T:6.3f}  action={action}  infer={infer_dt:.2f}ms")

        # =====================
        # step time
        # =====================
        time.sleep(DT)
        T += DT