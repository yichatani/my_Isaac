import os
import time
import zmq
import cv2
import numpy as np
from collections import deque
from isaacsim import SimulationApp

# =====================================================================
# Simulation App
# =====================================================================
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation
from omni.isaac.sensor import Camera
from isaacsim.core.utils.types import ArticulationActions
from omni.isaac.motion_generation import LulaKinematicsSolver
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats, quats_to_rot_matrices

# =====================================================================
# Config
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
asset_path = ROOT_DIR + "/franka_manipulation.usd"

urdf_path = (
    "/home/ani/isaacsim/exts/isaacsim.robot_motion.motion_generation/"
    "motion_policy_configs/FR3/fr3.urdf"
)
robot_description_path = (
    "/home/ani/isaacsim/exts/isaacsim.robot_motion.motion_generation/"
    "motion_policy_configs/FR3/rmpflow/fr3_robot_description.yaml"
)

camera_prim_path = "/fr3/fr3_hand_tcp/hand"

ZMQ_ADDR = "tcp://127.0.0.1:5556"

TARGET_H, TARGET_W = 448, 448

OBS_STEPS = 2
IMG_STEPS = 2
ACTION_HORIZON = 4

# =====================================================================
# Policy Client
# =====================================================================
class PolicyClient:
    def __init__(self, addr):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.connect(addr)
        print(f"[PolicyClient] connected to {addr}")

    def infer(self, obs):
        self.sock.send_pyobj(obs)
        return self.sock.recv_pyobj()  # (H, 8)

# =====================================================================
# Camera
# =====================================================================
def init_camera(path):
    cam = Camera(path, frequency=60, resolution=(1920, 1080))
    cam.initialize()
    return cam

def get_rgb(cam):
    frame = cam.get_current_frame()
    if frame is None:
        return None
    rgba = frame.get("rgba", None)
    if rgba is None or rgba.shape[0] == 0:
        return None
    rgb = rgba[:, :, :3]
    return cv2.resize(rgb, (TARGET_W, TARGET_H)).astype(np.uint8)

# =====================================================================
# Robot State
# =====================================================================
def get_tcp_pose(art, ik):
    q = art.get_joint_positions().squeeze()
    pos, rot = ik.compute_forward_kinematics("fr3_hand_tcp", q[:7])
    quat = rot_matrices_to_quats(np.array(rot).reshape(1, 3, 3))[0]
    width = q[7] + q[8]
    return np.concatenate([np.array(pos), quat, [width]]).astype(np.float32)

# =====================================================================
# Apply delta-pose action (LOCAL TCP FRAME)
# =====================================================================
def apply_delta_action(art, ik, delta):
    q = art.get_joint_positions().squeeze()

    pos, rot = ik.compute_forward_kinematics("fr3_hand_tcp", q[:7])
    pos = np.array(pos)
    quat = rot_matrices_to_quats(np.array(rot).reshape(1, 3, 3))[0]

    dp = delta[:3]
    dquat = delta[3:7]

    target_pos = pos + dp
    target_quat = quat + dquat
    target_quat /= np.linalg.norm(target_quat + 1e-8)

    q_target = ik.compute_inverse_kinematics(
        "fr3_hand_tcp", target_pos, target_quat
    )[0]

    cmd = q.copy()
    cmd[:7] = q_target
    art.apply_action(ArticulationActions(joint_positions=cmd))

# =====================================================================
# Main Loop
# =====================================================================
def main():
    open_stage(asset_path)

    sim = SimulationContext()
    sim.initialize_physics()

    art = Articulation("/fr3")
    art.initialize()

    ik = LulaKinematicsSolver(
        urdf_path=urdf_path,
        robot_description_path=robot_description_path,
    )

    camera = init_camera(camera_prim_path)

    policy = PolicyClient(ZMQ_ADDR)

    state_buf = deque(maxlen=OBS_STEPS)
    img_buf = deque(maxlen=IMG_STEPS)

    sim.play()

    print("[Isaac] Policy-driven control started")

    while sim.is_playing():

        rgb = get_rgb(camera)
        pose = get_tcp_pose(art, ik)

        if rgb is None:
            sim.step(render=True)
            continue

        state_buf.append(pose)
        img_buf.append(rgb)

        if len(state_buf) < OBS_STEPS or len(img_buf) < IMG_STEPS:
            sim.step(render=True)
            continue

        obs = {
            "state": np.stack(state_buf, axis=0),   # (2,8)
            "image": np.stack(img_buf, axis=0),     # (2,H,W,3)
        }

        action_seq = policy.infer(obs)  # (H,8)
        delta = action_seq[0]

        apply_delta_action(art, ik, delta)
        sim.step(render=True)

    sim.stop()
    simulation_app.close()

# =====================================================================
if __name__ == "__main__":
    main()
